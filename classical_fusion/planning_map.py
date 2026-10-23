"""Conservative planning export from one measured TSDF.

Unknown is not free. Bounds require calibrated depth, resolved obstacles and a
static scene between explicit, conservatively bounded local change events.
Unannounced dynamic obstacles are not covered. Not a physical certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from threading import RLock

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.ndimage import distance_transform_edt

from .active_tsdf import ActiveTSDF, ActiveTSDFConfig


@dataclass(frozen=True)
class PlanningMapConfig(ActiveTSDFConfig):
    voxel_size_m: float = .004
    truncation_m: float = .012
    surface_min_weight: float = 2.
    view_grid_stride: int = 5
    depth_edge_m: float = .025
    robust_delta_m: float = .006
    unknown_exploration_weight: float = .15
    # Assumed bounds to be independently measured on the actual camera/arm.
    depth_error_bound_m: float = .008
    evidence_max_depth_m: float = 4.
    extrinsic_translation_bound_m: float = .003
    extrinsic_rotation_bound_rad: float = np.deg2rad(.2)
    intrinsic_pixel_bound: float = 1.
    free_patch_radius_px: int = 8
    minimum_free_views: int = 2
    independent_view_translation_m: float = .04
    independent_view_rotation_rad: float = np.deg2rad(5.)
    planning_margin_m: float = .005
    snapshot_max_age_s: float = 5.

    def validate(self):
        super().validate()
        bounds=(self.depth_error_bound_m,self.extrinsic_translation_bound_m,
                self.extrinsic_rotation_bound_rad,self.intrinsic_pixel_bound,self.planning_margin_m)
        if not np.isfinite(bounds).all() or min(bounds)<0:
            raise ValueError('invalid planning error budget')
        if (not 1<=self.minimum_free_views<=255 or self.free_patch_radius_px<1
                or self.snapshot_max_age_s<=0 or self.independent_view_translation_m<=0
                or self.independent_view_rotation_rad<=0):
            raise ValueError('invalid planning evidence settings')
        if self.evidence_max_depth_m<self.max_depth_m:
            raise ValueError('evidence range must cover the reconstruction depth range')


class PlanningTSDF(ActiveTSDF):
    def __init__(self, config=None, device='cpu'):
        self._lock=RLock()
        super().__init__(config or PlanningMapConfig(),device)
        self.free_views=torch.zeros(self.shape,dtype=torch.uint8,device=self.device)
        self.surface_veto=torch.zeros(self.shape,dtype=torch.bool,device=self.device)
        self.evidence_poses=[]
        self.evidence_epochs=[]
        self.change_epoch=0
        self.cell_change_epoch=torch.zeros(self.shape,dtype=torch.int32,device=self.device)
        self.latest_observation_time=None

    def reset(self):
        with self._lock:
            super().reset()
            self.free_views.zero_(); self.surface_veto.zero_(); self.evidence_poses.clear()
            self.evidence_epochs.clear(); self.change_epoch=0; self.cell_change_epoch.zero_()
            self.latest_observation_time=None

    def invalidate_region(self,lower,upper,*,evidence_lower=None,evidence_upper=None,evidence_vertices=None):
        """Retire a bounded change support, with optional measured convex hull.

        Both occupied AND free evidence become unknown. Nothing is marked free
        merely because a grasp succeeded. Unchanged cells and their independent
        view counts are retained. Bounds must cover shape + pose error + TSDF
        truncation; unmodelled scene motion is outside this action-local model.
        Optional containment vertices refine physical support inside its AABB;
        the distance influence band and contradiction retirement stay intact.
        """
        lo=np.asarray(lower,dtype=float);hi=np.asarray(upper,dtype=float)
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(lo>=hi):
            raise ValueError('expected finite ordered region bounds [3]')
        if (evidence_lower is None)!=(evidence_upper is None):raise ValueError('both physical evidence bounds required')
        physical_lo=lo if evidence_lower is None else np.asarray(evidence_lower,float)
        physical_hi=hi if evidence_upper is None else np.asarray(evidence_upper,float)
        if (physical_lo.shape!=(3,) or physical_hi.shape!=(3,) or not np.isfinite([physical_lo,physical_hi]).all()
            or np.any(physical_lo>=physical_hi) or np.any(physical_lo<lo) or np.any(physical_hi>hi)):
            raise ValueError('physical change bounds must be finite and inside distance-update bounds')
        equations=None
        if evidence_vertices is not None:
            from scipy.spatial import ConvexHull,QhullError
            vertices=np.asarray(evidence_vertices,float)
            if (vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(vertices).all()
                or np.any(vertices<physical_lo-1e-9) or np.any(vertices>physical_hi+1e-9)):
                raise ValueError('finite containment vertices inside physical bounds required')
            try:equations=ConvexHull(vertices).equations
            except QhullError as exc:raise ValueError('nondegenerate physical containment hull required') from exc
        with self._lock:
            a=torch.as_tensor(lo,device=self.device,dtype=self.centers.dtype)
            b=torch.as_tensor(hi,device=self.device,dtype=self.centers.dtype)
            mask=((self.centers+self.spacing*.5>=a)&(self.centers-self.spacing*.5<=b)).all(1).reshape(self.shape)
            pa=torch.as_tensor(physical_lo,device=self.device,dtype=self.centers.dtype)
            pb=torch.as_tensor(physical_hi,device=self.device,dtype=self.centers.dtype)
            physical=((self.centers+self.spacing*.5>=pa)&(self.centers-self.spacing*.5<=pb)).all(1).reshape(self.shape)
            if equations is not None:
                # A face separating the WHOLE voxel from the measured padded
                # containment hull proves that its old free evidence need not
                # be retired merely because it lies in an AABB corner. Missing
                # edge-axis tests only OVER-retire, never under-retire cells.
                ids=torch.where(physical.view(-1))[0]
                # NumPy-derived ROI bounds can make spacing float64 while
                # torch grid centers remain float32. Use one explicit type for
                # geometric support arithmetic, independent of input origin.
                planes=torch.as_tensor(equations,device=self.device,dtype=torch.float64)
                half_spacing=self.spacing.to(torch.float64)*.5
                for chunk in planes.split(24):
                    if not len(ids):break
                    normals=chunk[:,:3]
                    intersects=(self.centers[ids].to(torch.float64)@normals.T+chunk[:,3]
                        -normals.abs()@half_spacing<=1e-6).all(1)
                    physical.view(-1)[ids[~intersects]]=False;ids=ids[intersects]
            # Resetting a nonpositive TSDF must not expose stale free votes and
            # promote a previously blocked cell. Such contradictions also lose
            # their free evidence, even outside the physical containment bound.
            physical |= mask&(self.weight>0)&(self.distance<=0)
            known=(self.weight>0)|(self.free_views>0)|self.surface_veto
            before=int(known.sum());retained_free=int((mask&~physical&(self.free_views>0)).sum())
            self.revision+=1;self.change_epoch+=1
            self.distance[mask]=self.cfg.truncation_m;self.weight[mask]=0
            # The TSDF influence band is NOT the physical motion/change set.
            # Outside a measured uncertainty-inclusive containment bound,
            # old free votes and obstacle vetoes remain valid in a static world.
            self.free_views[physical]=0;self.surface_veto[physical]=False
            self.cell_change_epoch[physical]=self.change_epoch
            after=int(((self.weight>0)|(self.free_views>0)|self.surface_veto).sum());removed=before-after
            return dict(cells_invalidated=int(physical.sum()),distance_cells_invalidated=int(mask.sum()),
                        physical_support='measured_convex_containment' if equations is not None else 'AABB',
                        free_evidence_preserved_in_distance_band=retained_free,known_before=before,
                        known_invalidated=removed,known_preserved=before-removed,
                        revision=self.revision,change_epoch=self.change_epoch)

    def integrate(self,*args,**kwargs):
        with self._lock:
            # Invalidate old snapshots BEFORE any update, including rejected
            # input, so concurrent queries cannot use a half-updated map.
            self.revision+=1
            return super().integrate(*args,**kwargs)

    def _on_depth_projection(self,depth,intrinsics,pose,u,v,z,ids):
        cfg=self.cfg
        self.latest_observation_time=time.monotonic()
        # Mark contradicted/near-surface cells on EVERY frame, even at a
        # repeated pose. Never let averaging erase evidence of an obstacle.
        ui,vi=u[ids].round().long(),v[ids].round().long()
        measured=depth[vi,ui]
        valid=torch.isfinite(measured)&(measured>=cfg.min_depth_m)&(measured<=cfg.evidence_max_depth_m)
        ray_scale=torch.sqrt(1+((u[ids]-intrinsics[0,2])/intrinsics[0,0])**2
                             +((v[ids]-intrinsics[1,2])/intrinsics[1,1])**2)
        half_diagonal=self.spacing.norm()*.5
        displacement=(half_diagonal+cfg.extrinsic_translation_bound_m
                      +2*z[ids]*ray_scale*np.sin(cfg.extrinsic_rotation_bound_rad*.5))
        guard=displacement+cfg.depth_error_bound_m
        veto=valid&((measured-z[ids]).abs()<=guard)
        self.surface_veto.view(-1)[ids[veto]]=True

        # A whole held view contributes at most one free-space vote. Do not
        # turn repeated frames or A->B->A revisits into independent evidence.
        matched=None
        if self.evidence_poses:
            past=torch.stack(self.evidence_poses)
            translation=(past[:,:3]-pose[:3]).norm(dim=1)
            angle=2*torch.acos((past[:,3:]*pose[3:]).sum(1).abs().clamp(0,1))
            matched=torch.where((translation<cfg.independent_view_translation_m)&(angle<cfg.independent_view_rotation_rad))[0].tolist()
            if matched:
                last_epoch=max(self.evidence_epochs[i] for i in matched)
                eligible=self.cell_change_epoch.view(-1)[ids]>last_epoch
                if not eligible.any():return
        radius=cfg.free_patch_radius_px
        all_valid=torch.isfinite(depth)&(depth>=cfg.min_depth_m)&(depth<=cfg.evidence_max_depth_m)
        safe=torch.where(all_valid,depth,0.)
        minimum=-functional.max_pool2d(-functional.pad(safe[None,None],(radius,)*4,value=0.),
                                      2*radius+1,1)[0,0]
        z_here=z[ids]
        # Bound the projection of the entire voxel plus extrinsic uncertainty.
        # For |dx|,|dz|<=d: |(x+dx)/(z+dz)-x/z| <= d*(1+|x/z|)/(z-d).
        offx=(u[ids]-intrinsics[0,2]).abs()/intrinsics[0,0]
        offy=(v[ids]-intrinsics[1,2]).abs()/intrinsics[1,1]
        spread=torch.maximum(intrinsics[0,0]*(1+offx),intrinsics[1,1]*(1+offy))
        spread=spread*displacement/(z_here-displacement).clamp_min(1e-6)+cfg.intrinsic_pixel_bound
        free=(z_here>displacement)&(spread<=radius-.5)&(minimum[vi,ui]>z_here+guard)
        if matched:free &= eligible
        if free.any():
            selected=ids[free]
            flat=self.free_views.view(-1)
            flat[selected]=(flat[selected].to(torch.int16)+1).clamp_max(255).to(torch.uint8)
            if matched:
                # Conservative: a partial revisit consumes this view's new
                # epoch even for changed cells outside its current ROI.
                for i in matched:self.evidence_epochs[i]=self.change_epoch
            else:
                self.evidence_poses.append(pose.clone());self.evidence_epochs.append(self.change_epoch)

    def planning_snapshot(self):
        with self._lock:
            known_free=(self.free_views>=self.cfg.minimum_free_views)&~self.surface_veto
            # Distant measured background can certify local free space even
            # when no TSDF surface is integrated inside the local range.
            known_free &= (self.weight==0)|(self.distance>0)
            return PlanningSnapshot(self,known_free.cpu().numpy())


@dataclass
class CollisionCheck:
    allowed: bool
    minimum_clearance_m: float
    tested_samples: int
    reason: str


class PlanningSnapshot:
    """EDT to all uncertified cells, not Euclidean interpretation of TSDF.

    Query subtracts both voxel extent and off-center position, a conservative
    lower bound on clearance to the unsafe voxel union. Padding blocks outside.
    """
    def __init__(self,mapper,known_free):
        self.mapper=mapper
        self.revision=mapper.revision
        self.built_at=mapper.latest_observation_time or time.monotonic()
        self.lower=mapper.lower.cpu().numpy().astype(np.float64)
        self.spacing=mapper.spacing.cpu().numpy().astype(np.float64)
        self.shape=np.asarray(mapper.shape)
        self.free_fraction=float(known_free.mean())
        self.cell_radius=float(np.linalg.norm(self.spacing)*.5)
        self.margin=mapper.cfg.planning_margin_m
        padded=np.pad(known_free,1,constant_values=False)
        self.clearance_grid=distance_transform_edt(padded,sampling=self.spacing)[1:-1,1:-1,1:-1].astype(np.float32)
        # Age starts at evidence snapshot acquisition, including EDT cost.
        self.max_age=mapper.cfg.snapshot_max_age_s

    def clearance(self,points,*,now_s=None):
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3:
            raise ValueError('expected world query points [N,3]')
        now=time.monotonic() if now_s is None else float(now_s)
        valid=np.isfinite(points).all(1)
        valid &= np.isfinite(now) and 0<=now-self.built_at<=self.max_age and self.mapper.revision==self.revision
        safe_points=np.where(np.isfinite(points),points,self.lower)
        valid &= ((safe_points>=self.lower)&(safe_points<self.lower+self.spacing*self.shape)).all(1)
        safe_points=np.where(valid[:,None],safe_points,self.lower)
        index=np.floor((safe_points-self.lower)/self.spacing).astype(np.int64)
        valid &= ((index>=0)&(index<self.shape)).all(1)
        index=np.clip(index,0,self.shape-1)
        center=self.lower+(index+.5)*self.spacing
        value=self.clearance_grid[index[:,0],index[:,1],index[:,2]].astype(float)
        value-=self.cell_radius+np.linalg.norm(safe_points-center,axis=1)+self.margin
        return np.where(valid,np.maximum(value,0.),0.),valid

    def check_swept_spheres(self,start,end,radii,*,tracking_bound_m=0.,
                           interpolation_deviation_bound_m=0.,max_step_m=.005,now_s=None):
        """Continuous LINEAR center sweeps of covering robot spheres.

        Cover every link/tool with spheres, not only the tool center. For a
        nonlinear joint/FK segment callers must supply a verified bound on
        departure from these center chords. Not a self/inter-arm checker.
        """
        start,end=np.asarray(start,dtype=float),np.asarray(end,dtype=float)
        radii=np.asarray(radii,dtype=float)
        if start.ndim!=2 or start.shape[1]!=3 or end.shape!=start.shape or radii.shape!=(len(start),) or len(start)==0:
            raise ValueError('expected start/end [S,3] and positive sphere radii [S]')
        if (not np.isfinite([start,end]).all() or not np.isfinite(radii).all() or (radii<=0).any()
                or not np.isfinite([tracking_bound_m,interpolation_deviation_bound_m,max_step_m]).all()
                or min(tracking_bound_m,interpolation_deviation_bound_m)<0 or max_step_m<=0):
            raise ValueError('invalid sweep geometry or error bounds')
        length=np.linalg.norm(end-start,axis=1)
        steps=max(1,int(np.ceil(length.max()/max_step_m)))
        if steps>100000:
            raise ValueError('sweep subdivision budget exceeded')
        t=np.linspace(0,1,steps+1)
        queries=start[:,None]+t[None,:,None]*(end-start)[:,None]
        values,valid=self.clearance(queries.reshape(-1,3),now_s=now_s)
        # Every intermediate center is within half a sample interval of one
        # tested center. Distance-to-obstacles is 1-Lipschitz.
        required=radii+tracking_bound_m+interpolation_deviation_bound_m+length/(2*steps)
        slack=values.reshape(len(start),-1)-required[:,None]
        allowed=bool(valid.all() and (slack>0).all())
        return CollisionCheck(allowed,float(slack.min()),len(values),
                              'clear_under_assumptions' if allowed else 'blocked_unknown_stale_or_insufficient_clearance')
