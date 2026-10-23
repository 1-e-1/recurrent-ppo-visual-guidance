"""Fail-closed environment authority. Inputs are a measured map and robot FK.

No USD stage, scene object list, object CAD, target extent or ground-truth pose
is accepted here. This is an ENVIRONMENT certificate; robot self/inter-arm
checks must ALSO pass before a caller executes a motion.
"""
from dataclasses import dataclass
import numpy as np


def robot_hulls(arm,q):
    return arm.collision_hulls_world(q) if hasattr(arm,'collision_hulls_world') else list(arm.hulls_world(q).items())


@dataclass(frozen=True)
class EnvironmentCertificate:
    allowed: bool
    reason: str
    boxes: int
    minimum_slack_m: float
    revision: int
    scope: str='RGBD_MEASURED_FREE_SPACE_ONLY'


class RGBDEnvironmentGuard:
    def __init__(self,left,right,snapshot,*,max_depth=12,max_boxes=4095):
        if max_depth<0 or max_boxes<1:raise ValueError('invalid certificate budget')
        self.arms=(left,right);self.snapshot=snapshot
        self.max_depth=max_depth;self.max_boxes=max_boxes

    def box(self,lo,hi,*,now_s=None):
        lo=np.asarray(lo,float);hi=np.asarray(hi,float)
        if lo.shape!=(16,) or hi.shape!=(16,) or not np.isfinite([lo,hi]).all() or np.any(lo>hi) or np.any(abs(np.r_[lo,hi])>np.pi):
            return False,'invalid_joint_box',float('-inf')
        q=((lo+hi)/2).reshape(2,8);delta=((hi-lo)/2).reshape(2,8)
        all_slack=[]
        for i,arm in enumerate(self.arms):
            if hasattr(arm,'prepare_motion_bounds'):arm.prepare_motion_bounds(q[i])
            points=arm.sphere_positions(q[i])
            radii=np.asarray(arm.radii,float)
            if points.shape!=(len(radii),3) or not len(radii) or not np.isfinite([points]).all() or not np.isfinite(radii).all() or np.any(radii<=0):
                return False,'invalid_robot_cover',float('-inf')
            displacement=np.array([arm.absolute_motion_bound(n,delta[i]) for n in arm.sphere_names])
            if displacement.shape!=radii.shape or not np.isfinite(displacement).all() or (displacement<0).any():
                return False,'invalid_robot_motion_bound',float('-inf')
            clear,valid=self.snapshot.clearance(points,now_s=now_s)
            slack=clear-radii-displacement
            all_slack.extend(slack.tolist())
            if not valid.all():return False,'outside_stale_or_revised_map',float(min(all_slack))
        minimum=float(min(all_slack))
        return minimum>0,'measured_free' if minimum>0 else 'unknown_occupied_or_insufficient_clearance',minimum

    def segment(self,start,end,*,now_s=None):
        start=np.asarray(start,float);end=np.asarray(end,float);boxes=0;minimum=float('inf')
        def check(a,b):
            nonlocal boxes,minimum
            if boxes>=self.max_boxes:return False,'certificate_budget_exhausted',0.
            boxes+=1;ok,why,slack=self.box(np.minimum(a,b),np.maximum(a,b),now_s=now_s)
            minimum=min(minimum,slack);return ok,why,slack
        if start.shape!=(16,) or end.shape!=(16,):
            return EnvironmentCertificate(False,'invalid_joint_shape',0,0.,self.snapshot.revision)
        # Endpoints must themselves be known safe. Do not spend an exponential
        # subdivision budget trying to certify an unknown start configuration.
        for q in (start,end):
            ok,why,_=check(q,q)
            if not ok:return EnvironmentCertificate(False,why,boxes,minimum,self.snapshot.revision)
        accepted_min=float('inf')
        def visit(a,b,depth):
            nonlocal accepted_min
            ok,why,slack=check(a,b)
            if ok:accepted_min=min(accepted_min,slack);return True,'measured_free'
            if depth>=self.max_depth or boxes>=self.max_boxes:return False,'certificate_budget_exhausted'
            mid=(a+b)/2;ok,why=visit(a,mid,depth+1)
            return visit(mid,b,depth+1) if ok else (False,why)
        ok,why=visit(start,end,0)
        return EnvironmentCertificate(ok,why,boxes,accepted_min if ok else minimum,self.snapshot.revision)


class RGBDHullEnvironmentGuard(RGBDEnvironmentGuard):
    """Tighter ROBOT hull / unknown-voxel check, not environment CAD fitting.

Sphere tests remain a fast sufficient gate. A failure may be refined using
the robot's known convex hulls. Unknown voxels are never removed or filled.
"""
    def __init__(self,*args,max_cell_tests=250000,**kwargs):
        super().__init__(*args,**kwargs)
        if max_cell_tests<1:raise ValueError('invalid voxel query budget')
        self.max_cell_tests=max_cell_tests;self.cell_tests=0;self.last_rejection=None

    def segment(self,*args,**kwargs):
        self.cell_tests=0;self.last_rejection=None;return super().segment(*args,**kwargs)

    def authorized_contact_cells(self,name,centers):
        return np.zeros(len(centers),bool)

    def query_grid(self,name):
        return self.snapshot.clearance_grid

    def authorized_contact_intersections(self,name,centers,vertices,normals,padding):
        return np.zeros(len(centers),bool)

    def box(self,lo,hi,*,now_s=None):
        ok,reason,slack=super().box(lo,hi,now_s=now_s)
        if ok or reason.startswith('invalid_'):return ok,reason,slack
        lo=np.asarray(lo,float);hi=np.asarray(hi,float);q=((lo+hi)/2).reshape(2,8);delta=((hi-lo)/2).reshape(2,8)
        s=self.snapshot;lower=s.lower;upper=lower+s.spacing*s.shape
        _,valid=s.clearance([(lower+upper)/2],now_s=now_s)
        if not valid.all():return False,'stale_or_revised_map',slack
        for arm_index,(arm,angles,width) in enumerate(zip(self.arms,q,delta)):
            if hasattr(arm,'prepare_motion_bounds'):arm.prepare_motion_bounds(angles)
            for name,(vertices,normals) in robot_hulls(arm,angles):
                vertices=np.asarray(vertices,float);normals=np.asarray(normals,float)
                if (vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or normals.ndim!=2 or normals.shape[1]!=3
                    or not len(normals) or not np.isfinite(vertices).all() or not np.isfinite(normals).all()):
                    return False,'invalid_robot_hull',float('-inf')
                lengths=np.linalg.norm(normals,axis=1)
                if (lengths<1e-12).any():return False,'invalid_robot_hull_normals',float('-inf')
                normals=normals/lengths[:,None]
                padding=s.margin+arm.absolute_motion_bound(name,width)
                a=vertices.min(0)-padding;b=vertices.max(0)+padding
                if np.any(a<=lower) or np.any(b>=upper):return False,'robot_hull_outside_map',slack
                first=np.maximum(0,np.floor((a-lower)/s.spacing).astype(int)-1)
                last=np.minimum(s.shape,np.ceil((b-lower)/s.spacing).astype(int)+1)
                region=tuple(slice(i,j) for i,j in zip(first,last))
                # EDT is exactly zero at occupied/uncertified cells.
                ids=np.argwhere(self.query_grid(name)[region]<=0)+first
                self.cell_tests+=len(ids)
                if self.cell_tests>self.max_cell_tests:return False,'voxel_query_budget_exhausted',slack
                if not len(ids):continue
                centers=lower+(ids+.5)*s.spacing
                # Necessary conditions for any unsafe voxel to intersect the
                # motion-expanded hull: voxel centre must satisfy every face
                # halfspace expanded by the exact axis-aligned voxel support
                # in that direction, plus motion/margin. This is no larger
                # than a half-diagonal and does not change map evidence.
                voxel_support=np.abs(normals)@(s.spacing*.5)
                support=(vertices@normals.T).max(0)+voxel_support+padding+1e-9
                candidates=((centers+s.spacing*.5>=a-1e-9)&(centers-s.spacing*.5<=b+1e-9)).all(1)
                for i in range(0,len(normals),24):
                    indices=np.flatnonzero(candidates)
                    if not len(indices):break
                    candidates[indices]=(centers[indices]@normals[i:i+24].T<=support[i:i+24]).all(1)
                if candidates.any():candidates &= ~self.authorized_contact_cells(name,centers)
                if candidates.any():
                    ids_partial=np.flatnonzero(candidates)
                    candidates[ids_partial] &= ~self.authorized_contact_intersections(
                        name,centers[ids_partial],vertices,normals,padding)
                if candidates.any():
                    self.last_rejection=dict(arm_index=arm_index,link=name,padding_m=float(padding),spacing_m=s.spacing.tolist(),
                        candidate_cells=int(candidates.sum()),
                        voxel_indices=ids[candidates][:20].tolist(),
                        voxel_centers_m=centers[candidates][:20].tolist())
                    return False,'unknown_or_occupied_voxel_intersects_robot_hull/'+name,slack
        # Guaranteed planned margin is already included; do not invent an
        # additional Euclidean slack estimate for this halfspace certificate.
        return True,'measured_robot_hull_cover',0.


class RGBDChordEnvironmentGuard(RGBDHullEnvironmentGuard):
    """Correlated joint-line envelope with a second-order chord remainder.

For a serial rigid chain and any material point, differentiating rotation
products twice bounds |p''(u)| by R*(sum |dq_j|)^2. R is the robot's existing
global chain-length plus local-geometry bound, not an estimated Jacobian norm.
The curve lies within R*(sum |dq_j|)^2/8 of its endpoint chord. Convex hulls
of endpoint robot vertices, enlarged by that remainder, cover the full sweep.
"""
    def chord(self,start,end,now_s=None):
        s=self.snapshot;lower=s.lower;upper=lower+s.spacing*s.shape
        _,valid=s.clearance([(lower+upper)/2],now_s=now_s)
        if not valid.all():return False,'stale_or_revised_map'
        a=np.asarray(start).reshape(2,8);b=np.asarray(end).reshape(2,8)
        # Refine a loose global remainder before enumerating irrelevant cells.
        # Exact error bounds, margins and final cell/node budgets are unchanged.
        pairs=[]
        for arm_index,(arm,qa,qb) in enumerate(zip(self.arms,a,b)):
            ha,hb=robot_hulls(arm,qa),robot_hulls(arm,qb)
            if [x[0] for x in ha]!=[x[0] for x in hb]:return False,'inconsistent_robot_hulls'
            # Split guards may deliberately select NO geometry from one arm.
            # Its motion cannot enlarge the other arm's stationary payload.
            # Do not subdivide and repeatedly spend that payload's cell budget
            # because an excluded camera arm moved. Broad/self guards still
            # certify that camera motion independently.
            if not ha:continue
            remainder=float(arm.motion_radius)*float(np.sum(abs(qb-qa)))**2/8
            if remainder>.020:return False,'loose_chord_requires_subdivision'
            pairs.append((arm_index,arm,qa,qb,ha,hb))
        for arm_index,arm,qa,qb,ha,hb in pairs:
            reach=float(arm.motion_radius)
            if not np.isfinite(reach) or reach<=0:return False,'invalid_global_robot_reach'
            padding=s.margin+reach*float(np.sum(abs(qb-qa)))**2/8
            for (name,A),(_,B) in zip(ha,hb):
                points=np.concatenate([A[0],B[0]])
                normals=np.concatenate([A[1],B[1]])
                lengths=np.linalg.norm(normals,axis=1)
                if not np.isfinite(points).all() or not np.isfinite(normals).all() or np.any(lengths<1e-12):return False,'invalid_robot_hull'
                normals=normals/lengths[:,None]
                lo,hi=points.min(0)-padding,points.max(0)+padding
                if np.any(lo<=lower) or np.any(hi>=upper):return False,'robot_chord_outside_map'
                first=np.maximum(0,np.floor((lo-lower)/s.spacing).astype(int)-1)
                last=np.minimum(s.shape,np.ceil((hi-lower)/s.spacing).astype(int)+1)
                ids=np.argwhere(self.query_grid(name)[tuple(slice(i,j) for i,j in zip(first,last))]<=0)+first
                self.cell_tests+=len(ids)
                if self.cell_tests>self.max_cell_tests:return False,'voxel_query_budget_exhausted'
                if not len(ids):continue
                centers=lower+(ids+.5)*s.spacing
                support=(points@normals.T).max(0)+np.abs(normals)@(s.spacing*.5)+padding+1e-9
                possible=((centers+s.spacing*.5>=lo-1e-9)&(centers-s.spacing*.5<=hi+1e-9)).all(1)
                for i in range(0,len(normals),24):
                    selected=np.flatnonzero(possible)
                    if not len(selected):break
                    possible[selected]=(centers[selected]@normals[i:i+24].T<=support[i:i+24]).all(1)
                if possible.any():possible &= ~self.authorized_contact_cells(name,centers)
                if possible.any():
                    ids_partial=np.flatnonzero(possible)
                    possible[ids_partial] &= ~self.authorized_contact_intersections(
                        name,centers[ids_partial],points,normals,padding)
                if possible.any():
                    self.last_rejection=dict(arm_index=arm_index,link=name,padding_m=padding,spacing_m=s.spacing.tolist(),candidate_cells=int(possible.sum()),
                        voxel_indices=ids[possible][:20].tolist(),voxel_centers_m=centers[possible][:20].tolist())
                    return False,'unknown_intersects_chord_envelope/'+name
        return True,'measured_chord_sweep'

    def segment(self,start,end,*,now_s=None):
        if not all(hasattr(arm,'motion_radius') for arm in self.arms):return super().segment(start,end,now_s=now_s)
        self.cell_tests=0;self.last_rejection=None
        a,b=np.asarray(start,float),np.asarray(end,float);boxes=0
        if a.shape!=(16,) or b.shape!=(16,):return EnvironmentCertificate(False,'invalid_joint_shape',0,0.,self.snapshot.revision)
        for q in (a,b):
            boxes+=1;ok,why,slack=self.box(q,q,now_s=now_s)
            if not ok:return EnvironmentCertificate(False,why,boxes,slack,self.snapshot.revision)
        stack=[(a,b,0)]
        while stack:
            if boxes>=self.max_boxes:return EnvironmentCertificate(False,'certificate_budget_exhausted',boxes,0.,self.snapshot.revision)
            x,y,depth=stack.pop();boxes+=1;ok,why=self.chord(x,y,now_s)
            if ok:continue
            if depth>=self.max_depth or why in ('voxel_query_budget_exhausted','stale_or_revised_map'):
                return EnvironmentCertificate(False,why,boxes,0.,self.snapshot.revision)
            mid=(x+y)/2;stack.extend([(mid,y,depth+1),(x,mid,depth+1)])
        return EnvironmentCertificate(True,'measured_chord_sweep',boxes,0.,self.snapshot.revision)
