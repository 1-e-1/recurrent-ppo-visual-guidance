"""Actual USD-mesh RGB-D ray rendering and image-only magnetic-cap guidance.

The renderer is a simulator sensor. It never returns object poses/IDs to the
guide; only RGB, axial depth, K, timestamp and calibrated camera extrinsics.
"""
from dataclasses import dataclass
import time
import numpy as np
import torch
import cv2
from pxr import Gf, Usd, UsdGeom, UsdShade
from .motion import vertices, transformed, world
from classical_fusion.planning_map import PlanningMapConfig,PlanningTSDF

def triangulate(prim):
    v=vertices(prim)
    if v is None:return None
    if prim.IsA(UsdGeom.Mesh):
        mesh=UsdGeom.Mesh(prim);counts=mesh.GetFaceVertexCountsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get()
        faces=[];offset=0
        for n in counts:
            poly=indices[offset:offset+n];offset+=n
            faces.extend([[poly[0],poly[j],poly[j+1]] for j in range(1,n-1)])
        return v,np.asarray(faces,np.int32)
    from scipy.spatial import ConvexHull
    return v,ConvexHull(v).simplices.astype(np.int32)

def color(prim):
    mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if mat:
        for p in Usd.PrimRange(mat.GetPrim()):
            for key in ('inputs:diffuseColor','inputs:diffuse_color_constant'):
                value=p.GetAttribute(key).Get()
                if value is not None:return np.asarray(value,float)
    return np.array([.65,.67,.7])

@dataclass
class Observation:
    rgb:np.ndarray
    depth:np.ndarray
    K:np.ndarray
    camera_world:np.ndarray  # Column-vector OpenGL camera-to-world
    stamp:float
    robot_mask:np.ndarray|None=None  # CAD/FK self mask, never target IDs
    color_valid:np.ndarray|None=None
    depth_sigma:np.ndarray|None=None
    metadata:dict|None=None

class MeshCamera:
    """Headless ray sensor for deterministic tests, using installed NVIDIA Warp."""
    def __init__(self,width=480,height=360,device='cuda:0',shaded=False,K=None,near_plane=.01):
        import warp as wp
        from .scene import ROOT
        wp.config.kernel_cache_dir=str(ROOT/'logs'/'delivery_warp_cache')
        wp.init()
        self.wp=wp;self.device=device;self.width=width;self.height=height
        f=width*.5/np.tan(np.deg2rad(60)/2)
        self.K=np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])
        if K is not None:self.K=np.asarray(K,float).copy()
        self.near_plane=near_plane
        self.cached_stage=None
        self.shaded=shaded

    def capture(self,stage,camera_world,stamp,noise_std=0.,seed=0,geometry_source=None):
        wp=self.wp
        from .ray_kernel import ray_image
        if geometry_source is not None and geometry_source.cached_stage is not stage:
            raise ValueError('Shared ray geometry must belong to this stage and exposure')
        if geometry_source is None and self.cached_stage is not stage:
            self.descriptors=[];faces=[];colors=[];robot_faces=[];offset=0
            for p in stage.Traverse():
                if not p.IsA(UsdGeom.Imageable):continue
                im=UsdGeom.Imageable(p)
                if im.ComputePurpose()=='guide' or im.ComputeVisibility()=='invisible' or 'collision' in str(p.GetPath()).lower():continue
                data=triangulate(p)
                if data is None:continue
                v,f=data;self.descriptors.append((p,v))
                faces.append(f+offset);colors.extend([color(p)]*len(f));offset+=len(v)
                robot_faces.extend([int(str(p.GetPath()).startswith(('/World/LeftArm/','/World/RightArm/','/World/RightMagnet/')))]*len(f))
            self.gpu_faces=wp.array(np.concatenate(faces).astype(np.int32).ravel(),dtype=wp.int32,device=self.device)
            self.gpu_colors=wp.array(np.asarray(colors,np.float32),dtype=wp.vec3,device=self.device)
            self.gpu_robot=wp.array(np.asarray(robot_faces,np.int32),dtype=wp.int32,device=self.device)
            self.cached_stage=stage;self.gpu_mesh=None
        if geometry_source is None:
            cache=UsdGeom.XformCache()
            pts=np.concatenate([transformed(v,np.array(cache.GetLocalToWorldTransform(p))) for p,v in self.descriptors]).astype(np.float32)
            if self.gpu_mesh is None:self.gpu_mesh=wp.Mesh(wp.array(pts,dtype=wp.vec3,device=self.device),self.gpu_faces)
            else:
                self.gpu_mesh.points.assign(pts);self.gpu_mesh.refit()
            self.geometry_stamp=stamp
        else:
            if geometry_source.device!=self.device or geometry_source.geometry_stamp!=stamp:
                raise ValueError('Cannot share ray geometry across devices or exposures')
        source=self if geometry_source is None else geometry_source
        mesh=source.gpu_mesh
        depth=wp.zeros(self.width*self.height,dtype=float,device=self.device)
        rgb=wp.zeros(self.width*self.height,dtype=wp.vec3,device=self.device)
        robot=wp.zeros(self.width*self.height,dtype=wp.int32,device=self.device)
        t=np.asarray(camera_world);R=t[:3,:3]
        wp.launch(ray_image,dim=self.width*self.height,inputs=[mesh.id,source.gpu_colors,source.gpu_robot,self.width,self.height,float(self.K[0,0]),float(self.K[1,1]),float(self.K[0,2]),float(self.K[1,2]),wp.vec3(*t[:3,3]),wp.mat33(*R.ravel()),int(self.shaded),float(self.near_plane),depth,rgb,robot],device=self.device)
        d=depth.numpy().reshape(self.height,self.width)
        if noise_std:
            noise=np.random.default_rng(seed).normal(0,noise_std,d.shape)
            d=np.where(d>0,d+noise,0).astype(np.float32)
        return Observation(rgb.numpy().reshape(self.height,self.width,3),d,self.K.copy(),t.copy(),stamp,robot_mask=robot.numpy().reshape(self.height,self.width).astype(bool))

@dataclass
class TargetEstimate:
    name:str
    position:np.ndarray
    covariance:np.ndarray
    pixels:int
    stamp:float
    kind:str='magnetic_interface'
    wall_stamp:float|None=None  # Monotonic request/exposure clock, never receipt time.
    measurement_method:str='original'

def circular_cap_center(points,radius=.020):
    """Fit the KNOWN 20 mm circular docking interface, not an arbitrary object.

    Partial disks have biased centroids. RANSAC on their observed convex boundary
    ignores the straight occlusion chord; radius is a task-interface prior, not
    a simulator pose. Return None if no sufficiently supported arc is visible.
    """
    from scipy.spatial import ConvexHull,QhullError
    from scipy.optimize import least_squares
    xy=np.asarray(points)[:,:2]
    if len(xy)<10:return None
    try:boundary=xy[ConvexHull(xy).vertices]
    except QhullError:return None
    if len(boundary)<6:return None
    # Each two boundary samples has up to two radius-constrained centres.
    best=None
    for i in range(len(boundary)):
        for j in range(i):
            delta=boundary[i]-boundary[j];length=np.linalg.norm(delta)
            if not .012<length<2*radius:continue
            middle=(boundary[i]+boundary[j])*.5;normal=np.array([-delta[1],delta[0]])/length
            for sign in (-1,1):
                center=middle+sign*normal*np.sqrt(radius*radius-length*length/4)
                residual=abs(np.linalg.norm(boundary-center,axis=1)-radius);inliers=residual<.0018
                inside=np.mean(np.linalg.norm(xy-center,axis=1)<radius+.002)
                if inliers.sum()<5 or inside<.93:continue
                score=int(inliers.sum())-.1*float(np.mean(residual[inliers])/.0018)
                if best is None or score>best[0]:best=(score,center,inliers)
    if best is None:return None
    _,center,inliers=best
    fit=least_squares(lambda c:np.linalg.norm(boundary[inliers]-c,axis=1)-radius,center,loss='soft_l1',f_scale=.001,max_nfev=30)
    return np.r_[fit.x,np.median(points[:,2])]

class VisualGuide:
    def __init__(self,palette,max_age=.25,sensor_profile=None):
        self.palette=palette;self.max_age=max_age
        self.mapper=PlanningTSDF(PlanningMapConfig(workspace_min=(.25,-.72,.65),workspace_max=(1.0,.48,1.60),voxel_size_m=.012,truncation_m=.036,surface_min_weight=1.,free_patch_radius_px=5,minimum_free_views=2,max_depth_m=2.,snapshot_max_age_s=5.),device='cpu')
        self.estimates={};self.frames=0;self.sensor_profile=sensor_profile
        self.last_map_stamp=-np.inf;self.map_updates=0
        self.reacquire_stale=False;self.pending_reacquisition={};self.reacquisitions=0
        self.fit_circular_cap=False
        self.docking_cap_radius=.020
        # Explicit sensor-side contract: calibrated RGB rejects returns from
        # differently coloured known hardware. Never changes mapping depth.
        self.registered_interface_override=False

    def update(self,obs,integrate_map=True):
        # Existing TSDF expects camera +X forward,+Y left,+Z up. OpenGL uses -Z.
        t=obs.camera_world;R=t[:3,:3]@np.array([[0,-1,0],[0,0,1],[-1,0,0.]])
        q=Gf.Matrix3d(R.T.tolist()).ExtractRotation().GetQuat()
        pose=np.r_[t[:3,3],float(q.GetReal()),np.array(q.GetImaginary())]
        depth=obs.depth.copy()
        if self.sensor_profile is not None and obs.robot_mask is not None:
            depth[cv2.dilate(obs.robot_mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)]=0
        if integrate_map and (self.sensor_profile is None or obs.stamp-self.last_map_stamp>=.5):
            self.mapper.integrate(depth,obs.K,pose)
            self.last_map_stamp=obs.stamp;self.map_updates+=1
        self.frames+=1
        for name,paint in self.palette.items():
            mask=(np.linalg.norm(obs.rgb-np.array(paint),axis=2)<.08)&np.isfinite(obs.depth)&(obs.depth>.05)
            if self.sensor_profile is not None:
                chroma=obs.rgb/np.maximum(obs.rgb.sum(2,keepdims=True),1e-6)
                paint=np.asarray(paint);paint=paint/paint.sum()
                mask=(np.linalg.norm(chroma-paint,axis=2)<.06)&(obs.rgb.sum(2)>.12)&(depth>0)
                if self.registered_interface_override:
                    if not self.fit_circular_cap or obs.color_valid is None:
                        raise ValueError('interface override requires registered RGB and circular fit')
                    mask=(np.linalg.norm(chroma-paint,axis=2)<.06)&(obs.rgb.sum(2)>.12)&np.isfinite(obs.depth)&(obs.depth>.05)
                if obs.color_valid is not None:mask&=obs.color_valid
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
            candidates=[]
            for label in range(1,count):
                if stats[label,cv2.CC_STAT_AREA]<6:continue
                v,u=np.where(labels==label);z=obs.depth[v,u]
                camera=np.column_stack(((u-obs.K[0,2])/obs.K[0,0]*z,-(v-obs.K[1,2])/obs.K[1,1]*z,-z))
                points=camera@t[:3,:3].T+t[:3,3]
                # Horizontal docking cap: fit the highest observed painted band.
                high=np.quantile(points[:,2],.8)
                top=points[points[:,2]>high-.004]
                if len(top)<4:continue
                center=np.r_[np.median(top[:,:2],axis=0),np.median(top[:,2])]
                if self.fit_circular_cap:
                    fitted=circular_cap_center(top,self.docking_cap_radius)
                    if fitted is None:continue
                    center=fitted
                covariance=np.cov(top.T)/len(top)+np.eye(3)*(.002**2)
                if obs.depth_sigma is not None:
                    covariance+=np.eye(3)*float(np.median(obs.depth_sigma[v,u])**2)
                candidates.append(TargetEstimate(name,center,covariance,len(top),obs.stamp,
                    wall_stamp=(obs.metadata or {}).get('exposure_wall_s')))
            if candidates:
                current=max(candidates,key=lambda x:x.pixels)
                previous=self.estimates.get(name)
                # Pick candidates are stationary until latched. Partial cap
                # occlusion must not make its visible centroid become a new goal.
                if previous is not None:
                    rejected=current.pixels<.4*previous.pixels or np.linalg.norm(current.position-previous.position)>.006
                    if rejected:
                        previous_age=obs.stamp-previous.stamp
                        if current.wall_stamp is not None and previous.wall_stamp is not None:
                            from .measurement_clock import observation_age
                            previous_age=observation_age(previous.stamp,previous.wall_stamp,motion_now=obs.stamp,wall_now=current.wall_stamp)
                        if not self.reacquire_stale or previous_age<=self.max_age:continue
                        pending=self.pending_reacquisition.get(name,[])
                        if pending and np.linalg.norm(current.position-pending[-1].position)>.003:pending=[]
                        pending.append(current);self.pending_reacquisition[name]=pending[-3:]
                        if len(pending)<3:continue
                        current.position=np.mean([x.position for x in pending[-3:]],axis=0)
                        self.pending_reacquisition.pop(name,None);self.reacquisitions+=1
                    else:
                        current.position=.25*current.position+.75*previous.position
                        self.pending_reacquisition.pop(name,None)
                self.estimates[name]=current
        return self.estimates

    def target(self,name,now):
        t=self.estimates.get(name)
        if t is None or not np.isfinite(now) or not 0<=now-t.stamp<=self.max_age:return None
        return t

    def correction(self,name,tcp,now,max_step=.01):
        target=self.target(name,now)
        if target is None:return None
        error=target.position-np.asarray(tcp)
        return error*min(1,max_step/max(np.linalg.norm(error),1e-9))

    def invalidate_after_transfer(self,regions=None,target=None):
        # Retire stale occupied AND free evidence. Local action bounds preserve
        # the rest of the map; None retains the legacy full reset fallback.
        if regions is None:
            self.mapper.reset();self.estimates.clear()
            self.pending_reacquisition.clear();stats=[]
        else:
            stats=[self.mapper.invalidate_region(region['lower'],region['upper']) for region in regions]
            self.estimates.pop(target,None);self.pending_reacquisition.pop(target,None)
        self.last_map_stamp=-np.inf
        return stats
