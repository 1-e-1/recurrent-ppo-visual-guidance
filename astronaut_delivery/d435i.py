"""D435i nominal RGB/stereo-depth geometry; not an emulation of librealsense.

All poses here use column vectors and OpenGL optical axes (+X right,+Y up,-Z
forward). Raw RealSense optical axes require diag(1,-1,-1) conversion.
"""
import json
import numpy as np
from .scene import ROOT
from .vision import MeshCamera,Observation

def load_profile(path=None):
    cfg=json.loads((ROOT/'configs/d435i.json' if path is None else __import__('pathlib').Path(path)).read_text())
    for stream in ('depth','color'):
        s=cfg[stream]
        if min(s['width'],s['height'],s['fps'])<=0:raise ValueError('Invalid stream dimensions')
    lo,hi=cfg['valid_depth_range_m']
    if not 0<lo<hi:raise ValueError('Invalid depth range')
    if cfg['calibration_status']!='NOMINAL_SIMULATION_ONLY':
        raise ValueError('Hardware calibration must be rectified and hand-eye verified before simulation import')
    return cfg

def intrinsics(stream):
    w,h=stream['width'],stream['height']
    return np.array([[w/(2*np.tan(np.deg2rad(stream['hfov_deg'])/2)),0,(w-1)/2],
                     [0,h/(2*np.tan(np.deg2rad(stream['vfov_deg'])/2)),(h-1)/2],[0,0,1.]])

def unproject(depth,K):
    v,u=np.indices(depth.shape)
    return np.stack(((u-K[0,2])*depth/K[0,0],-(v-K[1,2])*depth/K[1,1],-depth),axis=-1)

def project(points,K):
    z=-points[...,2];safe=np.maximum(z,1e-9)
    return np.stack((K[0,0]*points[...,0]/safe+K[0,2],-K[1,1]*points[...,1]/safe+K[1,2]),axis=-1),z

def reproject(depth,K,other_K,other_shape,T_depth_other):
    """Native depth pixels -> other camera pixels. T maps other GL to depth GL."""
    p=unproject(depth,K)
    p=(p-T_depth_other[:3,3])@T_depth_other[:3,:3]
    uv,z=project(p,other_K)
    finite=np.isfinite(uv).all(-1)&np.isfinite(z)
    ij=np.rint(np.clip(np.nan_to_num(uv),-1e6,1e6)).astype(np.int32);h,w=other_shape[:2]
    valid=finite&(depth>0)&(z>0)&(ij[...,0]>=0)&(ij[...,0]<w)&(ij[...,1]>=0)&(ij[...,1]<h)
    return ij,z,valid

def sample_registered(depth,K,other,T_depth_other,tolerance=.003):
    ij,z,inside=reproject(depth,K,other.K,other.depth.shape,T_depth_other)
    u=np.clip(ij[...,0],0,other.depth.shape[1]-1);v=np.clip(ij[...,1],0,other.depth.shape[0]-1)
    valid=inside&(other.depth[v,u]>0)&(np.abs(other.depth[v,u]-z)<tolerance)
    rgb=np.where(valid[...,None],other.rgb[v,u],0).astype(np.float32)
    return rgb,valid

class D435iCamera:
    def __init__(self,device='cuda:0',profile=None):
        self.profile=load_profile() if profile is None else profile
        self.streams={}
        for key in ('depth','color'):
            p=self.profile[key]
            # Official CAD models the cover glass as opaque geometry. Optical
            # centres sit 4.3 mm behind it; start rays beyond this own cover only.
            self.streams[key]=MeshCamera(p['width'],p['height'],device,K=intrinsics(p),near_plane=.01)
        self.K=self.streams['depth'].K
        self.color_local=np.eye(4);self.color_local[:3,3]=self.profile['color_origin_in_depth_gl_m']
        self.ir_local=np.eye(4);self.ir_local[0,3]=self.profile['stereo_baseline_m']
        self.last_native_color=None;self.last_quality={}

    def capture(self,stage,camera_world,stamp,noise_std=0.,seed=0,streams=None):
        streams=self.streams if streams is None else streams
        # Render both IR geometries: remove points with no stereo correspondence.
        depth=streams['depth'].capture(stage,camera_world,stamp)
        right=streams['depth'].capture(stage,camera_world@self.ir_local,stamp,geometry_source=streams['depth'])
        color=streams['color'].capture(stage,camera_world@self.color_local,stamp,geometry_source=streams['depth'])
        self.last_native_color=color.rgb
        _,stereo=sample_registered(depth.depth,depth.K,right,self.ir_local,self.profile['simulation_noise']['stereo_consistency_m'])
        aligned,color_valid=sample_registered(depth.depth,depth.K,color,self.color_local)
        lo,hi=self.profile['valid_depth_range_m'];d=depth.depth
        valid=stereo&(d>=lo)&(d<=hi)
        n=self.profile['simulation_noise'];rng=np.random.default_rng(seed)
        sigma=n['floor_sigma_m']+d*d*n['disparity_sigma_px']/(self.K[0,0]*self.profile['stereo_baseline_m'])
        valid&=rng.random(d.shape)>=n['dropout_probability']
        noisy=d+rng.normal(size=d.shape)*sigma
        valid&=(noisy>=lo)&(noisy<=hi)
        measured=np.where(valid,noisy,0).astype(np.float32)
        color_valid&=valid
        aligned[~color_valid]=0
        self.last_quality=dict(valid_depth_ratio=float(valid.mean()),rgb_depth_overlap_ratio=float(color_valid.mean()),
                               median_depth_sigma_m=float(np.median(sigma[valid])) if valid.any() else None)
        return Observation(aligned,measured,depth.K.copy(),camera_world.copy(),stamp,depth.robot_mask,color_valid,sigma,
                           dict(**self.last_quality,calibration_status=self.profile['calibration_status']))

    def capture_roi(self,stage,camera_world,stamp,target,seed=0):
        """Exact native-grid crop around a previously measured world target.

        Crop changes principal points, NOT focal lengths or physical FOV. On
        hardware this corresponds to cropping received frames, not sensor zoom.
        """
        if not hasattr(self,'roi_streams'):
            self.roi_streams={k:MeshCamera(size,size,self.streams[k].device,K=self.streams[k].K,near_plane=.01)
                              for k,size in [('depth',160),('color',320)]}
        offsets={}
        for key,local in [('depth',np.eye(4)),('color',self.color_local)]:
            world=camera_world@local;p=(np.asarray(target)-world[:3,3])@world[:3,:3]
            uv,z=project(p,self.streams[key].K)
            if z<=.05:raise ValueError('ROI target is behind camera')
            camera=self.roi_streams[key];native=self.streams[key]
            u0=int(np.clip(round(uv[0])-camera.width//2,0,native.width-camera.width))
            v0=int(np.clip(round(uv[1])-camera.height//2,0,native.height-camera.height))
            camera.K=native.K.copy();camera.K[0,2]-=u0;camera.K[1,2]-=v0;offsets[key]=[u0,v0]
        obs=self.capture(stage,camera_world,stamp,seed=seed,streams=self.roi_streams)
        obs.metadata.update(quality_domain='native_pixel_ROI',roi_offsets=offsets)
        return obs

    def view_score(self,camera_world,points):
        """Reachable view ranking uses observed 3D points, never scene part poses."""
        points=np.asarray(points);p=(points-camera_world[:3,3])@camera_world[:3,:3]
        dist=np.linalg.norm(p,axis=1);lo,hi=self.profile['target_standoff_m']
        if np.any(dist<lo) or np.any(dist>hi):return -np.inf
        pc=(p-self.color_local[:3,3])@self.color_local[:3,:3]
        uv,z=project(pc,self.streams['color'].K);s=self.profile['color']
        margin=np.minimum.reduce([uv[:,0]/s['width'],1-uv[:,0]/s['width'],uv[:,1]/s['height'],1-uv[:,1]/s['height']])
        if np.any(z<=0) or margin.min()<.04:return -np.inf
        return float(2*margin.min()-.2*np.mean(abs(dist-self.profile['preferred_standoff_m'])))
