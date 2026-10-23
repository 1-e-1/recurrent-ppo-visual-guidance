"""Rigid payload proxy made from measured geometry, not simulator part CAD."""
import numpy as np
from scipy.spatial import ConvexHull
from .motion import Payload
from classical_fusion.rgbd_guard import robot_hulls


def payload_from_measurement(name,envelope,tcp):
    local=(envelope.vertices-tcp[:3,3])@tcp[:3,:3]
    hull=ConvexHull(local)
    return Payload(name,local[hull.vertices],np.eye(4))


class PayloadArm:
    def __init__(self,arm,payload):
        self.arm,self.payload=arm,payload
        self.motion_radius=max(arm.motion_radius,arm.chain_length+np.linalg.norm(arm.tool_local[:3,3])+np.linalg.norm(payload.local_points,axis=1).max())
        self.radii=np.r_[arm.radii,payload.radius];self.sphere_names=arm.sphere_names+['payload']
        self._normals=ConvexHull(payload.local_points).equations[:,:3]
    def __getattr__(self,name):return getattr(self.arm,name)
    def sphere_positions(self,q):
        T=self.arm.tcp(q);center=self.payload.center@T[:3,:3].T+T[:3,3]
        return np.vstack([self.arm.sphere_positions(q),center])
    def absolute_motion_bound(self,name,delta):
        return self.arm.absolute_motion_bound(name,delta,self.payload) if name=='payload' else self.arm.absolute_motion_bound(name,delta)
    def payload_hull(self,q):
        T=self.arm.tcp(q)
        return self.payload.local_points@T[:3,:3].T+T[:3,3],self._normals@T[:3,:3].T
    def hulls_world(self,q):return dict(self.arm.hulls_world(q),payload=self.payload_hull(q))
    def collision_hulls_world(self,q):return robot_hulls(self.arm,q)+[('payload',self.payload_hull(q))]


def mask_payload_returns(depth,K,camera_world,local_points,tcp,padding=.012):
    p=np.asarray(local_points,float)
    if p.ndim!=2 or p.shape[1]!=3 or not 4<=len(p)<=3000 or not np.isfinite(p).all() or np.linalg.norm(p,axis=1).max()>.3:
        raise ValueError('invalid measured payload mask')
    vertices=p@tcp[:3,:3].T+tcp[:3,3];equations=ConvexHull(vertices).equations
    d=np.asarray(depth).copy();v,u=np.where(np.isfinite(d)&(d>0));z=d[v,u]
    points=np.c_[(u-K[0,2])*z/K[0,0],-(v-K[1,2])*z/K[1,1],-z]@camera_world[:3,:3].T+camera_world[:3,3]
    candidates=np.flatnonzero(((points>=vertices.min(0)-padding)&(points<=vertices.max(0)+padding)).all(1))
    for first in range(0,len(equations),24):
        if not len(candidates):break
        e=equations[first:first+24]
        candidates=candidates[(points[candidates]@e[:,:3].T+e[:,3]<=padding).all(1)]
    d[v[candidates],u[candidates]]=0
    return d
