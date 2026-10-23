"""Explicit measured-target contact permission, NOT a free-space certificate.

Only the magnetic tool may overlap the selected measured object's top band.
Other links and all voxels outside that bounded band retain RGB-D-only checks.
This permission relies on a static, isolated, horizontal docking-interface
hypothesis; it is not unconditional unknown-object contact safety.
"""
from dataclasses import dataclass,replace
import numpy as np
from scipy.spatial import ConvexHull
from .rgbd_guard import RGBDChordEnvironmentGuard,EnvironmentCertificate


@dataclass
class MeasuredContactPermit:
    target:np.ndarray
    footprint_equations:np.ndarray
    lower_offset:float=-.008
    upper_offset:float=.040

    def __post_init__(self):
        self.target=np.asarray(self.target,float).copy()
        e=np.asarray(self.footprint_equations,float)
        if (self.target.shape!=(3,) or not np.isfinite(self.target).all() or
            e.ndim!=2 or e.shape[1]!=3 or len(e)<3 or not np.isfinite(e).all()):
            raise ValueError('finite measured target and closed footprint required')
        norms=np.linalg.norm(e[:,:2],axis=1)
        if np.any(norms<1e-10):raise ValueError('degenerate footprint normal')
        e=e/norms[:,None]
        # The contact authority is local even if a malformed/unbounded set of
        # half spaces is supplied. It can never authorize a distant voxel.
        if np.any(e[:,:2]@self.target[:2]+e[:,2]>0):raise ValueError('target outside footprint')
        if not (-.008<=self.lower_offset<=0 and 0<self.upper_offset<=.040):
            raise ValueError('contact band exceeds declared limits')
        self.footprint_equations=e

    @classmethod
    def from_envelope(cls,envelope,target,observed_surface):
        t=np.asarray(target,float);points=np.asarray(observed_surface,float)
        footprint=ConvexHull(envelope.vertices[:,:2]).equations
        inside=(points[:,:2]@footprint[:,:2].T+footprint[:,2]<=0).all(1)
        if np.any(inside&(points[:,2]>t[2]+.015)&(points[:,2]<t[2]+.12)):
            raise ValueError('observed obstruction above docking surface')
        # Contact permission is an explicit task exception, never free evidence.
        # Its bounded top band accounts for physical capture (8 mm), map and
        # target depth budgets (2 x 8 mm), one voxel (8 mm), planning margin
        # (5 mm), and translation calibration (3 mm): 40 mm in this preset.
        # This budget DOES NOT prove absence of an unseen object in the band.
        return cls(t.copy(),footprint.copy(),upper_offset=.008+2*.008+.008+.005+.003)

    def cells(self,name,centers,spacing):
        if name!='magnet':return np.zeros(len(centers),bool)
        p=np.asarray(centers,float);s=np.asarray(spacing,float);e=self.footprint_equations
        if p.ndim!=2 or p.shape[1]!=3 or s.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(s).all() or np.any(s<=0):
            raise ValueError('invalid contact voxel query')
        footprint=(p[:,:2]@e[:,:2].T+e[:,2]+np.abs(e[:,:2])@(s[:2]*.5)<=-1e-9).all(1)
        footprint &= np.linalg.norm(p[:,:2]-self.target[:2],axis=1)+np.linalg.norm(s[:2]*.5)<=.12
        return footprint&(p[:,2]-s[2]*.5>=self.target[2]+self.lower_offset)&(p[:,2]+s[2]*.5<=self.target[2]+self.upper_offset)

    def intersections(self,name,centers,spacing,vertices,normals,padding):
        """Certify only hull/voxel intersections inside the SAME contact set.

        Unknown voxels crossing a footprint boundary are not wholly permitted.
        Instead, clip each voxel to EACH forbidden footprint halfspace. Every
        clipped solid must be separated from the robot convex sweep by more
        than the original margin plus motion remainder. Projection gaps are
        sufficient lower bounds; no optimiser success or free-space filling.
        Z/radius checks still require the entire voxel inside their bounds.
        """
        centers=np.asarray(centers,float);s=np.asarray(spacing,float)
        v=np.asarray(vertices,float);n=np.asarray(normals,float)
        allowed=np.zeros(len(centers),bool)
        if name!='magnet' or len(centers)>32:return allowed
        if (centers.ndim!=2 or centers.shape[1]!=3 or s.shape!=(3,) or np.any(s<=0)
            or v.ndim!=2 or v.shape[1]!=3 or not len(v) or n.ndim!=2 or n.shape[1]!=3
            or not len(n) or not all(np.isfinite(a).all() for a in (centers,s,v,n))
            or not np.isfinite(padding) or padding<0):return allowed
        eligible=(centers[:,2]-s[2]*.5>=self.target[2]+self.lower_offset)&(centers[:,2]+s[2]*.5<=self.target[2]+self.upper_offset)
        eligible&=np.linalg.norm(centers[:,:2]-self.target[:2],axis=1)+np.linalg.norm(s[:2]*.5)<=.12
        from .contact_intersection import footprint_intersections
        allowed[eligible]=footprint_intersections(self.footprint_equations,centers[eligible],s,v,n,padding)
        return allowed


class MeasuredContactGuard(RGBDChordEnvironmentGuard):
    def __init__(self,*args,permit,**kwargs):
        super().__init__(*args,**kwargs);self.permit=permit;self.permitted_cell_queries=0
        self._permit_query_cache=None

    def query_grid(self,name):
        if name!='magnet':return super().query_grid(name)
        from types import SimpleNamespace
        from .contact_query_grid import build_contact_query_grid
        p=self.permit
        key=(p.target.tobytes(),p.footprint_equations.tobytes(),p.lower_offset,p.upper_offset)
        cache=getattr(self,'_permit_query_cache',None)
        if cache is None or cache[0]!=key:
            # Reorder exactly the existing whole-cell exception, including its
            # radius limit. The map, free votes and boundary intersections do
            # not change; other robot links never receive this private mask.
            column=SimpleNamespace(equations=p.footprint_equations,
                lower=p.target[2]+p.lower_offset,upper=p.target[2]+p.upper_offset,
                links={'magnet'},cells=p.cells)
            grid=build_contact_query_grid(column,self.snapshot)
            self._permit_query_cache=(key,grid)
        return self._permit_query_cache[1]

    def authorized_contact_cells(self,name,centers):
        allowed=self.permit.cells(name,centers,self.snapshot.spacing)
        self.permitted_cell_queries+=int(allowed.sum());return allowed

    def authorized_contact_intersections(self,name,centers,vertices,normals,padding):
        allowed=self.permit.intersections(name,centers,self.snapshot.spacing,vertices,normals,padding)
        self.permitted_cell_queries+=int(allowed.sum());return allowed

    def segment(self,start,end,*,now_s=None):
        a,b=np.asarray(start,float),np.asarray(end,float)
        if a.shape!=(16,) or b.shape!=(16,) or not np.isfinite([a,b]).all():
            return EnvironmentCertificate(False,'invalid_contact_joint_path',0,0.,self.snapshot.revision)
        right=self.arms[1];qa,qb=a[8:],b[8:]
        beta=float(np.sum(abs(qb-qa)));remainder=right.motion_radius*beta**2/8
        for q in (qa,qb):
            T=right.tcp(q);vertices=right.hulls_world(q)['magnet'][0]
            if T[2,2]+beta**2/8>-.95 or np.linalg.norm(T[:2,3]-self.permit.target[:2])+remainder>.012:
                return EnvironmentCertificate(False,'outside_vertical_contact_corridor',0,0.,self.snapshot.revision)
            if vertices[:,2].min()-remainder-self.snapshot.margin<self.permit.target[2]-.002:
                return EnvironmentCertificate(False,'contact_penetration_bound',0,0.,self.snapshot.revision)
        result=super().segment(a,b,now_s=now_s)
        return replace(result,scope='RGBD_PLUS_EXPLICIT_MEASURED_TARGET_CONTACT_PERMISSION')
