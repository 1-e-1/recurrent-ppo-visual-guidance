"""Measured payload/source/support permissions with explicit contact scope.

This is CONDITIONAL manipulation of an isolated, supported measured object.
The selected source is not declared FREE. Only tool/payload may traverse its
bounded ownership column during upward extraction. Other robot links retain
their normal measured-space checks. The measured envelope must independently
contain the physical payload; the controller cannot prove unseen containment.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial import ConvexHull
from .rgbd_guard import RGBDChordEnvironmentGuard,EnvironmentCertificate
from .local_tool_guard import SelectedRobotHulls,SplitCertificate


class SupportContactColumn:
    def __init__(self,footprint_points,lower,upper,links,*,intersection_cell_budget=32):
        if type(intersection_cell_budget) is not int or intersection_cell_budget not in (32,128):
            raise ValueError('contact proof budget must be 32 or 128')
        self.intersection_cell_budget=intersection_cell_budget
        p=np.asarray(footprint_points,float)
        if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or not np.isfinite([lower,upper]).all() or lower>=upper:
            raise ValueError('invalid measured contact column')
        if not set(links)<={'payload','magnet'}:raise ValueError('contact cannot exempt robot links')
        if not len(links) or np.ptp(p,axis=0).max()>.30 or upper-lower>.25:
            raise ValueError('unbounded contact exception')
        self.equations=ConvexHull(p).equations;self.lower=lower;self.upper=upper;self.links=set(links)
    def cells(self,name,centers,spacing):
        if name not in self.links:return np.zeros(len(centers),bool)
        p=np.asarray(centers);half=np.asarray(spacing)*.5;e=self.equations
        inside=(p[:,:2]@e[:,:2].T+e[:,2]+abs(e[:,:2])@half[:2]<=1e-9).all(1)
        return inside&(p[:,2]-half[2]>=self.lower)&(p[:,2]+half[2]<=self.upper)

    def intersections(self,name,centers,spacing,vertices,normals,padding):
        """Same bounded column, but certify the actual swept intersection."""
        p=np.asarray(centers,float);s=np.asarray(spacing,float);allowed=np.zeros(len(p),bool)
        if name not in self.links or len(p)>getattr(self,'intersection_cell_budget',32):return allowed
        if p.ndim!=2 or p.shape[1]!=3 or s.shape!=(3,) or not np.isfinite(s).all() or np.any(s<=0):return allowed
        eligible=(p[:,2]-s[2]*.5>=self.lower)&(p[:,2]+s[2]*.5<=self.upper)
        from .contact_intersection import footprint_intersections
        # Partition computation only: identical footprint, vertical band and
        # motion padding. Each cell must pass the original geometric proof.
        ids=np.flatnonzero(eligible)
        for begin in range(0,len(ids),32):
            batch=ids[begin:begin+32]
            allowed[batch]=footprint_intersections(self.equations,p[batch],s,vertices,normals,padding)
        return allowed


class PayloadFineGuard(RGBDChordEnvironmentGuard):
    def __init__(self,*args,column=None,phase='transport',initial_bottom=None,source_xy=None,departure_minimum_z=None,**kwargs):
        super().__init__(*args,**kwargs);self.column=column;self.phase=phase;self.initial_bottom=initial_bottom;self.source_xy=source_xy
        self._contact_query_grid=None
        self.departure_minimum_z=departure_minimum_z
        if phase not in ('lift','transport','place','source_departure'):raise ValueError('invalid payload phase')
        if phase=='source_departure' and (column is None or departure_minimum_z is None or not np.isfinite(departure_minimum_z)):
            raise ValueError('bounded measured source ownership and support clearance required')
        if phase=='transport' and column is not None:raise ValueError('transport cannot exempt unknown space')
        if phase in ('lift','place') and (column is None or source_xy is None or initial_bottom is None):raise ValueError('contact corridor geometry required')
    def authorized_contact_cells(self,name,centers):
        return np.zeros(len(centers),bool) if self.column is None else self.column.cells(name,centers,self.snapshot.spacing)
    def authorized_contact_intersections(self,name,centers,vertices,normals,padding):
        return np.zeros(len(centers),bool) if self.column is None else self.column.intersections(name,centers,self.snapshot.spacing,vertices,normals,padding)
    def query_grid(self,name):
        if self.column is None or name not in self.column.links:return super().query_grid(name)
        if self._contact_query_grid is None:
            from .contact_query_grid import build_contact_query_grid
            self._contact_query_grid=build_contact_query_grid(self.column,self.snapshot)
        return self._contact_query_grid
    def segment(self,start,end,*,now_s=None):
        a,b=np.asarray(start,float),np.asarray(end,float)
        if a.shape!=(16,) or b.shape!=(16,) or not np.isfinite([a,b]).all():
            return EnvironmentCertificate(False,'invalid_payload_path',0,0.,self.snapshot.revision)
        corridor_boxes=0
        if self.phase=='source_departure':
            from .source_departure import certify
            corridor=certify(self.arms[1],a[8:],b[8:],self.departure_minimum_z,names=tuple(sorted(self.column.links)))
            corridor_boxes=corridor.boxes
            if not corridor.allowed:
                return EnvironmentCertificate(False,corridor.reason,corridor_boxes,0.,self.snapshot.revision)
        if self.phase in ('lift','place'):
            from .contact_corridor import certify_corridor
            corridor=certify_corridor(self.arms[1],a[8:],b[8:],self.source_xy,self.initial_bottom,
                max_boxes=127,max_depth=8)
            corridor_boxes=corridor.boxes
            if not corridor.allowed:
                return EnvironmentCertificate(False,corridor.reason,corridor_boxes,0.,self.snapshot.revision)
        result=super().segment(a,b,now_s=now_s)
        return replace(result,boxes=result.boxes+corridor_boxes,scope=(
            'RGBD_PAYLOAD_WITH_CONDITIONAL_OWNED_SOURCE_DEPARTURE' if self.phase=='source_departure'
            else 'RGBD_PAYLOAD_WITH_EXPLICIT_SOURCE_OR_SUPPORT_CONTACT'))


class PayloadEnvironmentGuard:
    def __init__(self,left,right,broad,fine,**contact):
        self.snapshot=broad;self.local_snapshot=fine;self.last_rejection=None
        self.broad=RGBDChordEnvironmentGuard(left,SelectedRobotHulls(right,lambda n:n not in ('magnet','payload')),
            broad,max_boxes=127,max_depth=8)
        self.fine=PayloadFineGuard(SelectedRobotHulls(left,lambda n:False),SelectedRobotHulls(right,lambda n:n in ('magnet','payload')),
            fine,max_boxes=127,max_depth=8,max_cell_tests=1000000,**contact)
    def segment(self,start,end,*,now_s=None):
        coarse=self.broad.segment(start,end,now_s=now_s)
        if not coarse.allowed:
            self.last_rejection=self.broad.last_rejection
            return SplitCertificate(False,'broad/'+coarse.reason,coarse.boxes,0.,self.snapshot.revision,self.local_snapshot.revision)
        fine=self.fine.segment(start,end,now_s=now_s);self.last_rejection=self.fine.last_rejection
        return SplitCertificate(fine.allowed,'payload/'+fine.reason,coarse.boxes+fine.boxes,0.,self.snapshot.revision,self.local_snapshot.revision,
            scope='RGBD_BROAD_ROBOT_FINE_PAYLOAD_WITH_EXPLICIT_CONTACT_HYPOTHESES')
