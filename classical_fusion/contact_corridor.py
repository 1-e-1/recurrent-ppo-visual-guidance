"""Bounded continuous contact-corridor proof, not sampled path acceptance.

Subdivision tightens the original R * ||delta q||_1^2 / 8 remainder. It never
widens the measured 8 mm lateral corridor or 2 mm lower penetration limit.
Only the carrying arm enters this bound; camera-only motion adds no remainder.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CorridorCertificate:
    allowed:bool
    reason:str
    boxes:int


def certify_corridor(arm,start,end,source_xy,initial_bottom,*,max_boxes=127,max_depth=8):
    if max_boxes<1 or max_depth<0:raise ValueError('invalid corridor certificate budget')
    a,b=np.asarray(start,float),np.asarray(end,float);xy=np.asarray(source_xy,float)
    radius=arm.motion_radius
    if (a.shape!=(8,) or b.shape!=(8,) or xy.shape!=(2,)
        or not np.isfinite(np.r_[a,b,xy,initial_bottom,radius]).all() or radius<=0):
        return CorridorCertificate(False,'invalid_contact_corridor_geometry',0)
    cache={};count=0;failure='contact_corridor_budget_exhausted'
    def margins(q):
        key=q.tobytes()
        if key not in cache:
            T=np.asarray(arm.tcp(q));v=np.asarray(arm.hulls_world(q)['payload'][0])
            if (T.shape!=(4,4) or v.ndim!=2 or v.shape[1]!=3 or not len(v)
                or not np.isfinite(T).all() or not np.isfinite(v).all()):return None
            cache[key]=np.array([.008-np.linalg.norm(T[:2,3]-xy),v[:,2].min()-initial_bottom+.002])
        return cache[key]
    def visit(x,y,depth):
        nonlocal count,failure
        if count>=max_boxes:return False
        count+=1;mx,my=margins(x),margins(y)
        if mx is None or my is None:
            failure='invalid_contact_corridor_geometry';return False
        slack=float(min(mx.min(),my.min()))
        if slack<0:
            failure='outside_upward_source_extraction_corridor';return False
        remainder=radius*np.sum(abs(y-x))**2/8
        if slack>=remainder:return True
        if depth>=max_depth:return False
        mid=(x+y)*.5
        return visit(x,mid,depth+1) and visit(mid,y,depth+1)
    allowed=visit(a,b,0)
    return CorridorCertificate(allowed,'continuous_contact_corridor' if allowed else failure,count)
