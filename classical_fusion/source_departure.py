"""Continuous lower-plane bound for departure from an owned source region.

This does NOT certify free space. A separate environment guard must restrict
the contact exception to the original measured isolated-object source column.
"""
import numpy as np
from .contact_corridor import CorridorCertificate


def certify(arm, start, end, minimum_z, *, names=('payload', 'magnet'), max_boxes=127, max_depth=8):
    a, b = np.asarray(start, float), np.asarray(end, float)
    if (a.shape != (8,) or b.shape != (8,) or not np.isfinite([a,b]).all()
            or not np.isfinite(minimum_z) or not np.isfinite(arm.motion_radius)
            or arm.motion_radius <= 0 or not names or not set(names) <= {'payload','magnet'}):
        return CorridorCertificate(False, 'invalid_owned_source_departure', 0)
    count=0; reason='source_departure_proof_budget'; cache={}
    def slack(q):
        key=q.tobytes()
        if key not in cache:
            hulls=arm.hulls_world(q)
            if any(n not in hulls for n in names): return -np.inf
            vertices=np.concatenate([hulls[n][0] for n in names])
            if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all(): return -np.inf
            cache[key]=float(vertices[:,2].min()-minimum_z)
        return cache[key]
    def visit(x,y,depth):
        nonlocal count,reason
        if count>=max_boxes:return False
        count+=1; margin=min(slack(x),slack(y))
        if margin<0:
            reason='source_departure_below_measured_support_clearance';return False
        if margin>=arm.motion_radius*np.sum(abs(y-x))**2/8:return True
        if depth>=max_depth:return False
        mid=(x+y)*.5
        return visit(x,mid,depth+1) and visit(mid,y,depth+1)
    allowed=visit(a,b,0)
    return CorridorCertificate(allowed,'continuous_owned_source_clearance' if allowed else reason,count)
