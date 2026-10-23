"""Balance task visibility and rejected measured-space evidence for a rescan."""
import numpy as np


def recovery_focus(target,decisions):
    target=np.asarray(target,float)
    if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('finite measured task target required')
    points=[]
    for row in decisions[-8:]:
        rejection=row.get('rejection') or {}
        p=np.asarray(rejection.get('voxel_centers_m',[]),float)
        if p.size and p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all():
            points.extend(p[np.linalg.norm(p-target,axis=1)<.6])
    if not points:return target.copy(),None
    frontier=np.median(points,axis=0)
    # Both locations must still satisfy the real camera/task checks. This
    # orientation proposal is neither a free-space claim nor an occlusion proof.
    return (target+frontier)*.5,frontier
