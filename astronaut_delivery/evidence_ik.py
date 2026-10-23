"""Rejected measured voxels guide a bounded IK proposal, never certify it.

No CAD environment lookup, distance completion, map mutation, or new contact
permission. The exact full-geometry guard must still accept the result.
"""
import numpy as np


def residual_from_rejections(arm,rejections,margin=.005,start=None):
    hints={}
    for row in rejections[-8:]:
        if not row or row.get('arm_index')!=1:continue
        name=row['link']
        if name not in arm.local_hulls or name in ('magnet','payload'):continue
        p=np.asarray(row.get('voxel_centers_m',[]),float);s=np.asarray(row.get('spacing_m',[]),float)
        if p.ndim!=2 or p.shape[1]!=3 or s.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(s).all() or np.any(s<=0):continue
        hints[name]=(p[:20],s)
    if not hints:return None
    hints=dict(list(hints.items())[-4:])
    def at(q):
        hulls=arm.hulls_world(q);values=[]
        for name,(points,spacing) in hints.items():
            vertices,normals=hulls[name];normals=np.asarray(normals);normals=normals/np.linalg.norm(normals,axis=1)[:,None]
            supports=(vertices@normals.T).max(0)+abs(normals)@(spacing*.5)
            separation=(points@normals.T-supports).max(1)
            values.extend(24*np.maximum(0,margin+.002-separation))
        return np.asarray(values)
    def residual(q):
        return at(q) if start is None else np.r_[at((np.asarray(start)+q)*.5),at(q)]
    return residual
