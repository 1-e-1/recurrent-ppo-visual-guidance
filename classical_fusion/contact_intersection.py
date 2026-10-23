"""Geometric proof for intersections inside an EXISTING footprint permission.

Callers enforce link ownership, the original vertical band and locality. This
helper changes neither the permission footprint nor any map evidence.
"""
from itertools import product,combinations
import numpy as np


def footprint_intersections(equations,centers,spacing,vertices,normals,padding):
    centers=np.asarray(centers,float);s=np.asarray(spacing,float)
    v=np.asarray(vertices,float);n=np.asarray(normals,float);e=np.asarray(equations,float)
    allowed=np.zeros(len(centers),bool)
    if len(centers)>32:return allowed
    if (centers.ndim!=2 or centers.shape[1]!=3 or s.shape!=(3,) or np.any(s<=0)
        or v.ndim!=2 or v.shape[1]!=3 or not len(v) or n.ndim!=2 or n.shape[1]!=3 or not len(n)
        or e.ndim!=2 or e.shape[1]!=3 or len(e)<3
        or not all(np.isfinite(a).all() for a in (centers,s,v,n,e))
        or not np.isfinite(padding) or padding<0):return allowed
    lengths=np.linalg.norm(e[:,:2],axis=1)
    if np.any(lengths<1e-12):return allowed
    e=e/lengths[:,None]
    axes=np.concatenate([n,np.eye(3),np.c_[e[:,:2],np.zeros(len(e))]])
    lengths=np.linalg.norm(axes,axis=1)
    if np.any(lengths<1e-12):return allowed
    axes/=lengths[:,None];projection=v@axes.T
    robot_lo,robot_hi=projection.min(0),projection.max(0)
    corners=np.array(list(product([-.5,.5],repeat=3)))
    edges=[(i,j) for i,j in combinations(range(8),2) if np.count_nonzero(corners[i]!=corners[j])==1]
    for k,c in enumerate(centers):
        cube=c+corners*s;certified=True
        for face in e:
            # Enlarge the forbidden part, including floating-point boundary.
            d=cube[:,:2]@face[:2]+face[2]+1e-9
            if d.max()<0:continue
            clipped=list(cube[d>=0])
            for i,j in edges:
                if d[i]*d[j]<0:clipped.append(cube[i]+(cube[j]-cube[i])*d[i]/(d[i]-d[j]))
            part=np.asarray(clipped)@axes.T
            gap=np.maximum(part.min(0)-robot_hi,robot_lo-part.max(0)).max()
            if not np.isfinite(gap) or gap<=padding+1e-9:
                certified=False;break
        allowed[k]=certified
    return allowed
