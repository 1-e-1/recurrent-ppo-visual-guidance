"""Conservative axial convex cover of a KNOWN robot triangle surface.

For each axial slab retain every triangle that intersects it, including all
three original vertices (no centroid-only assignment). A cross-section of a
closed solid lies in the convex hull of its boundary; its boundary triangles
are all retained. Thus the union covers the closed solid, while avoiding a
single convex hull spanning a stepped tool's empty exterior. This is not
reconstruction/completion of unknown environment objects.
"""
import numpy as np
from scipy.spatial import ConvexHull
from itertools import product


def axial_cover(points,faces,bands=8,axis=2):
    p=np.asarray(points,float);f=np.asarray(faces)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or len(p)<4 or
        f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu' or not len(f) or
        f.min()<0 or f.max()>=len(p) or axis not in (0,1,2) or not 1<=bands<=32):
        raise ValueError('valid known robot triangle mesh and bounded slab count required')
    edges=np.linspace(p[:,axis].min(),p[:,axis].max(),bands+1)
    tri=p[f];lo=tri[:,:,axis].min(1);hi=tri[:,:,axis].max(1);cover=[]
    for a,b in zip(edges[:-1],edges[1:]):
        selected=tri[(lo<=b+1e-10)&(hi>=a-1e-10)].reshape(-1,3)
        if not len(selected):continue
        selected=np.unique(selected,axis=0)
        if len(selected)<4 or np.linalg.matrix_rank(selected-selected[0],tol=1e-10)<3:
            lower=selected.min(0)-1e-9;upper=selected.max(0)+1e-9
            selected=np.array(list(product(*zip(lower,upper))))
        hull=ConvexHull(selected)
        cover.append((selected[hull.vertices],hull.equations))
    return cover
