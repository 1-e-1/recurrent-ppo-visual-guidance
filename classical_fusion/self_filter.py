"""Spatial-index acceleration of the same FK sphere self-return predicate."""
import numpy as np
from scipy.spatial import cKDTree


def sphere_return_mask(points,centers,radii):
    p=np.asarray(points,float);c=np.asarray(centers,float);r=np.asarray(radii,float)
    if p.ndim!=2 or p.shape[1]!=3 or c.ndim!=2 or c.shape[1]!=3 or r.shape!=(len(c),):
        raise ValueError('expected point arrays and sphere radii')
    if not np.isfinite(p).all() or not np.isfinite(c).all() or not np.isfinite(r).all() or (r<=0).any():
        raise ValueError('invalid self geometry')
    mask=np.zeros(len(p),bool)
    if not len(p) or not len(c):return mask
    tree=cKDTree(p)
    for center,radius,indices in zip(c,r,tree.query_ball_point(c,r+1e-12)):
        if not len(indices):continue
        ids=np.asarray(indices,int)
        # Preserve the exact strict predicate used by the dense implementation.
        mask[ids]|=np.sum((p[ids]-center)**2,axis=1)<radius**2
    return mask
