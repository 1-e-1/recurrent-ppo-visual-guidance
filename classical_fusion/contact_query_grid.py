"""Cache an existing contact permission in a PRIVATE query mask, not the map.

(unsafe cells intersect hull) minus wholly authorized cells equals
(unsafe cells minus wholly authorized cells) intersect hull. The second order
avoids enumerating the same permitted source volume at every chord subdivision.
No source TSDF, free vote or snapshot field is modified or exported as free.
"""
from itertools import combinations
import numpy as np


def build_contact_query_grid(column,snapshot):
    e=np.asarray(column.equations,float);vertices=[]
    for i,j in combinations(range(len(e)),2):
        matrix=e[[i,j],:2]
        if abs(np.linalg.det(matrix))<1e-12:continue
        p=np.linalg.solve(matrix,-e[[i,j],2])
        if np.isfinite(p).all() and (e[:,:2]@p+e[:,2]<=1e-8).all():vertices.append(p)
    original=snapshot.clearance_grid
    if not vertices:return original
    v=np.asarray(vertices);lo=np.r_[v.min(0),column.lower];hi=np.r_[v.max(0),column.upper]
    first=np.maximum(0,np.floor((lo-snapshot.lower)/snapshot.spacing).astype(int)-1)
    last=np.minimum(snapshot.shape,np.ceil((hi-snapshot.lower)/snapshot.spacing).astype(int)+1)
    shape=last-first
    if np.any(shape<=0) or np.prod(shape)>3_000_000:return original
    result=original.copy();count=int(np.prod(shape));name=next(iter(column.links))
    for begin in range(0,count,65536):
        ids=np.column_stack(np.unravel_index(np.arange(begin,min(begin+65536,count)),tuple(shape)))+first
        centers=snapshot.lower+(ids+.5)*snapshot.spacing
        allowed=column.cells(name,centers,snapshot.spacing)
        result[tuple(ids[allowed].T)]=1
    result.flags.writeable=False
    return result
