"""Bounded supporting-plane OUTER approximation of a Euclidean offset.

For unit n, support(conv(P) + ball(r), n) = max(P@n) + r.
Intersecting any finite set of these halfspaces contains that entire offset.
This preserves the existing Euclidean measurement-error budget; it does not
prove containment of unseen object geometry or make any map voxel free.
"""
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection


def outer_euclidean_offset(points,radius,maximum_planes=96):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():raise ValueError('finite 3D core required')
    if not np.isfinite(radius) or radius<=0 or not 26<=maximum_planes<=128:raise ValueError('positive radius and bounded plane budget required')
    hull=ConvexHull(points);core=points[hull.vertices]
    axes=np.array([v for v in product([-1.,0.,1.],repeat=3) if any(v)])
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    candidates=hull.equations[:,:3];candidates/=np.linalg.norm(candidates,axis=1)[:,None]
    normals=list(axes);closest=(candidates@axes.T).max(1)
    while len(normals)<maximum_planes:
        i=int(closest.argmin())
        if closest[i]>=np.cos(np.deg2rad(5.)):break
        normals.append(candidates[i].copy());closest=np.maximum(closest,candidates@candidates[i])
    normals=np.asarray(normals);normals/=np.linalg.norm(normals,axis=1)[:,None]
    # Cartesian directions guarantee boundedness. A tiny OUTWARD allowance
    # accounts for floating arithmetic; no inward erosion of the error ball.
    equations=np.c_[normals,-(core@normals.T).max(0)-radius-1e-8]
    vertices=HalfspaceIntersection(equations,core.mean(0)).intersections
    if not np.isfinite(vertices).all() or len(vertices)>2*maximum_planes-4:raise ValueError('invalid bounded offset polytope')
    result=ConvexHull(vertices)
    # Numerical construction must retain every declared supporting constraint
    # without cutting into the protected Euclidean offset.
    n=result.equations[:,:3];supports=(core@n.T).max(0)+radius*np.linalg.norm(n,axis=1)+result.equations[:,3]
    if supports.max()>1e-9:raise ValueError('numerical offset lost error-ball containment')
    return vertices,result
