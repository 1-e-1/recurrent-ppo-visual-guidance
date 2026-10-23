"""Object-envelope proposals from measured surfaces, never environment CAD.

Proposal is conditional on connected, above-support-plane objects. A convex
measurement envelope is NOT proof of unseen-volume containment; callers must
retain that scope and audit coverage before a manipulation claim.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull,cKDTree


def measured_surface_padding(config,spacing,K,camera_world,target):
    """Euclidean padding from the same declared sensor/calibration budgets.

    The depth budget is axial, hence multiplied by ray scale. Translation,
    rotation, pixel calibration and voxel quantization are separately bounded.
    This is conditional on those engineering bounds, not a hardware guarantee.
    """
    T=np.asarray(camera_world,float);K=np.asarray(K,float);p=np.asarray(target,float)
    local=(p-T[:3,3])@T[:3,:3];distance=float(np.linalg.norm(local));axial=-local[2]
    if axial<=.05 or not np.isfinite(local).all() or min(K[0,0],K[1,1])<=0:raise ValueError('invalid measured padding geometry')
    return float(config.depth_error_bound_m*distance/axial+config.extrinsic_translation_bound_m+
        2*distance*np.sin(config.extrinsic_rotation_bound_rad*.5)+
        config.intrinsic_pixel_bound*distance/min(K[0,0],K[1,1])+np.linalg.norm(spacing)*.5)


@dataclass
class MeasuredObjectEnvelope:
    vertices:np.ndarray
    faces:np.ndarray
    equations:np.ndarray
    support_z:float
    observed_points:int
    margin_m:float
    unseen_containment_proven:bool=False
    observed_surface_indices:np.ndarray|None=None

    def contains(self,points):
        return (np.asarray(points)@self.equations[:,:3].T+self.equations[:,3]<=1e-8).all(1)


def propose_envelope(surface,target,search_radius=.10,margin=.012,*,support_exclusion_m=.02):
    p=np.asarray(surface,float);t=np.asarray(target,float)
    if p.ndim!=2 or p.shape[1]!=3 or t.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('finite measured surface and target required')
    if not .06<=search_radius<=.2 or not .005<=margin<=.03:raise ValueError('invalid measured envelope scale')
    if not np.isfinite(support_exclusion_m) or not .004<=support_exclusion_m<=.02:
        raise ValueError('invalid support exclusion distance')
    xy=np.linalg.norm(p[:,:2]-t[:2],axis=1)
    # Thin tools can have a measured interface only 14 mm above the support.
    # This search parameter does not change collision clearance or map votes.
    annulus=p[(xy>.055)&(xy<.15)&(p[:,2]<t[2]-support_exclusion_m)&(p[:,2]>t[2]-.15)]
    if len(annulus)<30:raise ValueError('no observed support plane')
    bins=np.round(annulus[:,2]/.002).astype(int);keys,count=np.unique(bins,return_counts=True);peak=keys[np.argmax(count)]*.002
    near=annulus[abs(annulus[:,2]-peak)<.004]
    support=float(np.median(near[:,2]))
    candidate_ids=np.flatnonzero((xy<search_radius)&(p[:,2]>support+.008)&(p[:,2]<t[2]+.025))
    candidates=p[candidate_ids]
    if len(candidates)<12:raise ValueError('insufficient measured object surface')
    tree=cKDTree(candidates);seeds=tree.query_ball_point(t,.026)
    if not seeds:raise ValueError('target not connected to measured surface')
    found=set(seeds);todo=list(seeds)
    while todo:
        current=todo.pop()
        for i in tree.query_ball_point(candidates[current],.016):
            if i not in found:found.add(i);todo.append(i)
    points=candidates[sorted(found)]
    if len(points)<12 or np.linalg.norm(points[:,:2]-t[:2],axis=1).max()>search_radius-.012:
        raise ValueError('component truncated by search boundary')
    # Extend the measured footprint to the observed support plane. This is a
    # declared object-support hypothesis, not newly observed depth or free space.
    floor=points.copy();floor[:,2]=support
    points=np.r_[points,floor];hull=ConvexHull(points);core=points[hull.vertices]
    from .euclidean_envelope import outer_euclidean_offset
    expanded,hull=outer_euclidean_offset(core,margin)
    return MeasuredObjectEnvelope(expanded,hull.simplices,hull.equations,support,len(found),margin,
        observed_surface_indices=candidate_ids[sorted(found)])
