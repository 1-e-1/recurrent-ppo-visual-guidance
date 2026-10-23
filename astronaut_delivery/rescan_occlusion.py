"""Rank sensing proposals using known robot / measured held-shape occlusion.

This is ONLY a visibility heuristic, never environment free-space authority.
No simulator scene, instance mask, object count or unseen object model is read.
"""
from itertools import product
import numpy as np


def ray_occlusion(camera,points,hulls):
    camera=np.asarray(camera,float);points=np.asarray(points,float)
    if camera.shape!=(3,) or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(camera).all() or not np.isfinite(points).all():
        raise ValueError('finite camera and ray endpoints required')
    visible=np.ones(len(points),bool);length=np.zeros(len(points))
    for vertices,normals in hulls:
        ids=np.arange(len(points))
        vertices=np.asarray(vertices,float);normals=np.asarray(normals,float)
        offsets=-(vertices@normals.T).max(0)
        a=camera@normals.T+offsets;b=(points[ids]-camera)@normals.T
        parallel=np.abs(b)<1e-12
        ratio=-a/np.where(parallel,1.,b)
        entry=np.maximum(0.,np.where(b< -1e-12,ratio,-np.inf).max(1))
        leave=np.minimum(1.,np.where(b>1e-12,ratio,np.inf).min(1))
        impossible=(parallel&(a>1e-10)).any(1)
        blocked=(entry<=leave+1e-10)&(leave>1e-6)&(entry<1-1e-6)&~impossible
        visible[ids[blocked]]=False
        length[ids[blocked]]+=np.maximum(0.,leave[blocked]-entry[blocked])*np.linalg.norm(points[ids[blocked]]-camera,axis=1)
    return visible,length


def ray_visibility(camera,points,hulls):return ray_occlusion(camera,points,hulls)[0]


def rejected_ray_samples(decisions,target):
    target=np.asarray(target,float);points=[];spacing=[]
    for row in decisions[-8:]:
        w=row.get('rejection') or {};p=np.asarray(w.get('voxel_centers_m',[]),float)
        if p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all():
            points.extend(p[np.linalg.norm(p-target,axis=1)<.6])
            s=np.asarray(w.get('spacing_m',[.008]*3),float)
            if s.shape==(3,) and np.isfinite(s).all() and (s>0).all():spacing.append(s)
    if not points:return None
    p=np.unique(np.asarray(points),axis=0)
    if len(p)>20:p=p[np.linspace(0,len(p)-1,20,dtype=int)]
    # A finite ray bundle around the actual blocking cells, not a visibility
    # or depth-noise certificate. Real images and original guards decide later.
    radius=(np.max(spacing,axis=0)*.5 if spacing else np.full(3,.004))+.005
    corners=np.array(list(product([-1.,1.],repeat=3)))*radius
    return np.concatenate([p,(p[:,None]+corners).reshape(-1,3)])


def rank_offsets(camera,offsets,samples,hulls):
    if samples is None:return list(offsets),None
    before,overlap=ray_occlusion(camera,samples,hulls)
    rays=[ray_occlusion(camera+offset,samples,hulls) for offset in offsets]
    scores=[float(v.mean()) for v,l in rays];lengths=[float(l.mean()) for v,l in rays]
    # Length gives a descent direction even if all four nearby views remain
    # behind the held-object silhouette. Overlapping components may double
    # count; it remains a bounded ranking heuristic, not an occlusion proof.
    order=sorted(range(len(offsets)),key=lambda i:(-scores[i],lengths[i],i))
    return [offsets[i] for i in order],dict(current_visible_ray_fraction=float(before.mean()),
        current_mean_occluded_length_m=float(overlap.mean()),proposed_mean_occluded_lengths_m=[lengths[i] for i in order],
        proposed_visible_ray_fractions=[scores[i] for i in order],sample_count=len(samples),
        scope='Known right robot and measured payload occlusion heuristic only; not predicted sensor success or free-space permission')
