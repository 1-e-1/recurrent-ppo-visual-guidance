"""Geometry-only pre-descent proposals; no runtime motion authority or oracle."""
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .planned_placement_contact import planned_contact


def separating_gap(a,b):
    axes=np.r_[ConvexHull(a).equations[:,:3],ConvexHull(b).equations[:,:3]]
    aa=a@axes.T;bb=b@axes.T
    return float(np.maximum(aa.min(0)-bb.max(0),bb.min(0)-aa.max(0)).max())


def placement_candidates(local_points,current_rotation,goal,place_z,released_envelopes,surface,
                         *,minimum_gap=.003,maximum_candidates=4):
    local=np.asarray(local_points,float);rotation=np.asarray(current_rotation,float);goal=np.asarray(goal,float)
    released=[np.asarray(v,float) for v in released_envelopes]
    if (local.ndim!=2 or local.shape[1]!=3 or not 4<=len(local)<=3000
        or rotation.shape!=(3,3) or goal.shape!=(3,) or not np.isfinite(local).all()
        or not np.isfinite(rotation).all() or not np.isfinite(goal).all()
        or not np.isfinite([place_z,minimum_gap]).all() or minimum_gap<.003
        or not isinstance(maximum_candidates,int) or not 1<=maximum_candidates<=4
        or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0.)
        or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6,rtol=0.)):
        raise ValueError('finite measured rigid geometry and bounded candidate budget required')
    if any(v.ndim!=2 or v.shape[1]!=3 or not 4<=len(v)<=3000 or not np.isfinite(v).all() for v in released):
        raise ValueError('finite released measured hulls required')
    proposals=[];rejected=dict(packing=0,support=0)
    for yaw,offset in product((0.,-15.,15.,-30.,30.,-60.,60.,-90.,90.),
                             ((0.,0.,0.),(.004,0.,0.),(-.004,0.,0.),(0.,.004,0.),(0.,-.004,0.))):
        R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@rotation
        position=goal+offset;expected=local@R.T+position
        gap=min((separating_gap(expected,v) for v in released),default=float('inf'))
        if gap<=minimum_gap:rejected['packing']+=1;continue
        column,evidence=planned_contact(expected,place_z,surface)
        if column is None:rejected['support']+=1;continue
        # Prefer useful clearance up to 10mm, then less displacement/rotation.
        rank=(-min(gap,.010),float(np.linalg.norm(offset)),abs(yaw),yaw,offset)
        proposals.append(dict(rotation=R,position=position,column=column,expected=expected,
            yaw_delta_deg=yaw,offset_m=list(offset),minimum_measured_gap_m=None if not released else gap,
            support_evidence=evidence,_rank=rank))
    proposals.sort(key=lambda p:p['_rank'])
    for p in proposals:p.pop('_rank')
    return proposals[:maximum_candidates],dict(evaluated=45,admitted=len(proposals),rejected=rejected,
        scope='Measured geometry proposals only; original local map/IK/swept guards and fresh visual feedback still required')
