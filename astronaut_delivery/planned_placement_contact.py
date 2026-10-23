"""Offline candidate contact construction with the existing observed-support gate.

Not permission to revise an active contact column during descent. A controller
would need to select and reach this pose before descent and reobserve support.
"""
from itertools import product
import numpy as np
from classical_fusion.payload_contact_guard import SupportContactColumn


def planned_contact(expected,place_z,surface):
    vertices=np.asarray(expected,float);observed=np.asarray(surface,float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4
        or observed.ndim!=2 or observed.shape[1]!=3
        or not np.isfinite(vertices).all() or not np.isfinite(observed).all()
        or not np.isfinite(place_z)):raise ValueError('finite measured geometry required')
    footprint=(vertices[:,None,:2]+np.array(list(product([-.009,.009],repeat=2)))[None]).reshape(-1,2)
    column=SupportContactColumn(footprint,float(vertices[:,2].min())-.009,place_z+.04,['payload'])
    e=column.equations
    patch=observed[(observed[:,:2]@e[:,:2].T+e[:,2]<=0).all(1)]
    support=int(np.sum(abs(patch[:,2]-place_z)<.006))
    raised=int(np.sum((patch[:,2]>place_z+.012)&(patch[:,2]<place_z+.15)))
    return (column if support>=60 and raised==0 else None),dict(
        observed_support_points=support,raised_surface_points=raised,
        scope='Original 9mm footprint padding, vertical bounds and support gate; no map relabeling')
