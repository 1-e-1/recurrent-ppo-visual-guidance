"""Measured ROI capacity, separate from resolution and safety margins."""
import numpy as np


def measured_roi(source, destination, maximum_cells=12_000_000):
    if isinstance(maximum_cells, bool) or not isinstance(maximum_cells, int) or not 1<=maximum_cells<=24_000_000:
        raise ValueError('local map capacity must be an integer in [1, 24000000]')
    p=np.asarray([source,destination],dtype=float)
    if p.shape!=(2,3) or not np.isfinite(p).all():raise ValueError('finite measured endpoints required')
    lower=p.min(0)-[.14,.14,.10];upper=p.max(0)+[.14,.14,.24]
    cells=int(np.prod(np.ceil((upper-lower)/.002)))
    if cells>maximum_cells:raise RuntimeError('local_transfer_map_budget_exceeded')
    return lower,upper,cells
