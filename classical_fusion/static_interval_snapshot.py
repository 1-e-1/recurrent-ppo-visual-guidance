"""Static-epoch multi-view FREE certificates vs possible-surface intervals.

A near-surface uncertainty interval is a set of POSSIBLE surface locations,
not proof that every intersected cell contains a surface. Within an explicitly
unchanged epoch, two independently certified whole-cell free observations may
disambiguate that interval. Actual nonpositive TSDF cells remain blocked.

This does not authorize unknown cells, does not reduce sensor error budgets,
and is NOT valid across undeclared scene changes or out-of-bound sensor errors.
Known manipulation changes must revoke evidence before a new query.
"""
import numpy as np
from .robot_query_snapshot import RobotQuerySnapshot


class StaticIntervalSnapshot(RobotQuerySnapshot):
    def __init__(self,mapper,*,assume_static_epoch=False):
        if not assume_static_epoch:raise ValueError('explicit unchanged static epoch required')
        with mapper._lock:
            super().__init__(mapper)
            free=(mapper.free_views>=mapper.cfg.minimum_free_views)&((mapper.weight==0)|(mapper.distance>0))
            self.disambiguated_possible_surface_cells=int((free&mapper.surface_veto).sum())
            self.clearance_grid=free.cpu().numpy().astype(np.uint8);self.clearance_grid.flags.writeable=False
            self.free_fraction=float(self.clearance_grid.mean())
            self.representation='static_interval_consensus_not_permanent_possible_surface_union'
