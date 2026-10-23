"""Cheap measured-cell snapshot for direct robot-hull volume certificates.

No global EDT is constructed. Clearance returns the valid conservative lower
bound zero, so it can NEVER grant the sphere fast path. Hull/chord refinement
uses binary known-free cells directly; zeros still mean unknown/occupied.
"""
import time
import numpy as np


class RobotQuerySnapshot:
    def __init__(self,mapper):
        with mapper._lock:
            self.mapper=mapper;self.revision=mapper.revision
            free=(mapper.free_views>=mapper.cfg.minimum_free_views)&~mapper.surface_veto&((mapper.weight==0)|(mapper.distance>0))
            self.clearance_grid=free.cpu().numpy().astype(np.uint8)
            self.clearance_grid.flags.writeable=False
            self.free_fraction=float(self.clearance_grid.mean())
            self.lower=mapper.lower.cpu().numpy().astype(float);self.spacing=mapper.spacing.cpu().numpy().astype(float)
            self.shape=np.array(mapper.shape);self.margin=mapper.cfg.planning_margin_m
            self.cell_radius=float(np.linalg.norm(self.spacing)*.5)
            self.built_at=mapper.latest_observation_time or time.monotonic();self.max_age=mapper.cfg.snapshot_max_age_s
            self.representation='binary_evidence_no_metric_distance'

    def clearance(self,points,*,now_s=None):
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=3:raise ValueError('expected points [N,3]')
        now=time.monotonic() if now_s is None else float(now_s)
        valid=np.isfinite(p).all(1)&((p>=self.lower)&(p<self.lower+self.spacing*self.shape)).all(1)
        valid &= np.isfinite(now) and 0<=now-self.built_at<=self.max_age and self.mapper.revision==self.revision
        return np.zeros(len(p)),valid
