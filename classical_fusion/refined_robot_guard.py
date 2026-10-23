"""Geometric partition into complete robot components, never optimistic OR.

Fine evidence replaces the broad query only when it contains the entire padded
component sweep. Otherwise the original broad checker owns that component.
Once assigned to fine evidence, unknown fine cells cannot fall back to broad.
"""
import copy
from dataclasses import replace
import numpy as np
from .rgbd_guard import RGBDChordEnvironmentGuard,robot_hulls
from .local_tool_guard import SelectedRobotHulls


class RefinedRobotGuard:
    def __init__(self,base,snapshot,names,generation):
        self.base=base;self.snapshot=base.snapshot;self.local_snapshot=base.local_snapshot
        self.refinement_snapshot=snapshot;self.names=frozenset(names);self.generation=generation
        self.last_rejection=None
        left,right=base.broad.arms;self.right=SelectedRobotHulls(right,lambda n:n in self.names)
        self.partitioned=copy.copy(base)
        budget=dict(max_boxes=base.broad.max_boxes,max_depth=base.broad.max_depth,max_cell_tests=base.broad.max_cell_tests)
        self.partitioned.broad=RGBDChordEnvironmentGuard(left,SelectedRobotHulls(right,lambda n:n not in self.names),base.snapshot,**budget)
        self.refined=RGBDChordEnvironmentGuard(SelectedRobotHulls(left,lambda n:False),self.right,snapshot,**budget)

    def covered(self,start,end):
        a,b=np.asarray(start,float),np.asarray(end,float)
        if a.shape!=(16,) or b.shape!=(16,) or not np.isfinite([a,b]).all():return False
        s=self.refinement_snapshot;lo=s.lower;hi=lo+s.spacing*s.shape
        pad=s.margin+self.right.motion_radius*np.sum(abs(b[8:]-a[8:]))**2/8
        seen=set()
        for q in (a[8:],b[8:]):
            for name,(v,n) in robot_hulls(self.right,q):
                v=np.asarray(v,float)
                if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():return False
                if np.any(v.min(0)-pad<=lo) or np.any(v.max(0)+pad>=hi):return False
                seen.add(name)
        return seen==self.names and bool(seen)

    def segment(self,start,end,*,now_s=None):
        self.last_rejection=None
        if not self.covered(start,end):
            result=self.base.segment(start,end,now_s=now_s);self.last_rejection=self.base.last_rejection
            return result
        coarse=self.partitioned.segment(start,end,now_s=now_s)
        if not coarse.allowed:
            self.last_rejection=self.partitioned.last_rejection;return coarse
        fine=self.refined.segment(start,end,now_s=now_s);self.last_rejection=self.refined.last_rejection
        return replace(coarse,allowed=fine.allowed,reason='refined_robot/'+fine.reason,
            boxes=coarse.boxes+fine.boxes,minimum_slack_m=min(coarse.minimum_slack_m,fine.minimum_slack_m),
            refinement_revision=self.refinement_snapshot.revision,refinement_generation=self.generation,
            scope=coarse.scope+'_COMPLETE_COMPONENT_RAW_RGBD_REFINEMENT')
