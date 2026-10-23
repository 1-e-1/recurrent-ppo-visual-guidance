"""Split a robot certificate by geometry, NEVER by optimistic map precedence.

All non-tool robot solids must pass the broad measured map. The complete
magnetic tool sweep must independently pass the fine measured map. An unknown
fine voxel cannot inherit FREE from a coarse voxel. Both revisions are logged.
"""
from dataclasses import dataclass
from .rgbd_guard import RGBDChordEnvironmentGuard,robot_hulls
from .measured_contact import MeasuredContactGuard


class SelectedRobotHulls:
    def __init__(self,arm,selector):self.arm=arm;self.selector=selector
    def __getattr__(self,name):return getattr(self.arm,name)
    def hulls_world(self,q):return {k:v for k,v in self.arm.hulls_world(q).items() if self.selector(k)}
    def collision_hulls_world(self,q):return [(k,v) for k,v in robot_hulls(self.arm,q) if self.selector(k)]


@dataclass(frozen=True)
class SplitCertificate:
    allowed:bool
    reason:str
    boxes:int
    minimum_slack_m:float
    revision:int
    local_revision:int
    scope:str='RGBD_BROAD_ROBOT_AND_FINE_TOOL_WITH_EXPLICIT_MEASURED_TARGET_CONTACT'
    refinement_revision:int|None=None
    refinement_generation:int|None=None


class LocalToolEnvironmentGuard:
    def __init__(self,left,right,broad,fine,permit,**budget):
        self.snapshot=broad;self.local_snapshot=fine;self.last_rejection=None
        self.broad=RGBDChordEnvironmentGuard(left,SelectedRobotHulls(right,lambda n:n!='magnet'),broad,**budget)
        self.fine=MeasuredContactGuard(SelectedRobotHulls(left,lambda n:False),
            SelectedRobotHulls(right,lambda n:n=='magnet'),fine,permit=permit,**budget)

    def segment(self,start,end,*,now_s=None):
        self.last_rejection=None
        coarse=self.broad.segment(start,end,now_s=now_s)
        if not coarse.allowed:
            self.last_rejection=self.broad.last_rejection
            return SplitCertificate(False,'broad/'+coarse.reason,coarse.boxes,coarse.minimum_slack_m,self.snapshot.revision,self.local_snapshot.revision)
        fine=self.fine.segment(start,end,now_s=now_s);self.last_rejection=self.fine.last_rejection
        return SplitCertificate(fine.allowed,'fine/'+fine.reason,coarse.boxes+fine.boxes,
            min(coarse.minimum_slack_m,fine.minimum_slack_m),self.snapshot.revision,self.local_snapshot.revision)
