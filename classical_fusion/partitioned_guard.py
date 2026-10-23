"""Bounded conjunction of unchanged continuous certificates over the same chord."""
from dataclasses import replace,asdict,dataclass
import numpy as np
from classical_fusion.rgbd_guard import EnvironmentCertificate
from classical_fusion.local_tool_guard import SplitCertificate


def certify_partition(base,start,end,*,now_s=None,max_calls=31,max_depth=4):
    if (type(max_calls) is not int or not 1<=max_calls<=127
        or type(max_depth) is not int or not 0<=max_depth<=8):raise ValueError('bounded integer proof budget required')
    a,b=np.asarray(start,float),np.asarray(end,float)
    first=base.segment(a,b,now_s=now_s)
    trace=dict(calls=1,leaves=[],certificates=[],failure=None)
    if first.allowed:return first,trace
    if a.shape!=(16,) or b.shape!=(16,) or not np.isfinite([a,b]).all() or np.array_equal(a,b):
        return first,trace
    expected=(first.revision,getattr(first,'local_revision',None));boxes=first.boxes
    def visit(x,y,lo,hi,depth,existing=None):
        nonlocal boxes
        if existing is None:
            if trace['calls']>=max_calls:
                trace['failure']='partition_proof_budget_exhausted';return False
            c=base.segment(x,y,now_s=now_s);trace['calls']+=1;boxes+=c.boxes
        else:c=existing
        if (c.revision,getattr(c,'local_revision',None))!=expected:
            trace['failure']='partition_map_revision_changed';return False
        if c.allowed:
            trace['leaves'].append([lo,hi]);trace['certificates'].append(asdict(c));return True
        if depth>=max_depth:
            trace['failure']=c.reason;return False
        mid=(x+y)*.5;t=(lo+hi)*.5
        return visit(x,mid,lo,t,depth+1) and visit(mid,y,t,hi,depth+1)
    allowed=visit(a,b,0.,1.,0,first)
    revisions={}
    if allowed:
        for key in ('refinement_revision','refinement_generation'):
            values={c[key] for c in trace['certificates'] if c.get(key) is not None}
            if len(values)>1:
                allowed=False;trace['failure']='partition_'+key+'_changed';break
            if values and hasattr(first,key):revisions[key]=values.pop()
    return replace(first,allowed=allowed,reason='certified_chord_partition' if allowed else trace['failure'],
        boxes=boxes,minimum_slack_m=0.,scope=first.scope+'_PARTITIONED_SAME_CHORD',**revisions),trace


@dataclass(frozen=True)
class PartitionedEnvironmentCertificate(EnvironmentCertificate):
    partition:dict|None=None


@dataclass(frozen=True)
class PartitionedSplitCertificate(SplitCertificate):
    partition:dict|None=None


class PartitionedGuard:
    def __init__(self,base):
        if isinstance(base,PartitionedGuard):raise ValueError('nested partition budgets forbidden')
        self.base=base

    def __getattr__(self,name):return getattr(self.base,name)

    def segment(self,start,end,*,now_s=None):
        certificate,trace=certify_partition(self.base,start,end,now_s=now_s)
        if trace['calls']==1:return certificate
        cls=PartitionedSplitCertificate if isinstance(certificate,SplitCertificate) else PartitionedEnvironmentCertificate
        return cls(**asdict(certificate),partition=trace)
