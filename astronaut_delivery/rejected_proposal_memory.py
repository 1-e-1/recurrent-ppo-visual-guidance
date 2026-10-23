"""Conservative short-term tabu memory for unexecuted proposals; never authorizes motion."""
import numpy as np


def repeated_rejection(start,end,target,standoff,decisions):
    start,end,target=(np.asarray(x,float) for x in (start,end,target))
    if start.shape!=(16,) or end.shape!=(16,) or target.shape!=(3,) or not np.isfinite(np.r_[start,end,target,standoff]).all():
        raise ValueError('finite joint proposal and measured target required')
    checked=0
    for row in reversed(decisions):
        if row.get('status')!='PRE_EXECUTION_MEASUREMENT_REJECTED':continue
        checked+=1
        if checked>8:break
        p=row.get('proposal',{})
        a=np.asarray(p.get('start',[]));b=np.asarray(p.get('end',[]));t=np.asarray(p.get('target',[]))
        if a.shape!=(16,) or b.shape!=(16,) or t.shape!=(3,):continue
        if not np.allclose(a,start,atol=1e-7,rtol=0):continue
        if abs(float(p.get('standoff_m',np.inf))-standoff)>1e-6 or np.linalg.norm(t-target)>.005:continue
        if np.max(abs(b[:8]-end[:8]))<=.003 and np.max(abs(b[8:]-end[8:]))<=.01:
            return dict(reason=row.get('reason'),previous_time_s=row.get('time_s'),
                left_max_delta_rad=float(np.max(abs(b[:8]-end[:8]))),right_max_delta_rad=float(np.max(abs(b[8:]-end[8:]))))
    return None
