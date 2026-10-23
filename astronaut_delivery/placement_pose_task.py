"""Optional pose objective helpers; original safety guards remain separate."""
import numpy as np


def rotation_residual(actual,desired):
    a=np.asarray(actual,float);b=np.asarray(desired,float)
    if a.shape!=(3,3) or b.shape!=(3,3) or not np.isfinite([a,b]).all():
        raise ValueError('finite rotation matrices required')
    error=(a-b).ravel();norm=np.linalg.norm(error)
    return error*2*max(0.,norm-.008)/max(norm,1e-12)


def rotation_reached(actual,desired):
    if desired is None:return True
    a=np.asarray(actual,float);b=np.asarray(desired,float)
    return bool(a.shape==(3,3) and b.shape==(3,3) and np.isfinite([a,b]).all()
                and np.linalg.norm(a-b)<=.01)


def compose_evidence_residual(arguments,evidence_residual):
    """Do not replace a task objective when adding an evidence-aware proposal."""
    result=dict(arguments);original=result.get('extra_residual')
    if original is None:result['extra_residual']=evidence_residual
    else:result['extra_residual']=lambda q:np.r_[original(q),evidence_residual(q)]
    return result


def translation_step(current,desired,maximum_step):
    current=np.asarray(current,float);desired=np.asarray(desired,float)
    if (current.shape!=(3,) or desired.shape!=(3,) or not np.isfinite([current,desired]).all()
        or not np.isfinite(maximum_step) or maximum_step<=0):raise ValueError('finite bounded step required')
    error=desired-current;distance=np.linalg.norm(error)
    return current.copy() if distance<1e-12 else current+error*min(1.,maximum_step/distance)
