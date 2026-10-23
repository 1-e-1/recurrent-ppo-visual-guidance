"""Measured task progress gate; never replaces geometry or sensor checks."""
import numpy as np


def retains_progress(current, baseline, candidate, goal, fraction=.5):
    points=np.asarray([current,baseline,candidate,goal],float)
    if points.shape!=(4,3) or not np.isfinite(points).all() or not 0<fraction<=1:
        return False
    before=np.linalg.norm(points[0]-points[3])
    base=before-np.linalg.norm(points[1]-points[3])
    proposed=before-np.linalg.norm(points[2]-points[3])
    return bool(proposed>=max(0.,fraction*base)-1e-6)


def episode_failed(result):
    return bool(result.get('error') or not result.get('observed_queue_exhausted')
                or result.get('status')!='OBSERVED_RGBD_TASK_QUEUE_COMPLETED_PENDING_SCENE_AUDIT')
