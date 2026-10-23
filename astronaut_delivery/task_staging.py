"""Measured-target staging selection: pose feasibility is proposal-only."""
import numpy as np


def choose_standoff(arm, joints, target, final_standoff,*,orientation_deadband=False):
    target = np.asarray(target, float); joints = np.asarray(joints, float)
    if (target.shape != (3,) or joints.shape != (8,) or not np.isfinite(target).all()
            or not np.isfinite(joints).all() or not np.isfinite(final_standoff)
            or not 0 <= final_standoff < .10):
        raise ValueError('finite measured approach target and bounded final gap required')
    attempts = []
    for gap in (.10, .08, .06, .04):
        if gap <= final_standoff:
            continue
        goal = target + [0, 0, gap]
        # Full-range search checks whether a staging POSE can be proposed.
        # Executed segments still use the original +/-0.30rad local solver,
        # velocity limits and all continuous geometry/visual guards.
        q = arm.solve(goal, [0, 0, -1], seed=joints, max_delta=np.pi,
                      tries=12, rng=np.random.default_rng(43119),orientation_deadband=orientation_deadband)
        error = None if q is None else float(np.linalg.norm(arm.tcp(q)[:3, 3]-goal))
        feasible = q is not None and error <= .002
        attempts.append(dict(standoff_m=gap, vertical_IK_found=q is not None,
                             position_error_m=error, staging_candidate=feasible))
        if feasible:
            return gap, attempts
    return .10, attempts
