"""Exposure-clock accounting: receipt or a paused motion clock is not freshness."""
import math


def observation_age(motion_stamp,wall_stamp,*,motion_now,wall_now):
    try:
        values=tuple(float(x) for x in (motion_stamp,wall_stamp,motion_now,wall_now))
    except (TypeError,ValueError):
        return math.inf
    if not all(math.isfinite(x) for x in values):return math.inf
    motion_age=values[2]-values[0];wall_age=values[3]-values[1]
    if min(motion_age,wall_age)<0:return math.inf
    return max(motion_age,wall_age)


def refreshed_wall_stamps(previous,current,previous_stamps,exposure_wall):
    """Estimator objects must be replaced on a genuinely accepted detection.

    Equal motion timestamps are common at rest. Object identity, not timestamp
    equality or receipt time, distinguishes a new detection from retained belief.
    Missing historical clocks remain missing; dictionaries are never mutated.
    """
    try:stamp=float(exposure_wall)
    except (TypeError,ValueError):raise ValueError('finite exposure wall clock required') from None
    if not math.isfinite(stamp):raise ValueError('finite exposure wall clock required')
    result={}
    for name,estimate in current.items():
        if estimate is not previous.get(name):result[name]=stamp
        elif name in previous_stamps:result[name]=previous_stamps[name]
    return result
