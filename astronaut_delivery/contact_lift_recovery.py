"""Bounded contact-lift proposals; original online travel remains authority.

The original 8 mm lift corridor, 2 mm penetration bound, measured contact
column, geometry checks and fresh visual checks MUST remain the authority.
"""
import numpy as np


def contact_lift_candidates(source, current, destination, low_height):
    source, current, destination = [np.asarray(p, float) for p in (source, current, destination)]
    if (any(p.shape != (3,) for p in (source, current, destination))
            or not np.isfinite(np.r_[source, current, destination, low_height]).all()):
        raise ValueError('finite measured geometry required')
    # A single bounded fallback following a failed vertical staged lift.
    # Do not accumulate offsets from the current position or permit lowering.
    if np.linalg.norm(current[:2] - source[:2]) > .003 or low_height < current[2]:
        return []
    direction = destination[:2] - source[:2]
    length = np.linalg.norm(direction)
    if length < .025:
        return []
    return [np.r_[source[:2] + d * direction / length, low_height] for d in (.002, .004)]


def retry_contact_lift(task, position, destination):
    """One finite retry sequence, called ONLY after staged vertical IK failure.

    Every proposal re-enters the original online travel implementation. Never
    turn sensing/collision/actuator errors into another permission attempt.
    """
    if (not task.geometric_transport_recovery or task.transfer_phase != 'lift'
            or getattr(task, 'held', None) is None or not task.at_known_rest()):
        return False
    source = task.source_tcp[:3, 3]
    current = task.right.tcp(task.q[1])[:3, 3]
    candidates = contact_lift_candidates(source, current, destination, float(position[2]))
    for target in candidates:
        task.transfer_events.append(dict(event='bounded_contact_lift_proposal',
            time_s=task.time, position=target.tolist(), source_position=source.tolist(),
            scope='Proposal only; original lift contact corridor, geometry, vision and execution gates'))
        try:
            task.travel(target, 'lift')
        except RuntimeError as exc:
            if str(exc) != 'transfer/lift/no_measured_safe_IK' or not task.at_known_rest():
                raise
        else:
            return True
    return False
