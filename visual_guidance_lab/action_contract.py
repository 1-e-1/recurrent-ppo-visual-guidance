"""Pure batched proposal/commit contract, NOT a collision checker or actuator.

Never feed a rejected target back as measured state. The external guard must
certify a continuous trajectory (and brake if already moving); these tensors
alone do not authorize movement.
"""
import torch


def propose(q_measured, action, lower, upper, maximum_increment=.02):
    if q_measured.shape != action.shape or q_measured.shape[-1] != 16:
        raise ValueError('expected matching [...,16] joint/action tensors')
    if not 0 < maximum_increment <= .1:
        raise ValueError('invalid joint increment bound')
    if not all(torch.isfinite(x).all() for x in (q_measured, action, lower, upper)):
        raise ValueError('nonfinite joint/action limits')
    if torch.any(lower >= upper) or torch.any(q_measured < lower) or torch.any(q_measured > upper):
        raise ValueError('invalid measured joint state or limits')
    if torch.any(action.abs() > 1):
        raise ValueError('policy must supply bounded normalized actions')
    return torch.clamp(q_measured + maximum_increment * action, min=lower, max=upper)


def committed_increment(q_before, q_after, certified):
    """History contains actual measured motion, never requested motion.

An uncertified measured movement is an error, not a command to ignore it.
"""
    if q_before.shape != q_after.shape or q_before.shape[-1] != 16:
        raise ValueError('expected measured joint pairs [...,16]')
    if certified.shape != q_before.shape[:-1] or certified.dtype != torch.bool:
        raise ValueError('expected per-environment boolean certificate')
    if not torch.isfinite(q_before).all() or not torch.isfinite(q_after).all():
        raise ValueError('nonfinite measured state')
    delta=q_after-q_before
    if torch.any((~certified) & (delta.abs().amax(dim=-1) > 1e-7)):
        raise RuntimeError('uncertified measured motion')
    return delta
