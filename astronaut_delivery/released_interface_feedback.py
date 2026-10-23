"""Measured selection for retreat feedback; no motion or safety authority.

Use a fresh RGB-D observation of the released object's known interface, rather
than an occluded support marking. Rigid FK prediction gates identity only;
it must never become the returned visual measurement.
"""
import numpy as np


def released_feedback(guide,name,now,release_stamp,predicted_cap,*,minimum_pixels,
                      maximum_sigma_m,association_radius_m=.025):
    if (not np.isfinite([now,release_stamp,maximum_sigma_m,association_radius_m]).all()
        or release_stamp>now or maximum_sigma_m<=0 or association_radius_m<=0
        or minimum_pixels<1):raise ValueError('invalid feedback selection limits')
    predicted=np.asarray(predicted_cap,float)
    if predicted.shape!=(3,) or not np.isfinite(predicted).all():raise ValueError('invalid rigid prediction')
    observed=guide.target(name,now)
    if observed is None or observed.name!=name or getattr(observed,'kind',None)!='magnetic_interface':return None
    if not np.isfinite(observed.stamp) or not release_stamp<observed.stamp<=now:return None
    p=np.asarray(observed.position,float);cov=np.asarray(observed.covariance,float)
    if (p.shape!=(3,) or cov.shape!=(3,3) or not np.isfinite(p).all()
        or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=1e-12)):
        return None
    eigen=np.linalg.eigvalsh(cov)
    if eigen.min()<0 or eigen.max()>maximum_sigma_m**2:return None
    if observed.pixels<minimum_pixels or np.linalg.norm(p-predicted)>association_radius_m:return None
    return observed


def switch_retreat_feedback(controller):
    """At-rest, post-release handover; preserve permits, budgets and quality."""
    if (not getattr(controller,'released_interface_guidance',False)
        or controller.transfer_phase!='retreat' or controller.held is not None
        or not controller.at_known_rest()):return False
    release=next((e for e in reversed(controller.transfer_events)
                  if e.get('event')=='measured_release_command'),None)
    if release is None or controller.retreat_permit is None:return False
    if controller.active==release['target']:return False
    cfg=controller.allocator.cfg
    observed=released_feedback(controller.guide,release['target'],controller.time,
        release['time_s'],controller.retreat_permit.target,
        minimum_pixels=cfg.minimum_pixels,maximum_sigma_m=cfg.maximum_sigma_m)
    if observed is None:return False
    try:
        controller.allocator.context(controller.q[0],observed.position,observed.covariance,
                                     observed.pixels,controller.measurement_age(observed))
    except ValueError:return False
    previous=controller.active
    controller.active=observed.name
    controller.transfer_events.append(dict(event='released_interface_visual_handover',time_s=controller.time,
        previous=previous,target=observed.name,measurement_stamp_s=observed.stamp,
        measured_position=np.asarray(observed.position).tolist(),
        scope='Fresh released-interface feedback only; original allocation, swept safety, contact permissions and budgets unchanged'))
    return True
