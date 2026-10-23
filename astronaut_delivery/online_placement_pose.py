"""Measured packing-aware orientation before descent; never play offline joints."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .measured_placement_candidates import placement_candidates,separating_gap


def prepare_placement_pose(controller,goal):
    goal=np.asarray(goal,float).copy()
    if controller.held is None or controller.transfer_phase!='transport' or not controller.at_known_rest():
        raise RuntimeError('placement_pose_requires_held_transport_rest')
    released=[np.asarray(e['envelope'],float) for e in controller.transfer_events
              if e.get('event')=='measured_release_command']
    if not released:return goal
    origin=controller.right.tcp(controller.q[1]);local=controller.held.local_points
    expected=local@origin[:3,:3].T+goal
    margin=max(.003,float(controller.local_mapper.cfg.planning_margin_m))
    if min(separating_gap(expected,v) for v in released)>margin:return goal
    controller.observe(integrate=True)
    marking=controller.guide.target(controller.active,controller.time)
    anchor=np.r_[controller.place_xy,controller.place_z]
    if marking is None or np.linalg.norm(marking.position-anchor)>.005:
        raise RuntimeError('placement_support_changed_before_pose_selection')
    with controller.local_mapper._lock:surface=controller.local_mapper.mesh()[0].cpu().numpy()
    candidates,diagnostic=placement_candidates(local,origin[:3,:3],goal,controller.place_z,released,
                                              surface,minimum_gap=margin)
    controller.transfer_events.append(dict(event='measured_placement_pose_selection',time_s=controller.time,
        **diagnostic,candidates=[dict(yaw_delta_deg=p['yaw_delta_deg'],offset_m=p['offset_m'],
            minimum_measured_gap_m=p['minimum_measured_gap_m']) for p in candidates]))
    if not candidates:raise RuntimeError('no_measured_packing_pose')
    selected=candidates[0]
    interpolation=Slerp([0.,1.],Rotation.from_matrix(np.stack([origin[:3,:3],selected['rotation']])))
    # All calls use existing live travel -> allocation -> certificates -> fresh
    # exposure -> execution. No cached/offline q is an executable command.
    for index,u in enumerate(np.linspace(0.,1.,13)[1:]):
        controller._placement_rotation_goal=interpolation(u).as_matrix()
        high=max(float(origin[2,3]),float(controller.right.tcp(controller.q[1])[2,3]))
        position=np.r_[(1-u)*origin[:2,3]+u*selected['position'][:2],high]
        controller.transfer_events.append(dict(event='measured_placement_pose_waypoint',time_s=controller.time,
            index=index,position=position.tolist(),rotation=controller._placement_rotation_goal.tolist(),
            scope='Online bounded task pose; original transport, visual, reaction and execution checks'))
        controller.travel(position,'transport',maximum_steps=12)
    controller.observe(integrate=True)
    marking=controller.guide.target(controller.active,controller.time)
    if marking is None or np.linalg.norm(marking.position-anchor)>.005:
        raise RuntimeError('placement_support_changed_after_pose_selection')
    actual=controller.right.tcp(controller.q[1])
    final_expected=local@actual[:3,:3].T+selected['position']
    if min(separating_gap(final_expected,v) for v in released)<=margin:
        raise RuntimeError('achieved_placement_pose_insufficient_measured_gap')
    controller.place_xy=selected['position'][:2].copy()
    # Caller rebuilds and validates the original contact column from achieved
    # pose and current surface BEFORE any downward contact motion begins.
    return selected['position'].copy()
