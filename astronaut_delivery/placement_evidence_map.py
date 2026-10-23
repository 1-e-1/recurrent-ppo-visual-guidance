"""Bounded complete-tool map replacement at rest, using authenticated RGB-D."""
from dataclasses import replace
import time
import numpy as np


def placement_goal(goal):
    value=np.asarray(goal,float)
    if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('finite placement goal required')
    # Task proposal, NOT a relaxed collision or final placement tolerance.
    return value+np.array([0.,0.,.003])


def tool_bounds(vertices,current_tcp,goal,maximum_cells):
    v=np.asarray(vertices,float);tcp=np.asarray(current_tcp,float);target=np.asarray(goal,float)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or tcp.shape!=(3,) or target.shape!=(3,)
        or not np.isfinite(v).all() or not np.isfinite([tcp,target]).all()
        or not isinstance(maximum_cells,int) or not 1<=maximum_cells<=8000000):
        raise ValueError('finite complete-tool geometry and bounded cell budget required')
    destination=v+target-tcp
    all_vertices=np.concatenate([v,destination,destination+[0.,0.,.08]])
    lower=all_vertices.min(0)-.015;upper=all_vertices.max(0)+.015
    cells=int(np.prod(np.ceil((upper-lower)/.001)))
    if cells>maximum_cells:raise RuntimeError('placement_fine_map_capacity_exceeded')
    return lower,upper,cells


def prepare(controller,goal):
    if (controller.held is None or controller.transfer_phase!='transport'
        or not controller.at_known_rest() or not controller.static_interval_fusion
        or not controller.scan_source[3]):
        raise RuntimeError('placement_refinement_requires_held_static_transport_rest')
    from classical_fusion.rgbd_guard import robot_hulls
    from classical_fusion.rgbd_recording import replay_initial_scan
    from .measured_payload import PayloadArm
    from .temporal_evidence_map import TemporalEvidenceTSDF
    from .measured_task_memory import replay_changes
    arm=PayloadArm(controller.right,controller.held)
    hulls=[(name,v) for name,(v,n) in robot_hulls(arm,controller.q[1]) if name in ('magnet','payload')]
    if {name for name,v in hulls}!={'magnet','payload'}:raise RuntimeError('complete_tool_hulls_required')
    lower,upper,cells=tool_bounds(np.concatenate([v for name,v in hulls]),
        controller.right.tcp(controller.q[1])[:3,3],goal,min(8000000,controller.local_map_cell_budget))
    old=controller.local_mapper
    cfg=replace(old.cfg,workspace_min=tuple(lower),workspace_max=tuple(upper),voxel_size_m=.001)
    began=time.perf_counter();mapper=TemporalEvidenceTSDF(cfg,device=old.device)
    mapper.patch_radii=old.patch_radii
    scan,scene,calibration,unchanged=controller.scan_source
    provenance=replay_initial_scan(mapper,scan,expected_scene_token=scene,
        expected_calibration_id=calibration,assume_unchanged_scene=unchanged)
    changes=replay_changes(mapper,controller.transfer_events)
    # All subsequent tool checks use this map. No UNKNOWN-to-coarse fallback.
    # Normal mapping_layers, invalidation, revisions, raw observations and wall
    # freshness checks continue unchanged; no offline snapshot age override.
    controller.local_mapper=mapper
    controller.observe(integrate=True)
    controller.transfer_events.append(dict(event='complete_tool_placement_raw_refinement',time_s=controller.time,
        lower=lower.tolist(),upper=upper.tolist(),cells=cells,voxel_size_m=.001,
        placement_goal=np.asarray(goal).tolist(),raw_provenance=provenance,replayed_changes=changes,
        build_wall_s=time.perf_counter()-began,
        scope='Whole-tool measured map replacement; unchanged errors, contact, timing and execution checks'))
