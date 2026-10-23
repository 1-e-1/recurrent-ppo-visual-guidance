"""RGB-D-only conditional pick/lift/transport/place extension.

No environment USD/CAD queries enter this planner. The marked horizontal
support and isolated-object hypotheses are explicit. Unknown cells remain
blocked except bounded source/support contact permissions for tool/payload.
"""
import argparse,json,time
from dataclasses import replace,asdict
from itertools import product
import numpy as np
from threadpoolctl import threadpool_limits
from classical_fusion.planning_map import PlanningMapConfig
from classical_fusion.rgbd_recording import replay_initial_scan
from classical_fusion.robot_query_snapshot import RobotQuerySnapshot
from classical_fusion.static_interval_snapshot import StaticIntervalSnapshot
from classical_fusion.payload_contact_guard import PayloadEnvironmentGuard,SupportContactColumn
from .rgbd_cooperative import RGBDCooperative
from .temporal_evidence_map import TemporalEvidenceTSDF
from .measured_payload import payload_from_measurement,PayloadArm
from .vision import TargetEstimate
from scipy.spatial import ConvexHull


class RGBDTransfer(RGBDCooperative):
    def __init__(self,*args,local_map_cell_budget=12_000_000,quality_aware_handover=False,owned_source_departure=False,thin_support_search=False,geometric_transport_recovery=False,far_first_placement=False,partitioned_geometry_proof=False,partial_marking=False,released_interface_guidance=False,measured_placement_pose=False,placement_evidence_refinement=False,**kwargs):
        from .transfer_map_capacity import measured_roi
        measured_roi([0,0,0],[0,0,0],local_map_cell_budget)
        self.local_map_cell_budget=local_map_cell_budget
        self.quality_aware_handover=bool(quality_aware_handover)
        self.owned_source_departure=bool(owned_source_departure)
        self.thin_support_search=bool(thin_support_search)
        if geometric_transport_recovery and not owned_source_departure:
            raise ValueError('geometric extraction requires bounded owned-source departure')
        self.geometric_transport_recovery=bool(geometric_transport_recovery)
        self.partitioned_geometry_proof=bool(partitioned_geometry_proof)
        self.partial_marking=bool(partial_marking)
        self.released_interface_guidance=bool(released_interface_guidance)
        self.measured_placement_pose=bool(measured_placement_pose)
        self.placement_evidence_refinement=bool(placement_evidence_refinement)
        self.placement_assignment='farthest_camera_measured_mark' if far_first_placement else 'nearest_measured_mark'
        self.world_yaw_free=True
        super().__init__(*args,**kwargs)
        self.held=None;self.transfer_phase=None;self.transfer_events=[];self.placement_name=None
        self.source_target=None;self.source_tcp=None;self.source_column=None;self.place_column=None
        self.retreat_permit=None
        self.measured_layers=[]

    def export_task_geometry(self):
        from .rigid_measured_surface import transform_layer,export_composite
        layers=[]
        for item in self.measured_layers:
            pose=item.get('release_tcp')
            if pose is None:pose=self.right.tcp(self.q[1])
            layers.append(transform_layer(item['data'],pose))
        if layers:
            export_composite(self.output,layers)
            if self.local_mapper is not None:export_composite(self.output/'local_map',layers)

    def available_placements(self):
        return [e for e in self.guide.estimates.values() if e.kind=='placement_marking' and 0<=self.time-e.stamp<=.25]

    def placement_proposals(self):
        # A previously measured static marking can select an ROI or a gaze.
        # This method grants NO transport/contact permission and never alters
        # its timestamp. Fresh destination feedback is still required later.
        return [e for e in self.guide.estimates.values() if e.kind=='placement_marking'
            and e.stamp<=self.time and np.isfinite(e.position).all()]

    def select_placement_proposal(self,target):
        places=self.placement_proposals()
        if not places:raise RuntimeError('no_measured_placement_proposal')
        if getattr(self,'placement_assignment','nearest_measured_mark')=='farthest_camera_measured_mark':
            from .geometric_recovery import capacity_aware_placements
            admitted,diagnostics=capacity_aware_placements(places,self.left.camera(self.q[0])[:3,3],
                target.position,self.local_map_cell_budget)
            self.transfer_events.append(dict(event='measured_placement_capacity_ranking',time_s=self.time,
                candidates=diagnostics,maximum_cells=self.local_map_cell_budget))
            if not admitted:raise RuntimeError('no_measured_destination_within_map_capacity')
            return admitted[0]
        return min(places,key=lambda e:np.linalg.norm(e.position-target.position))

    def prepare_local_map(self,target):
        selected=self.select_placement_proposal(target)
        self.placement_name=selected.name;self.source_target=target.position.copy()
        self.transfer_events.append(dict(event='measured_destination_ROI_proposal',time_s=self.time,name=selected.name,
            position=selected.position.tolist(),measurement_stamp_s=selected.stamp,measurement_age_s=self.time-selected.stamp,
            scope='Historical RGB-D allowed for ROI/gaze proposal only; fresh measured destination required before transport'))
        from .transfer_map_capacity import measured_roi
        lo,hi,cells=measured_roi(target.position,selected.position,self.local_map_cell_budget)
        cfg=PlanningMapConfig(workspace_min=tuple(lo),workspace_max=tuple(hi),voxel_size_m=.002,
            truncation_m=.006,surface_min_weight=1.,snapshot_max_age_s=30.)
        began=time.perf_counter();mapper=TemporalEvidenceTSDF(cfg,device=self.mapper.device)
        mapper.patch_radii=(4,6,8,10,12,16,24,32)
        scan,scene,calibration,unchanged=self.scan_source
        replay_initial_scan(mapper,scan,expected_scene_token=scene,expected_calibration_id=calibration,assume_unchanged_scene=unchanged)
        from .measured_task_memory import replay_changes
        changes=replay_changes(mapper,self.transfer_events)
        self.local_mapper=mapper;self.local_map_build_wall_s=time.perf_counter()-began
        self.observe(integrate=True)
        print(json.dumps(dict(event='measured_transfer_map_replayed',cells=int(cells),wall_s=self.local_map_build_wall_s,
            source=target.position.tolist(),placement=selected.position.tolist(),replayed_change_regions=changes)),flush=True)

    def robot_guard_arguments(self):return {} if self.held is None else dict(held=self.held)

    def right_ik_arguments(self):
        options=super().right_ik_arguments()
        if self.held is not None or self.transfer_phase=='retreat':
            if getattr(self,'world_yaw_free',False):
                # World-Z yaw does not change any rigid payload point height.
                # Use the captured row, NOT a repeatedly drifting last pose.
                options['vertical_row']=self.source_tcp[2,:3].copy()
            else:
                options['rotation']=getattr(self,'_right_rotation_proposal',
                    getattr(self,'transport_rotation',self.source_tcp[:3,:3]))
        desired=getattr(self,'_placement_rotation_goal',None)
        if desired is not None:
            from .placement_pose_task import rotation_residual
            options['extra_residual']=lambda q:rotation_residual(self.right.tcp(q)[:3,:3],desired)
        return options

    def right_pose_reached(self,q):
        if getattr(self,'_placement_rotation_goal',None) is None:return True
        from .placement_pose_task import rotation_reached
        return rotation_reached(self.right.tcp(q)[:3,:3],getattr(self,'_placement_rotation_goal',None))

    def right_task_axis(self):
        if getattr(self,'world_yaw_free',False) and (self.held is not None or self.transfer_phase=='retreat'):return None
        return super().right_task_axis()

    def plan_step(self,target,standoff,step_m=.02):
        motion,why=super().plan_step(target,standoff,step_m)
        if getattr(self,'world_yaw_free',False):
            if motion is not None and 'vertical_row' in self.right_ik_arguments():
                self.paths[-1]['right_task_vertical_row']=self.source_tcp[2,:3].tolist()
                self.paths[-1]['right_task_orientation_scope']='Endpoint row error <= 0.001; whole interpolated sweep still requires original collision/contact certificates'
            return motion,why
        if motion is not None or why!='no_measured_safe_IK' or self.transfer_phase!='transport' or step_m>.005:
            return motion,why
        # Horizontal magnetic placement does not prescribe yaw. Release only
        # this unused task angle, after ordinary positional backtracking fails.
        # Every candidate still passes the ORIGINAL full segment certificates.
        from scipy.spatial.transform import Rotation
        original=self.right_ik_arguments()['rotation']
        try:
            for yaw in (15.,-15.,30.,-30.):
                rotation=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@self.source_tcp[:3,:3]
                if np.allclose(rotation,original,atol=1e-8):continue
                self._right_rotation_proposal=rotation
                motion,why=super().plan_step(target,standoff,step_m)
                self.decisions.append(dict(status='TRANSPORT_TASK_YAW_PROPOSAL',time_s=self.time,
                    world_yaw_from_capture_deg=yaw,accepted=motion is not None,
                    scope='Unused horizontal-task yaw only; no added free space or waived geometry/vision checks'))
                if motion is not None:
                    self.paths[-1]['right_task_rotation']=rotation.tolist()
                    self.paths[-1]['world_yaw_from_capture_deg']=yaw
                    return motion,why
        finally:
            self.__dict__.pop('_right_rotation_proposal',None)
        return None,why

    def commit_right_orientation(self):
        # Called only after successful execution, never while just proposing
        # a path or after braking part-way along it.
        rotation=self.paths[-1].get('right_task_rotation')
        if rotation is not None:self.transport_rotation=np.asarray(rotation,float).copy()

    def allocation_context(self,target):
        if self.held is None and self.transfer_phase!='retreat':return super().allocation_context(target)
        measured=self.guide.target(self.active,self.time)
        if measured is None:raise ValueError('placement_visual_feedback_expired')
        return super().allocation_context(measured)

    def make_environment(self,target,standoff):
        if self.transfer_phase=='retreat':
            from classical_fusion.local_tool_guard import LocalToolEnvironmentGuard
            broad=self.broad_snapshot()
            fine=StaticIntervalSnapshot(self.local_mapper,assume_static_epoch=True) if self.static_interval_fusion else RobotQuerySnapshot(self.local_mapper)
            env=LocalToolEnvironmentGuard(self.left,self.right,broad,fine,self.retreat_permit,max_boxes=127,max_depth=8)
            return self.refined_environment(env),broad
        if self.held is None:
            env,broad=super().make_environment(target,standoff)
            return self.refined_environment(env),broad
        broad=self.broad_snapshot()
        fine=StaticIntervalSnapshot(self.local_mapper,assume_static_epoch=True) if self.static_interval_fusion else RobotQuerySnapshot(self.local_mapper)
        right=PayloadArm(self.right,self.held)
        column=self.source_column if self.transfer_phase=='lift' else self.place_column if self.transfer_phase=='place' else None
        contact=dict(phase=self.transfer_phase,column=column)
        if self.transfer_phase=='lift':contact.update(initial_bottom=self.initial_payload_bottom,source_xy=self.source_tcp[:2,3])
        if self.transfer_phase=='place':contact.update(initial_bottom=self.expected_placed_bottom,source_xy=self.place_xy)
        if self.transfer_phase=='transport' and self.owned_source_departure:
            contact.update(phase='source_departure',column=self.source_column,
                departure_minimum_z=self.source_support_z+.025)
        return self.refined_environment(PayloadEnvironmentGuard(self.left,right,broad,fine,**contact)),broad

    def invalidate(self,lower,upper,reason,**physical):
        rows=[]
        for name,mapper in self.mapping_layers():
            if mapper is not None:rows.append(dict(map=name,change=mapper.invalidate_region(lower,upper,**physical)))
        event=dict(time_s=self.time,event='local_evidence_retired_to_UNKNOWN',reason=reason,
            lower=np.asarray(lower).tolist(),upper=np.asarray(upper).tolist(),maps=rows,
            physical_evidence_bounds={k:np.asarray(v).tolist() for k,v in physical.items()})
        self.transfer_events.append(event)

    def acquire_placement_view(self):
        """Bridge tasks while preserving the CURRENT measured source target.

        A historical observed slot may propose a gaze, never authorize contact
        or be retimestamped. Only a new RGB-D slot estimate completes handover.
        """
        previous=self.guide.estimates.get(self.placement_name)
        if previous is None:raise RuntimeError('no_measured_destination_for_gaze_proposal')
        self.active=self.held.name;self.transfer_phase='lift'
        if getattr(self,'history_reacquisition',False):
            self.acquire_destination_at_rest()
            return
        for _ in range(3):
            destination=self.guide.target(self.placement_name,self.time)
            quality_error=None
            if destination is not None:
                if not self.quality_aware_handover:return
                # Being detected is not the same as satisfying the NEXT phase's
                # larger support footprint. Check that unchanged next-phase set
                # BEFORE abandoning the currently tracked source interface.
                original_cfg=self.allocator.cfg
                try:
                    self.allocator.cfg=replace(original_cfg,cap_radius_m=.045,endpoint_reserve=.01)
                    self.allocator.context(self.q[0],destination.position,destination.covariance,
                        destination.pixels,self.time-destination.stamp)
                except ValueError as exc:quality_error=str(exc)
                finally:self.allocator.cfg=original_cfg
                self.transfer_events.append(dict(event='next_phase_visual_quality_gate',time_s=self.time,
                    destination=self.placement_name,allowed=quality_error is None,reason=quality_error,
                    scope='Fresh measured destination tested against actual next-phase visual set before switching target'))
                if quality_error is None:return
            source=self.guide.target(self.active,self.time)
            if source is None:raise RuntimeError('source_feedback_lost_during_visual_handover')
            began=time.perf_counter();reference=None;restoration=False
            try:self.allocation_context(source)
            except ValueError as exc:
                # Restore the fresh source before requiring a whole-path
                # operational view. Capture can change the measured margin.
                motion,reference=self.plan_operational_recovery(source,str(exc));restoration=True
            else:
                # When already detected but unsuitable, centre more closely on
                # the destination; the entire move must still keep source valid.
                focus=(source.position+previous.position)*.5 if quality_error is None else destination.position
                motion=self.plan_observation_recovery(source,look_at=focus)
            self.planning_wall.append(time.perf_counter()-began)
            if motion is None:raise RuntimeError('no_certified_visual_task_handover')
            self.paths[-1].update(payload_local_points=self.held.local_points.tolist(),payload_name=self.held.name,payload_contact_phase='lift')
            self.transfer_events.append(dict(event='visual_task_handover_rescan',time_s=self.time,source_view_restoration=restoration,
                current_source=self.active,proposed_destination=self.placement_name,
                historical_destination_age_s=self.time-previous.stamp,scope='Historical estimate proposes gaze only; fresh destination required'))
            old=len(self.steps);original=self.active
            try:
                if reference is not None:self.active=reference
                self.execute(motion)
            except RuntimeError as exc:
                self.active=original;self.reobserve_after_visual_stop(exc)
            finally:self.active=original
            for row in self.steps[old:]:
                if row['phase']!='braking':row['phase']='visual_task_handover'
        if self.quality_aware_handover:
            destination=self.guide.target(self.placement_name,self.time)
            if destination is None:raise RuntimeError('destination_reacquisition_budget_exhausted')
            original_cfg=self.allocator.cfg
            try:
                self.allocator.cfg=replace(original_cfg,cap_radius_m=.045,endpoint_reserve=.01)
                self.allocator.context(self.q[0],destination.position,destination.covariance,destination.pixels,self.time-destination.stamp)
            except ValueError as exc:raise RuntimeError('destination_next_phase_quality_budget_exhausted/'+str(exc)) from exc
            finally:self.allocator.cfg=original_cfg
        elif self.guide.target(self.placement_name,self.time) is None:raise RuntimeError('destination_reacquisition_budget_exhausted')

    def acquire_destination_at_rest(self):
        """Fresh next-phase feedback, without requiring a now-occluded source.

        Historical estimates only propose gaze. The camera-only helper keeps
        the right arm at certified rest and retains all current geometry checks.
        """
        original=self.active;self.active=self.placement_name
        cfg=self.allocator.cfg
        try:
            self.allocator.cfg=replace(cfg,cap_radius_m=.045,endpoint_reserve=.01)
            for attempt in range(7):
                destination=self.guide.target(self.active,self.time)
                reason='destination_not_fresh'
                if destination is not None:
                    try:
                        self.allocator.context(self.q[0],destination.position,destination.covariance,
                            destination.pixels,self.time-destination.stamp)
                    except ValueError as exc:reason=str(exc)
                    else:
                        self.transfer_events.append(dict(event='fresh_destination_handover_at_rest',
                            time_s=self.time,destination=self.active,measurement_stamp_s=destination.stamp))
                        return
                if attempt==6 or not self.reacquire_from_measured_history():
                    raise RuntimeError('destination_certified_reacquisition_failed/'+reason)
        finally:
            self.active=original;self.allocator.cfg=cfg

    def plan_transport_raise(self,waypoint):
        from .geometric_recovery import raised_transport_waypoint
        tcp=self.right.tcp(self.q[1])[:3,3]
        raised=raised_transport_waypoint(tcp,self.initial_transport_height,self.transport_raise_count)
        if raised is None:return None,None
        began=time.perf_counter()
        motion,reason=self.plan_step(replace(waypoint,position=raised),0.,step_m=.012)
        self.planning_wall.append(time.perf_counter()-began)
        self.transfer_events.append(dict(event='bounded_transport_raise_proposal',time_s=self.time,
            position=raised.tolist(),accepted=motion is not None,reason=reason,
            accepted_before=self.transport_raise_count,maximum_accepted=6,
            ceiling_m=self.initial_transport_height+.072,
            scope='Proposal only; original online geometry, vision, robot and execution gates'))
        if motion is not None:
            self.transport_raise_count+=1
            self.paths[-1]['phase']='transport_height_adjustment'
        return motion,raised

    def lift_with_geometric_recovery(self,high,place,height_info):
        try:self.travel(np.r_[self.source_tcp[:2,3],high],'lift')
        except RuntimeError as exc:
            if (not self.geometric_transport_recovery or str(exc)!='transfer/lift/no_measured_safe_IK'
                or not self.at_known_rest()):raise
            from .geometric_recovery import extraction_waypoints
            route=extraction_waypoints(self.source_tcp[:3,3],self.right.tcp(self.q[1])[:3,3],
                place.position,self.source_support_z,height_info['payload_below_tcp_m'],high)
            if route is None:raise
            self.transfer_events.append(dict(event='staged_extraction_proposal',time_s=self.time,
                waypoints=[dict(phase=phase,position=p.tolist()) for phase,p in route],
                scope='Unchanged fixed vertical rotation row; lift and transport keep their original certificates'))
            for phase,position in route:
                try:self.travel(position,phase)
                except RuntimeError as staged_error:
                    if phase!='lift' or str(staged_error)!='transfer/lift/no_measured_safe_IK':raise
                    from .contact_lift_recovery import retry_contact_lift
                    if not retry_contact_lift(self,position,place.position):raise

    def travel(self,position,phase,maximum_steps=48):
        self.transfer_phase=phase
        position=np.asarray(position,float).copy()
        from .visual_stall_gate import VisualStallGate,attempt_recovery
        gate=VisualStallGate() if getattr(self,'history_reacquisition',False) else None
        for _ in range(maximum_steps):
            self.task_metadata['requested_transfer_goal']=dict(phase=phase,position=position.tolist(),time_s=self.time)
            if phase=='retreat' and getattr(self,'released_interface_guidance',False):
                from .released_interface_feedback import switch_retreat_feedback
                switch_retreat_feedback(self)
            owner=(getattr(getattr(self,'held',None),'name',None),self.active,phase)
            if gate is not None and attempt_recovery(self,gate,owner):continue
            visual=self.guide.target(self.active,self.time)
            if visual is None:
                if self.recover_missing_task_view():continue
                raise RuntimeError('transfer_visual_feedback_expired/'+phase)
            waypoint=TargetEstimate('measured_waypoint',np.asarray(position,float),visual.covariance,visual.pixels,visual.stamp)
            began=time.perf_counter();base_step=.012 if phase in ('lift','place') else .02
            reference=None
            try:self.allocation_context(visual)
            except ValueError as exc:
                motion,reference=self.plan_operational_recovery(visual,str(exc));why='visual_quality/'+str(exc)
                if motion is None:raise RuntimeError('no_certified_quality_reacquisition/'+phase+'/'+str(exc))
            else:
                for factor in (1.,.5,.25):
                    motion,why=self.plan_step(waypoint,0.,step_m=base_step*factor)
                    if motion is not None or why=='arrived':break
            self.planning_wall.append(time.perf_counter()-began)
            if why=='arrived':return
            if motion is None and phase=='transport' and self.geometric_transport_recovery:
                motion,raised=self.plan_transport_raise(waypoint)
                if motion is not None:position[2]=raised[2]
            if motion is None and phase=='transport':
                from .measured_transport_height import lower_within_measured_band
                tcp=self.right.tcp(self.q[1])[:3,3]
                lower=lower_within_measured_band(tcp[2],self.transport_floor)
                if lower is not None and (not self.geometric_transport_recovery or self.transport_raise_count==0):
                    escape=replace(waypoint,position=np.r_[tcp[:2],lower])
                    began=time.perf_counter();motion,_=self.plan_step(escape,0.,step_m=.006)
                    self.planning_wall.append(time.perf_counter()-began)
                    if motion is not None:
                        position[2]=lower;self.paths[-1]['phase']='transport_height_adjustment'
                        self.transfer_events.append(dict(event='measured_transport_height_adjustment',time_s=self.time,
                            proposed_tcp_height_m=lower,minimum_observed_clearance_height_m=self.transport_floor,
                            scope='Bounded proposal; full measured robot and payload certificates unchanged'))
            if motion is None and self.refresh_expired_evidence():continue
            if motion is None and why=='no_measured_safe_IK' and self.prepare_robot_refinement():continue
            if motion is None and self.recovery_count<6:
                began=time.perf_counter();motion=self.plan_observation_recovery(visual)
                self.planning_wall.append(time.perf_counter()-began)
            if motion is None:raise RuntimeError('transfer/'+phase+'/'+why)
            actual_phase=self.paths[-1].get('phase',phase)
            self.paths[-1].update(phase=actual_phase)
            if self.held is not None:
                self.paths[-1].update(payload_local_points=self.held.local_points.tolist(),payload_name=self.held.name,payload_contact_phase=phase)
            elif phase=='retreat':
                self.paths[-1]['released_target_contact']=dict(target=self.retreat_permit.target.tolist(),
                    footprint_equations=self.retreat_permit.footprint_equations.tolist(),
                    lower_offset=self.retreat_permit.lower_offset,upper_offset=self.retreat_permit.upper_offset)
            if phase in ('lift','place'):
                column=self.source_column if phase=='lift' else self.place_column
                self.paths[-1]['contact_column']=dict(equations=column.equations.tolist(),lower=column.lower,upper=column.upper,links=sorted(column.links))
            old=len(self.steps);original=self.active
            try:
                if reference is not None:self.active=reference
                self.execute(motion)
            except RuntimeError as exc:
                self.active=original;self.reobserve_after_visual_stop(exc)
                if gate is not None:gate.stopped(owner,str(exc))
                self.transfer_events.append(dict(event='visual_brake_completed_at_rest',time_s=self.time,phase=phase,reason=str(exc)))
            else:
                self.commit_right_orientation()
                if gate is not None:gate.completed(owner)
            finally:self.active=original
            for row in self.steps[old:]:
                if row['phase']!='braking':row['phase']=actual_phase
            print(json.dumps(dict(event='rgbd_payload_segment',phase=phase,segments=len(self.paths),time_s=self.time)),flush=True)
        raise RuntimeError('transfer_phase_budget_exhausted/'+phase)

    def after_capture(self,status):
        if self.last_contact_envelope is None:raise RuntimeError('missing_measured_payload_envelope')
        self.__dict__.pop('_placement_rotation_goal',None)
        self.task_metadata.update(single_transfer_released=False,
            post_release_retreat_completed=False,single_transfer_cycle_completed=False)
        self.source_tcp=self.right.tcp(self.q[1]);envelope=self.last_contact_envelope
        self.source_support_z=float(envelope.support_z)
        self.transport_rotation=self.source_tcp[:3,:3].copy()
        self.held=payload_from_measurement(self.active,envelope,self.source_tcp)
        self.sensor_service.payload_local_points=self.held.local_points
        self.initial_payload_bottom=float(envelope.vertices[:,2].min())
        footprint=envelope.vertices[:,:2]
        footprint=(footprint[:,None,:]+np.array(list(product([-.009,.009],repeat=2)))[None]).reshape(-1,2)
        self.source_column=SupportContactColumn(footprint,self.initial_payload_bottom-.009,
            self.source_target[2]+.04,['payload','magnet'])
        capture_index=sum(e.get('event')=='measured_payload_attached' for e in self.transfer_events)
        envelope_file='captured_measured_envelope.npz' if capture_index==0 else f'captured_measured_envelope_{capture_index:03d}.npz'
        surface_file=None
        if self.last_contact_surface is not None and envelope.observed_surface_indices is not None:
            from .rigid_measured_surface import extract_layer
            import torch
            v,f,revision=self.last_contact_surface;ids=envelope.observed_surface_indices
            with self.local_mapper._lock:
                g,H,ok=self.local_mapper.differentials(torch.as_tensor(v[ids],device=self.local_mapper.device))
            layer=extract_layer(v,f,ids,self.source_tcp,g.cpu().numpy(),H.cpu().numpy(),ok.cpu().numpy())
            self.output.joinpath('measured_objects').mkdir(exist_ok=True)
            surface_file=f'measured_objects/{capture_index:03d}_{self.held.name}.npz'
            np.savez_compressed(self.output/surface_file,**layer)
            self.measured_layers.append(dict(name=self.held.name,data=layer,source_map_revision=revision,source_file=surface_file))
        self.transfer_events.append(dict(event='measured_payload_attached',time_s=self.time,name=self.held.name,envelope_file=envelope_file,
            retained_measured_surface=surface_file,
            tcp=self.source_tcp.tolist(),local_points=self.held.local_points.tolist(),measured_support_z=envelope.support_z,
            assumption='Observed convex envelope contains isolated rigid payload; unseen containment requires independent audit'))
        np.savez_compressed(self.output/envelope_file,vertices=envelope.vertices,faces=envelope.faces,equations=envelope.equations,
            support_z=envelope.support_z,margin_m=envelope.margin_m,tcp=self.source_tcp,local_points=self.held.local_points)
        self.invalidate(envelope.vertices.min(0)-.008,envelope.vertices.max(0)+.008,'owned object lifted; old occupancy and free evidence are no longer valid',
            evidence_lower=envelope.vertices.min(0),evidence_upper=envelope.vertices.max(0),evidence_vertices=envelope.vertices)
        self.observe(integrate=True)
        self.acquire_placement_view()
        self.active=self.placement_name
        self.allocator.cfg=replace(self.allocator.cfg,cap_radius_m=.045,endpoint_reserve=.01)
        place=self.guide.target(self.active,self.time)
        if place is None:raise RuntimeError('placement_marking_not_current_after_capture')
        # Height is measured source support-to-TCP, not a known part dimension.
        support_offset=self.source_tcp[2,3]-envelope.support_z
        from .measured_transport_height import propose_height
        with self.local_mapper._lock:surface=self.local_mapper.mesh()[0].cpu().numpy()
        high,height_info=propose_height(self.source_tcp,envelope,surface,place.position,self.local_mapper.spacing.cpu().numpy(),
            planning_margin=self.local_mapper.cfg.planning_margin_m,owned_source_departure=self.owned_source_departure)
        self.transport_floor=max(height_info['highest_observed_surface_m'],float(place.position[2]))+height_info['payload_below_tcp_m']+.025
        self.transfer_events.append(dict(event='measured_transport_height_proposal',time_s=self.time,**height_info))
        self.initial_transport_height=high;self.transport_raise_count=0
        if self.owned_source_departure:
            self.transfer_events.append(dict(event='bounded_owned_source_departure',time_s=self.time,
                source_column_equations=self.source_column.equations.tolist(),
                source_column_lower=self.source_column.lower,source_column_upper=self.source_column.upper,
                minimum_payload_and_magnet_z=self.source_support_z+.025,
                scope='Conditional isolated measured rigid object ownership after confirmed attachment; no map voxel becomes FREE. Only original source column may be traversed by payload/magnet; other geometry and unknown space retain full checks.'))
        self.lift_with_geometric_recovery(high,place,height_info)
        place=self.guide.target(self.active,self.time)
        if place is None:raise RuntimeError('placement_marking_lost_before_transport')
        self.place_xy=place.position[:2].copy();self.place_z=float(place.position[2])
        self.travel(np.r_[self.place_xy,high],'transport')
        place=self.guide.target(self.active,self.time)
        if place is None or np.linalg.norm(place.position-np.r_[self.place_xy,self.place_z])>.005:
            raise RuntimeError('placement_support_changed')
        goal=np.r_[self.place_xy,self.place_z+support_offset+.001]
        if self.placement_evidence_refinement:
            from .placement_evidence_map import placement_goal
            goal=placement_goal(goal)
        if getattr(self,'measured_placement_pose',False):
            from .online_placement_pose import prepare_placement_pose
            goal=prepare_placement_pose(self,goal)
        current_tcp=self.right.tcp(self.q[1]);payload=PayloadArm(self.right,self.held).payload_hull(self.q[1])[0]
        expected=payload+(goal-current_tcp[:3,3])
        self.expected_placed_bottom=float(expected[:,2].min())
        destination_footprint=expected[:,:2]
        destination_footprint=(destination_footprint[:,None,:]+np.array(list(product([-.009,.009],repeat=2)))[None]).reshape(-1,2)
        self.place_column=SupportContactColumn(destination_footprint,self.expected_placed_bottom-.009,self.place_z+.04,['payload'],
            intersection_cell_budget=128 if self.placement_evidence_refinement else 32)
        self.transfer_events.append(dict(event='placement_contact_proof_budget',time_s=self.time,
            maximum_cells=self.place_column.intersection_cell_budget,batch_cells=32,
            scope='Bounded subdivision of unchanged per-cell geometry proof; no map or contact-region enlargement'))
        with self.local_mapper._lock:surface=self.local_mapper.mesh()[0].cpu().numpy()
        e=self.place_column.equations
        inside=(surface[:,:2]@e[:,:2].T+e[:,2]<=0).all(1)
        patch=surface[inside]
        if np.sum(abs(patch[:,2]-self.place_z)<.006)<60 or np.any((patch[:,2]>self.place_z+.012)&(patch[:,2]<self.place_z+.15)):
            raise RuntimeError('placement_support_not_observed_clear')
        if self.placement_evidence_refinement:
            from .placement_evidence_map import prepare
            prepare(self,goal)
        self.travel(goal,'place')
        self.__dict__.pop('_placement_rotation_goal',None)
        if self.rest_certificate is None or not np.array_equal(self.rest_certificate['q'],self.q):raise RuntimeError('release_without_current_rest_certificate')
        packet=self.sensor_service.acquire(self.q,self.time,actuation=dict(action='release'))
        self.accept_packet(packet,integrate=False)
        self.actuation_events.append(dict(time_s=self.time,command='release',target=self.held.name,feedback=packet['actuator'],motion_commanded=False))
        if packet['actuator']['attached']:raise RuntimeError('magnetic_release_failed')
        actual_envelope=PayloadArm(self.right,self.held).payload_hull(self.q[1])[0]
        self.transfer_events.append(dict(event='measured_release_command',time_s=self.time,target=self.held.name,
            measured_destination=place.position.tolist(),tcp=self.right.tcp(self.q[1]).tolist(),envelope=actual_envelope.tolist()))
        # Transport the measured cap through the known rigid attachment, never
        # query the simulator's released object pose or load an old map revision.
        release_tcp=self.right.tcp(self.q[1])
        for item in self.measured_layers:
            if item['name']==self.held.name:item['release_tcp']=release_tcp.copy()
        cap_local=(self.source_target-self.source_tcp[:3,3])@self.source_tcp[:3,:3]
        released_cap=cap_local@release_tcp[:3,:3].T+release_tcp[:3,3]
        from classical_fusion.measured_contact import MeasuredContactPermit
        self.retreat_permit=MeasuredContactPermit(released_cap,ConvexHull(actual_envelope[:,:2]).equations)
        self.sensor_service.payload_local_points=None;self.held=None
        self.invalidate(actual_envelope.min(0)-.008,actual_envelope.max(0)+.008,'placed object changes local environment ownership',
            evidence_lower=actual_envelope.min(0),evidence_upper=actual_envelope.max(0),evidence_vertices=actual_envelope)
        self.observe(integrate=True)
        self.task_metadata['single_transfer_released']=True
        self.travel(np.r_[release_tcp[:2,3],release_tcp[2,3]+.08],'retreat')
        self.task_metadata['post_release_retreat_completed']=True
        self.task_metadata['single_transfer_cycle_completed']=True
        return 'SINGLE_MEASURED_TRANSFER_AND_RETREAT_PENDING_INDEPENDENT_AUDIT'

    def run(self,*args,**kwargs):
        self.task_metadata.update(transfer_events=self.transfer_events,
            transfer_scope=getattr(self,'transfer_scope','One measured marked-interface object; independent actual payload collision and placement audits required'),
            limitations=['No hardware validation','No whole-system hard realtime claim','Conditional isolated horizontal supported object and marked destination',
                'Environment from RGB-D only; payload inertia excluded, geometry included','Multi-object randomized completion not yet verified'])
        return super().run(*args,**kwargs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--initial-scan',required=True);p.add_argument('--scene-token',required=True);p.add_argument('--calibration-id',required=True)
    p.add_argument('--assume-unchanged-scene',action='store_true');p.add_argument('--mode',choices=['set','pose','hold'],default='set')
    p.add_argument('--steps',type=int,default=40);p.add_argument('--target',default='bracket');a=p.parse_args()
    with threadpool_limits(1):
        task=RGBDTransfer(a.output,a.scene,a.initial_scan,a.scene_token,a.calibration_id,assume_unchanged_scene=a.assume_unchanged_scene,
            mode=a.mode,asynchronous=True,contact_permission=True,capture=True,local_contact_map=True,static_interval_fusion=True)
        task.run(a.steps,.007,a.target)


if __name__=='__main__':main()
