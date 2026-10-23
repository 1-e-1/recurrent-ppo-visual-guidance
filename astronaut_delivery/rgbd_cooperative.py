"""Measured-environment dual-arm guidance experiment. No inherited CAD planner.

First noncontact task: visually approach a measured docking target to a safe
standoff. Deliberately NOT labelled completed magnetic transfer. Sensor output
instance masks are overwritten with FK self masks before any perception use.
"""
import argparse,hashlib,json,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from classical_fusion.rgbd_guard import RGBDChordEnvironmentGuard
from classical_fusion.rgbd_recording import integrate_gl_frame
from classical_fusion.robot_query_snapshot import RobotQuerySnapshot
from .rgbd_bootstrap import run as bootstrap
from .simultaneous import SimultaneousGuard
from .vision import VisualGuide,Observation,TargetEstimate
from .visual_sufficiency import VisualSetAllocator,SufficiencyConfig
from .continuous_visual import certify as visual_certificate
from .reaction import CombinedReaction,reaction_metrics
from .smooth_path import SmoothPath
from .measured_surface import export_surface
from .execution_freshness import FreshExecutionMixin
from .robot_evidence_refinement import RobotEvidenceRefinementMixin
from .history_reacquisition import HistoryReacquisitionMixin


class RGBDCooperative(FreshExecutionMixin,RobotEvidenceRefinementMixin,HistoryReacquisitionMixin):
    def __init__(self,output,scene,initial_scan,scene_token,calibration_id,*,assume_unchanged_scene=False,mode='set',asynchronous=False,contact_permission=False,capture=False,local_contact_map=False,static_interval_fusion=False,sensor_fault=None,adaptive_visual_proof=False,avoid_rejected_proposals=False,structured_ik=False,adaptive_task_staging=False,task_inequality_ik=False,history_reacquisition=False,precontact_refinement=False):
        from .rgbd_sensor_faults import validate_fault
        sensor_fault=validate_fault(sensor_fault)
        if sensor_fault is not None and not asynchronous:raise ValueError('rendered faults require isolated asynchronous sensor')
        self.wall_started=time.perf_counter()
        self.adaptive_visual_proof=bool(adaptive_visual_proof)
        self.structured_ik=bool(structured_ik)
        self.adaptive_task_staging=bool(adaptive_task_staging)
        self.task_inequality_ik=bool(task_inequality_ik)
        self.history_reacquisition=bool(history_reacquisition)
        self.precontact_refinement=bool(precontact_refinement)
        self.avoid_rejected_proposals=bool(avoid_rejected_proposals)
        self.timing=dict(full_observe_inclusive_s=[],map_commit_s=[],raw_flush_s=[],execute_inclusive_s=[])
        if mode not in ('set','pose','hold'):raise ValueError('unknown visual allocation policy')
        if capture and (not contact_permission or not asynchronous):raise ValueError('capture requires explicit measured contact and isolated actuator')
        if local_contact_map and not contact_permission:raise ValueError('local contact mode requires explicit contact permission')
        if static_interval_fusion and (not local_contact_map or not assume_unchanged_scene):raise ValueError('interval fusion requires local map and static epoch')
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        from .source_freeze import freeze
        self.source_root=Path(__file__).resolve().parents[1]
        self.sources=freeze(self.source_root,self.output,dict(scene=scene,initial_scan=initial_scan,
            scene_token=scene_token,calibration_id=calibration_id,assume_unchanged_scene=assume_unchanged_scene,
            mode=mode,asynchronous=asynchronous,contact_permission=contact_permission,capture=capture,
            local_contact_map=local_contact_map,static_interval_fusion=static_interval_fusion,
            local_map_cell_budget=getattr(self,'local_map_cell_budget',12000000),
            quality_aware_handover=getattr(self,'quality_aware_handover',False),
            static_interval_scope='broad_and_fine' if static_interval_fusion else 'disabled',
            guidance_clock='motion_and_monotonic_wall_with_pre_execution_refresh',
            robot_refinement='one_active_2mm_raw_replayed_component_block_max_2M_cells_three_builds',
            visual_certificate_max_boxes=4095 if self.adaptive_visual_proof else 127,
            adaptive_visual_proof=self.adaptive_visual_proof,sensor_stream_primed_before_motion=True,
            avoid_rejected_proposals=self.avoid_rejected_proposals,
            structured_ik=self.structured_ik,
            adaptive_task_staging=self.adaptive_task_staging,
            task_inequality_ik=self.task_inequality_ik,
            history_reacquisition=self.history_reacquisition,precontact_refinement=self.precontact_refinement,
            owned_source_departure=getattr(self,'owned_source_departure',False),
            geometric_transport_recovery=getattr(self,'geometric_transport_recovery',False),
            partitioned_geometry_proof=getattr(self,'partitioned_geometry_proof',False),
            partial_marking=getattr(self,'partial_marking',False),
            released_interface_guidance=getattr(self,'released_interface_guidance',False),
            measured_placement_pose=getattr(self,'measured_placement_pose',False),
            partial_marking_estimator='nuisance_yaw_information_with_current_support' if getattr(self,'partial_marking',False) else 'disabled',
            support_exclusion_m=.004 if getattr(self,'thin_support_search',False) else .02,
            residual_policy=getattr(self,'residual_profile',None),
            contact_query='cached_private_whole_cell_permission_mask_max_3M_cells',
            right_task_orientation='fixed_world_vertical_row_free_yaw' if getattr(self,'world_yaw_free',False) else 'legacy_axis_or_full_rotation',
            rescan_selection='known_robot_and_measured_payload_ray_bundle_ranking',
            sensor_fault=sensor_fault,
            measured_object_envelope='euclidean_error_outer_support_polytope_max_96_planes',
            placement_assignment=getattr(self,'placement_assignment','nearest_measured_mark'),
            sensing_budget_scope=getattr(self,'sensing_budget_scope','six_rescan_proposals_per_run'),
            maximum_observed_task_cycles=getattr(self,'maximum_cycles',1),
            contact_corridor='continuous_subdivision_127_boxes_depth_8_original_8mm_and_2mm_bounds'))
        initial,state=bootstrap(self.output/'commissioning',scene=scene,initial_scan=initial_scan,
            scene_token=scene_token,calibration_id=calibration_id,assume_unchanged_scene=assume_unchanged_scene,
            initial_right='ready',return_runtime=True)
        if not initial['combined_certified_candidates']:raise RuntimeError('commissioning gate failed')
        self.__dict__.update(state);self.left,self.right=self.arms;self.mode=mode
        self.self_guard=SimultaneousGuard(self.robot_guard);self.reaction=CombinedReaction(self.stage,*self.arms)
        # Known docking colours, not positions, object extents or instance IDs.
        self.guide=VisualGuide(getattr(self,'interface_palette',None) or dict(bolt=[.85,.12,.10],washer=[.10,.75,.18],bracket=[.12,.25,.90]),sensor_profile=self.camera.profile)
        self.guide.mapper=self.mapper;self.guide.fit_circular_cap=True;self.guide.reacquire_stale=True
        self.allocator=VisualSetAllocator(self.left,self.reaction,SufficiencyConfig(objective='wrench',accept_feasible_iterate=True))
        self.time=0.;self.steps=[];self.observations=[];self.paths=[];self.decisions=[];self.counter=0
        self.start_joints=self.q.copy();self.active=None;self.context=None
        self.contact_permission=contact_permission;self.contact_permits=[]
        self.recovery_count=0;self.recovery_count_total=0
        self.capture_on_arrival=capture;self.actuation_events=[]
        self.rest_certificate=None;self.last_contact_envelope=None;self.last_contact_surface=None;self.full_transfer_completed=False;self.task_metadata={}
        self.local_mapper=None;self.local_contact_map=local_contact_map
        self.static_interval_fusion=static_interval_fusion
        self.scan_source=(initial_scan,scene_token,calibration_id,assume_unchanged_scene);self.local_map_build_wall_s=None
        self.sensor_service=None;self.pending_raw=[];self.observation_ages=[];self.planning_wall=[]
        self.wall_clock_guidance=True;self.wall_observation_ages=[]
        self.robot_refinement=None;self.refinement_count=0
        if asynchronous:
            from .rgbd_sensor_service import RGBDSensorService
            self.sensor_service=RGBDSensorService(scene,**(dict(interface_palette=self.interface_palette) if getattr(self,'interface_palette',None) else {}),**(dict(partial_marking=True) if getattr(self,'partial_marking',False) else {}),**({} if sensor_fault is None else
                dict(fault_config=sensor_fault,fault_output=str(self.output/'sensor_faults'))))
        self.output.joinpath('raw').mkdir()
        self.setup_wall_s=time.perf_counter()-self.wall_started

    def observe(self,integrate=True):
        start=time.perf_counter();T=self.left.camera(self.q[0]);stamp=self.time
        if self.sensor_service is not None:
            packet=self.sensor_service.acquire(self.q,stamp)
            value=self.accept_packet(packet,integrate)
            self.timing['full_observe_inclusive_s'].append(time.perf_counter()-start)
            return value
        # Renderer alone accesses the scene. Nothing it knows about objects is
        # passed into the map, estimator, allocator or geometry guard.
        focus=self.guide.estimates.get(self.active)
        if not integrate and focus is not None:
            obs=self.camera.capture_roi(self.stage,T,stamp,focus.position,seed=17000+self.counter)
        else:obs=self.camera.capture(self.stage,T,stamp,seed=17000+self.counter)
        obs.metadata=dict(obs.metadata or {},exposure_wall_s=start)
        depth=self.robot_guard.mask_self_returns(obs.depth,obs.K,T,self.q)
        obs.robot_mask=(obs.depth>0)&(depth==0)
        self.guide.update(obs,integrate_map=False)
        if integrate:
            for _,mapper in self.mapping_layers():integrate_gl_frame(mapper,obs.rgb,depth,obs.K,T)
        name=f'rgbd_{self.counter:05d}.npz';path=self.output/'raw'/name
        np.savez_compressed(path,rgb=obs.rgb,depth=obs.depth,K=obs.K,camera_world=T,exposure_s=stamp,q=self.q,
            requested_wall_s=start)
        self.observations.append(dict(time_s=stamp,requested_wall_s=start,received_wall_s=time.perf_counter(),file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            integrated=integrate,targets={k:dict(position=v.position.tolist(),pixels=v.pixels,stamp=v.stamp,wall_stamp=v.wall_stamp) for k,v in self.guide.estimates.items()},
            valid_depth_fraction=float(np.mean(obs.depth>0)),wall_s=time.perf_counter()-start))
        self.counter+=1;return obs

    def accept_packet(self,packet,integrate=False):
        if packet is None:return None
        began=time.perf_counter()
        obs=Observation(packet['rgb'],packet['depth'],packet['K'],packet['camera_world'],packet['stamp'],
            packet['self_mask'],packet['color_valid'],packet['depth_sigma'])
        age=self.time-obs.stamp
        if age<0 or age>.25:return None
        if getattr(self,'wall_clock_guidance',False):
            wall_stamp=packet.get('requested_wall_s',np.nan)
            if not np.isfinite(wall_stamp) or wall_stamp>began:raise RuntimeError('invalid_sensor_exposure_clock')
            for value in packet['estimates'].values():
                target_wall=value.get('wall_stamp')
                if target_wall is not None and (not np.isfinite(target_wall) or target_wall>wall_stamp):
                    raise RuntimeError('invalid_target_exposure_clock')
            self.wall_observation_ages.append(began-wall_stamp)
        self.observation_ages.append(age)
        self.guide.estimates={k:TargetEstimate(**v) for k,v in packet['estimates'].items()}
        if integrate:
            commit_started=time.perf_counter()
            depth=np.where(obs.robot_mask,0,obs.depth)
            for _,mapper in self.mapping_layers():integrate_gl_frame(mapper,obs.rgb,depth,obs.K,obs.camera_world)
            if hasattr(self,'timing'):self.timing['map_commit_s'].append(time.perf_counter()-commit_started)
        name=f'rgbd_{self.counter:05d}.npz'
        row=dict(time_s=obs.stamp,received_s=self.time,requested_wall_s=packet.get('requested_wall_s'),received_wall_s=began,file=name,integrated=integrate,sensor_seed=packet['seed'],
            active_visual_target=self.active,
            targets={k:dict(position=v.position.tolist(),covariance=v.covariance.tolist(),pixels=v.pixels,stamp=v.stamp,kind=v.kind,wall_stamp=v.wall_stamp,measurement_method=v.measurement_method) for k,v in self.guide.estimates.items()},
            valid_depth_fraction=float(np.mean(obs.depth>0)),wall_s=time.perf_counter()-began,sensor_wall_s=packet['sensor_wall_s'])
        self.remember_measured_views(packet,row)
        self.observations.append(row);self.pending_raw.append((name,packet,row));self.counter+=1
        if packet.get('sensor_fault') is not None:row['sensor_fault']=packet['sensor_fault']
        if integrate:self.flush_raw()
        return obs

    def flush_raw(self):
        began=time.perf_counter();count=len(self.pending_raw)
        for name,packet,row in self.pending_raw:
            path=self.output/'raw'/name
            np.savez_compressed(path,rgb=packet['rgb'],depth=packet['depth'],K=packet['K'],camera_world=packet['camera_world'],
                exposure_s=packet['stamp'],requested_wall_s=packet.get('requested_wall_s',np.nan),
                received_wall_s=row.get('received_wall_s',np.nan),q=packet['q'],robot_mask=packet['self_mask'],
                color_valid=packet['color_valid'],depth_sigma=packet['depth_sigma'])
            row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        self.pending_raw.clear()
        if count and hasattr(self,'timing'):self.timing['raw_flush_s'].append(time.perf_counter()-began)

    def prepare_local_map(self,target):
        from classical_fusion.planning_map import PlanningMapConfig
        from classical_fusion.rgbd_recording import replay_initial_scan
        from .temporal_evidence_map import TemporalEvidenceTSDF
        began=time.perf_counter();p=target.position
        cfg=PlanningMapConfig(workspace_min=tuple(p+[-.13,-.13,-.04]),workspace_max=tuple(p+[.13,.13,.20]),
            voxel_size_m=.002,truncation_m=.006,surface_min_weight=1.,snapshot_max_age_s=30.)
        mapper=TemporalEvidenceTSDF(cfg,device=self.mapper.device);mapper.patch_radii=(4,6,8,10,12,16,24,32)
        scan,scene,calibration,unchanged=self.scan_source
        replay_initial_scan(mapper,scan,expected_scene_token=scene,expected_calibration_id=calibration,assume_unchanged_scene=unchanged)
        self.local_mapper=mapper;self.local_map_build_wall_s=time.perf_counter()-began
        self.observe(integrate=True)
        print(json.dumps(dict(event='local_RGBD_map_replayed',voxel_m=.002,wall_s=self.local_map_build_wall_s)),flush=True)

    def broad_snapshot(self):
        if getattr(self,'static_interval_fusion',False):
            from classical_fusion.static_interval_snapshot import StaticIntervalSnapshot
            if not self.scan_source[3]:raise ValueError('broad interval query requires explicitly static epoch')
            return StaticIntervalSnapshot(self.mapper,assume_static_epoch=True)
        return RobotQuerySnapshot(self.mapper)

    def make_environment(self,target,standoff):
        snapshot=self.broad_snapshot();env=RGBDChordEnvironmentGuard(*self.arms,snapshot,max_boxes=127,max_depth=8)
        if self.contact_permission and standoff<.035:
            from classical_fusion.measured_object import propose_envelope,measured_surface_padding
            from classical_fusion.measured_contact import MeasuredContactGuard,MeasuredContactPermit
            object_map=self.local_mapper if self.local_mapper is not None else self.mapper
            with object_map._lock:
                mesh=object_map.mesh();surface=mesh[0].cpu().numpy()
            padding=measured_surface_padding(object_map.cfg,object_map.spacing.cpu().numpy(),self.camera.K,
                self.left.camera(self.q[0]),target.position)
            envelope=propose_envelope(surface,target.position,margin=padding,
                support_exclusion_m=.004 if getattr(self,'thin_support_search',False) else .02)
            self.last_contact_envelope=envelope
            self.last_contact_surface=(surface,mesh[1].cpu().numpy(),object_map.revision)
            permit=MeasuredContactPermit.from_envelope(envelope,target.position,surface)
            env=MeasuredContactGuard(*self.arms,snapshot,permit=permit,max_boxes=127,max_depth=8)
            if self.local_mapper is not None:
                from classical_fusion.local_tool_guard import LocalToolEnvironmentGuard
                fine=RobotQuerySnapshot(self.local_mapper)
                if self.static_interval_fusion:
                    from classical_fusion.static_interval_snapshot import StaticIntervalSnapshot
                    fine=StaticIntervalSnapshot(self.local_mapper,assume_static_epoch=True)
                env=LocalToolEnvironmentGuard(*self.arms,snapshot,fine,permit,max_boxes=127,max_depth=8)
            self.contact_permits.append(dict(time_s=self.time,target=target.position.tolist(),
                measurement_padding_m=padding,
                footprint_equations=permit.footprint_equations.tolist(),lower_offset=permit.lower_offset,upper_offset=permit.upper_offset,
                scope='Conditional selected-target magnetic contact, not measured free space',
                assumptions=['Static isolated horizontal docking interface','Measured envelope is not proof of unseen containment']))
        return env,snapshot

    def robot_guard_arguments(self):return {}

    def right_ik_arguments(self):
        arguments=dict(max_delta=.30)
        if getattr(self,'task_inequality_ik',False):arguments['orientation_deadband']=True
        return arguments

    def right_task_axis(self):return [0,0,-1]

    def allocation_context(self,target):
        return self.allocator.context(self.q[0],target.position,target.covariance,target.pixels,self.time-target.stamp)

    def certify_visual_path(self,start,end,context):
        if not getattr(self,'adaptive_visual_proof',False):
            return visual_certificate(self.allocator,start,end,context,max_boxes=127)
        from .continuous_visual import certify_refined
        result,attempts=certify_refined(self.allocator,start,end,context)
        if len(attempts)>1:
            self.decisions.append(dict(status='ADAPTIVE_VISUAL_PROOF',time_s=self.time,
                attempts=attempts,accepted=result.allowed,
                scope='Same continuous interval bounds, target, margins and depth; computational refinement only'))
        return result

    def after_capture(self,status):return status

    def export_task_geometry(self):pass

    def plan_visual_acquisition(self,target):
        """Acquire operational view while the execution arm remains at rest.

        This is a distinct sensing-only phase: current RGB-D localization is
        required, but the manipulation visual set may initially be unmet. The
        endpoint MUST meet that unchanged set. Whole-robot environment and
        self/inter-arm certificates and fresh online feedback remain required.
        """
        if self.steps and max(np.linalg.norm(self.steps[-1]['v']),np.linalg.norm(self.steps[-1]['a']))>1e-6:
            raise RuntimeError('visual_acquisition_requires_C2_rest')
        # Preserve the existing measured contact permission when the tool is
        # already near the interface. A hard-coded 10 cm gap incorrectly
        # revoked that permission during a camera-only restoration.
        actual_standoff=float(self.right.tcp(self.q[1])[2,3]-target.position[2])
        env,snapshot=self.make_environment(target,actual_standoff)
        q0=self.q.ravel().copy();camera=self.left.camera(self.q[0])[:3,3]
        restored=None
        if hasattr(self.allocator,'restore_view'):
            try:restored=self.allocator.restore_view(self.q[0],target.position,target.covariance,target.pixels,self.time-target.stamp)
            except ValueError:pass
        proposals=([('visual_margin_QP',restored)] if restored is not None else [])
        proposals += [('look_at_IK',offset) for offset in ([0,0,0],[0,0,.04],[0,0,.08],[0,.04,0],[0,-.04,0])]
        for index,(kind,value) in enumerate(proposals):
            diagnostic=dict(status='OPERATIONAL_VIEW_ACQUISITION_PROPOSAL',time_s=self.time,index=index,rejections=[])
            def valid(left):
                pair=np.r_[left,self.q[1]]
                if not self.self_guard.box(pair,pair,**self.robot_guard_arguments())[0]:
                    diagnostic['rejections'].append('robot_endpoint');return False
                try:self.allocator.context(left,target.position,target.covariance,target.pixels,self.time-target.stamp)
                except ValueError as exc:
                    diagnostic['rejections'].append('endpoint_visual/'+str(exc));return False
                c=env.segment(q0,pair)
                if not c.allowed:diagnostic['rejections'].append(dict(environment=asdict(c),rejection=env.last_rejection))
                return c.allowed
            diagnostic['kind']=kind
            if kind=='visual_margin_QP':left=value if valid(value) else None
            else:
                position=camera+value;axis=target.position-position;axis/=np.linalg.norm(axis)
                left=self.left.solve(position,axis,seed=self.q[0],max_delta=.30,tries=4,valid=valid,
                    rng=np.random.default_rng(9200+4*len(self.paths)+index))
            if left is None:
                diagnostic['result']='no_valid_IK';self.decisions.append(diagnostic);continue
            end=np.r_[left,self.q[1]];certificate=env.segment(q0,end)
            if not certificate.allowed or not self.self_guard.segment(q0,end,**self.robot_guard_arguments()):
                diagnostic['result']='whole_path_rejected';self.decisions.append(diagnostic);continue
            diagnostic['result']='accepted';self.decisions.append(diagnostic)
            motion=SmoothPath.make(q0,end,minimum=1.2)
            self.paths.append(dict(start=q0.tolist(),end=end.tolist(),duration_s=motion.duration,time_s=self.time,
                target=target.position.tolist(),standoff_m=actual_standoff,map_revision=snapshot.revision,
                environment_certificate=asdict(certificate),robot_certificate=True,phase='visual_acquisition_right_rest',
                operational_visual_sufficiency_required=False,endpoint_operational_visual_set_passed=True,
                visual_certificate=dict(allowed=None,scope='No operational whole-segment visual claim during acquisition; fresh feedback required'),
                visual_target_name=self.active,visual_target_position=target.position.tolist(),
                scope='Measured sensing-only acquisition; execution arm does not move'))
            return motion
        return None

    def plan_step(self,target,standoff,step_m=.02):
        env,snapshot=self.make_environment(target,standoff)
        q0=self.q.ravel().copy();tcp=self.right.tcp(self.q[1])[:3,3]
        desired=target.position+np.array([0,0,standoff]);error=desired-tcp
        pose_reached=getattr(self,'right_pose_reached',lambda q:True)
        if np.linalg.norm(error)<(.001 if standoff<.035 else .004) and pose_reached(self.q[1]):
            certificate=env.segment(q0,q0)
            if not certificate.allowed or not self.self_guard.box(q0,q0,**self.robot_guard_arguments())[0]:return None,'arrival_not_certified'
            self.rest_certificate=dict(certificate=asdict(certificate),q=self.q.copy())
            return None,'arrived'
        from .placement_pose_task import translation_step
        goal=translation_step(tcp,desired,step_m)
        reasons=[];ik_rejections=[];ik_checks=dict(callbacks=0,robot_endpoint_rejections=0)
        def valid(qright):
            ik_checks['callbacks']+=1
            if not pose_reached(qright):return False
            pair=np.r_[self.q[0],qright]
            if not self.self_guard.box(pair,pair,**self.robot_guard_arguments())[0]:
                ik_checks['robot_endpoint_rejections']+=1;return False
            # A safe endpoint is not enough: select an IK branch whose whole
            # approach from the CURRENT state also has measured authority.
            c=env.segment(q0,pair);reasons.append(c.reason)
            if not c.allowed:ik_rejections.append(env.last_rejection)
            return c.allowed
        right_goal=self.right.solve(goal,self.right_task_axis(),seed=self.q[1],tries=8,valid=valid,rng=np.random.default_rng(4000+len(self.paths)),**self.right_ik_arguments())
        if right_goal is None:
            from .evidence_ik import residual_from_rejections
            residual=residual_from_rejections(self.right,ik_rejections,start=self.q[1])
            if residual is not None:
                from .placement_pose_task import compose_evidence_residual
                arguments=compose_evidence_residual(self.right_ik_arguments(),residual)
                right_goal=self.right.solve(goal,self.right_task_axis(),seed=self.q[1],tries=4,valid=valid,
                    rng=np.random.default_rng(8100+len(self.paths)),**arguments)
                self.decisions.append(dict(time_s=self.time,status='MEASURED_EVIDENCE_IK_PROPOSAL',accepted=right_goal is not None,
                    scope='Proposal only; unchanged whole-geometry and measured-free certificates required'))
        if right_goal is None and getattr(self,'structured_ik',False):
            from .structured_ik import nullspace_starts
            arguments=self.right_ik_arguments();axis=self.right_task_axis()
            starts=nullspace_starts(self.right,self.q[1],axis=axis,
                **{k:v for k,v in arguments.items() if k in ('max_delta','rotation','vertical_row')})
            began=time.perf_counter()
            if len(starts):
                right_goal=self.right.solve(goal,axis,seed=self.q[1],valid=valid,
                    initial_points=starts,**arguments)
            self.decisions.append(dict(status='STRUCTURED_NULLSPACE_IK',time_s=self.time,
                starts=len(starts),accepted=right_goal is not None,wall_s=time.perf_counter()-began,
                scope='Deterministic task-nullspace restarts; original trust region and full certificates unchanged'))
        if right_goal is None:
            # A later non-voxel rejection (e.g. contact corridor) must not erase
            # actual measured-cell witnesses needed by the bounded rescan.
            witness=next((r for r in reversed(ik_rejections) if r),env.last_rejection)
            self.decisions.append(dict(time_s=self.time,status='IK_REJECTED',reasons=reasons,rejection=witness,ik_checks=ik_checks));return None,'no_measured_safe_IK'
        try:
            context=self.allocation_context(target)
        except ValueError as exc:return None,'visual_set/'+str(exc)
        proposals=[]
        for mode in (('set','pose') if self.mode=='set' else (self.mode,)):
            if mode=='hold':
                left_goal=self.q[0].copy();row=dict(mode='hold',accepted=True,optimizer_wall_s=0.,scope='No camera reaction optimization; unchanged visual and safety certificates')
            else:left_goal,row=self.allocator.propose(self.q,right_goal,context,mode)
            for scale in ((0.,) if mode=='hold' else (1.,.5,.25,0.)):
                endpoint=np.r_[self.q[0]+scale*(left_goal-self.q[0]),right_goal]
                delta=endpoint-q0;u=np.array([.2113248654,.5,.7886751346])
                s=10*u**3-15*u**4+6*u**5;v=30*u**2-60*u**3+30*u**4;a=60*u-180*u**2+120*u**3
                W=self.reaction.wrench_batch(q0+s[:,None]*delta,v[:,None]*delta,a[:,None]*delta)
                cost=float(np.sum((W/np.array([.35]*3+[.15]*3))**2)+.001*np.dot(delta[:8],delta[:8]))
                proposals.append((cost,scale,endpoint,row,mode))
        ordered=sorted(proposals,key=lambda p:p[0])
        residual_hook=getattr(self,'residual_proposal',None)
        if residual_hook is not None:
            candidate=residual_hook(ordered[0],target,standoff,context)
            if candidate is not None:ordered.insert(0,candidate)
        for cost,scale,endpoint,row,proposal_mode in ordered:
            if getattr(self,'avoid_rejected_proposals',False):
                from .rejected_proposal_memory import repeated_rejection
                repeated=repeated_rejection(q0,endpoint,target.position,standoff,self.decisions)
                if repeated is not None:
                    self.decisions.append(dict(status='RECENT_REJECTED_PROPOSAL_SKIPPED',time_s=self.time,
                        scale=scale,proposal_mode=proposal_mode,**repeated,
                        scope='Unexecuted recent rejection at same joint state and nearby measured goal; only removes a candidate'))
                    continue
            c=env.segment(q0,endpoint)
            visual=self.certify_visual_path(q0[:8],endpoint[:8],context)
            reserve_ok=self.allocator.endpoint_valid(q0[:8],endpoint[:8],context)
            self_ok=self.self_guard.segment(q0,endpoint,**self.robot_guard_arguments()) if c.allowed and visual.allowed else False
            self.decisions.append(dict(time_s=self.time,scale=scale,environment=asdict(c),visual=visual.dictionary(),
                robot_geometry=self_ok,allocation=row,rejection=env.last_rejection,actual_unit_clock_cost=cost,proposal_mode=proposal_mode,visual_endpoint_reserve_passed=reserve_ok))
            if c.allowed and visual.allowed and reserve_ok and self_ok:
                motion=SmoothPath.make(q0,endpoint,minimum=1.2)
                self.paths.append(dict(start=q0.tolist(),end=endpoint.tolist(),duration_s=motion.duration,
                    time_s=self.time,target=target.position.tolist(),standoff_m=standoff,map_revision=snapshot.revision,
                    environment_certificate=asdict(c),visual_certificate=visual.dictionary(),robot_certificate=True,
                    visual_target_name=self.active,visual_target_position=context['target'].tolist(),
                    scope='Static environment during this segment; mapping commits at rest endpoints'))
                return motion,'accepted'
        return None,'no_continuous_joint_certificate'

    def plan_quality_recovery(self,target):
        """Use a fresh sufficient visual witness to reacquire a poor task view.

        The original estimate is a gaze proposal ONLY, never retimestamped or
        substituted as a manipulation target. Execution arm stays at rest. The
        caller monitors the returned witness during sensing, then rechecks the
        original target before any execution-arm increment.
        """
        if self.recovery_count>=6:return None,None
        if self.steps and max(np.linalg.norm(self.steps[-1]['v']),np.linalg.norm(self.steps[-1]['a']))>1e-6:
            raise RuntimeError('quality_recovery_requires_C2_rest')
        original=self.active
        candidates=sorted(self.guide.estimates.values(),key=lambda e:np.linalg.norm(e.position-target.position))
        try:
            for reference in candidates:
                if reference.name==original:continue
                try:self.allocator.context(self.q[0],reference.position,reference.covariance,reference.pixels,self.time-reference.stamp)
                except ValueError:continue
                self.active=reference.name
                motion=self.plan_observation_recovery(reference,look_at=target.position,require_view_gain=True)
                if motion is None:continue
                self.paths[-1].update(phase='quality_reacquisition_right_rest',original_task_target=original,
                    recovery_visual_witness=reference.name,original_task_quality_required_before_right_motion=True)
                return motion,reference.name
        finally:self.active=original
        return None,None

    def plan_operational_recovery(self,target,reason):
        # Fresh but geometrically misplaced and poorly localized are different
        # failures. A gaze-only geometric restoration need not improve pixels.
        if reason=='initial_view_outside_visual_set':return self.plan_visual_acquisition(target),None
        return self.plan_quality_recovery(target)

    def plan_observation_recovery(self,target,look_at=None,require_view_gain=False):
        """Bounded RGB-D rescan: right arm rests, left takes a certified view.

        Candidate directions are task-independent camera displacements. No
        environment truth is used to select a 'known good' viewpoint.
        """
        if self.recovery_count>=6:
            self.decisions.append(dict(status='RGBD_RESCAN_TASK_BUDGET_EXHAUSTED',time_s=self.time,limit=6))
            return None
        env,snapshot=self.make_environment(target,float(self.right.tcp(self.q[1])[2,3]-target.position[2]))
        q0=self.q.ravel().copy();camera=self.left.camera(self.q[0])[:3,3]
        try:context=self.allocation_context(target)
        except ValueError as exc:
            self.decisions.append(dict(status='RGBD_RESCAN_REFERENCE_REJECTED',time_s=self.time,reason=str(exc)))
            return None
        radial=target.position-camera;radial[2]=0;radial/=max(np.linalg.norm(radial),1e-8)
        lateral=np.cross(radial,[0,0,1.])
        offsets=[np.array([0,0,.06]),.04*radial+[0,0,.04],.06*lateral,-.06*lateral]
        from .evidence_view_focus import recovery_focus
        focus,frontier=recovery_focus(target.position,self.decisions)
        if look_at is not None:
            focus=np.asarray(look_at,float)
            if focus.shape!=(3,) or not np.isfinite(focus).all():raise ValueError('finite measured view proposal required')
            offsets=[np.zeros(3),*offsets]
        if require_view_gain:
            from .visual_sufficiency import projected_support_proxy
            initial_support=projected_support_proxy(camera,focus)
            direction=focus-camera;direction/=max(np.linalg.norm(direction),1e-8)
            # Reorienting at the same position cannot improve this support
            # proxy. Try actual small translations, not repeated gaze-only IK.
            offsets=[.04*direction,.02*direction,np.array([0,0,.04]),np.array([0,0,.02]),*offsets[1:]]
        if look_at is None and not require_view_gain and frontier is not None:
            from .rescan_occlusion import rejected_ray_samples,rank_offsets
            from classical_fusion.rgbd_guard import robot_hulls
            arm=self.right
            if getattr(self,'held',None) is not None:
                from .measured_payload import PayloadArm
                arm=PayloadArm(arm,self.held)
            offsets.extend([np.array([0,0,-.06]),-.04*radial+[0,0,-.04]])
            offsets,ranking=rank_offsets(camera,offsets,rejected_ray_samples(self.decisions,target.position),
                [h for _,h in robot_hulls(arm,self.q[1])])
            self.decisions.append(dict(status='RGBD_RESCAN_OCCLUSION_RANKING',time_s=self.time,**(ranking or {})))
        # Alternate the lateral choice if upward motion has exhausted reach.
        for index,offset in enumerate(offsets):
            position=camera+offset;axis=focus-position;axis/=np.linalg.norm(axis)
            diagnostic=dict(status='RGBD_RESCAN_PROPOSAL',time_s=self.time,index=index,offset_m=offset.tolist(),endpoint_rejections=[])
            def valid(left):
                if require_view_gain and projected_support_proxy(self.left.camera(left)[:3,3],focus)<1.04*initial_support:
                    diagnostic['endpoint_rejections'].append(dict(reason='insufficient_projected_support_gain'));return False
                pair=np.r_[left,self.q[1]]
                robot_ok=self.self_guard.box(pair,pair,**self.robot_guard_arguments())[0]
                certificate=env.segment(pair,pair) if robot_ok else None
                if not robot_ok or not certificate.allowed:
                    diagnostic['endpoint_rejections'].append(dict(robot=robot_ok,environment=None if certificate is None else asdict(certificate),rejection=env.last_rejection))
                return robot_ok and certificate.allowed
            left=self.left.solve(position,axis,seed=self.q[0],tries=4,valid=valid,
                rng=np.random.default_rng(7100+4*self.recovery_count+index))
            if left is None:
                diagnostic['result']='no_endpoint_IK';self.decisions.append(diagnostic);continue
            endpoint=np.r_[left,self.q[1]];certificate=env.segment(q0,endpoint)
            visual=self.certify_visual_path(q0[:8],endpoint[:8],context)
            reserve=self.allocator.endpoint_valid(q0[:8],endpoint[:8],context)
            robot_ok=self.self_guard.segment(q0,endpoint,**self.robot_guard_arguments()) if certificate.allowed and visual.allowed else False
            diagnostic.update(environment=asdict(certificate),rejection=env.last_rejection,visual=visual.dictionary(),endpoint_reserve=reserve,robot=robot_ok)
            diagnostic['result']='accepted' if certificate.allowed and visual.allowed and reserve and robot_ok else 'path_rejected'
            self.decisions.append(diagnostic)
            if diagnostic['result']!='accepted':continue
            motion=SmoothPath.make(q0,endpoint,minimum=1.2);self.recovery_count+=1
            self.recovery_count_total=getattr(self,'recovery_count_total',0)+1
            self.paths.append(dict(start=q0.tolist(),end=endpoint.tolist(),duration_s=motion.duration,
                time_s=self.time,target=target.position.tolist(),standoff_m=float(self.right.tcp(self.q[1])[2,3]-target.position[2]),
                map_revision=snapshot.revision,environment_certificate=asdict(certificate),visual_certificate=visual.dictionary(),
                robot_certificate=True,phase='active_RGBD_coverage_recovery',
                visual_target_name=self.active,visual_target_position=context['target'].tolist(),
                scope='Certified left-only rescan; right arm rests, unknown space remains blocked'))
            self.decisions.append(dict(time_s=self.time,status='ACTIVE_RGBD_RESCAN',camera_displacement_m=offset.tolist(),
                measured_evidence_frontier=None if frontier is None else frontier.tolist(),look_at_proposal=focus.tolist()))
            return motion
        return None

    def execute(self,motion):
        import gc
        began=time.perf_counter()
        enabled=gc.isenabled();gc.disable()
        try:
            self.prepare_execution_measurement(motion)
            return self._execute(motion)
        finally:
            if enabled:gc.enable()
            if hasattr(self,'timing'):self.timing['execute_inclusive_s'].append(time.perf_counter()-began)

    def stationary_capture(self,*,after_visual_brake=False):
        """No motion permission: binary latch confirmation at a certified rest.

        After terminal occlusion, a recent static RGB-D belief may justify a
        zero-motion actuator attempt. It is never relabelled as a fresh image,
        and cannot authorize a resumed blind trajectory or payload transport.
        """
        if not self.capture_on_arrival or not self.contact_permission or self.sensor_service is None:
            raise RuntimeError('capture_not_enabled')
        if not self.steps or not self.paths:raise RuntimeError('no_certified_arrival')
        last=self.steps[-1]
        if max(np.linalg.norm(last['v']),np.linalg.norm(last['a']))>1e-6:raise RuntimeError('capture_requires_C2_rest')
        target=self.guide.estimates.get(self.active)
        if target is None:raise RuntimeError('missing_measured_capture_target')
        age=self.time-target.stamp;maximum_age=.5 if after_visual_brake else .25
        if not 0<=age<=maximum_age or not self.scan_source[3]:raise RuntimeError('terminal_static_belief_expired')
        if np.sqrt(np.linalg.eigvalsh(target.covariance).max())>.005:raise RuntimeError('uncertain_capture_target')
        if np.linalg.norm(self.right.tcp(self.q[1])[:3,3]-target.position)>.008:raise RuntimeError('outside_measured_capture_neighborhood')
        certificate=self.paths[-1]['environment_certificate']
        if not after_visual_brake:
            if self.rest_certificate is None or not np.array_equal(self.rest_certificate['q'],self.q):raise RuntimeError('current_rest_not_certified')
            certificate=self.rest_certificate['certificate']
        if self.paths[-1]['standoff_m']>=.035 or 'CONTACT' not in certificate['scope']:raise RuntimeError('not_a_contact_approach')
        if not certificate['allowed'] or certificate['revision']!=self.mapper.revision:raise RuntimeError('capture_certificate_revised')
        if self.local_mapper is not None and certificate.get('local_revision')!=self.local_mapper.revision:
            raise RuntimeError('local_capture_certificate_revised')
        packet=self.sensor_service.acquire(self.q,self.time,actuation=dict(action='capture',target=self.active))
        self.accept_packet(packet,integrate=False)
        self.actuation_events.append(dict(time_s=self.time,command='capture',target=self.active,feedback=packet['actuator'],
            after_visual_brake=after_visual_brake,last_measured_target_age_s=age,motion_commanded=False,
            scope='Binary kinematic latch only; no target pose or geometry feedback, no blind resumed motion'))
        if not packet['actuator']['attached']:raise RuntimeError('physical_magnetic_capture_failed')

    def _execute(self,motion):
        # RGB-D feedback is online during both arms' motion. Spatial commits
        # occur at rest endpoints, avoiding silently using revoked revisions.
        # This synchronous prototype reports wall cost; it is NOT hard realtime.
        elapsed=0.;tick=0;stop_reason=None
        while elapsed<motion.duration-1e-9:
            began=time.perf_counter();h=min(1/60,motion.duration-elapsed)
            middle=motion.state(elapsed+h*.5);W=self.reaction.wrench(*middle[:3])
            elapsed+=h;self.time+=h;q,v,a,j=motion.state(elapsed);self.q=q.reshape(2,8)
            self.robot.set_positions(*self.q);tick+=1
            if self.sensor_service is not None and stop_reason is None:
                try:
                    if tick%4==0:
                        estimate=self.guide.estimates.get(self.active)
                        self.sensor_service.submit(self.q,self.time,estimate.position if estimate else None)
                    self.accept_packet(self.sensor_service.poll())
                    stop_reason=self.feedback_stop_reason()
                except Exception as exc:stop_reason='sensor_exception_smooth_brake/'+str(exc)
                if stop_reason is not None:
                    brake=motion.stop(elapsed)
                    if brake is not None:motion=brake;elapsed=0.
            elif tick%4==0 and stop_reason is None:
                try:
                    self.observe(integrate=False)
                    stop_reason=self.feedback_stop_reason()
                except Exception as exc:stop_reason='sensor_exception_smooth_brake/'+str(exc)
                if stop_reason is not None:
                    brake=motion.stop(elapsed)
                    if brake is not None:motion=brake;elapsed=0.
            visual_target=self.guide.estimates.get(self.active)
            self.steps.append(dict(time_s=self.time,dt_s=h,q=q.tolist(),v=v.tolist(),a=a.tolist(),jerk=j.tolist(),wrench=W.tolist(),
                guidance_age_s=None if visual_target is None else self.measurement_age(visual_target),
                raw_age_s=self.raw_measurement_age(),feedback_stop_reason=stop_reason,
                monotonic_wall_s=time.perf_counter(),
                wrench_state=dict(q=middle[0].tolist(),v=middle[1].tolist(),a=middle[2].tolist()),
                wall_s=time.perf_counter()-began,phase='braking' if stop_reason else 'guided_approach'))
            if self.sensor_service is not None:time.sleep(max(0.,h-(time.perf_counter()-began)))
        if stop_reason:raise RuntimeError(stop_reason)
        self.observe(integrate=True)

    def feedback_stop_reason(self):
        if getattr(self,'_view_rollback_active',False):
            return None if self.raw_measurement_age()<=.25 else 'rgbd_loss_smooth_brake'
        target=self.guide.target(self.active,self.time)
        if target is None or self.measurement_age(target)>.25:return 'visual_loss_smooth_brake'
        cfg=self.allocator.cfg;cov=np.asarray(target.covariance,float)
        if (target.pixels<cfg.minimum_pixels or cov.shape!=(3,3) or not np.isfinite(cov).all()
            or not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min()<-1e-12
            or np.linalg.eigvalsh(cov).max()>cfg.maximum_sigma_m**2):return 'visual_quality_smooth_brake'
        return None

    def recover_missing_task_view(self):
        """Bounded sensing-only return, NOT blind manipulation or new free space.

        A failed CAMERA-ONLY scan may have lost every task marker. Return along
        its actual joint line to the prior camera start, after re-certifying
        current robot/payload and RGB-D map revisions. Operational task vision
        is not claimed during acquisition; raw-frame liveness is still checked.
        """
        if getattr(self,'history_reacquisition',False) and self.reacquire_from_measured_history():return True
        rest=np.r_[self.steps[-1]['v'],self.steps[-1]['a']] if self.steps else np.array([np.nan])
        historical=self.guide.estimates.get(self.active)
        if (not np.isfinite(rest).all() or np.linalg.norm(rest)>1e-6 or not self.paths
            or not self.scan_source[3] or getattr(self,'view_rollback_count',0)>=2 or historical is None
            or not np.isfinite(historical.stamp) or historical.stamp>self.time
            or not np.isfinite(historical.position).all()):return False
        previous=self.paths[-1]
        if (previous.get('phase') not in ('active_RGBD_coverage_recovery','quality_reacquisition_right_rest')
            or previous.get('visual_target_name')!=self.active or not previous.get('robot_certificate')
            or not previous.get('environment_certificate',{}).get('allowed')
            or not previous.get('visual_certificate',{}).get('allowed')):return False
        a,b=np.asarray(previous['start'],float),np.asarray(previous['end'],float);current=self.q.ravel().copy()
        if a.shape!=(16,) or b.shape!=(16,) or not np.isfinite([a,b,current]).all():return False
        if not np.array_equal(a[8:],b[8:]) or not np.allclose(current[8:],a[8:],atol=1e-10,rtol=0):return False
        line=b-a;length2=float(line@line)
        if length2<1e-12:return False
        u=float((current-a)@line/length2)
        if not -1e-8<=u<=1+1e-8 or np.linalg.norm(current-(a+u*line))>1e-7 or np.linalg.norm(current-a)<1e-6:return False
        end=np.r_[a[:8],current[8:]];began=time.perf_counter()
        try:
            gap=float(self.right.tcp(self.q[1])[2,3]-historical.position[2])
            env,snapshot=self.make_environment(historical,gap)
            certificate=env.segment(current,end)
            if not certificate.allowed or not self.self_guard.segment(current,end,**self.robot_guard_arguments()):return False
        finally:self.planning_wall.append(time.perf_counter()-began)
        motion=SmoothPath.make(current,end,minimum=1.2)
        row=dict(start=current.tolist(),end=end.tolist(),duration_s=motion.duration,time_s=self.time,
            target=historical.position.tolist(),standoff_m=gap,map_revision=snapshot.revision,
            environment_certificate=asdict(certificate),robot_certificate=True,phase='certified_view_rollback_right_rest',
            operational_visual_sufficiency_required=False,visual_target_name=self.active,visual_target_position=historical.position.tolist(),
            visual_certificate=dict(allowed=None,scope='Sensing-only return with raw-frame watchdog; task marker may be absent'),
            historical_target_stamp_s=historical.stamp,return_to_sensing_start_s=previous['time_s'],
            scope='Current-map certified left-only reversal; no right motion or stale target execution authority')
        # Retain the exact ownership/contact metadata for independent auditing.
        for key in ('payload_local_points','payload_name','payload_contact_phase','contact_column','released_target_contact'):
            if key in previous:row[key]=previous[key]
        self.paths.append(row);self.view_rollback_count=getattr(self,'view_rollback_count',0)+1
        self.task_metadata['view_rollback_attempts']=self.view_rollback_count
        self.decisions.append(dict(status='CERTIFIED_SENSING_VIEW_ROLLBACK',time_s=self.time,
            target=self.active,historical_target_stamp_s=historical.stamp,attempt=self.view_rollback_count,
            scope=row['scope']))
        first=len(self.steps);self._view_rollback_active=True
        try:self.execute(motion)
        finally:
            self._view_rollback_active=False
            for item in self.steps[first:]:
                if item['phase']!='braking':item['phase']=row['phase']
        return True  # attempt executed; caller MUST require new task feedback

    def reobserve_after_visual_stop(self,error):
        if str(error)=='pre_execution_measurement_rejected':
            if not self.at_known_rest():raise RuntimeError('pre_execution_rejection_not_at_rest')
            self.observe(integrate=True)
            return
        if str(error) not in ('visual_quality_smooth_brake','visual_loss_smooth_brake'):raise error
        rest=np.r_[self.steps[-1]['v'],self.steps[-1]['a']] if self.steps else np.array([np.nan])
        if not np.isfinite(rest).all() or np.linalg.norm(rest)>1e-6:
            raise RuntimeError('visual_reobservation_requires_C2_rest')
        self.observe(integrate=True)
        self.decisions.append(dict(status='FULL_RGBD_REOBSERVED_AFTER_VISUAL_STOP',time_s=self.time,
            reason=str(error),target=self.active,new_target_available=self.guide.target(self.active,self.time) is not None,
            scope='Observation only; next motion still requires all original certificates and quality gates'))

    def approach(self,steps,standoff):
        """One measured approach; reusable without resetting robot or maps."""
        staging=.10
        if standoff<.10 and getattr(self,'adaptive_task_staging',False):
            target=self.guide.target(self.active,self.time)
            if target is not None:
                from .task_staging import choose_standoff
                began=time.perf_counter()
                staging,attempts=choose_standoff(self.right,self.q[1],target.position,standoff,
                    orientation_deadband=getattr(self,'task_inequality_ik',False))
                self.planning_wall.append(time.perf_counter()-began)
                self.decisions.append(dict(status='MEASURED_TASK_STAGING',time_s=self.time,
                    target=target.position.tolist(),selected_standoff_m=staging,attempts=attempts,
                    scope='Measured target and robot pose feasibility only; no motion permission. Original joint trust region and all certificates unchanged'))
        schedule=[staging,standoff] if standoff<.10 else [standoff];phase=0
        for _ in range(steps):
            target=self.guide.target(self.active,self.time)
            if target is None:
                if self.recover_missing_task_view():continue
                raise RuntimeError('no_current_RGBD_target')
            try:self.allocation_context(target)
            except ValueError as exc:
                began=time.perf_counter();reference=None
                acquisition,reference=self.plan_operational_recovery(target,str(exc))
                self.planning_wall.append(time.perf_counter()-began)
                if acquisition is None:
                    if getattr(self,'history_reacquisition',False) and self.reacquire_from_measured_history():continue
                    raise RuntimeError('no_certified_operational_view_acquisition')
                first=len(self.steps);original=self.active
                try:
                    if reference is not None:self.active=reference
                    self.execute(acquisition)
                except RuntimeError as exc:
                    self.active=original;self.reobserve_after_visual_stop(exc)
                finally:self.active=original
                for row in self.steps[first:]:
                    if row['phase']!='braking':row['phase']='visual_acquisition_right_rest'
                continue
            began=time.perf_counter()
            for step_m in (.02,.01,.005):
                motion,why=self.plan_step(target,schedule[phase],step_m=step_m)
                if motion is not None or why=='arrived':break
            self.planning_wall.append(time.perf_counter()-began)
            if why=='arrived':
                if phase<len(schedule)-1:
                    if self.local_contact_map and self.local_mapper is None:self.prepare_local_map(target)
                    phase+=1;continue
                if self.capture_on_arrival:
                    self.stationary_capture();return 'MAGNETIC_CAPTURE_CONFIRMED'
                return 'GUIDED_STANDOFF_REACHED'
            if motion is None:
                if self.refresh_expired_evidence():continue
                if why=='no_measured_safe_IK' and getattr(self,'precontact_refinement',False) and self.prepare_robot_refinement():continue
                if why=='no_measured_safe_IK' and self.recovery_count<6:
                    began=time.perf_counter();motion=self.plan_observation_recovery(target);self.planning_wall.append(time.perf_counter()-began)
                if motion is None:raise RuntimeError(why)
            try:self.execute(motion)
            except RuntimeError as exc:
                if str(exc)=='visual_loss_smooth_brake' and self.capture_on_arrival:
                    # Same zero-motion static-belief latch permission used by
                    # run(), now available to later objects too. Never resume
                    # blind motion or refresh a historical target timestamp.
                    try:self.stationary_capture(after_visual_brake=True)
                    except RuntimeError:pass
                    else:return 'MAGNETIC_CAPTURE_CONFIRMED_AFTER_BRAKE'
                # Current line has stopped at C2 rest. No blind continuation:
                # reacquire a full frame and let the next iteration recheck.
                self.reobserve_after_visual_stop(exc);continue
            print(json.dumps(dict(event='rgbd_segment_executed',segments=len(self.paths),time_s=self.time)),flush=True)
        return 'TIMEOUT'

    def run(self,steps=20,standoff=.10,target_name='bracket'):
        run_wall_started=time.perf_counter()
        self.output.joinpath('task_request.json').write_text(json.dumps(dict(approach_step_budget=steps,
            standoff_m=standoff,target_selection=target_name)),encoding='utf-8')
        self.active=target_name;status='TIMEOUT';error=None
        try:
            self.observe(integrate=True)
            if target_name=='auto':
                candidates=[e for e in self.guide.estimates.values() if e.kind=='magnetic_interface' and self.guide.target(e.name,self.time) is not None]
                if not candidates:raise RuntimeError('no_observed_magnetic_interface')
                tcp=self.right.tcp(self.q[1])[:3,3]
                sufficient=[]
                for candidate in candidates:
                    try:self.allocation_context(candidate);sufficient.append(candidate)
                    except ValueError:pass
                self.active=min(sufficient or candidates,key=lambda e:np.linalg.norm(e.position-tcp)).name
            if self.sensor_service is not None:
                estimate=self.guide.estimates.get(self.active)
                if estimate is not None:self.accept_packet(self.sensor_service.acquire(self.q,self.time,focus=estimate.position))
            status=self.approach(steps,standoff)
        except Exception as exc:
            status='STOPPED';error=str(exc)
            self.task_metadata['failure_traceback']=traceback.format_exc()
            if error=='visual_loss_smooth_brake' and self.capture_on_arrival:
                try:
                    self.stationary_capture(after_visual_brake=True)
                    status='MAGNETIC_CAPTURE_CONFIRMED_AFTER_BRAKE';error=None
                except Exception as capture_error:error+='; '+str(capture_error)
        if status.startswith('MAGNETIC_CAPTURE_CONFIRMED'):
            try:status=self.after_capture(status)
            except Exception as continuation_error:
                status='STOPPED';error=str(continuation_error);self.task_metadata['failure_traceback']=traceback.format_exc()
        self.flush_raw()
        if self.sensor_service is not None:self.sensor_service.close()
        cycle=np.array([x['wall_s'] for x in self.steps])
        from .source_freeze import unchanged
        result=dict(status=status,error=error,full_transfer_completed=self.full_transfer_completed,environment_authority='RGBD_ONLY',
            guidance_clock='motion_and_monotonic_wall_with_pre_execution_refresh',
            received_wall_observation_age_s=self.wall_observation_ages,
            explicit_measured_target_contact_permission=self.contact_permission,
            local_contact_map=self.local_mapper is not None,local_map_build_wall_s=self.local_map_build_wall_s,
            static_interval_fusion=self.static_interval_fusion,
            static_interval_scope='broad_and_fine' if self.static_interval_fusion else 'disabled',
            active_RGBD_recovery_segments=self.recovery_count_total,
            recovery_counter_scope='Accepted rescan proposals, including ones rejected by a later execution refresh; actual execution remains in paths/steps',
            current_observed_task_recovery_proposals=self.recovery_count,
            timing=dict(**self.timing,setup_wall_s=self.setup_wall_s,
                task_wall_s_including_setup=time.perf_counter()-self.wall_started,
                run_wall_s_before_artifact_export=time.perf_counter()-run_wall_started,
                scope='Wall clock including compute waits; named inclusive intervals overlap and MUST NOT be summed. Final artifact export excluded.'),
            mode=self.mode,initial_joints=self.start_joints.tolist(),final_joints=self.q.tolist(),
            candidate_selection='bounded_actual_unit_clock_cost_v2',asynchronous_sensor=self.sensor_service is not None,
            robot_refinement_builds=self.refinement_count,
            wall_observation_age_p95_s=float(np.quantile(self.wall_observation_ages,.95)) if self.wall_observation_ages else None,
            observation_age_p95_s=float(np.quantile(self.observation_ages,.95)) if self.observation_ages else None,
            cycle_p99_ms=float(np.quantile(cycle,.99)*1000) if len(cycle) else None,
            cycle_max_ms=float(cycle.max()*1000) if len(cycle) else None,
            cycle_deadline_misses=int(np.sum(cycle>1/60)),planning_wall_s=self.planning_wall,
            executed_segments=len(self.paths),executed_ticks=len(self.steps),simulation_time_s=self.time,
            both_arms_moving_ticks=int(sum(np.linalg.norm(x['v'][:8])>1e-5 and np.linalg.norm(x['v'][8:])>1e-5 for x in self.steps)),
            reaction=reaction_metrics([x['wrench'] for x in self.steps],[x['dt_s'] for x in self.steps]) if self.steps else None,
            source_sha256=self.sources,source_unchanged_during_run=unchanged(self.source_root,self.sources),real_time_verified=False,
            limitations=['Guidance/capture experiment only; no carried-payload motion or placement yet','No whole-system hard realtime claim; planning, full-frame mapping and raw flush occur at rest',
                          'Static environment during certified segments; map commits at C2 rest endpoints','Known coloured circular docking interface, not arbitrary semantic grasping'])
        result.update(self.task_metadata)
        for name,value in [('result.json',result),('paths.json',self.paths),('steps.json',self.steps),('observations.json',self.observations),('decisions.json',self.decisions),('contact_permits.json',self.contact_permits),('actuation_events.json',self.actuation_events)]:
            (self.output/name).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
        export_surface(self.mapper,self.output)
        if self.local_mapper is not None:
            local_output=self.output/'local_map';local_output.mkdir();export_surface(self.local_mapper,local_output)
        self.export_task_geometry()
        if self.robot_refinement is not None:
            block=self.robot_refinement;folder=self.output/f'robot_refinement_{block["generation"]:02d}'
            folder.mkdir();export_surface(block['mapper'],folder)
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','transfer_events')},indent=2),flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--initial-scan',required=True);p.add_argument('--scene-token',required=True);p.add_argument('--calibration-id',required=True)
    p.add_argument('--assume-unchanged-scene',action='store_true');p.add_argument('--mode',choices=['set','pose','hold'],default='set')
    p.add_argument('--steps',type=int,default=20);p.add_argument('--standoff',type=float,default=.10);p.add_argument('--target',default='bracket');p.add_argument('--async-sensor',action='store_true')
    p.add_argument('--contact-permission',action='store_true',help='Explicit measured docking-band permission; does not make unknown space free');p.add_argument('--capture',action='store_true');p.add_argument('--local-contact-map',action='store_true');p.add_argument('--static-interval-fusion',action='store_true');a=p.parse_args()
    with threadpool_limits(1):
        task=RGBDCooperative(a.output,a.scene,a.initial_scan,a.scene_token,a.calibration_id,assume_unchanged_scene=a.assume_unchanged_scene,mode=a.mode,asynchronous=a.async_sensor,contact_permission=a.contact_permission,capture=a.capture,local_contact_map=a.local_contact_map,static_interval_fusion=a.static_interval_fusion)
        task.run(a.steps,a.standoff,a.target)


if __name__=='__main__':main()
