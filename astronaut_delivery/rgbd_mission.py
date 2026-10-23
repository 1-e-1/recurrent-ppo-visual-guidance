"""Persistent measured-interface task queue; never receives scene object count.

Queue exhaustion means all OBSERVED interfaces completed, not proof that the
world contains no other object. Independent audit alone checks the scene count.
"""
import argparse,json,time
from dataclasses import replace
import numpy as np
from threadpoolctl import threadpool_limits
from .rgbd_transfer import RGBDTransfer
from .measured_task_memory import MeasuredTaskMemory


class RGBDMission(RGBDTransfer):
    def __init__(self,*args,maximum_cycles=8,**kwargs):
        if not 1<=maximum_cycles<=32:raise ValueError('bounded mission cycle budget required')
        self.memory=MeasuredTaskMemory();self.maximum_cycles=maximum_cycles
        self.sensing_budget_scope='six_rescan_proposals_per_observed_object'
        self.recovery_budget_targets=set()
        self.transfer_scope='Persistent queue of observed uniquely coloured interfaces; no supplied object count; independent scene-completion audit required'
        super().__init__(*args,**kwargs)
        self.task_metadata.update(completed_observed_targets=self.memory.completed,
            scene_object_count_supplied=False,observed_queue_exhausted=False,mission_cycle_budget=maximum_cycles)

    def accept_packet(self,packet,integrate=False):
        observation=super().accept_packet(packet,integrate)
        if observation is not None:self.memory.observe(self.guide.estimates)
        return observation

    def available_placements(self):
        return [e for e in super().available_placements() if self.memory.place_available(e.position)]

    def placement_proposals(self):
        return [e for e in super().placement_proposals() if self.memory.place_available(e.position)]

    def begin_next_recovery_budget(self,previous,name):
        # A new measured task gets the SAME finite sensing allowance. A retry
        # of the same task cannot reset it, nor can an unobserved scene object.
        if not self.memory.completed or previous!=self.memory.completed[-1] or name not in {e.name for e in self.memory.pending()}:
            raise ValueError('recovery budget requires completed previous and observed pending task')
        if name in self.recovery_budget_targets:raise ValueError('cannot renew budget for the same task')
        self.recovery_budget_targets.add(name)
        self.transfer_events.append(dict(event='new_observed_task_recovery_budget',time_s=self.time,
            previous_target=previous,previous_used=self.recovery_count,new_target=name,proposal_limit=6,
            scope='New measured object task only; no same-task renewal or relaxed geometry/vision limits'))
        self.recovery_count=0

    def acquire_next_view(self,name):
        historical=self.memory.seen[name]
        for _ in range(3):
            if self.guide.target(name,self.time) is not None:return
            current=self.guide.target(self.active,self.time)
            if current is None:raise RuntimeError('no_fresh_target_for_next_object_rescan')
            began=time.perf_counter();motion=self.plan_observation_recovery(current,look_at=(current.position+historical.position)*.5)
            self.planning_wall.append(time.perf_counter()-began)
            if motion is None:raise RuntimeError('no_certified_next_object_rescan')
            self.paths[-1]['phase']='next_object_visual_handover'
            try:self.execute(motion)
            except RuntimeError as exc:
                if not getattr(self,'history_reacquisition',False):raise
                self.reobserve_after_visual_stop(exc)
        if self.guide.target(name,self.time) is None:raise RuntimeError('next_object_not_reobserved')

    def after_capture(self,status):
        for _ in range(self.maximum_cycles):
            name=self.active
            status=super().after_capture(status)
            self.memory.complete(name,np.r_[self.place_xy,self.place_z])
            self.transfer_events.append(dict(event='observed_object_cycle_completed',name=name,time_s=self.time,
                placement=np.r_[self.place_xy,self.place_z].tolist(),scene_count_known=False))
            self.observe(integrate=True)
            # Preserve the current fine-map evidence before a different ROI
            # replaces it. Do not lose the successful cycle's mapping audit.
            if self.local_mapper is not None:
                from .measured_surface import export_surface
                from .rigid_measured_surface import export_composite,transform_layer
                folder=self.output/f'cycle_{len(self.memory.completed)-1:03d}_local_map';folder.mkdir()
                export_surface(self.local_mapper,folder)
                (folder/'scene_state_at_export.json').write_text(json.dumps(dict(time_s=self.time,
                    final_joints=self.q.tolist(),transfer_events=self.transfer_events,
                    scope='Measured task events and encoder state at map export; no environment truth'),indent=2),encoding='utf-8')
                layers=[transform_layer(item['data'],item['release_tcp']) for item in self.measured_layers if 'release_tcp' in item]
                if layers:export_composite(folder,layers)
            pending=self.memory.pending()
            if not pending:
                self.task_metadata['observed_queue_exhausted']=True
                return 'OBSERVED_RGBD_TASK_QUEUE_COMPLETED_PENDING_SCENE_AUDIT'
            if len(self.memory.completed)>=self.maximum_cycles:break
            tcp=self.right.tcp(self.q[1])[:3,3]
            selected=min(pending,key=lambda e:np.linalg.norm(e.position-tcp))
            self.begin_next_recovery_budget(name,selected.name)
            self.acquire_next_view(selected.name)
            self.active=selected.name;self.transfer_phase=None;self.retreat_permit=None
            self.last_contact_envelope=None;self.rest_certificate=None;self.local_mapper=None
            self.allocator.cfg=replace(self.allocator.cfg,cap_radius_m=.020,endpoint_reserve=0.)
            status=self.approach(48,.007)
            if not status.startswith('MAGNETIC_CAPTURE_CONFIRMED'):raise RuntimeError('next_object_approach/'+status)
        raise RuntimeError('observed_mission_cycle_budget_exhausted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--initial-scan',required=True);p.add_argument('--scene-token',required=True);p.add_argument('--calibration-id',required=True)
    p.add_argument('--assume-unchanged-scene',action='store_true');p.add_argument('--mode',choices=['set','pose','hold'],default='set')
    p.add_argument('--maximum-cycles',type=int,default=8)
    p.add_argument('--structured-ik',action='store_true')
    p.add_argument('--adaptive-task-staging',action='store_true')
    p.add_argument('--owned-source-departure',action='store_true')
    p.add_argument('--task-inequality-ik',action='store_true')
    p.add_argument('--history-reacquisition',action='store_true')
    p.add_argument('--precontact-refinement',action='store_true')
    p.add_argument('--thin-support-search',action='store_true')
    p.add_argument('--geometric-transport-recovery',action='store_true')
    p.add_argument('--far-first-placement',action='store_true')
    p.add_argument('--partitioned-geometry-proof',action='store_true')
    p.add_argument('--partial-marking',action='store_true')
    p.add_argument('--released-interface-guidance',action='store_true')
    p.add_argument('--measured-placement-pose',action='store_true')
    p.add_argument('--placement-evidence-refinement',action='store_true')
    p.add_argument('--adaptive-visual-proof',action='store_true')
    p.add_argument('--quality-aware-handover',action='store_true')
    p.add_argument('--avoid-rejected-proposals',action='store_true')
    p.add_argument('--local-map-cell-budget',type=int,default=12000000,
        help='Measured fine-map cell capacity only; does not change 2 mm voxels or safety margins; max 24M')
    p.add_argument('--sensor-fault',choices=['blackout','rgb_blackout','delay']);a=p.parse_args()
    with threadpool_limits(1):
        task=RGBDMission(a.output,a.scene,a.initial_scan,a.scene_token,a.calibration_id,assume_unchanged_scene=a.assume_unchanged_scene,
            mode=a.mode,asynchronous=True,contact_permission=True,capture=True,local_contact_map=True,static_interval_fusion=True,
            maximum_cycles=a.maximum_cycles,local_map_cell_budget=a.local_map_cell_budget,
            structured_ik=a.structured_ik,
            adaptive_task_staging=a.adaptive_task_staging,
            owned_source_departure=a.owned_source_departure,
            task_inequality_ik=a.task_inequality_ik,
            history_reacquisition=a.history_reacquisition,precontact_refinement=a.precontact_refinement,
            thin_support_search=a.thin_support_search,
            geometric_transport_recovery=a.geometric_transport_recovery,far_first_placement=a.far_first_placement,
            partitioned_geometry_proof=a.partitioned_geometry_proof,
            partial_marking=a.partial_marking,
            released_interface_guidance=a.released_interface_guidance,
            measured_placement_pose=a.measured_placement_pose,
            placement_evidence_refinement=a.placement_evidence_refinement,
            adaptive_visual_proof=a.adaptive_visual_proof,
            quality_aware_handover=a.quality_aware_handover,
            avoid_rejected_proposals=a.avoid_rejected_proposals,
            sensor_fault=None if a.sensor_fault is None else dict(mode=a.sensor_fault))
        task.run(48,.007,'auto')


if __name__=='__main__':main()
