"""Direct 16-joint PPO actions before IK; retain measured safety and task latch.

This deliberately reuses the verified RGB-D safety/execution backend first.
It is a sequential training environment, not an IsaacLab vector environment.
"""
import argparse
import json
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits
from astronaut_delivery.smooth_path import SmoothPath
from classical_fusion.rgbd_guard import RGBDHullEnvironmentGuard
from .online_mission import VisualPolicyMission
from .random_tasks import PALETTE
from .training_support import atomic_json,require_disk_headroom


class DirectPolicyMission(VisualPolicyMission):
    def __init__(self,*args,episode_seed,reset_jitter=.12,max_policy_actions=2048,**kwargs):
        if not 0 <= reset_jitter <= .5:raise ValueError('bounded reset jitter required')
        if not 1<=max_policy_actions<=2048:raise ValueError('bounded policy action budget required')
        self.max_policy_actions=max_policy_actions
        self.interface_palette=PALETTE
        self.policy_backend='direct_16_joint_measured_rgbd_sequential'
        self.policy_limitations=['classical task phases, latch, and missing-view recovery',
            'coarse temporal observation histogram; safety uses separate measured map',
            'sequential backend, not parallel IsaacLab',
            'observed completions require independent full-scene audit']
        super().__init__(*args,**kwargs)
        self.inherited_updates=len(self.visual_policy.updates)
        self.seen_voxels=set();self.novel_cells=0;self.temporal_features=np.zeros(64,np.float32)
        rng=np.random.default_rng(episode_seed);base=self.q.copy();attempts=[]
        snapshot=self.mapper.planning_snapshot();env=RGBDHullEnvironmentGuard(*self.arms,snapshot)
        accepted=None
        for _ in range(64):
            candidate=np.clip(base+rng.uniform(-reset_jitter,reset_jitter,(2,8)),-np.pi+1e-7,np.pi-1e-7)
            robot_ok=self.self_guard.box(candidate.ravel(),candidate.ravel())[0]
            cert=env.segment(candidate.ravel(),candidate.ravel()) if robot_ok else None
            attempts.append(dict(robot_ok=bool(robot_ok),environment_ok=bool(cert and cert.allowed)))
            if robot_ok and cert.allowed:accepted=candidate;break
        reset=dict(seed=int(episode_seed),jitter_rad=reset_jitter,attempts=attempts,
            accepted=accepted is not None,scope='Simulation reset, NOT an executed startup trajectory; no visibility/success screening')
        (self.output/'random_reset.json').write_text(json.dumps(reset,indent=2),encoding='utf-8')
        if accepted is None:
            if self.sensor_service is not None:self.sensor_service.close()
            raise RuntimeError('random_initial_joint_admission_exhausted')
        self.q=accepted;self.start_joints=accepted.copy();self.robot.set_positions(*accepted)

    def visual_observation(self,target,standoff):
        obs=super().visual_observation(target,standoff)
        # Keep temporal observation features. This is NOT a free-space map.
        self.temporal_features=np.maximum(self.temporal_features,obs['map'])
        obs['map']=self.temporal_features.copy()
        p=self.latest_policy_packet;d=p['depth'];yy,xx=np.indices(d.shape)
        valid=np.isfinite(d)&(d>.30)&(d<2.)&~p['self_mask']
        if p.get('color_valid') is not None:valid &= p['color_valid']
        yy,xx=yy[valid][::8],xx[valid][::8];z=d[yy,xx];K=p['K'];T=p['camera_world']
        points=np.c_[(xx-K[0,2])*z/K[0,0],-(yy-K[1,2])*z/K[1,1],-z]@T[:3,:3].T+T[:3,3]
        points=points[((points>[.15,-.75,.65])&(points<[.85,-.01,1.15])).all(1)]
        cells={tuple(v) for v in np.floor(points/.012).astype(int)}
        self.novel_cells=len(cells-self.seen_voxels);self.seen_voxels.update(cells)
        return obs

    def finish_policy_pending(self,next_observation=None,terminal=False,failed=False):
        if self.policy_pending is None:return
        pending=self.policy_pending
        recent=self.steps[pending['steps']:]
        distance=np.linalg.norm(pending['goal']-self.right.tcp(self.q[1])[:3,3])
        progress=(pending['distance']-distance)/.02
        axis=self.right.tcp(self.q[1])[:3,2]
        alignment=float(np.dot(axis,pending['axis']))
        completed=len(self.memory.completed)-pending['completed']
        rejected=not pending.get('certified',False)
        information=0. if next_observation is None else min(self.novel_cells/500.,1.)
        reward=float(np.clip(progress,-2,2)+.2*(alignment-pending['alignment'])+
            .05*information+10*completed-.3*rejected-.01-(2 if failed else 0))
        self.visual_policy.finish(pending['sample'],reward,next_observation,terminal)
        self.policy_events.append(dict(time_s=self.time,reward=reward,progress=float(progress),
            information_gain=information,observed_object_completions=completed,
            candidate_certified=not rejected,executed_ticks=len(recent),terminal=terminal,
            action=pending['action'].tolist(),reaction_reward=0.,full_scene_success_reward_supplied=False))
        self.policy_pending=None
        atomic_json(self.output/'learning_progress.json',dict(
            samples=len(self.policy_events),new_updates=len(self.visual_policy.updates)-self.inherited_updates,
            total_updates=len(self.visual_policy.updates),last=self.policy_events[-1]))
        if not self.visual_policy.buffer and self.visual_policy.training:
            self.visual_policy.save(self.output/'visual_policy_latest.pt')

    def plan_step(self,target,standoff,step_m=.02):
        if len(self.policy_events)+int(self.policy_pending is not None)>=getattr(self,'max_policy_actions',2048):
            raise RuntimeError('policy_action_budget_exhausted')
        # Called at rest, before proposing another movement. Reserve enough
        # space for pending evidence flush and final audit artifacts.
        if hasattr(self,'output'):require_disk_headroom(self.output,24.)
        env,snapshot=self.make_environment(target,standoff)
        q0=self.q.ravel().copy();goal=target.position+np.array([0,0,standoff])
        tcp=self.right.tcp(self.q[1]);distance=float(np.linalg.norm(goal-tcp[:3,3]))
        pose_reached=getattr(self,'right_pose_reached',lambda q:True)
        if distance<(.001 if standoff<.035 else .004) and pose_reached(self.q[1]):
            certificate=env.segment(q0,q0)
            if not certificate.allowed or not self.self_guard.box(q0,q0,**self.robot_guard_arguments())[0]:
                return None,'arrival_not_certified'
            self.rest_certificate=dict(certificate=asdict(certificate),q=self.q.copy())
            return None,'arrived'
        try:
            obs=self.visual_observation(target,standoff)
            context=self.allocation_context(target)
        except ValueError as exc:return None,'visual_set/'+str(exc)
        self.finish_policy_pending(obs)
        action,sample=self.visual_policy.sample(obs)
        axis=self.right_task_axis()
        if axis is None:axis=tcp[:3,2].copy()
        self.policy_pending=dict(sample=sample,action=action,goal=goal,distance=distance,
            axis=axis,alignment=float(np.dot(tcp[:3,2],axis)),
            completed=len(self.memory.completed),steps=len(self.steps),certified=False)
        endpoint=np.clip(q0+.02*action,-np.pi+1e-7,np.pi-1e-7)
        # No task-progress admission: RL must learn which safe actions help.
        certificate=env.segment(q0,endpoint)
        visual=self.certify_visual_path(q0[:8],endpoint[:8],context)
        reserve=self.allocator.endpoint_valid(q0[:8],endpoint[:8],context)
        robot_ok=self.self_guard.segment(q0,endpoint,**self.robot_guard_arguments()) if certificate.allowed and visual.allowed else False
        accepted=bool(certificate.allowed and visual.allowed and reserve and robot_ok)
        self.policy_pending['certified']=accepted
        self.decisions.append(dict(time_s=self.time,proposal_mode='direct_visual_ppo',
            environment=asdict(certificate),visual=visual.dictionary(),robot_geometry=bool(robot_ok),
            visual_endpoint_reserve_passed=bool(reserve),accepted=accepted,rejection=env.last_rejection))
        if not accepted:
            # Rejected action is not integrated. A certified rest consumes an
            # actual control interval; next call produces a fresh observation.
            endpoint=q0.copy();certificate=env.segment(q0,q0)
            visual=self.certify_visual_path(q0[:8],q0[:8],context)
            if not certificate.allowed or not visual.allowed or not self.self_guard.box(q0,q0,**self.robot_guard_arguments())[0]:
                return None,'unsafe_rest_after_policy_rejection'
        motion=SmoothPath.make(q0,endpoint,minimum=.4)
        row=dict(start=q0.tolist(),end=endpoint.tolist(),duration_s=motion.duration,time_s=self.time,
            target=target.position.tolist(),standoff_m=standoff,map_revision=snapshot.revision,
            environment_certificate=asdict(certificate),visual_certificate=visual.dictionary(),
            robot_certificate=True,visual_target_name=self.active,visual_target_position=context['target'].tolist(),
            policy_rejected=not accepted,scope='Direct policy proposal; unchanged RGB-D/robot continuous certificates')
        # Preserve carried-geometry audit metadata normally added by RGBDTransfer.plan_step.
        if self.held is not None:
            row.update(payload_name=self.held.name,payload_local_points=self.held.local_points.tolist(),
                payload_contact_phase=self.transfer_phase)
        self.paths.append(row)
        return motion,'accepted'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','scene','initial-scan','scene-token','calibration-id','checkpoint'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--episode-seed',required=True,type=int);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--reset-jitter',type=float,default=.12);p.add_argument('--evaluate',action='store_true')
    a=p.parse_args()
    with threadpool_limits(1):
        task=DirectPolicyMission(a.output,a.scene,a.initial_scan,a.scene_token,a.calibration_id,
            checkpoint=a.checkpoint,training=not a.evaluate,episode_seed=a.episode_seed,reset_jitter=a.reset_jitter,
            max_policy_actions=a.steps,
            assume_unchanged_scene=True,asynchronous=True,contact_permission=True,capture=True,local_contact_map=True,
            static_interval_fusion=True,local_map_cell_budget=16000000,owned_source_departure=True,
            task_inequality_ik=True,history_reacquisition=True,thin_support_search=True,
            partial_marking=True,released_interface_guidance=True,measured_placement_pose=True,
            placement_evidence_refinement=True)
        task.run(a.steps,.007,'auto')
