"""First closed-loop integration: vision policy proposals plus classical guard.

Sequential existing RGB-D backend, NOT the final parallel IsaacLab environment.
Right-arm classical feasibility is still needed to reach the proposal hook.
Capture/release remain measured state-machine operations; no reaction objective.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from astronaut_delivery.rgbd_mission import RGBDMission
from .recurrent_ppo import RecurrentPPO
from .train_bootstrap import encode_frame,PHASES
from .task_progress import retains_progress,episode_failed


def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob('*.py')}


class VisualPolicyMission(RGBDMission):
    def __init__(self,*args,checkpoint,training=True,**kwargs):
        self.visual_policy=RecurrentPPO(checkpoint,training)
        self.latest_policy_packet=None;self.policy_pending=None;self.policy_events=[]
        self.policy_previous_q=None;self.policy_previous_stamp=None
        self.visual_sources=source_hashes()
        kwargs['mode']='hold'  # No wrench-minimizing left-arm allocator.
        super().__init__(*args,**kwargs)

    def accept_packet(self,packet,integrate=False):
        obs=super().accept_packet(packet,integrate)
        if obs is not None:self.latest_policy_packet=packet
        return obs

    def visual_observation(self,target,standoff):
        # Fresh sensor frame, no map commit and no relabelled historical image.
        self.accept_packet(self.sensor_service.acquire(self.q,self.time),integrate=False)
        packet=self.latest_policy_packet
        if packet is None or not 0<=self.time-packet['stamp']<=.25:raise ValueError('no fresh policy RGB-D')
        visual=self.guide.target(self.active,self.time)
        if visual is None:raise ValueError('no fresh policy target')
        raw=dict(depth=packet['depth'],rgb=packet['rgb'],K=packet['K'],
                 camera_world=packet['camera_world'],robot_mask=packet['self_mask'])
        image,features=encode_frame(raw);q=self.q.ravel().astype(np.float32)
        previous=np.zeros(16,np.float32) if self.policy_previous_q is None else q-self.policy_previous_q
        velocity=np.zeros(16,np.float32) if not self.steps else np.array(self.steps[-1]['v'],np.float32)
        task=np.zeros(24,np.float32);task[:3]=visual.position
        task[3:6]=target.position+[0,0,standoff]
        task[6:9]=np.sqrt(np.maximum(np.diag(visual.covariance),0))
        task[9:12]=[visual.pixels/1000.,self.time-visual.stamp,standoff]
        phase=getattr(self,'transfer_phase',None) or 'approach'
        if phase in PHASES:task[12+PHASES.index(phase)]=1
        task[18]=self.policy_previous_stamp is not None
        task[19]=0 if self.policy_previous_stamp is None else min(self.time-self.policy_previous_stamp,10)/10.
        self.policy_previous_q=q.copy();self.policy_previous_stamp=self.time
        return dict(image=image,q=q,velocity=velocity,previous=previous,map=features,task=task)

    def finish_policy_pending(self,next_observation=None,terminal=False,failed=False):
        if self.policy_pending is None:return
        p=self.policy_pending;recent=self.steps[p['steps']:]
        progress=(p['distance']-np.linalg.norm(p['goal']-self.right.tcp(self.q[1])[:3,3]))/.02
        decisions=self.decisions[p['decisions']:]
        certified=any(d.get('proposal_mode')=='visual_recurrent_ppo' and d.get('robot_geometry')
            and d.get('environment',{}).get('allowed') and d.get('visual',{}).get('allowed')
            and d.get('visual_endpoint_reserve_passed') for d in decisions)
        completed=len(self.memory.completed)-p['completed']
        duration=sum(x['dt_s'] for x in recent)
        reward=float(np.clip(progress,-2,2)+5*completed-.25*(not certified)-.01*duration-(2 if failed else 0))
        self.visual_policy.finish(p['sample'],reward,next_observation,terminal)
        self.policy_events.append(dict(time_s=self.time,reward=reward,progress=float(progress),
            observed_object_completions=completed,candidate_certified=bool(certified),
            executed_ticks=len(recent),terminal=terminal,action=p['action'].tolist(),
            reaction_reward=0.,full_scene_success_reward_supplied=False))
        self.policy_pending=None

    def residual_proposal(self,base,target,standoff,context):
        try:obs=self.visual_observation(target,standoff)
        except ValueError as exc:
            self.decisions.append(dict(status='VISUAL_POLICY_NO_OBSERVATION',reason=str(exc),time_s=self.time));return None
        self.finish_policy_pending(obs)
        action,sample=self.visual_policy.sample(obs)
        goal=target.position+[0,0,standoff]
        self.policy_pending=dict(sample=sample,action=action,goal=goal,
            distance=np.linalg.norm(goal-self.right.tcp(self.q[1])[:3,3]),completed=len(self.memory.completed),
            steps=len(self.steps),decisions=len(self.decisions))
        q=self.q.ravel();baseline=np.asarray(base[2]);end=baseline.copy()
        end[:8]=np.clip(q[:8]+.02*action[:8],-np.pi+1e-7,np.pi-1e-7)
        # Preserve the classical task waypoint. Learn viewpoint + redundant IK
        # correction rather than giving priority to safe but aimless microsteps.
        position=self.right.tcp(baseline[8:])[:3,3]
        options=self.right_ik_arguments();options['max_delta']=.02
        seed=np.clip(baseline[8:]+.02*action[8:],-np.pi+1e-7,np.pi-1e-7)
        fit=self.right.solve(position,self.right_task_axis(),seed=seed,tries=1,**options)
        if fit is None:return None
        end[8:]=fit
        if np.max(abs(end[:8]-q[:8]))>.02+1e-8 or np.max(abs(end[8:]-baseline[8:]))>.02+1e-8:return None
        if np.max(abs(end[8:]-q[8:]))>.30+1e-8:return None
        if not retains_progress(self.right.tcp(q[8:])[:3,3],position,
                                self.right.tcp(end[8:])[:3,3],goal):
            self.decisions.append(dict(status='VISUAL_POLICY_INSUFFICIENT_TASK_PROGRESS',time_s=self.time));return None
        if not getattr(self,'right_pose_reached',lambda q:True)(end[8:]):return None
        return 0.,1.,end,dict(mode='visual_recurrent_ppo',scope='Fresh RGB-D proposal, no motion permission'),'visual_recurrent_ppo'

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        self.finish_policy_pending(terminal=True,failed=episode_failed(result))
        self.visual_policy.save(self.output/'visual_policy_final.pt')
        unchanged=source_hashes()==self.visual_sources
        (self.output/'visual_training.json').write_text(json.dumps(dict(
            source_hashes=self.visual_sources,source_unchanged=unchanged,events=self.policy_events,
            updates=self.visual_policy.updates,mode='perception_first_no_reaction_objective',
            policy_backend=getattr(self,'policy_backend','sequential_measured_rgbd_not_parallel_isaaclab'),
            limitations=getattr(self,'policy_limitations',['classical IK prerequisite','classical latch and safety fallback',
                         'coarse current RGB-D spatial features, not full temporal map encoder',
                         'no information gain reward yet','observed completions are not full-scene audit'])),indent=2),encoding='utf-8')
        if not unchanged:raise RuntimeError('visual policy source changed during run')
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','scene','initial-scan','scene-token','calibration-id','checkpoint'):p.add_argument('--'+name,required=True)
    p.add_argument('--assume-unchanged-scene',action='store_true');p.add_argument('--evaluate',action='store_true')
    a=p.parse_args()
    with threadpool_limits(1):
        task=VisualPolicyMission(a.output,a.scene,a.initial_scan,a.scene_token,a.calibration_id,
            checkpoint=a.checkpoint,training=not a.evaluate,assume_unchanged_scene=a.assume_unchanged_scene,
            asynchronous=True,contact_permission=True,capture=True,local_contact_map=True,
            static_interval_fusion=True,local_map_cell_budget=16000000,adaptive_task_staging=True,
            owned_source_departure=True,task_inequality_ik=True,history_reacquisition=True,
            precontact_refinement=True,thin_support_search=True,geometric_transport_recovery=True,
            far_first_placement=True,partitioned_geometry_proof=True,partial_marking=True,
            released_interface_guidance=True,measured_placement_pose=True,placement_evidence_refinement=True)
        task.run(48,.007,'auto')
