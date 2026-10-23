"""Deterministic RGB-D guided, collision-checked sequential transfer demo.

Pure kinematics. USD-mesh ray sensor, no neural weights or privileged target
positions in perception. Known fixture/part CAD is an independent safety model.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from PIL import Image
from pxr import Gf,Usd,UsdGeom
from .scene import ROOT,SCENE
from .motion import Arm,CollisionGuard
from .vision import MeshCamera,VisualGuide
from .magnet import MagneticLatch
sys.path.insert(0,str(ROOT/'scripts'))
from astronaut_kinematics import DualArmKinematics

class Delivery:
    def __init__(self,output,device='cuda:0',continuous=True,sensor='d435i',scene_override=None):
        self.cfg=json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
        self.output=Path(output)
        if self.output.exists() and any(self.output.iterdir()):raise ValueError('Use a new empty output directory; existing run artifacts are preserved')
        self.output.mkdir(parents=True,exist_ok=True)
        self.sensor_name=sensor;self.scene_path=SCENE;self.sensor_profile=None
        if sensor=='d435i':
            from .d435i import D435iCamera,load_profile
            from .d435i_scene import D435_SCENE
            self.scene_path=D435_SCENE;self.sensor_profile=load_profile()
        if scene_override is not None:self.scene_path=Path(scene_override)
        self.stage=Usd.Stage.Open(str(self.scene_path));self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.left=Arm(self.stage,'left');self.right=Arm(self.stage,'right')
        self.guard=CollisionGuard(self.stage,self.left,self.right,self.cfg['collision_margin_m'])
        self.robot=DualArmKinematics(self.stage)
        self.latch=MagneticLatch(self.stage,self.guard,self.cfg['magnet_capture_distance_m'])
        self.camera=D435iCamera(device,self.sensor_profile) if sensor=='d435i' else MeshCamera(device=device)
        self.guide=VisualGuide({p['name']:p['color'] for p in self.cfg['parts']},self.cfg['observation_max_age_s'],self.sensor_profile)
        self.q=np.zeros((2,8));self.time=0.;self.tick=0;self.frames=[];self.events=[]
        self.continuous=continuous;self.records=[];self.completed=[];self.estop=False
        self.last_obs=None;self.rng=np.random.default_rng(self.cfg['seed'])
        self.camera_ticks=round(1/(self.cfg['camera_rate_hz']*self.cfg['control_dt_s']))
        self.used_views=[];self.view_quality=[]
        assert self.camera_ticks>=1
        assert self.cfg['joint_increment_rad']<=self.cfg['joint_speed_rad_s']*self.cfg['control_dt_s']+1e-12
        self.record('reset')

    def event(self,event,**info):
        row=dict(event=event,time_s=self.time,**info);self.events.append(row)
        print(json.dumps(row),flush=True)
        (self.output/'events.json').write_text(json.dumps(self.events,indent=2),encoding='utf-8')

    def observe(self):
        self.last_obs=self.camera.capture(self.stage,self.left.camera(self.q[0]),self.time,self.cfg['depth_noise_std_m'],self.cfg['seed']+self.tick)
        self.guide.update(self.last_obs,integrate_map=self.latch.payload is None)
        if self.guide.frames%4==1:self.frames.append((self.last_obs.rgb*255).astype(np.uint8))

    def survey(self,phase='initial_view',name=None):
        """Bounded active perception, only while the execution arm is paused.

        Bootstrap with the task workspace centre; subsequent candidates use
        image-derived targets. All motions retain continuous collision guards.
        """
        cfg=self.sensor_profile
        if self.latch.payload is not None:raise RuntimeError('cannot_reposition_camera_while_holding')
        points=[v.position for key,v in self.guide.estimates.items() if name is None or key==name]
        if not points:points=[cfg['workspace_look_at_m']]
        center=np.mean(points,axis=0);candidates=[]
        for index,position in enumerate(cfg['view_candidates_m']):
            if index in self.used_views:continue
            position=np.asarray(position);axis=center-position;axis/=np.linalg.norm(axis)
            q=self.left.solve(position,axis,seed=self.q[0],tries=20,rng=self.rng,
                 valid=lambda q:self.guard.check('left',q,self.q[1])[0] and self.guard.check('right',self.q[1],q)[0])
            if q is None:continue
            score=self.camera.view_score(self.left.camera(q),points)-.015*np.linalg.norm(q-self.q[0])
            if np.isfinite(score):candidates.append((score,index,q))
        for score,index,q in sorted(candidates,key=lambda row:row[0],reverse=True):
            self.used_views.append(index)
            self.move('left',q,phase+f'_{index}')
            # Wait three processing periods after the camera stops. No integration
            # of moving-camera frames and no fabricated fresh timestamps.
            for _ in range(3):
                self.time+=self.camera_ticks*self.cfg['control_dt_s'];self.tick+=self.camera_ticks
                self.observe()
            obs=self.last_obs;v,u=np.indices(obs.depth.shape)
            roi=(u>obs.depth.shape[1]*.2)&(u<obs.depth.shape[1]*.8)&(v>obs.depth.shape[0]*.2)&(v<obs.depth.shape[0]*.8)
            valid=(obs.depth>0)&~obs.robot_mask
            ratio=float(valid[roi].mean())
            row=dict(view=index,roi_valid_ratio=ratio,score=score,targets=list(self.guide.estimates),**self.camera.last_quality)
            self.view_quality.append(row);self.event('d435i_view_quality',**row)
            self.record(phase+'/settled')
            if ratio>=cfg['minimum_roi_valid_ratio'] and (name is None or self.guide.target(name,self.time) is not None):return
        raise RuntimeError('no_informative_safe_d435i_view/'+str(name))

    def acquire_target(self,name):
        self.observe();estimate=self.guide.target(name,self.time)
        if estimate is None and self.sensor_profile is not None:
            self.survey('reacquire_'+name,name);estimate=self.guide.target(name,self.time)
        if estimate is None:raise RuntimeError('no_fresh_visual_target/'+name)
        return estimate

    def record(self,phase):
        parts={p.GetName():np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).tolist() for p in self.stage.GetPrimAtPath('/World/Parts').GetChildren()}
        self.records.append(dict(time_s=self.time,q=self.q.tolist(),parts=parts,phase=phase,held=self.latch.payload.name if self.latch.payload else None))

    def move(self,side,goal,phase,contact_part=None,support_contact=False):
        if goal is None:raise RuntimeError(phase+': no collision-free IK solution')
        if side=='left' and self.latch.payload is not None:raise RuntimeError('left_survey_requires_empty_right_magnet')
        index=0 if side=='left' else 1;other=self.q[1-index]
        opts=dict(contact_part=contact_part,held=self.latch.payload if index==1 else None,support_contact=support_contact)
        path=self.guard.plan(side,self.q[index],goal,other,seed=self.cfg['seed']+len(self.events),**opts)
        if path is None:raise RuntimeError(phase+': no safe RRT path')
        self.event('planned',phase=phase,knots=len(path))
        if self.continuous:
            certified=[path[0]]
            for segment,(a,b) in enumerate(zip(path[:-1],path[1:])):
                if self.guard.edge(side,a,b,other,continuous=True,**opts):
                    certified.append(b)
                else:
                    self.event('continuous_replan',phase=phase,segment=segment)
                    repaired=None
                    for retry in range(3):
                        candidate=self.guard.plan(side,a,b,other,seed=self.cfg['seed']+1000+segment*10+retry,resolution=.01/(retry+1),**opts)
                        if candidate is not None and all(self.guard.edge(side,x,y,other,continuous=True,**opts) for x,y in zip(candidate[:-1],candidate[1:])):
                            repaired=candidate;break
                    if repaired is None:raise RuntimeError(phase+': no continuously certified repair')
                    certified.extend(repaired[1:])
                self.event('segment_certified',phase=phase,segment=segment)
            path=np.array(certified)
        # Freeze the other arm during planning/execution. Any future concurrent
        # left motion must invalidate/recheck the right-arm path before advancing.
        for a,b in zip(path[:-1],path[1:]):
            # Raised-cosine time scaling gives zero speed at each segment end;
            # pi/2 bounds its maximum slope, maintaining the configured speed cap.
            steps=max(1,int(np.ceil(np.pi/2*np.max(abs(b-a))/self.cfg['joint_increment_rad'])))
            fractions=.5-.5*np.cos(np.pi*np.linspace(0,1,steps+1)[1:])
            for fraction in fractions:
                q=a+(b-a)*fraction
                if self.estop:raise RuntimeError('emergency_stop_hold')
                ok,reason=self.guard.check(side,q,other,**opts)
                if not ok:raise RuntimeError('runtime_collision_hold/'+reason)
                if side=='left':
                    ok,reason=self.guard.check('right',self.q[1],q)
                    if not ok:raise RuntimeError('right_supervisor_hold/'+reason)
                self.q[index]=q;self.robot.set_positions(*self.q)
                self.latch.update(self.right.tcp(self.q[1]))
                self.time+=self.cfg['control_dt_s'];self.tick+=1
                if side=='right' and self.tick%self.camera_ticks==0:self.observe()
                if self.tick%8==0:self.record(phase)
        self.record(phase);self.event('arrived',phase=phase)

    def right_ik(self,position,axis,contact=None,support=False,offset=None):
        opts=dict(contact_part=contact,held=self.latch.payload,support_contact=support)
        valid=lambda q:self.guard.check('right',q,self.q[0],**opts)[0]
        if self.latch.payload is not None:
            # Prefer an IK branch directly connected to the current carry pose;
            # a merely valid endpoint can require a needlessly folded detour.
            direct=self.right.solve(position,np.array(axis),seed=self.q[1],tries=12,rng=self.rng,offset=offset,
                valid=lambda q:valid(q) and self.guard.edge('right',self.q[1],q,self.q[0],resolution=.01,**opts))
            if direct is not None:return direct
        return self.right.solve(position,np.array(axis),seed=self.q[1],tries=35,rng=self.rng,offset=offset,valid=valid)

    def servo_contact(self,name):
        """Reobserve and replan bounded Cartesian increments; stop on stale RGB-D."""
        for step in range(10):
            # Independent simulated proximity/contact signal can terminate the
            # visual approach before the cap disappears behind the magnetic tip.
            ok,gap=self.latch.attach(name,self.right.tcp(self.q[1]))
            if ok:
                self.event('attached',part=name,capture_gap_m=gap)
                return
            self.observe()
            estimate=self.guide.target(name,self.time)
            if estimate is None:raise RuntimeError('visual_servo_stale_hold/'+name)
            tcp=self.right.tcp(self.q[1])[:3,3]
            correction=self.guide.correction(name,tcp,self.time,max_step=.025)
            residual=float(np.linalg.norm(estimate.position-tcp))
            if residual<.0035:raise RuntimeError('visual_contact_without_magnetic_confirmation/'+name)
            self.event('visual_servo',part=name,iteration=step,residual_m=residual,observation_age_s=self.time-estimate.stamp)
            self.move('right',self.right_ik(tcp+correction,[0,0,-1],contact=name),name+f'/servo_{step}',contact_part=name)
        raise RuntimeError('visual_servo_iteration_limit/'+name)

    def run(self,count=3):
        if self.sensor_profile is not None:
            self.survey()
            self.survey('mapping_second_view')
        else:
            target=np.array([.60,.45,1.20]);direction=np.array([.55,-.2,.815])-target;direction/=np.linalg.norm(direction)
            view=self.left.solve(target,direction,seed=self.q[0],tries=20,valid=lambda q:self.guard.check('left',q,self.q[1])[0])
            self.move('left',view,'initial_view');self.observe()
        self.event('visual_targets',targets={k:v.position.tolist() for k,v in self.guide.estimates.items()})
        for spec in self.cfg['parts'][:count]:
            name=spec['name'];estimate=self.acquire_target(name)
            self.move('right',self.right_ik(estimate.position+[0,0,.11],[0,0,-1]),name+'/approach')
            self.observe();estimate=self.guide.target(name,self.time)
            if estimate is None:raise RuntimeError('visual_target_occluded/'+name)
            self.event('visual_guidance_update',part=name,position=estimate.position.tolist(),pixels=estimate.pixels)
            # Guarded final approach uses the measured cap; proximity latch below
            # independently confirms true simulated surface distance (not a snap).
            self.servo_contact(name)
            offset=self.latch.payload.local_pose
            pos=(self.right.tcp(self.q[1])@offset)[:3,3]
            self.move('right',self.right_ik(pos+[0,0,.08],[0,0,1],support=True,offset=offset),name+'/lift',support_contact=True)
            destination=np.array(spec['destination'])
            self.move('right',self.right_ik(destination+[0,0,.08],[0,0,1],offset=offset),name+'/transfer')
            self.move('right',self.right_ik(destination,[0,0,1],support=True,offset=offset),name+'/place',support_contact=True)
            ok,reason=self.latch.release(self.right.tcp(self.q[1]),destination)
            if not ok:raise RuntimeError('release_failed/'+reason)
            self.completed.append(name);self.record('released');self.event('released',part=name)
            self.move('right',self.right_ik(self.right.tcp(self.q[1])[:3,3]+[0,0,.12],[0,0,-1],contact=name),name+'/retract',contact_part=name)
            self.guide.invalidate_after_transfer();self.observe()
        self.event('complete',parts=self.completed)

    def save(self,error=None):
        (self.output/'trajectory.json').write_text(json.dumps(self.records),encoding='utf-8')
        surface,_=self.guide.mapper.surface();points=surface.cpu().numpy()
        np.savez_compressed(self.output/'measured_map.npz',points=points)
        if self.frames:
            images=[Image.fromarray(x) for x in self.frames]
            images[0].save(self.output/'left_camera.gif',save_all=True,append_images=images[1:],duration=80,loop=0)
        # Flatten snapshot has all final part poses, but no animated/GT input to perception.
        self.stage.Flatten().Export(str(self.output/'final_state.usdc'))
        result=dict(status='PASS' if error is None else 'STOPPED',error=error,completed=self.completed,seed=self.cfg['seed'],simulation_time_s=self.time,camera_frames=self.guide.frames,collision_checks=self.guard.checks,continuous_audit=self.continuous,map_points=len(points),sensor=self.sensor_name+' / USD mesh ray-cast RGB-D; not RTX or a physical stereo matcher',scene_path=str(self.scene_path),map_updates=self.guide.map_updates,view_quality=self.view_quality,sensor_profile=self.sensor_profile,safety_model='Known USD CAD hulls + conservative full-link covers including camera; not real hardware certification')
        (self.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--parts',type=int,default=3);p.add_argument('--device',default='cuda:0');p.add_argument('--output',default=str(ROOT/'reports'/'astronaut_delivery_d435i_run'));p.add_argument('--sampled-debug',action='store_true');p.add_argument('--sensor',choices=['d435i','legacy'],default='d435i')
    args=p.parse_args()
    with threadpool_limits(1):
        run=Delivery(args.output,args.device,not args.sampled_debug,args.sensor)
        try:run.run(args.parts)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()

if __name__=='__main__':main()
