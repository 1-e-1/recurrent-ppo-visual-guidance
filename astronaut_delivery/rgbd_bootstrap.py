"""Strict RGB-D commissioning gate, with NO CAD safety fallback.

The simulator sees its scene solely to render RGB-D. The environment guard
receives measured depth/K/camera pose and known robot FK, never scene objects.
No arm is moved before this gate. This is not yet a complete delivery runner.
"""
import argparse,json,time,hashlib
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from pxr import Usd,Gf
from threadpoolctl import threadpool_limits
from .online_scene import ONLINE_SCENE
from .run import DualArmKinematics
from .simultaneous import TightBoundArm,SimultaneousGuard
from .d435i import D435iCamera
from .temporal_evidence_map import TemporalEvidenceTSDF
from .measured_surface import export_surface
from .robot_only_geometry import RobotOnlyGuard
from classical_fusion.planning_map import PlanningMapConfig
from classical_fusion.rgbd_guard import RGBDHullEnvironmentGuard


def run(output,frames=8,device='cuda:0',scene=None,initial_pose='workspace_view',
        initial_scan=None,scene_token=None,calibration_id=None,assume_unchanged_scene=False,voxel_m=.008,initial_right='zero',return_runtime=False):
    folder=Path(output)
    if frames<1:raise ValueError('need real observations')
    if voxel_m not in (.008,.012):raise ValueError('validated broad-workspace voxel presets are 8 or 12 mm')
    folder.mkdir(parents=True,exist_ok=False)
    # Simulator-only stage. It is never passed to RGBDEnvironmentGuard.
    stage=Usd.Stage.Open(str(scene or ONLINE_SCENE));stage.SetEditTarget(stage.GetSessionLayer())
    robot=DualArmKinematics(stage);arms=[TightBoundArm(stage,s) for s in ('left','right')]
    q=np.zeros((2,8));camera=D435iCamera(device);robot_guard=RobotOnlyGuard(stage,*arms)
    if initial_pose=='workspace_view':
        # Define a useful experiment INITIAL CONDITION using robot FK only,
        # not a CAD-checked startup motion or a hidden object pose. This reset
        # is explicitly not presented as a certified trajectory from q=0.
        position=np.array(camera.profile['view_candidates_m'][1],float)
        direction=np.array(camera.profile['workspace_look_at_m'])-position
        direction/=np.linalg.norm(direction)
        initial=arms[0].solve(position,direction,seed=q[0],tries=6,rng=np.random.default_rng(982),
            valid=lambda x:robot_guard.check('left',x,q[1])[0] and robot_guard.check('right',q[1],x)[0])
        if initial is None:raise RuntimeError('FK-only initial view IK unavailable')
        q[0]=initial
    elif initial_pose!='zero':raise ValueError('unknown initial condition')
    robot.set_positions(*q)
    # Broad robot/workcell volume, not a CAD-derived obstacle bounding box.
    cfg=PlanningMapConfig(workspace_min=(-.5,-.9,.3),workspace_max=(1.3,.9,1.9),
        voxel_size_m=voxel_m,truncation_m=3*voxel_m,minimum_free_views=2,surface_min_weight=1.,snapshot_max_age_s=30.)
    mapper=TemporalEvidenceTSDF(cfg,device=device)
    mapper.patch_radii=(4,6,8,10,12,16,24,32)
    initial_evidence=None
    if initial_scan:
        from classical_fusion.rgbd_recording import replay_initial_scan
        initial_evidence=replay_initial_scan(mapper,initial_scan,expected_scene_token=scene_token,
            expected_calibration_id=calibration_id,assume_unchanged_scene=assume_unchanged_scene)
    exposures=[];camera_world=arms[0].camera(q[0])
    R=camera_world[:3,:3]@np.array([[0,-1,0],[0,0,1],[-1,0,0.]])
    quat=Gf.Matrix3d(R.T.tolist()).ExtractRotation().GetQuat()
    pose=np.r_[camera_world[:3,3],float(quat.GetReal()),np.array(quat.GetImaginary())]
    for i in range(frames):
        obs=camera.capture(stage,camera_world,i/30,seed=9000+i)
        # FK-only self masking; removed samples are unknown, never free votes.
        depth=robot_guard.mask_self_returns(obs.depth,obs.K,camera_world,q)
        mapper.integrate(depth,obs.K,pose)
        path=folder/f'rgbd_{i:03d}.npz'
        np.savez_compressed(path,rgb=obs.rgb,depth=obs.depth,K=obs.K,camera_world=obs.camera_world,
                            exposure_s=obs.stamp,q=q)
        exposures.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             stamp=obs.stamp,valid_depth_fraction=float(np.mean(obs.depth>0))))
    if initial_right=='ready':
        if not initial_scan:raise ValueError('Measured ready-pose selection requires raw initial scan')
        snapshot=mapper.planning_snapshot();initial_guard=RGBDHullEnvironmentGuard(*arms,snapshot)
        def ready_valid(value):
            candidate=q.copy();candidate[1]=value
            return (robot_guard.check('right',value,q[0])[0] and robot_guard.check('left',q[0],value)[0]
                    and initial_guard.segment(candidate.ravel(),candidate.ravel()).allowed)
        ready=None;rng=np.random.default_rng(9312)
        for position in ([.45,-.5,1.10],[.55,-.45,1.05],[.5,-.25,1.1],[.65,-.4,1.1],[.35,-.45,1.15]):
            ready=arms[1].solve(position,[0,0,-1.],seed=q[1],tries=24,rng=rng,valid=ready_valid)
            if ready is not None:break
        if ready is None:raise RuntimeError('No measured-safe ready initial condition within IK budget')
        q[1]=ready;robot.set_positions(*q)
    elif initial_right!='zero':raise ValueError('unknown right initial condition')
    calibration_digest=hashlib.sha256(json.dumps(camera.profile,sort_keys=True).encode()).hexdigest()
    recording=dict(format='rgbd-initial-map-v1',scene_token=scene_token or Path(scene or ONLINE_SCENE).stem,
        calibration_id=calibration_digest,records=[dict(file=e['file'],sha256=e['sha256']) for e in exposures],
        note='Raw stationary acquisition, not sufficient initial coverage. Reuse requires independently justified unchanged scene and matching calibration.')
    (folder/'recording.json').write_text(json.dumps(recording,indent=2),encoding='utf-8')
    snapshot=mapper.planning_snapshot();guard=RGBDHullEnvironmentGuard(*arms,snapshot);self_guard=SimultaneousGuard(robot_guard)
    # A bounded set of small joint commands, not a precomputed safe CAD path.
    certificates=[]
    for side in range(2):
        for joint in (0,3,7):
            for sign in (-1,1):
                target=q.copy();target[side,joint]+=sign*.01
                cert=guard.segment(q.ravel(),target.ravel())
                data=asdict(cert)
                if not np.isfinite(data['minimum_slack_m']):data['minimum_slack_m']=None
                self_ok=self_guard.segment(q.ravel(),target.ravel()) if cert.allowed else False
                certificates.append(dict(side=side,joint=joint,increment_rad=sign*.01,
                    robot_geometry_certified=self_ok,combined_allowed=cert.allowed and self_ok,
                    unknown_voxel_tests=guard.cell_tests,rejection_diagnostic=guard.last_rejection,**data))
    # Separate one-view diagnostic: two independent views are not the ONLY
    # obstacle. This never changes the actual planning acceptance threshold.
    original=mapper.cfg;mapper.cfg=replace(original,minimum_free_views=1)
    diagnostic=mapper.planning_snapshot();mapper.cfg=original
    optical=[]
    for side,arm in enumerate(arms):
        points=arm.sphere_positions(q[side]);local=(points-camera_world[:3,3])@camera_world[:3,:3]
        z=-local[:,2];clear,valid=diagnostic.clearance(points)
        optical.append(dict(side=side,cover_spheres=len(points),
            centers_behind_camera=int(np.sum(z<=0)),
            centers_below_minimum_depth=int(np.sum((z>0)&(z<camera.profile['valid_depth_range_m'][0]))),
            one_view_sphere_centers_admissible=int(np.sum(valid&(clear>arm.radii)))))
    export_surface(mapper,folder)
    passed=sum(c['allowed'] for c in certificates)
    report=dict(status='BLOCKED_INITIAL_RGBD_COVERAGE' if passed==0 else 'ENVIRONMENT_GATE_ONLY_NOT_DELIVERY',
        completed=[],executed_motion=False,environment_authority='RGBD_ONLY',environment_CAD_fallback=False,
        uses_renderer_instance_masks=False,uses_known_part_geometry=False,
        acquisition='One stationary left-arm D435i initial condition; no multi-view teleportation or inherited CAD-controlled trajectory',
        initial_pose=initial_pose,initial_right=initial_right,initial_joints=q.tolist(),
        initial_recording=initial_evidence,
        initialization_scope='Reset defines initial simulation state, not a certified startup motion; initial robot/body collision feasibility is not claimed by this environment-only diagnostic.',
        frames=frames,exposures=exposures,independent_evidence_views=len(mapper.evidence_poses),
        free_fraction=snapshot.free_fraction,candidate_count=len(certificates),environment_certified_candidates=passed,
        combined_certified_candidates=sum(c['combined_allowed'] for c in certificates),
        candidates=certificates,initial_observability=optical,
        map_config=asdict(cfg),
        remaining='Full-arm measured initial clearance, measured payload/contact envelopes, online planning integration and complete delivery remain required. Robot-only self/inter-arm checks are implemented; the astronaut is checked as measured environment. No motion is executed by this commissioning diagnostic.',
        reason='A loaded initial recording is not automatically sufficient coverage. Every robot link and its swept margin must be supported by current free-space evidence; missing returns and unknown cells never become free by assumption.')
    project=Path(__file__).resolve().parents[1]
    report['source_sha256']={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in (
        'classical_fusion/rgbd_guard.py','classical_fusion/rgbd_recording.py',
        'astronaut_delivery/rgbd_bootstrap.py','astronaut_delivery/research_delivery.py','astronaut_delivery/robot_only_geometry.py')}
    (folder/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('exposures','candidates','map_config')},indent=2),flush=True)
    if return_runtime:
        return report,dict(stage=stage,robot=robot,arms=arms,mapper=mapper,q=q,camera=camera,robot_guard=robot_guard)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=8)
    p.add_argument('--device',default='cuda:0');p.add_argument('--scene')
    p.add_argument('--initial-pose',choices=['zero','workspace_view'],default='workspace_view')
    p.add_argument('--initial-right',choices=['zero','ready'],default='zero')
    p.add_argument('--initial-scan');p.add_argument('--scene-token');p.add_argument('--calibration-id')
    p.add_argument('--assume-unchanged-scene',action='store_true');args=p.parse_args()
    with threadpool_limits(1):run(args.output,args.frames,args.device,args.scene,args.initial_pose,
        args.initial_scan,args.scene_token,args.calibration_id,args.assume_unchanged_scene,initial_right=args.initial_right)


if __name__=='__main__':main()
