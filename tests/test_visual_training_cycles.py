from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
from visual_guidance_lab.random_tasks import draw,FAMILIES,PALETTE
from visual_guidance_lab.train_cycles import plan,summarize


def test_seed_schedule_disjoint_and_failures_in_denominator():
    schedule=plan(3,4,2)
    train={x['seed'] for x in schedule if x['training']}
    validation={x['seed'] for x in schedule if not x['training']}
    assert len(train)==12 and len(validation)==6 and not train&validation
    rows=[dict(training=False,passed=True),dict(training=False,passed=False,error='reset failed')]
    assert summarize(rows)['validation']['success_rate']==.5


def test_draw_reproducible_varied_and_nonoverlapping():
    assert draw(100000)==draw(100000)
    kinds=set();counts=set();failed=[]
    for seed in range(100000,100100):
        try:scene=draw(seed)
        except RuntimeError:failed.append(seed);continue
        parts=scene['parts'];counts.add(len(parts));assert .70<=scene['table_top_z_m']<=.82
        for i,a in enumerate(parts):
            kinds.add(a['kind']);assert a['name'] in PALETTE
            R=np.array(a['rotation']);np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12)
            for b in parts[:i]:
                assert np.linalg.norm(np.array(a['position'][:2])-b['position'][:2])>a['placement_radius_m']+b['placement_radius_m']+.014
    assert kinds==set(FAMILIES) and counts==set(range(1,7))
    assert len(failed)<30  # Failed placements must be surfaced, not resampled by runner.


@dataclass
class Certificate:
    allowed:bool=True


@pytest.mark.parametrize('accepted',[True,False])
def test_direct_joint_actions_do_not_require_ik_or_accumulate_rejected(accepted):
    from visual_guidance_lab.direct_mission import DirectPolicyMission
    task=DirectPolicyMission.__new__(DirectPolicyMission)
    task.q=np.zeros((2,8));task.steps=[];task.decisions=[];task.paths=[];task.time=0.;task.active='item_0';task.held=None
    task.policy_events=[];task.policy_pending=None
    task.right=SimpleNamespace(tcp=lambda q:np.eye(4))
    task.right_task_axis=lambda:np.array([0,0,-1.])
    task.robot_guard_arguments=lambda:{}
    env=SimpleNamespace(segment=lambda a,b:Certificate(accepted or np.array_equal(a,b)),last_rejection=None)
    task.make_environment=lambda *args:(env,SimpleNamespace(revision=1))
    task.visual_observation=lambda *args:{}
    task.allocation_context=lambda t:dict(target=t.position)
    task.finish_policy_pending=lambda *args:None
    task.visual_policy=SimpleNamespace(sample=lambda obs:(np.full(16,.5),{}))
    task.memory=SimpleNamespace(completed=[])
    task.self_guard=SimpleNamespace(segment=lambda *a,**kw:True,box=lambda *a,**kw:(True,'clear'))
    task.allocator=SimpleNamespace(endpoint_valid=lambda *a:True)
    task.certify_visual_path=lambda *a:SimpleNamespace(allowed=True,dictionary=lambda:dict(allowed=True))
    motion,status=task.plan_step(SimpleNamespace(position=np.ones(3)),.1)
    np.testing.assert_allclose(motion.end,np.full(16,.01 if accepted else 0.))
    np.testing.assert_array_equal(task.q,np.zeros((2,8)))
    assert status=='accepted' and task.policy_pending['certified']==accepted


def test_six_object_usd_generation_has_supported_geometry(tmp_path):
    from visual_guidance_lab.random_tasks import build
    from astronaut_delivery.motion import vertices,transformed,world
    from pxr import Usd,UsdGeom
    import json
    seed=next(s for s in range(100000,100100) if len(draw(s)['parts'])==6)
    scene=build(tmp_path/'scene',seed)
    cfg=json.loads((scene.parent/'randomization.json').read_text())
    stage=Usd.Stage.Open(str(scene));parts=list(stage.GetPrimAtPath('/World/Parts').GetChildren())
    assert len(parts)==6
    for body in parts:
        points=np.concatenate([transformed(vertices(p),np.array(world(p)))
            for p in Usd.PrimRange(body) if vertices(p) is not None])
        assert abs(points[:,2].min()-(cfg['table_top_z_m']+.002))<1e-6
        assert body.GetAttribute('magnet:ferromagnetic').Get()
    assert len([p for p in stage.GetPrimAtPath('/World/Station/Operation').GetChildren()
                if p.GetName().startswith('Slot')])==24
