import json
from pathlib import Path
import pytest
import torch
from visual_guidance_lab.action_contract import propose, committed_increment
from visual_guidance_lab.policy import VisualJointPolicy


def test_rejected_target_never_accumulates():
    q=torch.zeros(2,16);a=torch.ones_like(q);lo=q-1;hi=q+1
    p=propose(q,a,lo,hi)
    assert torch.equal(p,propose(q,a,lo,hi))
    assert torch.equal(committed_increment(q,q,torch.zeros(2,dtype=torch.bool)),q)
    with pytest.raises(RuntimeError):
        committed_increment(q,p,torch.zeros(2,dtype=torch.bool))


def test_limit_and_action_validation():
    q=torch.full((1,16),.999);lo=torch.full_like(q,-1);hi=torch.ones_like(q)
    assert propose(q,hi,lo,hi).max()==1
    with pytest.raises(ValueError):propose(q,hi*2,lo,hi)


def test_image_and_history_affect_policy_and_reset_isolated():
    torch.manual_seed(91);m=VisualJointPolicy();b=2
    image=torch.rand(b,5,32,32);q=torch.zeros(b,16);features=torch.zeros(b,64)
    task=torch.zeros(b,24);state=torch.rand(b,128);reset=torch.zeros(b,dtype=torch.bool)
    args=(image,q,q,q,features,task)
    out=m(*args,state,reset)
    out['action_mean'].sum().backward()
    assert m.image[0].weight.grad.abs().sum()>0
    changed=m(*args,state+1,reset)
    assert not torch.allclose(out['action_mean'],changed['action_mean'])
    reset[0]=True
    one=m(*args,state,reset);two=m(*args,state+1,reset)
    assert torch.equal(one['state'][0],two['state'][0])
    assert not torch.equal(one['state'][1],two['state'][1])


def test_protocol_disjoint_and_full_scene_denominator():
    p=json.loads((Path(__file__).resolve().parents[1]/'visual_guidance_lab/protocol.json').read_text())
    train=p['training_seed_range_inclusive'];valid=p['validation_seed_range_inclusive'];test=p['test_seed_range_inclusive']
    assert train[1]<valid[0] and valid[1]<test[0]
    assert test[1]-test[0]+1==p['formal_scene_count']==1000
    assert p['minimum_scene_successes']==990
    assert p['all_started_failures_and_interruptions_count']
    assert not p['reaction_optimization']
