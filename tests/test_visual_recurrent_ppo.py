import numpy as np
import torch
from visual_guidance_lab.policy import VisualJointPolicy
from visual_guidance_lab.recurrent_ppo import RecurrentPPO


def observation():
    return dict(image=np.zeros((5,48,64),np.float32),q=np.zeros(16,np.float32),
        velocity=np.zeros(16,np.float32),previous=np.zeros(16,np.float32),
        map=np.zeros(64,np.float32),task=np.zeros(24,np.float32))


def test_recurrent_replay_matches_collected_likelihood_and_updates(tmp_path):
    path=tmp_path/'start.pt';torch.save(dict(model=VisualJointPolicy().state_dict()),path)
    agent=RecurrentPPO(path);obs=observation()
    before=agent.model.actor.weight.detach().clone()
    for i in range(4):
        action,row=agent.sample(obs)
        assert action.shape==(16,) and np.max(abs(action))<=1
        agent.finish(row,float(i),obs,terminal=i==3)
    assert len(agent.updates)==1 and agent.updates[0]['samples']==4
    assert agent.updates[0]['initial_ratio_max_error']<1e-5
    assert not torch.equal(before,agent.model.actor.weight)
    agent.save(tmp_path/'trained.pt')


def test_eval_does_not_learn(tmp_path):
    path=tmp_path/'start.pt';torch.save(dict(model=VisualJointPolicy().state_dict()),path)
    agent=RecurrentPPO(path,training=False);obs=observation()
    _,row=agent.sample(obs);agent.finish(row,1.,terminal=True)
    assert not agent.updates and not agent.buffer
