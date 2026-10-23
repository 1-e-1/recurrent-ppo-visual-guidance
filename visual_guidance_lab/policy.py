"""Small recurrent observation encoder; no PPO/environment claim.

RGB-D must already be calibrated/normalized, with a fifth valid-depth channel.
Map/task features must be derived from actual observations, not object truth.
Action distribution/likelihood is intentionally left to the training adapter.
"""
import torch
from torch import nn


class VisualJointPolicy(nn.Module):
    def __init__(self, map_dim=64, task_dim=24, hidden=128):
        super().__init__()
        self.image=nn.Sequential(nn.Conv2d(5,16,5,2,2),nn.ELU(),
            nn.Conv2d(16,32,3,2,1),nn.ELU(),nn.Conv2d(32,32,3,2,1),
            nn.ELU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.ELU())
        self.fusion=nn.Sequential(nn.Linear(64+64+map_dim+task_dim,hidden),nn.ELU())
        self.memory=nn.GRUCell(hidden,hidden)
        self.actor=nn.Linear(hidden,16)
        self.latch=nn.Linear(hidden,3)
        self.value=nn.Linear(hidden,1)
        self.map_dim,self.task_dim,self.hidden=map_dim,task_dim,hidden

    def forward(self, rgbd_valid, q, velocity, previous_executed, map_features,
                task_features, state, episode_start):
        b=q.shape[0]
        if q.shape!=(b,16) or velocity.shape!=q.shape or previous_executed.shape!=q.shape:
            raise ValueError('expected [B,16] proprioception')
        if rgbd_valid.ndim!=4 or rgbd_valid.shape[:2]!=(b,5):
            raise ValueError('expected [B,5,H,W] RGB/depth/valid input')
        if map_features.shape!=(b,self.map_dim) or task_features.shape!=(b,self.task_dim):
            raise ValueError('map/task feature shape mismatch')
        if state.shape!=(b,self.hidden) or episode_start.shape!=(b,) or episode_start.dtype!=torch.bool:
            raise ValueError('recurrent state/reset shape mismatch')
        inputs=(rgbd_valid,q,velocity,previous_executed,map_features,task_features,state)
        if not all(torch.isfinite(x).all() for x in inputs):
            raise ValueError('nonfinite observation; depth must include explicit validity mask')
        state=torch.where(episode_start[:,None],torch.zeros_like(state),state)
        proprio=torch.cat((q.sin(),q.cos(),velocity,previous_executed),dim=-1)
        z=self.fusion(torch.cat((self.image(rgbd_valid),proprio,map_features,task_features),dim=-1))
        state=self.memory(z,state)
        return dict(action_mean=self.actor(state),latch_logits=self.latch(state),
                    value=self.value(state).squeeze(-1),state=state)
