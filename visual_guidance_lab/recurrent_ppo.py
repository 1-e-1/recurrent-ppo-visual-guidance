"""On-policy recurrent PPO for RGB-D joint proposals, not motion authority.

Store latent sampled actions, not shielded/executed actions. Replay the rollout
chronologically with truncated BPTT; no random shuffling of hidden states.
"""
import numpy as np
import torch
from .policy import VisualJointPolicy


class RecurrentPPO:
    def __init__(self,checkpoint,training=True,seed=20260916):
        torch.set_num_threads(1);torch.manual_seed(seed)
        self.model=VisualJointPolicy()
        data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model.load_state_dict(data['model'])
        self.log_std=torch.nn.Parameter(torch.full((16,),float(data.get('initial_log_std',-2.))))
        self.parameters=list(self.model.parameters())+[self.log_std]
        self.optimizer=torch.optim.Adam(self.parameters,lr=1e-4)
        self.training=training;self.state=torch.zeros(1,self.model.hidden)
        self.buffer=[];self.updates=[];self.first=True
        if 'log_std' in data:
            with torch.no_grad():self.log_std.copy_(data['log_std'])
            if training:
                self.optimizer.load_state_dict(data['optimizer'])
                self.updates=list(data['updates']);torch.set_rng_state(data['rng'])

    def forward(self,observation,state,reset=False):
        values=[torch.as_tensor(observation[k],dtype=torch.float32).unsqueeze(0)
                for k in ('image','q','velocity','previous','map','task')]
        return self.model(*values,state,torch.tensor([reset]))

    def sample(self,observation):
        with torch.no_grad():
            before=self.state.clone();out=self.forward(observation,before,self.first)
            d=torch.distributions.Normal(out['action_mean'][0],self.log_std.exp())
            latent=d.sample() if self.training else d.mean
            row=dict(observation={k:np.asarray(v,np.float32).copy() for k,v in observation.items()},
                state=before,reset=self.first,latent=latent.clone(),logp=d.log_prob(latent).sum(),
                value=out['value'][0])
            self.state=out['state'].detach();self.first=False
        return latent.tanh().numpy(),row

    def finish(self,row,reward,next_observation=None,terminal=False):
        if not self.training:return
        if not np.isfinite(reward):raise ValueError('nonfinite reward')
        with torch.no_grad():
            if terminal:bootstrap=0.
            else:
                if next_observation is None:raise ValueError('nonterminal bootstrap needs actual next observation')
                bootstrap=float(self.forward(next_observation,self.state)['value'][0])
        self.buffer.append(dict(row,reward=float(reward),bootstrap=bootstrap,terminal=bool(terminal)))
        if len(self.buffer)>=16 or terminal:self.update()

    def replay(self):
        state=self.buffer[0]['state'].detach();logps=[];values=[];entropy=[]
        for row in self.buffer:
            out=self.forward(row['observation'],state,row['reset']);state=out['state']
            d=torch.distributions.Normal(out['action_mean'][0],self.log_std.exp())
            logps.append(d.log_prob(row['latent']).sum());values.append(out['value'][0]);entropy.append(d.entropy().sum())
        return torch.stack(logps),torch.stack(values),torch.stack(entropy)

    def update(self):
        if not self.buffer:return
        advantages=[];gae=0.
        for row in reversed(self.buffer):
            delta=row['reward']+.99*row['bootstrap']-float(row['value'])
            gae=delta+.99*.95*(not row['terminal'])*gae;advantages.append(gae)
        advantages=torch.tensor(advantages[::-1]);returns=advantages+torch.stack([r['value'] for r in self.buffer])
        normalized=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
        old=torch.stack([r['logp'] for r in self.buffer]);initial_ratio_error=None
        for epoch in range(4):
            logp,value,entropy=self.replay();ratio=(logp-old).exp()
            if epoch==0:initial_ratio_error=float((ratio.detach()-1).abs().max())
            policy=-torch.minimum(ratio*normalized,ratio.clamp(.8,1.2)*normalized).mean()
            loss=policy+.5*(value-returns).square().mean()-.001*entropy.mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite recurrent PPO objective')
            self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.parameters,.5);self.optimizer.step()
        self.updates.append(dict(samples=len(self.buffer),mean_reward=float(np.mean([r['reward'] for r in self.buffer])),
            loss=float(loss.detach()),initial_ratio_max_error=initial_ratio_error))
        self.buffer=[]

    def save(self,path):
        if self.buffer:raise RuntimeError('flush rollout before checkpoint')
        from .training_support import atomic_checkpoint
        atomic_checkpoint(path,dict(model=self.model.state_dict(),log_std=self.log_std.detach(),
            optimizer=self.optimizer.state_dict(),updates=self.updates,rng=torch.get_rng_state(),
            scope='Online development PPO joint proposal; classical latch and fallback; no task-success claim'))
