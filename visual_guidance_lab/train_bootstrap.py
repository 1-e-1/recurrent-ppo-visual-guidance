"""Fresh visual-policy imitation bootstrap from causal recorded measurements.

This is NOT PPO or evidence of task completion. Actions are measured executed
increments, not rejected/planned endpoints. Validation uses separate scenes;
all are legacy development scenes, never the reserved formal test scenes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .policy import VisualJointPolicy

ROOT=Path(__file__).resolve().parents[1]
PHASES=('approach','lift','transport','place','retreat','visual_acquisition_right_rest')


def causal_index(stamps, query):
    return int(np.searchsorted(stamps,query+1e-9,side='right')-1)


def encode_frame(raw):
    depth=raw['depth'];rgb=raw['rgb'];K=raw['K'];T=raw['camera_world']
    valid=np.isfinite(depth)&(depth>0)&(depth<3.)&~raw['robot_mask'].astype(bool)
    ys=np.linspace(0,depth.shape[0]-1,48).astype(int)
    xs=np.linspace(0,depth.shape[1]-1,64).astype(int)
    ix=np.ix_(ys,xs);d=depth[ix];ok=valid[ix]
    color=rgb[ix].astype(np.float32)/255.
    image=np.concatenate((color.transpose(2,0,1),np.where(ok,d/3.,0)[None],ok[None]),axis=0).astype(np.float32)
    # Coarse current measured spatial evidence, not an obstacle/free certificate.
    yy,xx=np.meshgrid(ys,xs,indexing='ij');z=d[ok]
    p=np.stack(((xx[ok]-K[0,2])*z/K[0,0],-(yy[ok]-K[1,2])*z/K[1,1],-z),axis=1)
    world=p@T[:3,:3].T+T[:3,3]
    hist=np.histogramdd(world,bins=(4,4,4),range=((-1.,1.),(-1.,1.),(0.,2.)))[0]
    features=np.log1p(hist.ravel()).astype(np.float32)/8.
    return image,features


def scene_samples(folder,maximum):
    read=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    observations=read('observations.json');paths=read('paths.json');steps=read('steps.json')
    st=np.array([s['time_s'] for s in steps]);pt=np.array([p['time_s'] for p in paths])
    rows=[];previous=None;previous_time=None
    candidates=np.unique(np.linspace(0,len(observations)-1,min(maximum,len(observations))).astype(int))
    for index in candidates:
        obs=observations[index];stamp=obs['time_s'];pi=causal_index(pt,stamp)
        if pi<0:continue
        path=paths[pi];target=obs['targets'].get(path.get('visual_target_name'))
        if target is None or not 0<=stamp-target['stamp']<=.25:continue
        si=causal_index(st,stamp);future=int(np.searchsorted(st,stamp+.20))
        if future>=len(steps) or st[future]-stamp>.25:continue
        # Do not supervise across a manipulation/phase boundary.
        if si>=0 and steps[future].get('phase')!=steps[si].get('phase'):continue
        file=folder/'raw'/obs['file'];data=file.read_bytes()
        if hashlib.sha256(data).hexdigest()!=obs['sha256']:raise ValueError('raw hash mismatch')
        with np.load(file,allow_pickle=False) as raw:
            if abs(float(raw['exposure_s'])-stamp)>1e-6:raise ValueError('exposure mismatch')
            q=np.asarray(raw['q'],dtype=np.float32).reshape(16)
            if not np.allclose(q,np.asarray(obs['q']).reshape(16),atol=1e-6):raise ValueError('joint record mismatch')
            image,features=encode_frame(raw)
        delta=np.array(steps[future]['q'],dtype=np.float32)-q
        # Retain the actual action direction without per-joint distortion.
        label=delta/max(.02,float(np.abs(delta).max()))
        velocity=np.zeros(16,np.float32) if si<0 else np.array(steps[si]['v'],np.float32)
        history=np.zeros(16,np.float32) if previous is None else q-previous
        task=np.zeros(24,np.float32);task[:3]=target['position']
        task[3:6]=np.array(path['target'])+[0,0,path.get('standoff_m',0.)]
        task[6:9]=np.sqrt(np.maximum(np.diag(target['covariance']),0))
        task[9:12]=[target['pixels']/1000.,stamp-target['stamp'],path.get('standoff_m',0.)]
        phase=path.get('phase','approach')
        if phase in PHASES:task[12+PHASES.index(phase)]=1
        task[18]=previous_time is not None
        task[19]=0 if previous_time is None else min(stamp-previous_time,10)/10.
        rows.append(dict(image=image,q=q,velocity=velocity,previous=history,map=features,task=task,label=label))
        previous=q;previous_time=stamp
    return rows


def loss_for(model,rows,train,optimizer=None):
    losses=[]
    for start in range(0,len(rows),8):
        chunk=rows[start:start+8];state=torch.zeros(1,model.hidden);loss=0
        for i,row in enumerate(chunk):
            tensors=[torch.from_numpy(row[k]).unsqueeze(0) for k in ('image','q','velocity','previous','map','task')]
            output=model(*tensors,state,torch.tensor([i==0]));state=output['state']
            label=torch.from_numpy(row['label']).unsqueeze(0)
            loss=loss+(output['action_mean'].tanh()-label).square().mean()
        loss=loss/len(chunk)
        if train:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=30);parser.add_argument('--max-frames-per-scene',type=int,default=128)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(20260916)
    batches=('rgbd_classical_stall_cache_dev_01','rgbd_classical_stall_cache_expanded_dev_01','rgbd_classical_stall_cache_remaining_dev_01')
    sources=[];train=[];valid=[];began=time.time()
    for batch in batches:
        root=ROOT/'reports'/batch;progress=json.loads((root/'progress.json').read_text())
        for row in progress['rows']:
            if not row.get('passed'):continue
            folder=root/('run_'+str(row['seed'])+'_set')
            audit=json.loads((folder/'mission_audit.json').read_text())
            if not audit.get('scene_all_objects_delivered_audited'):raise ValueError('missing successful demonstration audit')
            samples=scene_samples(folder,args.max_frames_per_scene)
            split='validation' if row['seed'] in (82017,93003) else 'train'
            (valid if split=='validation' else train).append(samples)
            sources.append(dict(seed=row['seed'],run=str(folder),samples=len(samples),split=split,
                observation_manifest_sha256=hashlib.sha256((folder/'observations.json').read_bytes()).hexdigest()))
            (output/'sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
            print(json.dumps(sources[-1]),flush=True)
    if not train or not valid or any(not s for s in train+valid):raise ValueError('empty scene split')
    model=VisualJointPolicy();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    zero=float(np.mean([np.mean([np.mean(r['label']**2) for r in scene]) for scene in valid]))
    best=float('inf');history=[]
    for epoch in range(args.epochs):
        model.train();training=float(np.mean([loss_for(model,s,True,optimizer) for s in train]))
        model.eval()
        with torch.no_grad():validation=float(np.mean([loss_for(model,s,False) for s in valid]))
        if validation<best:
            best=validation;torch.save(dict(model=model.state_dict(),epoch=epoch,validation_mse=validation,
                scope='Development-only BC initialization, latch/value heads untrained; no online task result'),output/'policy_best.pt')
        history.append(dict(epoch=epoch,train_mse=training,validation_mse=validation))
        report=dict(status='complete' if epoch==args.epochs-1 else 'training',epochs=history,
            zero_action_validation_mse=zero,best_validation_mse=best,wall_s=time.time()-began,
            training_type='fresh visual recurrent behavioral cloning; NOT PPO',task_success_rate=None,
            caveat='Old demonstrations include reaction-shaped motions; this is initialization only, not the new task objective')
        (output/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(history[-1]),flush=True)


if __name__=='__main__':main()
