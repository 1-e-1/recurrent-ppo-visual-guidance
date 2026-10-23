"""Opt-in sensor-side corruption of NEW renders, with before/after evidence.

No fault flag is used by the controller to authorize or select robot motion.
Durations use monotonic wall time so a stopped robot cannot freeze a fault.
"""
import hashlib,json
from pathlib import Path
import numpy as np


def validate_fault(config):
    if config is None:return None
    if set(config)-{'mode','trigger_motion_s','duration_wall_s','delay_wall_s'}:raise ValueError('unknown sensor-fault field')
    c=dict(mode=config['mode'],trigger_motion_s=float(config.get('trigger_motion_s',5.)),
        duration_wall_s=float(config.get('duration_wall_s',.6)),delay_wall_s=float(config.get('delay_wall_s',.3)))
    if c['mode'] not in ('blackout','rgb_blackout','delay'):raise ValueError('unknown sensor fault')
    if not all(np.isfinite(c[k]) for k in ('trigger_motion_s','duration_wall_s','delay_wall_s')):raise ValueError('finite fault times required')
    if c['trigger_motion_s']<0 or not 0<c['duration_wall_s']<=5 or not 0<=c['delay_wall_s']<=1:raise ValueError('bounded fault schedule required')
    return c


class RenderedSensorFault:
    def __init__(self,config,output=None):
        self.config=validate_fault(config)
        if self.config is None:raise ValueError('explicit fault required')
        self.started=None;self.finished_recorded=False;self.count=0
        self.output=None if output is None else Path(output)
        if self.output is not None:
            self.output.mkdir(parents=True,exist_ok=False)
            (self.output/'records.jsonl').touch(exist_ok=False)

    def prepare(self,obs,motion_stamp,wall_now):
        if not np.isfinite([motion_stamp,wall_now]).all():raise ValueError('finite fault clocks required')
        if self.started is None and motion_stamp>=self.config['trigger_motion_s']:self.started=wall_now
        active=self.started is not None and 0<=wall_now-self.started<self.config['duration_wall_s']
        info=dict(self.config,active=active,started_wall_s=self.started,
            ends_wall_s=None if self.started is None else self.started+self.config['duration_wall_s'],capture_completed_wall_s=wall_now)
        clean=(obs.rgb,obs.depth)
        if active and self.config['mode'] in ('blackout','rgb_blackout'):
            obs.rgb=np.zeros_like(obs.rgb)
            if obs.color_valid is not None:obs.color_valid=np.zeros_like(obs.color_valid)
            if self.config['mode']=='blackout':obs.depth=np.zeros_like(obs.depth)
        delay=self.config['delay_wall_s'] if active and self.config['mode']=='delay' else 0.
        return info,clean,delay

    def record(self,obs,clean,request,info,delay,delivered_wall):
        if self.output is None or self.started is None:return
        if not info['active'] and self.finished_recorded:return
        name=f'fault_{self.count:04d}.npz';path=self.output/name
        # Original rendered pixels and actual estimator input are both retained,
        # even if a delayed reply is later dropped by latest-packet semantics.
        np.savez_compressed(path,clean_rgb=clean[0],clean_depth=clean[1],rgb=obs.rgb,depth=obs.depth,
            K=obs.K,camera_world=obs.camera_world,q=np.asarray(request['q']),exposure_s=request['stamp'],
            requested_wall_s=request['requested_wall_s'])
        row=dict(info,file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            request_id=request['id'],motion_stamp_s=request['stamp'],requested_wall_s=request['requested_wall_s'],
            added_delay_s=delay,delivery_ready_wall_s=delivered_wall,
            scope='New sensor render before/after corruption; not simulation geometry supplied to planner')
        with (self.output/'records.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        self.count+=1
        if not info['active']:self.finished_recorded=True
