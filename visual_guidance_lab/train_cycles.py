"""Random reset -> PPO training -> frozen validation, with a persistent ledger.

New random scene + authenticated raw RGB-D scan per episode. No truth
observations and no success-based seed replacement. Existing audited backend
is sequential; IsaacLab rendering smoke tests are NOT called task parallelism.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import zipfile
from .bounded_job import run
from .training_support import atomic_json,source_changes,require_disk_headroom,process_is_running


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan(cycles,train_per_cycle,eval_per_cycle,train_start=100000,eval_start=900000):
    if min(cycles,train_per_cycle,eval_per_cycle)<1:raise ValueError('positive cycle sizes required')
    if not (100000<=train_start and train_start+cycles*train_per_cycle<=900000 and
            900000<=eval_start and eval_start+cycles*eval_per_cycle<=900200):raise ValueError('seed split capacity exceeded')
    return [dict(cycle=c,training=training,seed=(train_start+c*train_per_cycle+i if training else eval_start+c*eval_per_cycle+i))
        for c in range(cycles) for training,n in ((True,train_per_cycle),(False,eval_per_cycle)) for i in range(n)]


def summarize(rows):
    result={}
    for name,training in [('training',True),('validation',False)]:
        group=[r for r in rows if r['training']==training]
        result[name]=dict(episodes=len(group),successes=sum(bool(r.get('passed')) for r in group),
            samples=sum(r.get('samples',0) for r in group),new_updates=sum(r.get('new_updates',0) for r in group),
            success_rate=sum(bool(r.get('passed')) for r in group)/len(group) if group else None)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--cycles',type=int,default=2);p.add_argument('--train-per-cycle',type=int,default=2)
    p.add_argument('--eval-per-cycle',type=int,default=1);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--max-objects',type=int,default=2);p.add_argument('--tilt-deg',type=float,default=0.)
    p.add_argument('--reset-jitter',type=float,default=.12);p.add_argument('--checkpoint')
    p.add_argument('--episode-timeout',type=float,default=3600)
    p.add_argument('--train-start',type=int,default=100000);p.add_argument('--eval-start',type=int,default=900000)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();root=Path(args.output).resolve();project=Path(__file__).resolve().parents[1]
    resume=args.resume;rows=[];elapsed_before=0.
    if resume:
        config=read(root/'configuration.json')
        if (root/'runner.json').exists() and process_is_running(read(root/'runner.json')):
            raise RuntimeError('training runner still active; refusing duplicate resume')
        if any(process_is_running(read(f)) for f in root.rglob('running.json')):
            raise RuntimeError('training subprocess still active; do not resume concurrently')
        changed=source_changes(project,config['source_sha256'])
        if changed:raise RuntimeError('Resume requires frozen source; changed: '+str(changed))
        prior=read(root/'progress.json');rows=prior['episodes'];elapsed_before=prior['wall_s']
        for key in vars(args):
            if key not in ('resume','output') and key in config:setattr(args,key,config[key])
    schedule=plan(args.cycles,args.train_per_cycle,args.eval_per_cycle,args.train_start,args.eval_start)
    if not 1<=args.steps<=2048 or not 1<=args.max_objects<=6 or not 0<=args.tilt_deg<=30:
        raise ValueError('invalid training bounds')
    if not resume:root.mkdir(parents=True,exist_ok=False)
    start=time.time();active=None
    import psutil
    own_process=psutil.Process()
    atomic_json(root/'runner.json',dict(pid=own_process.pid,create_time=own_process.create_time(),resume=resume))
    # Preserve exact source bytes, not only hashes, before any subprocess starts.
    files=[x for name in ('visual_guidance_lab','astronaut_delivery','classical_fusion','configs','scripts')
           for x in (project/name).rglob('*') if x.is_file() and x.suffix in ('.py','.json') and '__pycache__' not in x.parts]
    hashes={x.relative_to(project).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    if not resume:
        with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for f in files:archive.write(f,f.relative_to(project))
    config=dict(vars(args),schedule=schedule,source_sha256=hashes,
        backend='sequential direct-action existing RGB-D guard; NOT IsaacLab task parallelism',
        reset='Uniform bounded jitter around measured-safe commissioning pose, followed by safety admission; not full-range uniform joints',
        initial_scan_frames=786,initial_scan='Simulated external commissioning RGB-D; preserved raw records',
        formal_99_percent_evaluation=False,reaction_reward=0.)
    if not resume:atomic_json(root/'configuration.json',config)
    def save(status='running'):
        report=dict(status=status,active=active,wall_s=elapsed_before+time.time()-start,summary=summarize(rows),episodes=rows,
                    latest_training_checkpoint=str(checkpoint))
        atomic_json(root/'progress.json',report)
        print(json.dumps(dict(status=status,active=active,summary=report['summary'])),flush=True)
    def job(module,arguments,destination,timeout=1800):
        outcome=run(module,list(map(str,arguments)),destination,timeout)
        if outcome['exit_code']!=0 or outcome['timed_out']:raise RuntimeError(f'{module}: {outcome}')
        return outcome
    if resume:checkpoint=Path(prior['latest_training_checkpoint'])
    elif args.checkpoint:checkpoint=Path(args.checkpoint).resolve()
    else:
        import torch
        from .policy import VisualJointPolicy
        torch.manual_seed(20260916)
        checkpoint=root/'initial_policy.pt'
        torch.save(dict(model=VisualJointPolicy().state_dict(),initial_log_std=-.7,
            scope='Fresh random network; not prior PPO or imitation weights'),checkpoint)
    save()
    for case in schedule:
        if any(r['seed']==case['seed'] and r['training']==case['training'] for r in rows):continue
        if (root/'STOP_AFTER_EPISODE').exists():save('paused_by_stop_file');return
        if source_changes(project,hashes):
            active=dict(**case,phase='source_changed');save('paused_source_changed');return
        # Do not delete evidence or silently skip selected seeds when capacity runs out.
        try:require_disk_headroom(root,30.)
        except RuntimeError:
            active=dict(**case,phase='storage_guard');save('paused_storage_guard');return
        label=('train_' if case['training'] else 'eval_')+str(case['seed'])
        folder=root/label
        if folder.exists():
            # A crash is a retained failed episode, not an invisible retry.
            if (folder/'episode.json').exists():row=read(folder/'episode.json')
            else:
                row=dict(**case,passed=False,samples=0,new_updates=0,error='interrupted_before_episode_commit')
                latest=folder/'mission/visual_policy_latest.pt'
                if case['training'] and latest.exists():
                    import torch
                    data=torch.load(latest,map_location='cpu',weights_only=True)
                    previous=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    updates=data['updates'][len(previous.get('updates',[])):]
                    row.update(checkpoint_out=str(latest),new_updates=len(updates),samples=sum(u['samples'] for u in updates))
                atomic_json(folder/'episode.json',row)
            if case['training'] and row.get('checkpoint_out'):checkpoint=Path(row['checkpoint_out'])
            rows.append(row);save();continue
        folder.mkdir();began=time.time()
        row=dict(**case,passed=False,samples=0,new_updates=0,checkpoint=str(checkpoint))
        active=dict(**case,phase='generate');save()
        try:
            job('visual_guidance_lab.random_tasks',['--output',folder/'scene','--seed',case['seed'],
                '--max-objects',args.max_objects,'--tilt-deg',args.tilt_deg],folder/'generate_job')
            scene=folder/'scene'/f"random_task_{case['seed']}.usda"
            active=dict(**case,phase='initial_rgbd');save()
            scans=[]
            for name,flag,stamp in [('scan','--dense-workspace',0),('corridor','--right-corridor-only',30),('near','--near-table-only',60)]:
                job('astronaut_delivery.rgbd_initial_scan',['--output',folder/name,'--scene',scene,flag,
                    '--start-time',stamp],folder/(name+'_job'))
                scans.append(folder/name/'recording.json')
            job('astronaut_delivery.rgbd_merge_recordings',['--inputs',*scans,'--output',folder/'recording.json'],folder/'merge_job',120)
            manifest=read(folder/'recording.json')
            if len(manifest['records'])!=786:raise ValueError('initial RGB-D budget mismatch')
            active=dict(**case,phase='training' if case['training'] else 'frozen_validation');save()
            before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            arguments=['--output',folder/'mission','--scene',scene,'--initial-scan',folder/'recording.json',
                '--scene-token',manifest['scene_token'],'--calibration-id',manifest['calibration_id'],
                '--checkpoint',checkpoint,'--episode-seed',case['seed'],'--steps',args.steps,'--reset-jitter',args.reset_jitter]
            if not case['training']:arguments.append('--evaluate')
            process=run('visual_guidance_lab.direct_mission',list(map(str,arguments)),folder/'mission_job',args.episode_timeout)
            row['process']=process;mission=folder/'mission'
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=before:raise RuntimeError('input checkpoint mutated')
            learning=mission/'learning_progress.json'
            if learning.exists():
                info=read(learning);row.update(samples=info['samples'],new_updates=info['new_updates'])
            final=mission/'visual_policy_final.pt'
            latest=mission/'visual_policy_latest.pt'
            if case['training'] and (final.exists() or latest.exists()):
                checkpoint=final if final.exists() else latest
                row['checkpoint_out']=str(checkpoint)
            if not case['training'] and row['new_updates']!=0:raise RuntimeError('validation changed policy')
            if (mission/'result.json').exists():
                row['controller']=read(mission/'result.json').get('status')
                row['error']=read(mission/'result.json').get('error')
                active=dict(**case,phase='audit');save()
                audit=run('astronaut_delivery.rgbd_mission_audit',['--run',str(mission),'--scene',str(scene)],folder/'audit_job',600)
                clock=run('scripts.audit_rgbd_wall_clock',['--run',str(mission)],folder/'clock_job',600)
                proof=read(mission/'mission_audit.json') if (mission/'mission_audit.json').exists() else {}
                timing=read(mission/'wall_clock_audit.json') if (mission/'wall_clock_audit.json').exists() else {}
                frozen=read(mission/'visual_training.json') if (mission/'visual_training.json').exists() else {}
                row['passed']=bool(process['exit_code']==0 and not process['timed_out'] and
                    audit['exit_code']==0 and clock['exit_code']==0 and proof.get('scene_all_objects_delivered_audited')
                    and timing.get('passed') and frozen.get('source_unchanged'))
        except Exception as exc:
            row['error']=repr(exc)
            (folder/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        row['wall_s']=time.time()-began;rows.append(row)
        atomic_json(folder/'episode.json',row);active=None;save()
    save('completed_schedule_not_target_achieved')


if __name__=='__main__':main()
