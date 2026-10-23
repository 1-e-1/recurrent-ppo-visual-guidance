"""One bounded online development episode followed by independent audits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .bounded_job import run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((Path(args.source_run)/'run_configuration.json').read_text())
    print('configuration_keys',list(cfg),flush=True)
    scan=Path(cfg['initial_scan']);manifest=json.loads(scan.read_text())
    arguments=['--output',str(root/'mission'),'--scene',cfg['scene'],'--initial-scan',str(scan),
        '--scene-token',manifest['scene_token'],'--calibration-id',manifest['calibration_id'],
        '--checkpoint',str(Path(args.checkpoint).resolve()),'--assume-unchanged-scene']
    outcome=run('visual_guidance_lab.online_mission',arguments,root/'job',2400)
    report=dict(process=outcome,full_scene_passed=False,training_updates=None,
                scope='Reused development scene, NOT the 1000-scene evaluation')
    mission=root/'mission'
    if (mission/'result.json').exists():
        report['controller']=json.loads((mission/'result.json').read_text())['status']
        for name,command in (
            ('mission',[sys.executable,'-m','astronaut_delivery.rgbd_mission_audit','--run',str(mission),'--scene',cfg['scene']]),
            ('clock',[sys.executable,'scripts/audit_rgbd_wall_clock.py','--run',str(mission)])):
            with (root/(name+'_audit.log')).open('w',encoding='utf-8') as log:
                try:
                    done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    report[name+'_audit_exit']=done.returncode
                except subprocess.TimeoutExpired:report[name+'_audit_exit']='timeout'
        if (mission/'visual_training.json').exists():
            training=json.loads((mission/'visual_training.json').read_text())
            report['training_updates']=len(training['updates'])
            report['policy_samples']=len(training['events'])
            report['policy_certified_candidates']=sum(x['candidate_certified'] for x in training['events'])
            a=json.loads((mission/'mission_audit.json').read_text()) if (mission/'mission_audit.json').exists() else {}
            c=json.loads((mission/'wall_clock_audit.json').read_text()) if (mission/'wall_clock_audit.json').exists() else {}
            report['full_scene_passed']=bool(outcome['exit_code']==0 and not outcome['timed_out']
                and a.get('scene_all_objects_delivered_audited') and c.get('passed') and training['source_unchanged'])
    (root/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
