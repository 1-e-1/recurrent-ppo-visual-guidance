"""Finite supervised subprocess; only the spawned process tree is terminated."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil


def run(module,arguments,output,timeout):
    if timeout<=0:raise ValueError('positive timeout required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    start=time.time()
    with (root/'stdout.log').open('w',encoding='utf-8') as out,(root/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u','-m',module,*arguments],stdout=out,stderr=err,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        identity=psutil.Process(child.pid).create_time()
        (root/'running.json').write_text(json.dumps(dict(pid=child.pid,create_time=identity,
            module=module,timeout_s=timeout)),encoding='utf-8')
        timed_out=False
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:
                p=psutil.Process(child.pid)
                if p.create_time()==identity:
                    children=p.children(recursive=True)
                    for target in reversed(children):
                        try:target.kill()
                        except psutil.NoSuchProcess:pass
                    p.kill()
            except psutil.NoSuchProcess:pass
            code=child.wait(timeout=30)
    result=dict(module=module,exit_code=code,timed_out=timed_out,wall_s=time.time()-start,
                scope='Process outcome only; not task success')
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--module',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    a=parser.parse_args();arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    run(a.module,arguments,a.output,a.timeout)
