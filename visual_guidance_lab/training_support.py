"""Crash-safe ledger/checkpoints and bounded long-run resource checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def atomic_checkpoint(path,value):
    import torch
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    torch.save(value,temporary)
    os.replace(temporary,path)


def source_changes(project,expected):
    project=Path(project)
    return [name for name,digest in expected.items() if not (project/name).is_file()
            or hashlib.sha256((project/name).read_bytes()).hexdigest()!=digest]


def require_disk_headroom(path,minimum_gib=24.):
    available=shutil.disk_usage(path).free/2**30
    if available<minimum_gib:
        raise RuntimeError(f'training_storage_guard/free_GiB={available:.2f}/minimum={minimum_gib}')
    return available


def process_is_running(identity):
    import psutil
    try:
        p=psutil.Process(identity['pid'])
        return p.create_time()==identity['create_time'] and p.is_running()
    except (psutil.NoSuchProcess,KeyError):return False
