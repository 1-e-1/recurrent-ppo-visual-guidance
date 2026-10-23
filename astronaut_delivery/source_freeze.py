"""Archive actual run sources, and detect mutation before reporting evidence."""
import hashlib,json,zipfile
from pathlib import Path


def fingerprint(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('astronaut_delivery','classical_fusion','configs','scripts','codex_zbot')
        for p in sorted((root/folder).rglob('*'))
        if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json') and '__pycache__' not in p.parts}


def freeze(root,output,configuration):
    root=Path(root);output=Path(output);hashes=fingerprint(root)
    with zipfile.ZipFile(output/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,digest in hashes.items():
            data=(root/name).read_bytes()
            if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('source changed while archiving: '+name)
            archive.writestr(name,data)
    (output/'run_configuration.json').write_text(json.dumps(configuration,indent=2,default=str),encoding='utf-8')
    (output/'source_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    return hashes


def unchanged(root,hashes):
    return fingerprint(root)==hashes
