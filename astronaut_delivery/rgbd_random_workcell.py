"""Seeded simulation scene generation, isolated from measured-only control.

Randomizes 1--3 bodies, size, horizontal pose, separation and support height.
Standardized colored horizontal docking interfaces are a stated task prior.
No reach/visibility/success admission filter; failed draws remain failures.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from pxr import Usd
from .scene import ROOT
from .randomized_benchmark import make_scene


RANGES=dict(count=[1,3],x_m=[.515,.66],y_m=[-.55,-.18],table_top_z_m=[.72,.80],
    body_xy_m=[.042,.066],body_height_m=[.032,.058],yaw_deg=[-180,180],minimum_separation_m=[.014,.035],
    shapes=['box','cylinder','hex_prism','wedge'])


def draw(seed,*,table_height_range=None):
    rng=np.random.default_rng(seed);cfg=json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
    number=int(rng.integers(1,4));indices=rng.choice(len(cfg['parts']),number,replace=False)
    heights=RANGES['table_top_z_m'] if table_height_range is None else table_height_range
    if len(heights)!=2 or not np.isfinite(heights).all() or not .65<=heights[0]<heights[1]<=.85:
        raise ValueError('bounded ordered support height range required')
    height=float(rng.uniform(*heights));gap=float(rng.uniform(*RANGES['minimum_separation_m']));parts=[]
    for index in indices:
        part=copy.deepcopy(cfg['parts'][int(index)])
        part.update(shape=str(rng.choice(RANGES['shapes'])),dimensions=[float(rng.uniform(.042,.066)),float(rng.uniform(.042,.066)),float(rng.uniform(.032,.058))],
            yaw_deg=float(rng.uniform(-180,180)))
        for attempt in range(1000):
            xy=np.array([rng.uniform(*RANGES['x_m']),rng.uniform(*RANGES['y_m'])]);radius=np.linalg.norm(part['dimensions'][:2])/2
            if all(np.linalg.norm(xy-other['position'][:2])>radius+np.linalg.norm(other['dimensions'][:2])/2+gap for other in parts):break
        else:raise RuntimeError('bounded_geometric_scene_draw_failed')
        part['position']=[float(xy[0]),float(xy[1]),height+.002];part['destination'][2]=height+.002;parts.append(part)
    cfg.update(parts=parts,table_top_z_m=height,random_seed=int(seed),random_count=number,
        minimum_separation_m=gap,randomization_scope='Random body shape/size/count, planar position/yaw, support height; horizontal standardized interfaces')
    return cfg


def build(output,seed,*,table_height_range=None):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    cfg=draw(seed,table_height_range=table_height_range);old_path=make_scene(folder,cfg);stage=Usd.Stage.Open(str(old_path))
    stage.GetRootLayer().subLayerPaths=[str(ROOT/'isaacsim_model/astronaut_delivery_rgbd_cabin.usda').replace('\\','/')]
    present={p['name'] for p in cfg['parts']}
    for child in stage.GetPrimAtPath('/World/Parts').GetChildren():
        if child.GetName() not in present:stage.OverridePrim(child.GetPath()).SetActive(False)
    # Unique scene token avoids a generic "scene" identity shared by all seeds.
    path=folder/f'rgbd_random_{seed}.usda';stage.GetRootLayer().Export(str(path))
    ranges=dict(RANGES,table_top_z_m=list(table_height_range or RANGES['table_top_z_m']))
    (folder/'randomization.json').write_text(json.dumps(dict(seed=seed,ranges=ranges,draw=cfg,
        filtered_by_CAD_IK=False,filtered_by_visibility=False,filtered_by_task_success=False,
        authority='Scene generator and audit only; controller receives RGB-D, never this metadata'),indent=2),encoding='utf-8')
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--table-height-range',nargs=2,type=float);a=p.parse_args()
    print(build(a.output,a.seed,table_height_range=a.table_height_range))
