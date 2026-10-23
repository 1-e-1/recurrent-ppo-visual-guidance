"""Random task generator. Metadata is simulator/audit-only, never observations.

The six-colour magnetic adaptor catalogue is a fixed task-interface prior;
the policy is not told which catalogue entries or how many exist in a scene.
"""
import argparse
import json
from pathlib import Path
import numpy as np

PALETTE = dict(zip([f'item_{i}' for i in range(6)],
    [[.85,.10,.08],[.08,.82,.12],[.10,.18,.90],
     [.85,.72,.06],[.82,.08,.76],[.06,.78,.82]]))
FAMILIES = ('box','cylinder','hex_prism','wedge','tool_bolt','tool_nut',
            'tool_wrench','tool_screwdriver')
BOUNDS = dict(objects=[1,6], table_height_m=[.70,.82], yaw_deg=[-180,180],
    tilt_deg=[-30,30], center_x_m=[.52,.69], center_y_m=[-.63,-.13],
    minimum_gap_m=.014, initial_joint_jitter_rad=.12)


def draw(seed, max_objects=6, tilt_deg=30.):
    if not 1 <= max_objects <= 6 or not 0 <= tilt_deg <= 30:
        raise ValueError('invalid curriculum bounds')
    rng=np.random.default_rng(seed)
    height=float(rng.uniform(*BOUNDS['table_height_m']))
    count=int(rng.integers(1,max_objects+1));parts=[]
    for identity in rng.choice(6,count,replace=False):
        name=f'item_{identity}';kind=str(rng.choice(FAMILIES))
        dims=rng.uniform([.042,.041,.025],[.066,.058,.058])
        if kind in ('tool_wrench','tool_screwdriver'):
            dims[0]=rng.uniform(.085,.12)
            if kind=='tool_wrench':dims[2]=rng.uniform(.009,.016)
            else:dims[1:]=rng.uniform(.024,.032)
        yaw=float(rng.uniform(-180,180));tilt=rng.uniform(-tilt_deg,tilt_deg,2)
        # Conservative rotated AABB footprint/support, including the adaptor.
        from scipy.spatial.transform import Rotation
        R=Rotation.from_euler('xyz',[*tilt,yaw],degrees=True).as_matrix()
        corners=np.array([[x,y,z] for x in (-max(.020,dims[0]/2),max(.020,dims[0]/2))
            for y in (-max(.020,dims[1]/2),max(.020,dims[1]/2)) for z in (0,dims[2]+.004)])@R.T
        radius=float(np.linalg.norm(corners[:,:2],axis=1).max())
        for attempt in range(2000):
            xy=rng.uniform([.52,-.63],[.69,-.13])
            if all(np.linalg.norm(xy-np.array(p['position'][:2]))>
                   radius+p['placement_radius_m']+.014 for p in parts):break
        else:raise RuntimeError(f'geometric_placement_budget_exhausted/seed={seed}/count={count}')
        parts.append(dict(name=name,kind=kind,shape=kind,dimensions=dims.tolist(),
            yaw_deg=yaw,tilt_xy_deg=tilt.tolist(),rotation=R.tolist(),
            position=[*xy.tolist(),height+.002-float(corners[:,2].min())],
            destination=[.29,-.38,height+.002],color=PALETTE[name],placement_radius_m=radius))
    return dict(parts=parts,table_top_z_m=height,random_seed=int(seed),bounds=BOUNDS,
        curriculum=dict(max_objects=max_objects,tilt_deg=tilt_deg),
        scene_metadata_authority='renderer/reset/audit only; never actor or critic')


def build(output,seed,max_objects=6,tilt_deg=30.):
    from pxr import Gf, Usd, UsdGeom
    from astronaut_delivery.randomized_benchmark import make_scene
    from astronaut_delivery.scene import ROOT,set_transform,cube,material
    from astronaut_delivery.motion import vertices,transformed,world
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    cfg=draw(seed,max_objects,tilt_deg)
    path=make_scene(folder,cfg);stage=Usd.Stage.Open(str(path))
    stage.GetRootLayer().subLayerPaths=[str(ROOT/'isaacsim_model/astronaut_delivery_rgbd_cabin.usda').replace('\\','/')]
    names={p['name'] for p in cfg['parts']}
    for child in stage.GetPrimAtPath('/World/Parts').GetChildren():
        if child.GetName() not in names:stage.OverridePrim(child.GetPath()).SetActive(False)
    for spec in cfg['parts']:
        body=stage.GetPrimAtPath('/World/Parts/'+spec['name'])
        T=np.eye(4);T[:3,:3]=spec['rotation'];T[:3,3]=spec['position']
        UsdGeom.Xformable(body).GetOrderedXformOps()[0].Set(Gf.Matrix4d(T.T.tolist()))
        # Resolve the actual mesh bottom once, privately on the generator side.
        points=np.concatenate([transformed(vertices(p),np.array(world(p)))
            for p in Usd.PrimRange(body) if vertices(p) is not None])
        dz=cfg['table_top_z_m']+.002-float(points[:,2].min())
        T[2,3]+=dz;spec['position'][2]+=dz
        UsdGeom.Xformable(body).GetOrderedXformOps()[0].Set(Gf.Matrix4d(T.T.tolist()))
    # Same six visible destination marks in EVERY scene: no count leakage.
    white=material(stage,'random_slot_white',(.75,.85,.9),.1)
    for child in stage.GetPrimAtPath('/World/Station/Operation').GetChildren():
        if child.GetName().startswith('Slot'):stage.OverridePrim(child.GetPath()).SetActive(False)
    for name,cx,cy,size in [('Storage',.61,-.38,(.38,.72)),('Operation',.29,-.38,(.23,.72))]:
        for part,z,dims in [('Top',.735,(*size,.05)),('Pad',.761,(size[0]-.025,size[1]-.025,.002))]:
            set_transform(stage.GetPrimAtPath(f'/World/Station/{name}/{part}'),(cx,cy,z),dims)
        for child in stage.GetPrimAtPath('/World/Station/'+name).GetChildren():
            if child.GetName().startswith(('Edge','Fastener')):stage.OverridePrim(child.GetPath()).SetActive(False)
    for i,y in enumerate(np.linspace(-.67,-.09,6)):
        for j,(dx,dy,sx,sy) in enumerate([(-.045,0,.004,.085),(.045,0,.004,.085),(0,-.045,.085,.004),(0,.045,.085,.004)]):
            cube(stage,f'/World/Station/Operation/SlotNew{i}_{j}',(.29+dx,y+dy,.764),(sx,sy,.002),white,False)
    final=folder/f'random_task_{seed}.usda';stage.GetRootLayer().Export(str(final))
    (folder/'randomization.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--max-objects',type=int,default=6);p.add_argument('--tilt-deg',type=float,default=30.)
    a=p.parse_args();print(build(a.output,a.seed,a.max_objects,a.tilt_deg))
