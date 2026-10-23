"""Procedural simulation tool geometry, not manufacturing CAD or recognition labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
from .scene import ROOT, material, cube, cylinder, set_transform

KINDS=('tool_bolt','tool_nut','tool_wrench','tool_screwdriver')


def ring_mesh(cx,cy,rx,ry,inner_ratio,z0,z1,*,hex_outer=False,open_jaw=False):
    n=48
    angle=np.linspace(-.75*np.pi,.75*np.pi,n+1) if open_jaw else np.arange(n)*2*np.pi/n
    # Closed annulus or capped open annular sector; holes remain actual holes.
    outer=(np.cos(np.pi/6)/np.cos((angle+np.pi/6)%(np.pi/3)-np.pi/6)) if hex_outer else np.ones(len(angle))
    points=np.array([[cx+rx*r*np.cos(a),cy+ry*r*np.sin(a),z]
        for z in (z0,z1) for radii in (outer,np.full(len(angle),inner_ratio)) for a,r in zip(angle,radii)])
    k=len(angle);quads=[]
    for i in range(k-1 if open_jaw else k):
        j=(i+1)%k
        quads.extend([(i,j,2*k+j,2*k+i),(k+j,k+i,3*k+i,3*k+j),
                      (j,i,k+i,k+j),(2*k+i,2*k+j,3*k+j,3*k+i)])
    if open_jaw:quads.extend([(0,2*k,3*k,k),(k-1,2*k-1,4*k-1,3*k-1)])
    faces=np.array([[a,b,c] for a,b,c,d in quads]+[[a,c,d] for a,b,c,d in quads],dtype=np.int32)
    return points,faces


def mesh(stage,path,data,mat):
    points,faces=data;m=UsdGeom.Mesh.Define(stage,path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in points]);m.CreateFaceVertexCountsAttr([3]*len(faces))
    m.CreateFaceVertexIndicesAttr(faces.ravel().tolist());m.CreateSubdivisionSchemeAttr('none')
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat);UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    return m


def build_tool(stage,root,kind,dimensions,steel):
    x,y,z=map(float,dimensions)
    if kind not in KINDS or min(x,y,z)<=0:raise ValueError('unsupported positive tool geometry')
    grip=material(stage,'tool_grip',(.12,.13,.14),.05)
    if kind=='tool_bolt':
        cylinder(stage,root+'/Shaft',(0,0,z*.36),min(x,y)*.19,z*.72,steel)
        # Separate annular ridges are a visual thread approximation, not an ISO screw thread.
        for i in range(7):cylinder(stage,root+f'/ThreadRidge{i}',(0,0,z*(.07+.085*i)),min(x,y)*.23,z*.025,steel)
        from scipy.spatial import ConvexHull
        points=np.array([[x*.5*np.cos(a),y*.5*np.sin(a),h] for h in (.72*z,z) for a in np.arange(6)*np.pi/3])
        faces=ConvexHull(points).simplices
        mesh(stage,root+'/HexHead',(points,faces),steel)
    elif kind=='tool_nut':
        mesh(stage,root+'/HexNut',ring_mesh(0,0,x/2,y/2,.45,0,z,hex_outer=True),steel)
    elif kind=='tool_wrench':
        cube(stage,root+'/Handle',(0,0,z/2),(x*.66,y*.24,z*.65),steel)
        mesh(stage,root+'/OpenJaw',ring_mesh(-x*.32,0,x*.18,y/2,.58,0,z,open_jaw=True),steel)
        mesh(stage,root+'/BoxEnd',ring_mesh(x*.34,0,x*.16,y*.40,.56,0,z),steel)
    else:
        h=cylinder(stage,root+'/Grip',(-x*.29,0,z/2),min(y,z)/2,x*.42,grip);h.CreateAxisAttr('X')
        shaft=cylinder(stage,root+'/SteelShaft',(x*.16,0,z/2),min(y,z)*.09,x*.48,steel);shaft.CreateAxisAttr('X')
        cube(stage,root+'/FlatTip',(x*.445,0,z/2),(x*.11,y*.22,z*.045),steel)
        for i in range(6):
            a=2*np.pi*i/6
            cube(stage,root+f'/GripFlute{i}',(-x*.29,np.cos(a)*min(y,z)*.45,z/2+np.sin(a)*min(y,z)*.45),
                 (x*.33,y*.035,z*.035),grip)


def catalog(path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    steel=material(stage,'tool_steel',(.48,.52,.56),.8)
    sizes=[(.055,.05,.06),(.052,.052,.025),(.12,.05,.009),(.13,.025,.025)]
    for i,(kind,size) in enumerate(zip(KINDS,sizes)):
        root='/World/'+kind;body=UsdGeom.Xform.Define(stage,root).GetPrim()
        set_transform(body,(i*.16,0,0));build_tool(stage,root,kind,size,steel)
    stage.GetRootLayer().Save()
    return path


def random_scene(output,seed,*,table_height_range=None):
    from .rgbd_random_workcell import draw
    from .randomized_benchmark import make_scene
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    cfg=draw(seed,table_height_range=table_height_range);rng=np.random.default_rng(seed+907000);placed=[]
    for p in cfg['parts']:
        kind=str(rng.choice(KINDS));long=kind in ('tool_wrench','tool_screwdriver')
        dims=[float(rng.uniform(.085,.125) if long else rng.uniform(.045,.060)),
              float(rng.uniform(.041,.052)),float(rng.uniform(.009,.016) if kind=='tool_wrench' else rng.uniform(.025,.052))]
        if kind=='tool_screwdriver':dims[1]=dims[2]=float(rng.uniform(.024,.032))
        for _ in range(1000):
            xy=rng.uniform([.515,-.55],[.66,-.18]);radius=max(np.linalg.norm(dims[:2])/2,.020)
            if all(np.linalg.norm(xy-old[:2])>radius+old[2]+cfg['minimum_separation_m'] for old in placed):break
        else:raise RuntimeError('tool_scene_placement_budget_exhausted')
        placed.append([*xy,radius]);p.update(shape=kind,dimensions=dims,position=[*map(float,xy),cfg['table_top_z_m']+.002])
    cfg['randomization_scope']='Tool exterior/size/yaw/count/position/support height; known magnetic adaptor remains; no CAD IK/visibility admission'
    path=make_scene(folder,cfg);stage=Usd.Stage.Open(str(path))
    stage.GetRootLayer().subLayerPaths=[str(ROOT/'isaacsim_model/astronaut_delivery_rgbd_cabin.usda').replace('\\','/')]
    names={p['name'] for p in cfg['parts']}
    for child in stage.GetPrimAtPath('/World/Parts').GetChildren():
        if child.GetName() not in names:stage.OverridePrim(child.GetPath()).SetActive(False)
    final=folder/f'rgbd_tools_{seed}.usda';stage.GetRootLayer().Export(str(final))
    (folder/'randomization.json').write_text(json.dumps(dict(seed=seed,draw=cfg,
        table_top_z_range_m=list(table_height_range or [.72,.80]),
        authority='generator/audit only, not controller input',shape_catalog=KINDS,
        note='1--3 observed-identity slots, not all four kinds guaranteed in one draw; no motion validation yet'),indent=2),encoding='utf-8')
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--catalog');p.add_argument('--scene-output');p.add_argument('--seed',type=int,default=91001)
    p.add_argument('--table-height-range',nargs=2,type=float);a=p.parse_args()
    if a.catalog:print(catalog(a.catalog))
    if a.scene_output:print(random_scene(a.scene_output,a.seed,table_height_range=a.table_height_range))
