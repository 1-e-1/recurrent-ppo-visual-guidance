"""Paired-factor scene randomization, with no pose oracle in the visual guide.

Reach screening uses CAD IK ONLY for defining the evaluation domain. Every draw
and rejected admission is logged. Trials are never filtered by delivery success.
"""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from .scene import ROOT,set_transform,cube,cylinder,material
from .online_scene import ONLINE_SCENE
from .motion import Arm,CollisionGuard,world

def draw(seed,mode):
    rng=np.random.default_rng(seed);cfg=json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
    h=float(rng.uniform(.72,.80)) if mode in ('height','combined') else .76
    specs=[]
    for index,part in enumerate(cfg['parts']):
        spec=copy.deepcopy(part);spec['position'][2]=h+.002;spec['destination'][2]=h+.002
        spec['dimensions']=[.055,.05,.051];spec['shape']='original';spec['yaw_deg']=0.
        if mode in ('shape','combined'):
            spec['shape']=str(rng.choice(['box','cylinder','hex_prism','wedge']))
            spec['dimensions']=[float(rng.uniform(.042,.066)),float(rng.uniform(.042,.066)),float(rng.uniform(.032,.058))]
            spec['yaw_deg']=float(rng.uniform(-180,180))
        for attempt in range(1000):
            if mode in ('position','combined'):
                spec['position'][:2]=[float(rng.uniform(.515,.66)),float(rng.uniform(-.55,-.18))]
            r=.5*np.linalg.norm(spec['dimensions'][:2])
            if all(np.linalg.norm(np.array(spec['position'][:2])-other['position'][:2])>r+.5*np.linalg.norm(other['dimensions'][:2])+.012 for other in specs):break
        else:raise RuntimeError('Placement rejection budget exhausted')
        specs.append(spec)
    cfg.update(parts=specs,table_top_z_m=h)
    return cfg

def make_scene(folder,cfg):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/'scene.usda'
    stage=Usd.Stage.CreateNew(str(path));stage.GetRootLayer().subLayerPaths=[str(ONLINE_SCENE).replace('\\','/')]
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    dh=cfg['table_top_z_m']-.76
    # Move complete tables (including pads/slots), never just the visible top.
    for name in ('Storage','Operation'):
        # The source parents are untyped organizational prims. Authoring only
        # xformOp attributes on them does NOT transform their geometry in USD.
        table=UsdGeom.Xform.Define(stage,'/World/Station/'+name)
        set_transform(table.GetPrim(),(0,0,dh))
    steel=material(stage,'random_body',(.32,.37,.42),.8)
    for spec in cfg['parts']:
        root='/World/Parts/'+spec['name'];body=stage.GetPrimAtPath(root)
        if not body:
            from pxr import Sdf
            body=UsdGeom.Xform.Define(stage,root).GetPrim()
            set_transform(body)
            UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr(True)
            body.CreateAttribute('magnet:ferromagnetic',Sdf.ValueTypeNames.Bool).Set(True)
            body.CreateAttribute('magnet:contactLocal',Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0,0,.055))
        t=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),spec['yaw_deg']));t.SetTranslateOnly(Gf.Vec3d(*spec['position']))
        xf=UsdGeom.Xformable(body);xf.GetOrderedXformOps()[0].Set(t)
        if spec['shape']=='original':continue
        for child in list(body.GetChildren()):stage.OverridePrim(child.GetPath()).SetActive(False)
        x,y,z=spec['dimensions'];shape=spec['shape']
        if shape.startswith('tool_'):
            from .maintenance_tools import build_tool
            build_tool(stage,root,shape,(x,y,z),steel)
        elif shape=='box':cube(stage,root+'/RandomBody',(0,0,z/2),(x,y,z),steel)
        elif shape=='cylinder':cylinder(stage,root+'/RandomBody',(0,0,z/2),min(x,y)/2,z,steel)
        else:
            from scipy.spatial import ConvexHull
            if shape=='hex_prism':
                angle=np.arange(6)*np.pi/3
                points=np.array([[x/2*np.cos(a),y/2*np.sin(a),height] for height in (0,z) for a in angle])
            else:points=np.array([[-x/2,-y/2,0],[-x/2,y/2,0],[x/2,-y/2,0],[x/2,y/2,0],[-x/2,-y/2,z],[-x/2,y/2,z],[x/2,-y/2,z*.65],[x/2,y/2,z*.65]])
            hull=ConvexHull(points);mesh=UsdGeom.Mesh.Define(stage,root+'/RandomBody');mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
            mesh.CreateFaceVertexCountsAttr([3]*len(hull.simplices));mesh.CreateFaceVertexIndicesAttr(hull.simplices.ravel().tolist())
            from pxr import UsdShade
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(steel);UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        # The object envelope varies; the horizontal magnetic interface remains
        # standardized. This is NOT arbitrary-size unmarked-target recognition.
        paint=material(stage,spec['name'],spec['color'],.1)
        cylinder(stage,root+'/RandomCap',(0,0,z+.002),.020,.004,paint)
        body.GetAttribute('magnet:contactLocal').Set(Gf.Vec3d(0,0,z+.004))
    stage.GetRootLayer().Save();(folder/'task.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return path

def screen(path,cfg):
    stage=Usd.Stage.Open(str(path));left,right=Arm(stage,'left'),Arm(stage,'right');guard=CollisionGuard(stage,left,right)
    logs=json.loads((ROOT/'reports/cooperative_reaction_final/trajectory.json').read_text());checks=[]
    for spec in cfg['parts']:
        seed=np.array([r for r in logs if r['phase']==spec['name']+'/approach'][-1]['q'])
        p=stage.GetPrimAtPath('/World/Parts/'+spec['name']);cap=(np.array(world(p)).T@np.r_[p.GetAttribute('magnet:contactLocal').Get(),1])[:3]
        q=right.solve(cap+[0,0,.11],np.array([0,0,-1]),seed=seed[1],tries=3,rng=np.random.default_rng(19),
                      valid=lambda q:guard.check('right',q,seed[0])[0])
        checks.append(dict(part=spec['name'],pregrasp_ik_exists=q is not None))
    return all(c['pregrasp_ik_exists'] for c in checks),checks

def trial(base,mode,seed,timeout,controller='cooperative'):
    from threadpoolctl import threadpool_limits
    folder=Path(base)/f'{mode}_{seed}';folder.mkdir(parents=True,exist_ok=False)
    cfg=draw(seed,mode);scene=make_scene(folder,cfg)
    with threadpool_limits(1):admitted,checks=screen(scene,cfg)
    admission=dict(seed=seed,mode=mode,admitted=admitted,checks=checks,
                   meaning='Collision-free pregrasp endpoint IK at a known reference left pose only; not full task or camera visibility feasibility')
    (folder/'admission.json').write_text(json.dumps(admission,indent=2),encoding='utf-8')
    row=dict(mode=mode,seed=seed,admitted=admitted,scene=str(scene),checks=checks,controller=controller)
    if not admitted:return dict(row,status='ADMISSION_REJECTED',completed=0)
    start=time.perf_counter()
    cmd=[sys.executable,'-B','-m','astronaut_delivery.cooperative','--output',str(folder/'run'),'--scene',str(scene),'--task-config',str(folder/'task.json'),'--seed',str(seed),'--parts','3']
    if controller=='sufficiency':
        cmd[3]='astronaut_delivery.sufficiency_delivery';cmd=cmd[:-2]+['--objective','wrench','--anytime-feasible','--map-policy','local']
    try:
        with (folder/'execution.log').open('w',encoding='utf-8') as log:
            process=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        path=folder/'run/result.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        row.update(status=result.get('status','PROCESS_FAILED'),error=result.get('error'),completed=len(result.get('completed',[])),exit_code=process.returncode,
                   simulation_time_s=result.get('simulation_time_s'),reaction_metrics=result.get('reaction_metrics'))
        if result.get('status')=='PASS':
            with (folder/'audit.log').open('w',encoding='utf-8') as log:
                audit=subprocess.run([sys.executable,'-B','-m','astronaut_delivery.cooperative_audit','--run',str(folder/'run')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=90)
            row['audit_pass']=audit.returncode==0
        else:row['audit_pass']=False
        if path.exists():
            events=json.loads((folder/'run/events.json').read_text())
            row['capture_gaps_m']=[e['capture_gap_m'] for e in events if e['event']=='attached']
    except subprocess.TimeoutExpired:row.update(status='TIMEOUT',completed=0,audit_pass=False)
    row['wall_s']=time.perf_counter()-start;return row

def summary(rows):
    groups={}
    for mode in ('position','shape','height','combined'):
        subset=[r for r in rows if r['mode']==mode];admitted=[r for r in subset if r['admitted']]
        passed=sum(r['status']=='PASS' and r.get('audit_pass',False) for r in admitted)
        groups[mode]=dict(draws=len(subset),admitted=len(admitted),audited_success=passed,
                          success_fraction=passed/len(admitted) if admitted else None,completed_parts=sum(r['completed'] for r in admitted))
    return dict(groups=groups,trials=rows,scope='Small pilot; three known-colour objects, fixed 20 mm magnetic interface; initial static randomization, not moving obstacles',
                timing='Runs may be parallel. Wall times are NOT real-time deadline evidence.',ranges=dict(x_m=[.515,.66],y_m=[-.55,-.18],table_height_m=[.72,.80],body_xy_m=[.042,.066],body_height_m=[.032,.058],yaw_deg=[-180,180]),
                rejection_policy='No success-based resampling. Failed IK admissions, stopped runs and timeouts are retained.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[101,202]);p.add_argument('--workers',type=int,default=2);p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--modes',nargs='+',choices=['position','shape','height','combined'],default=['position','shape','height','combined'])
    p.add_argument('--controller',choices=['cooperative','sufficiency'],default='cooperative');args=p.parse_args()
    base=Path(args.output).resolve();base.mkdir(parents=True,exist_ok=False);rows=[]
    import hashlib
    sources=list((ROOT/'astronaut_delivery').glob('*.py'))+list((ROOT/'classical_fusion').glob('*.py'))
    (base/'protocol.json').write_text(json.dumps(dict(controller=args.controller,seeds=args.seeds,modes=args.modes,
        workers=args.workers,timeout_s=args.timeout,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        fixed_before_execution=True,external_preregistration=False),indent=2),encoding='utf-8')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(trial,base,mode,seed,args.timeout,args.controller) for mode in args.modes for seed in args.seeds]
        for future in as_completed(futures):
            row=future.result();rows.append(row);rows.sort(key=lambda r:(r['mode'],r['seed']))
            (base/'benchmark_summary.json').write_text(json.dumps(summary(rows),indent=2),encoding='utf-8');print(json.dumps(row),flush=True)

if __name__=='__main__':main()
