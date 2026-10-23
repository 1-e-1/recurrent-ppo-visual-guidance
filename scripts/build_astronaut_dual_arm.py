"""Build a local, reference-based, zero-gravity kinematic USD assembly.

Run with D:\\conda_envs\\env_isaaclab\\python.exe.
Original CAD and robot assets are never modified.
"""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdLux

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'isaacsim_model'
SOURCE = Path(str(__import__('pathlib').Path(__import__('os').environ.get('ZBOT_V63_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'external_assets/zbot-v63')))))
USD = OUT/'astronaut_dual_arm_kinematic.usd'
SAGITTAL_MIRROR = np.diag([1.,-1.,1.,1.])
# A reflection is not a quaternion. Compensate local Y to keep body frames RH.
LOCAL_MIRROR = np.diag([1.,-1.,1.,1.])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def matrix(rotation=None, translation=(0,0,0)):
    m = np.eye(4)
    if rotation is not None:
        m[:3,:3] = rotation
    m[3,:3] = translation
    return Gf.Matrix4d(m.tolist())

def transform(prim, mat):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = next((o for o in xf.GetOrderedXformOps() if o.GetOpName()=='xformOp:transform'),None)
    # Reuse a previously authored matrix op even after clearing the order.
    if prim.HasAttribute('xformOp:transform'):
        op = UsdGeom.XformOp(prim.GetAttribute('xformOp:transform'))
        xf.SetXformOpOrder([op])
    else:
        op = xf.AddTransformOp()
    op.Set(mat)

def world(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

def frame(joint, index):
    p = getattr(joint,f'GetLocalPos{index}Attr')().Get()
    q = getattr(joint,f'GetLocalRot{index}Attr')().Get()
    m = Gf.Matrix4d().SetRotate(Gf.Quatd(q))
    m.SetTranslateOnly(Gf.Vec3d(p))
    return m

def set_frame(joint,index,m):
    getattr(joint,f'CreateLocalPos{index}Attr')().Set(Gf.Vec3f(m.ExtractTranslation()))
    getattr(joint,f'CreateLocalRot{index}Attr')().Set(Gf.Quatf(m.ExtractRotationQuat()))

def add_fixed(stage,name,body0,body1):
    j = UsdPhysics.FixedJoint.Define(stage,'/World/Connections/'+name)
    if body0:
        j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    w0 = world(stage.GetPrimAtPath(body0)) if body0 else Gf.Matrix4d(1)
    w1 = world(stage.GetPrimAtPath(body1))
    set_frame(j,0,w1*w0.GetInverse())
    set_frame(j,1,Gf.Matrix4d(1))
    j.CreateCollisionEnabledAttr(False)
    return j

def ref(stage,path,file,source_prim):
    p = stage.DefinePrim(path,'Xform')
    p.GetReferences().AddReference(file,source_prim)
    return p

def build():
    backup=OUT/'astronaut_dual_arm_before_symmetry_fix.usd'
    if USD.exists() and not backup.exists():
        shutil.copy2(USD,backup)
    manifest = {}
    for name in ('C_structure.usd','cixi.usd','assets/nasa_astronaut/astronaut.usd','assets/nasa_astronaut/texture_0.png','assets/nasa_astronaut/texture_1.png'):
        src = SOURCE/name
        dst = OUT/name.replace('assets/nasa_astronaut/','astronaut/')
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and digest(dst)!=digest(src):
            raise RuntimeError(f'Refusing to overwrite a different user asset: {dst}')
        if not dst.exists():
            shutil.copy2(src,dst)
        manifest[str(dst.relative_to(OUT))] = dict(source=str(src),sha256=digest(src))
    for name in ('zbot_8s_new_left.usd','zbot_8s_new_right.usd'):
        manifest[name] = dict(source='user supplied',sha256=digest(OUT/name))
    from pose_astronaut_work import build_pose
    build_pose()
    cad = json.loads((OUT/'solidworks_reference.json').read_text(encoding='utf-8'))
    comps = {c['name']:c for c in cad['components']}
    lift_t = np.array(comps['7_lift_2_2-1']['transform'][9:12])
    # CAD X -> world Y (anatomical left), CAD Y -> world Z,
    # CAD Z -> world X (astronaut forward); proper rotation, det = +1.
    cad_R = np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=float)
    # Waist/crotch mounting band; astronaut feet are normalized to world Z=0.
    # CAD lift's rear edge is -15.6 mm in its local Z. Position that edge at -0.14 m.
    cad_origin = np.array([-0.124401114206128,0,0.97])
    def cad_point(p):
        return (np.asarray(p)-lift_t)@cad_R+cad_origin
    stage = Usd.Stage.CreateNew(str(USD)) if not USD.exists() else Usd.Stage.Open(str(USD))
    stage.GetRootLayer().Clear()
    root = UsdGeom.Xform.Define(stage,'/World').GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1)
    stage.SetTimeCodesPerSecond(60)
    scene = UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
    scene.CreateGravityMagnitudeAttr(0)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    astro = ref(stage,'/World/Astronaut','astronaut/astronaut_work_pose.usd','/Astronaut')
    transform(astro,matrix(translation=(0,0,-0.05580005645751962)))
    c = UsdGeom.Xform.Define(stage,'/World/C_structure').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(c)
    # Correct front/back about the C bounding-box centre, without changing height.
    # Its body frame now has a genuine vertical local Z, unlike the CAD axes.
    c_center=np.array([.08,0.,(.9550656127929688+1.109)/2])
    turn=np.diag([-1.,-1.,1.])
    transform(c,matrix(turn,c_center))
    geom = ref(stage,'/World/C_structure/Geometry','C_structure.usd','/World')
    # Exported C mesh is millimetres with its bounding-box minimum at zero.
    # Max planes from CAD are exact; GetPartBox minimum has 19 um approximation error.
    c_local_min = np.array([-.361,.139-.15393438720703125,-.015598885793871928])
    transform(geom,matrix(np.eye(3)*.001,c_local_min)*matrix(cad_R,cad_origin)*matrix(translation=-c_center))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath('/World/C_structure/Geometry/mesh'))
    arm_poses={}
    for side,num in (('LeftArm',1),('RightArm',2)):
        key = f'7_arm7_2_2-{num}/7_Zbot3.0_2_2-1/7_mainpart1_2_2-2'
        data = np.array(comps[key]['transform'])
        direction = data[:9].reshape(3,3)[1]@cad_R
        direction /= np.linalg.norm(direction)
        yaxis = np.array([1.,0,0])
        xaxis = np.cross(yaxis,direction)
        R = np.stack([xaxis,yaxis,direction])
        # CAD mainpart1 origin is the first module centre. USD first joint is
        # 53 mm above its base frame. Preserve CAD module position and direction.
        origin = cad_point(data[9:12])-.053*direction
        # Follow the rotated side mounts, exchanging anatomical L/R. For this
        # symmetric support this keeps the mirrored orientations, moves X only.
        origin[0]=2*c_center[0]-origin[0]
        if side=='RightArm':
            # Preserve the right asset's native 180-degree base roll. The previous
            # build overwrote it, rotating corresponding joint axes by ~90 degrees.
            left_pose=np.array(world(stage.GetPrimAtPath('/World/LeftArm')))
            mirrored=LOCAL_MIRROR@left_pose@SAGITTAL_MIRROR
            R,origin=mirrored[:3,:3],mirrored[3,:3]
        p = ref(stage,'/World/'+side,f'zbot_8s_new_{"left" if num==1 else "right"}.usd','/zbot')
        transform(p,matrix(R,origin))
        arm_poses[side] = dict(cad_component=key,position_m=origin.tolist(),rotation_rows=R.tolist())
        p.CreateAttribute('planning:canonicalJointSign',Sdf.ValueTypeNames.DoubleArray).Set([1. if num==1 else -1.]*8)
        p.CreateAttribute('planning:localReflection',Sdf.ValueTypeNames.Matrix4d).Set(Gf.Matrix4d(LOCAL_MIRROR.tolist()))
    # Reconcile stale body1 frames to the supplied straight-arm geometry, while
    # preserving the original body0 pivot and axis. This defines q=0 unambiguously.
    corrections=[]
    for p in list(stage.Traverse()):
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0,b1 = j.GetBody0Rel().GetTargets(),j.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            raise RuntimeError(f'Unexpected unconnected source joint: {p.GetPath()}')
        w0,w1 = world(stage.GetPrimAtPath(b0[0])),world(stage.GetPrimAtPath(b1[0]))
        before = frame(j,1)*w1
        desired = frame(j,0)*w0
        corrections.append(dict(joint=str(p.GetPath()),translation_error_m=(before.ExtractTranslation()-desired.ExtractTranslation()).GetLength(),rotation_error_deg=(before*desired.GetInverse()).ExtractRotation().GetAngle()))
        set_frame(j,1,desired*w1.GetInverse())
    # Camera intrinsics and b8-local mounting copied from the supplied template.
    template = Usd.Stage.Open(str(ROOT/'zbot_8s_new.usd'))
    cam_path = '/World/LeftArm/b8/Camera'
    Sdf.CreatePrimInLayer(stage.GetRootLayer(),'/World/LeftArm/b8')
    Sdf.CopySpec(template.GetRootLayer(),'/zbot/b8/Camera',stage.GetRootLayer(),cam_path)
    cam = stage.GetPrimAtPath(cam_path)
    cam.CreateAttribute('guide:resolution',Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(1280,960))
    # Separate rigid body, not nested beneath another rigid body.
    magnet = UsdGeom.Xform.Define(stage,'/World/RightMagnet').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(magnet)
    # CAD magnet +Y thickness maps along b8 +Z. Back face starts at the
    # terminal interface plane (the same 52.88 mm TCP convention as the camera).
    tip = world(stage.GetPrimAtPath('/World/RightArm/b8'))
    transform(magnet,matrix(translation=(0,0,.05288))*tip)
    mg = ref(stage,'/World/RightMagnet/Geometry','cixi.usd','/World')
    mg_R = np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=float)*.001
    transform(mg,matrix(mg_R,(-.0525,.0525,0)))
    for p in Usd.PrimRange(mg):
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)
    add_fixed(stage,'World_Astronaut',None,'/World/Astronaut/Body')
    add_fixed(stage,'Astronaut_C_structure','/World/Astronaut/Body','/World/C_structure')
    for side in ('LeftArm','RightArm'):
        add_fixed(stage,'C_structure_'+side,'/World/C_structure',f'/World/{side}/base/a1')
    add_fixed(stage,'RightArm_cixi','/World/RightArm/b8','/World/RightMagnet')
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(p)
            api.CreateKinematicEnabledAttr(True)
            api.CreateVelocityAttr(Gf.Vec3f(0))
            api.CreateAngularVelocityAttr(Gf.Vec3f(0))
        if p.IsA(UsdPhysics.Joint):
            # Constraints are solved exactly by our FK, never by PhysX dynamics.
            UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
            p.CreateAttribute('kinematics:enabled',Sdf.ValueTypeNames.Bool).Set(True)
    light=UsdLux.DomeLight.Define(stage,'/World/Lighting/Dome')
    light.CreateIntensityAttr(800)
    light.CreateColorAttr(Gf.Vec3f(.88,.93,1))
    view=UsdGeom.Camera.Define(stage,'/World/OverviewCamera')
    transform(view.GetPrim(),Gf.Matrix4d().SetLookAt(Gf.Vec3d(3.1,3.3,2.3),Gf.Vec3d(0,0,.95),Gf.Vec3d(0,0,1)).GetInverse())
    view.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    view.CreateFocalLengthAttr(28)
    stage.GetRootLayer().customLayerData = dict(kinematicController='scripts/astronaut_kinematics.py',cameraSettings=dict(Perspective=dict(position=Gf.Vec3d(3.1,3.3,2.3),target=Gf.Vec3d(0,0,.95)),boundCamera='/World/OverviewCamera'),description='Fixed astronaut; CAD-referenced C mount; 2x8 DOF; left eye-in-hand; right fixed magnet; exact FK, no dynamics')
    stage.GetRootLayer().Save()
    runtime_report=OUT/'isaacsim_validation.json'
    if runtime_report.exists():
        old=json.loads(runtime_report.read_text(encoding='utf-8'))
        old['status']='STALE_REBUILD_REQUIRES_RUNTIME_CHECK'
        runtime_report.write_text(json.dumps(old,indent=2),encoding='utf-8')
    report=dict(usd=str(USD),assets=manifest,cad_frame_rotation_rows=cad_R.tolist(),cad_origin_world_m=cad_origin.tolist(),arms=arm_poses,joint_frame_corrections=corrections,notes=['CAD mount pose is referenced; old CAD module diameter is 105 mm, supplied USD is 100 mm, not a certified flange fit.','Mount vertical band selected at approximately 0.955 to 1.109 m, astronaut feet at 0.','No baskets, no independent left magnet. PhysX solver disabled on joints; exact FK consumes real USD joint schemas.'])
    report['c_front_correction']=dict(rotation_axis='world vertical Z / new C body local Z',degrees=180,pivot_world_m=c_center.tolist(),follow_side_mounts=True)
    report['astronaut_work_pose']=dict(asset='astronaut/astronaut_work_pose.usd',sha256=digest(OUT/'astronaut'/'astronaut_work_pose.usd'),static_mesh_pose=True)
    (OUT/'assembly_build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(usd=str(USD),arms=arm_poses,max_joint_position_correction_m=max(c['translation_error_m'] for c in corrections),max_joint_rotation_correction_deg=max(c['rotation_error_deg'] for c in corrections)),indent=2))

if __name__=='__main__':
    build()
