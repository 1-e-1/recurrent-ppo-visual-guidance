"""Procedural industrial workcell; retains original astronaut assembly via sublayer."""
from pathlib import Path
import json
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'isaacsim_model'
SCENE=ASSETS/'astronaut_delivery_workcell.usd'

def set_transform(prim,position=(0,0,0),scale=(1,1,1)):
    xf=UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    mat=Gf.Matrix4d().SetScale(Gf.Vec3d(*scale))
    mat.SetTranslateOnly(Gf.Vec3d(*position))
    if prim.HasAttribute('xformOp:transform'):
        op=UsdGeom.XformOp(prim.GetAttribute('xformOp:transform'));xf.SetXformOpOrder([op])
    else:
        op=xf.AddTransformOp()
    op.Set(mat)

def material(stage,name,color,metallic=.3):
    mat=UsdShade.Material.Define(stage,'/World/Station/Materials/'+name)
    shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.45)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    return mat

def cube(stage,path,position,size,mat,collision=True):
    p=UsdGeom.Cube.Define(stage,path)
    p.CreateSizeAttr(1)
    set_transform(p.GetPrim(),position,size)
    UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(mat)
    if collision: UsdPhysics.CollisionAPI.Apply(p.GetPrim())
    return p

def cylinder(stage,path,position,radius,height,mat,collision=True):
    p=UsdGeom.Cylinder.Define(stage,path)
    p.CreateRadiusAttr(radius);p.CreateHeightAttr(height);p.CreateAxisAttr('Z')
    set_transform(p.GetPrim(),position)
    UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(mat)
    if collision: UsdPhysics.CollisionAPI.Apply(p.GetPrim())
    return p

def build():
    cfg=json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
    s=Usd.Stage.CreateNew(str(SCENE)) if not SCENE.exists() else Usd.Stage.Open(str(SCENE))
    s.GetRootLayer().Clear()
    s.GetRootLayer().subLayerPaths=['astronaut_dual_arm_kinematic.usd']
    s.SetDefaultPrim(s.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(s,'Z');UsdGeom.SetStageMetersPerUnit(s,1)
    steel=material(s,'steel',(.32,.37,.42),.8)
    dark=material(s,'matte',(.045,.065,.085),.1)
    trim=material(s,'trim',(.05,.40,.50),.6)
    white=material(s,'label',(.75,.85,.9),.1)
    screen=material(s,'screen',(.015,.07,.11),.1)
    green=material(s,'status',(.03,.65,.4),.1)
    for name,x,y,length,width in [('Storage',.59,-.36,.28,.48),('Operation',.32,-.40,.22,.45)]:
        prefix='/World/Station/'+name
        cube(s,prefix+'/Top',(x,y,.735),(length,width,.05),steel)
        cube(s,prefix+'/Pad',(x,y,.761),(length-.025,width-.025,.002),dark)
        cube(s,prefix+'/Column',(x+.06,y,.42),(.045,.10,.58),steel)
        cube(s,prefix+'/Foot',(x,y,.13),(length,width,.045),steel)
        for sy in (-1,1):
            cube(s,prefix+f'/Edge{sy+1}',(x,y+sy*width/2,.752),(length,.01,.025),trim)
        # Physical fasteners and an instrument panel make it a workcell, not slabs.
        for i,(dx,dy) in enumerate([(-.5,-.5),(.5,-.5),(-.5,.5),(.5,.5)]):
            cylinder(s,prefix+f'/Fastener{i}',(x+dx*(length-.035),y+dy*(width-.035),.764),.006,.006,dark,False)
    cube(s,'/World/Station/Console/Back',(.88,-.05,1.03),(.035,1.25,.50),steel)
    cube(s,'/World/Station/Console/Screen',(.858,.17,1.11),(.008,.26,.18),screen)
    for i in range(4):
        cube(s,f'/World/Station/Console/Readout{i}',(.851,.17,1.17-i*.035),(.006,.19-i*.018,.009),green,False)
    for i in range(4):
        cube(s,f'/World/Station/Console/Key{i}',(.848,.05+i*.07,.963),(.017,.038,.023),trim)
    cube(s,'/World/Station/Console/Rail',(.85,-.41,1.17),(.04,.40,.025),trim)
    for i in range(4):
        cube(s,f'/World/Station/Console/ToolHook{i}',(.83,-.55+i*.10,1.115),(.06,.015,.09),steel)
    for i,part in enumerate(cfg['parts']):
        prefix='/World/Parts/'+part['name']
        body=UsdGeom.Xform.Define(s,prefix).GetPrim()
        set_transform(body,part['position'])
        UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr(True)
        body.CreateAttribute('magnet:ferromagnetic',Sdf.ValueTypeNames.Bool).Set(True)
        body.CreateAttribute('task:partId',Sdf.ValueTypeNames.String).Set(part['name'])
        body.CreateAttribute('magnet:contactLocal',Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0,0,.055))
        paint=material(s,part['name'],part['color'],.1)
        if part['kind']=='bolt':
            cylinder(s,prefix+'/Shaft',(0,0,.020),.012,.04,steel)
            cylinder(s,prefix+'/Head',(0,0,.044),.026,.018,steel)
            for z in (.005,.012,.019,.026):
                cylinder(s,prefix+f'/Thread{int(z*1000)}',(0,0,z),.014,.003,steel,False)
        elif part['kind']=='washer':
            cylinder(s,prefix+'/Base',(0,0,.012),.032,.024,steel)
            cylinder(s,prefix+'/Boss',(0,0,.037),.021,.026,steel)
        else:
            cube(s,prefix+'/Foot',(0,0,.008),(.065,.055,.016),steel)
            cube(s,prefix+'/Upright',(.024,0,.028),(.016,.055,.045),steel)
            cylinder(s,prefix+'/Boss',(0,0,.035),.016,.032,steel)
        # Colored ferromagnetic docking cap is the optical target, not simulator
        # segmentation metadata. Depth + cap RGB supply its position.
        cylinder(s,prefix+'/MagneticCap',(0,0,.053),.020,.004,paint)
        destination=part['destination']
        for dx,dy,sx,sy in [(-.045,0,.004,.085),(.045,0,.004,.085),(0,-.045,.085,.004),(0,.045,.085,.004)]:
            cube(s,f'/World/Station/Operation/Slot{i}_{dx}_{dy}'.replace('-','m').replace('.','p'),(destination[0]+dx,destination[1]+dy,.764),(sx,sy,.002),white,False)
    # Environment-only screenshot, independent of the camera-on-arm optical path.
    overview=UsdGeom.Camera(s.GetPrimAtPath('/World/OverviewCamera'))
    xf=UsdGeom.Xformable(overview.GetPrim());xf.ClearXformOpOrder()
    op=UsdGeom.XformOp(overview.GetPrim().GetAttribute('xformOp:transform'));xf.SetXformOpOrder([op])
    op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.35,-2.45,1.95),Gf.Vec3d(.32,-.22,.95),Gf.Vec3d(0,0,1)).GetInverse())
    s.GetRootLayer().Save()
    print(SCENE)
    return SCENE

if __name__=='__main__':build()
