"""Convert the pinned official D435 mesh (also used by D435i) and mount it.

The small COLLADA reader deliberately rejects unimplemented transforms rather
than silently importing arbitrary DAE incorrectly. No downloaded code executes.
"""
import hashlib,json
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade,UsdPhysics,Vt
from .scene import ROOT,ASSETS,SCENE
from .d435i import load_profile,intrinsics

D435_SCENE=ASSETS/'astronaut_delivery_d435i.usd'
SENSOR=ASSETS/'sensors/d435i'
REVISION='9a11121700cb4780e273e34141f6402fe184321d'

def matrix(prim,m):
    xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder()
    attr=prim.GetAttribute('xformOp:transform')
    op=UsdGeom.XformOp(attr) if attr else xf.AddTransformOp()
    xf.SetXformOpOrder([op]);op.Set(Gf.Matrix4d(np.asarray(m).T.tolist()))

def geometry():
    root=ET.parse(SENSOR/'source/d435.dae').getroot()
    ns={'c':root.tag.split('}')[0][1:]}
    assert root.find('c:asset/c:unit',ns).attrib['meter']=='1'
    assert root.find('c:asset/c:up_axis',ns).text=='Z_UP'
    for tag in ('matrix','translate','rotate','scale','lookat','skew','instance_node'):
        if root.findall('.//c:library_visual_scenes//c:'+tag,ns):raise ValueError('Unsupported COLLADA transform: '+tag)
    instances={x.attrib['url'][1:] for x in root.findall('.//c:instance_geometry',ns)}
    points=[];faces=[];offset=0
    # Official URDF: mesh -> camera_link Rz(pi/2)Rx(pi/2); camera_link -> GL.
    link_gl=np.array([[0,-1,0],[0,0,1],[-1,0,0.]])
    mesh_link=np.array([[0,0,1],[1,0,0],[0,1,0.]])
    rotation=link_gl@mesh_link;translation=link_gl@np.array([.0043,-.0175,0])
    for g in root.findall('c:library_geometries/c:geometry',ns):
        if g.attrib['id'] not in instances:continue
        mesh=g.find('c:mesh',ns)
        sources={}
        for source in mesh.findall('c:source',ns):
            accessor=source.find('c:technique_common/c:accessor',ns)
            sources[source.attrib['id']]=np.fromstring(source.find('c:float_array',ns).text,sep=' ').reshape(-1,int(accessor.attrib['stride']))
        verts={v.attrib['id']:v.find('c:input',ns).attrib['source'][1:] for v in mesh.findall('c:vertices',ns)}
        for tri in mesh.findall('c:triangles',ns):
            inputs=tri.findall('c:input',ns);inp=next(i for i in inputs if i.attrib['semantic']=='VERTEX')
            v=sources[verts[inp.attrib['source'][1:]]][:,:3]
            stride=1+max(int(i.attrib.get('offset',0)) for i in inputs)
            indices=np.fromstring(tri.find('c:p',ns).text,sep=' ',dtype=np.int32).reshape(-1,stride)[:,int(inp.attrib.get('offset',0))].reshape(-1,3)
            if rotation.shape!=(3,3) or not np.isclose(np.linalg.det(rotation),1):raise ValueError('Mesh rotation must be proper')
            points.append(v@rotation.T+translation);faces.append(indices+offset);offset+=len(v)
    return np.concatenate(points),np.concatenate(faces)

def author_camera(stage,path,stream,local):
    cam=UsdGeom.Camera.Define(stage,path);matrix(cam.GetPrim(),local)
    K=intrinsics(stream);focal=1.93
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(float(focal*stream['width']/K[0,0]))
    cam.CreateVerticalApertureAttr(float(focal*stream['height']/K[1,1]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(.01,4.))
    cam.GetPrim().CreateAttribute('sensor:resolution',Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(stream['width'],stream['height']))
    cam.GetPrim().CreateAttribute('sensor:fps',Sdf.ValueTypeNames.Int).Set(stream['fps'])
    return cam

def build():
    cfg=load_profile();v,f=geometry()
    asset=SENSOR/'d435i.usdc'
    s=Usd.Stage.CreateNew(str(asset)) if not asset.exists() else Usd.Stage.Open(str(asset));s.GetRootLayer().Clear()
    root=UsdGeom.Xform.Define(s,'/D435i');s.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(s,1);UsdGeom.SetStageUpAxis(s,'Y')
    mesh=UsdGeom.Mesh.Define(s,'/D435i/HousingMesh')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f),3,np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.astype(np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(.74902,.74902,.74902)])
    mat=UsdShade.Material.Define(s,'/D435i/Material');shader=UsdShade.Shader.Define(s,'/D435i/Material/Surface')
    shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.74902,.74902,.74902))
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.55);shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.35)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    for name,stream,t in [('DepthCamera','depth',[0,0,0]),('ColorCamera','color',cfg['color_origin_in_depth_gl_m']),('RightIRCamera','depth',[.05,0,0])]:
        local=np.eye(4);local[:3,3]=t;author_camera(s,'/D435i/'+name,cfg[stream],local)
    imu=np.eye(4);imu[:3,3]=[.00552,.0051,.01174]
    matrix(UsdGeom.Xform.Define(s,'/D435i/IMUFrame').GetPrim(),imu)
    root.GetPrim().SetCustomData({'calibration_status':cfg['calibration_status'],'source_revision':REVISION})
    s.GetRootLayer().Save()
    scene=Usd.Stage.CreateNew(str(D435_SCENE)) if not D435_SCENE.exists() else Usd.Stage.Open(str(D435_SCENE));scene.GetRootLayer().Clear()
    scene.GetRootLayer().subLayerPaths=[SCENE.name];scene.SetDefaultPrim(scene.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(scene,1.);UsdGeom.SetStageUpAxis(scene,'Z')
    mount=np.asarray(cfg['nominal_flange_to_depth_gl'],float)
    housing=UsdGeom.Xform.Define(scene,'/World/LeftArm/b8/D435i');housing.GetPrim().GetReferences().AddReference('sensors/d435i/d435i.usdc','/D435i');matrix(housing.GetPrim(),mount)
    author_camera(scene,'/World/LeftArm/b8/Camera',cfg['depth'],mount)
    # Conservative exact mesh AABB, not a guessed catalogue centre. Guide purpose
    # keeps proxies out of RGB-D while CollisionGuard explicitly includes them.
    for name,lo,hi in [('CameraHousing',v.min(0)-.0002,v.max(0)+.0002),('CameraMount',np.array([-.012,-.009,.022]),np.array([.012,.009,.040]))]:
        proxy=UsdGeom.Cube.Define(scene,'/World/LeftArm/b8/D435i/'+name)
        proxy.CreateSizeAttr(1);xf=np.diag(np.r_[hi-lo,1.]);xf[:3,3]=(lo+hi)/2;matrix(proxy.GetPrim(),xf)
        if name=='CameraHousing':proxy.CreatePurposeAttr('guide')
        else:UsdShade.MaterialBindingAPI.Apply(proxy.GetPrim()).Bind(UsdShade.Material(scene.GetPrimAtPath('/World/LeftArm/b8/D435i/Material')))
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
    scene.GetRootLayer().Save()
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SENSOR/'source').iterdir() if p.is_file()}
    manifest=dict(repository='https://github.com/realsenseai/realsense-ros',revision=REVISION,license='Apache-2.0; source/LICENSE',files_sha256=files,
                  conversion='Official D435 mesh reused by official D435i URDF; mesh transformed to OpenGL depth optical frame; no geometry decimation.',
                  housing_bounds_depth_gl_m=[v.min(0).tolist(),v.max(0).tolist()],triangles=len(f),vertices=len(v),mounting='Nominal adapter; not measured hardware hand-eye calibration')
    (SENSOR/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2));print(D435_SCENE)

if __name__=='__main__':build()
