"""Non-destructive static work pose for the unrigged astronaut mesh.

Smooth shoulder/elbow deformation in bind space; preserved UV/topology;
collision hulls rebuilt from ALL posed visual triangles, not old T-pose hulls.
This is an illustrative fixed posture, not an anatomically validated human rig.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

ASSET=Path(__file__).resolve().parents[1]/'isaacsim_model'/'astronaut'
GROUND=.05580005645751962
BODY_OFFSET=np.array([0.,0.,1.1-GROUND])

def smooth(x,a,b):
    t=np.clip((x-a)/(b-a),0,1)
    return t*t*(3-2*t)

def arm_parameters(side):
    shoulder=np.array([.08,side*.275,1.365])
    elbow=np.array([.09,side*.50,1.325])
    wrist=np.array([.10,side*.70,1.29])
    upper=np.array([.16,side*.11,-.13])
    fore=np.array([.19,-side*.075,.07])
    upper*=np.linalg.norm(elbow-shoulder)/np.linalg.norm(upper)
    fore*=np.linalg.norm(wrist-elbow)/np.linalg.norm(fore)
    r0=np.array(Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(*(elbow-shoulder)),Gf.Vec3d(*upper))))
    r1=np.array(Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(*(wrist-elbow)),Gf.Vec3d(*fore))))
    return shoulder,elbow,shoulder+upper,r0,r1

def deform(points):
    p=np.asarray(points,dtype=float)
    result=p.copy()
    for side in (1,-1):
        shoulder,elbow,new_elbow,r0,r1=arm_parameters(side)
        lateral=p[:,1]*side
        shoulder_weight=smooth(lateral,.245,.355)*smooth(p[:,2],1.13,1.23)*smooth(p[:,0],-.13,-.025)
        elbow_weight=smooth(lateral,.445,.555)
        upper=(p-shoulder)@r0+shoulder
        fore=(p-elbow)@r1+new_elbow
        bent=upper*(1-elbow_weight[:,None])+fore*elbow_weight[:,None]
        result+=(bent-p)*shoulder_weight[:,None]
    return result

def set_mesh(mesh,points,faces=None):
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points,dtype=np.float32)))
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
    if faces is not None:
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr('none')

def group(center):
    x,y,z=center
    side='left' if y>=0 else 'right'
    if abs(y)>.28 and z>1.14 and x>-.08:
        return side+('_upper_arm' if abs(y)<.47 else '_forearm' if abs(y)<.67 else '_glove')
    if z>1.49:
        return 'helmet_upper' if z>1.68 else 'helmet_lower'
    if z<.88:
        return side+('_boot' if z<.22 else '_calf' if z<.58 else '_thigh')
    return ('backpack' if x<-.09 else 'torso')+('_upper' if z>1.22 else '_middle' if z>1.04 else '_pelvis')

def build_pose():
    source=Usd.Stage.Open(str(ASSET/'astronaut.usd'))
    target=ASSET/'astronaut_work_pose.usd'
    stage=Usd.Stage.CreateNew(str(target)) if not target.exists() else Usd.Stage.Open(str(target))
    stage.GetRootLayer().Clear()
    root=stage.DefinePrim('/Astronaut','Xform')
    root.GetReferences().AddReference('astronaut.usd','/Astronaut')
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage,'Z')
    UsdGeom.SetStageMetersPerUnit(stage,1)
    groups={}
    checks=[]
    for original in source.Traverse():
        if not original.IsA(UsdGeom.Mesh) or '/Visual/' not in str(original.GetPath()):
            continue
        mesh=UsdGeom.Mesh(original)
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)+BODY_OFFSET
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        moved=deform(points)
        out=UsdGeom.Mesh(stage.OverridePrim(original.GetPath()))
        set_mesh(out,moved-BODY_OFFSET)
        face_normals=np.cross(moved[indices[:,1]]-moved[indices[:,0]],moved[indices[:,2]]-moved[indices[:,0]])
        normals=np.zeros_like(moved)
        for j in range(3):
            np.add.at(normals,indices[:,j],face_normals)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
        out.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        out.SetNormalsInterpolation('vertex')
        assert np.isfinite(moved).all() and np.isfinite(normals).all()
        old_area=np.linalg.norm(np.cross(points[indices[:,1]]-points[indices[:,0]],points[indices[:,2]]-points[indices[:,0]]),axis=1)
        new_area=np.linalg.norm(face_normals,axis=1)
        assert not np.any((old_area>1e-10)&(new_area<1e-12)),'New degenerate triangle'
        for center,tri in zip(points[indices].mean(1),moved[indices]):
            groups.setdefault(group(center),[]).append(tri)
        stationary=(np.abs(points[:,1])<.24)|(points[:,2]<1.12)|(points[:,0]<-.13)
        checks.append(dict(mesh=str(original.GetPath()),vertices=len(points),triangles=len(indices),max_stationary_displacement_m=float(np.linalg.norm(moved[stationary]-points[stationary],axis=1).max()),topology_and_uv_preserved=True))
    # Disable source collider subtree and author new compound colliders.
    stage.OverridePrim('/Astronaut/Body/Collision').SetActive(False)
    hulls=[]
    for name,triangles in sorted(groups.items()):
        points=np.unique(np.asarray(triangles).reshape(-1,3),axis=0)
        hull=ConvexHull(points)
        faces=hull.simplices.copy()
        tri=points[faces]
        flip=(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*hull.equations[:,:3]).sum(1)<0
        faces[flip]=faces[flip,::-1]
        used,inverse=np.unique(faces,return_inverse=True)
        mesh=UsdGeom.Mesh.Define(stage,'/Astronaut/Body/WorkPoseCollision/'+name)
        set_mesh(mesh,points[used]-BODY_OFFSET,inverse.reshape(-1,3))
        mesh.CreatePurposeAttr('guide')
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        containment=float((points@hull.equations[:,:3].T+hull.equations[:,3]).max())
        assert containment<1e-7
        hulls.append(dict(name=name,vertices=len(used),source_triangles=len(triangles),max_containment_error_m=containment))
    for side in ('left','right'):
        path=f'/Astronaut/Body/{side}_hand_endpoint'
        original=source.GetPrimAtPath(path)
        old=np.array(UsdGeom.Xformable(original).GetLocalTransformation().ExtractTranslation())+BODY_OFFSET
        moved=deform(old[None,:])[0]-BODY_OFFSET
        stage.OverridePrim(path).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*moved))
    root.CreateAttribute('pose:description',Sdf.ValueTypeNames.String).Set('Static bent-elbow work pose; no skeletal animation or anthropometric certification')
    stage.GetRootLayer().Save()
    report=dict(asset=str(target),mesh_checks=checks,collision_hulls=hulls,all_visual_triangles_assigned=sum(x['source_triangles'] for x in hulls)==sum(x['triangles'] for x in checks),notes=['Original astronaut.usd and textures unchanged.','Visual deformation and collision rebuild share identical posed points.','Fixed static work posture, not a humanoid control rig.'])
    assert report['all_visual_triangles_assigned']
    (ASSET/'work_pose_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('WORK_POSE',len(hulls),'hulls',sum(x['vertices'] for x in checks),'visual vertices')
    return target

if __name__=='__main__':
    build_pose()
