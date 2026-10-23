"""Single-payload proximity latch; explicit simulation contact and release."""
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics
from .motion import Payload,Hull,world

class MagneticLatch:
    def __init__(self,stage,guard,capture_distance=.008):
        self.stage,self.guard,self.capture_distance=stage,guard,capture_distance
        self.payload=None;self.joint_path='/World/DeliveryAttachment'

    def attach(self,name,tcp):
        if self.payload is not None:return False,'already_holding'
        part=self.stage.GetPrimAtPath('/World/Parts/'+name)
        if not part or not part.GetAttribute('magnet:ferromagnetic').Get():return False,'not_magnetic'
        pose=np.array(world(part)).T
        contact=(pose@np.r_[np.array(part.GetAttribute('magnet:contactLocal').Get()),1])[:3]
        gap=np.linalg.norm(tcp[:3,3]-contact)
        if not np.isfinite(gap) or gap>self.capture_distance:return False,'outside_capture_range'
        if np.dot(tcp[:3,2],pose[:3,2])>-.7:return False,'wrong_contact_side'
        points=self.guard.parts[name].points
        local=(points-tcp[:3,3])@tcp[:3,:3]
        self.payload=Payload(name,local,np.linalg.inv(tcp)@pose)
        joint=UsdPhysics.FixedJoint.Define(self.stage,self.joint_path)
        joint.CreateBody0Rel().SetTargets(['/World/RightMagnet'])
        joint.CreateBody1Rel().SetTargets([part.GetPath()])
        magnet=np.array(world(self.stage.GetPrimAtPath('/World/RightMagnet'))).T
        rel=np.linalg.inv(magnet)@pose
        rm=Gf.Matrix4d(rel.T.tolist())
        joint.CreateLocalPos0Attr(Gf.Vec3f(rm.ExtractTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(rm.ExtractRotationQuat()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0));joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateJointEnabledAttr(False) # Exact kinematic attachment, no dynamics.
        joint.CreateCollisionEnabledAttr(False)
        return True,float(gap)

    def update(self,tcp):
        if self.payload is None:return
        pose=tcp@self.payload.local_pose
        part=self.stage.GetPrimAtPath('/World/Parts/'+self.payload.name)
        UsdGeom.Xformable(part).GetOrderedXformOps()[0].Set(Gf.Matrix4d(pose.T.tolist()))

    def release(self,tcp,destination,tolerance=.018):
        if self.payload is None:return False,'empty'
        pose=tcp@self.payload.local_pose
        if np.linalg.norm(pose[:3,3]-destination)>tolerance:return False,'outside_drop_zone'
        self.update(tcp)
        points=self.payload.local_points@tcp[:3,:3].T+tcp[:3,3]
        self.guard.parts[self.payload.name]=Hull.make('part/'+self.payload.name,points)
        self.stage.RemovePrim(self.joint_path)
        self.payload=None
        return True,'released_without_pose_jump'
