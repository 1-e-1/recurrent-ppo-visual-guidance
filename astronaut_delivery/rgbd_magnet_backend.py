"""Simulation-only magnetic actuator. Never a source of planner geometry.

The private scene determines physical proximity and carries the object by an
exact kinematic attachment. Public feedback is binary attachment state, not
object pose, shape, contact distance, or a collision/free-space certificate.
Payload inertia is deliberately not added to the reaction model.
"""
import numpy as np
from pxr import Gf,UsdGeom,UsdPhysics
from .motion import world


class KinematicMagnetBackend:
    def __init__(self,stage,capture_distance=.008):
        if not 0<capture_distance<=.008:raise ValueError('invalid physical capture distance')
        self._stage=stage;self._distance=capture_distance;self._name=None;self._relative=None
        self._joint='/World/RGBDDeliveryAttachment';self._generation=0

    @staticmethod
    def _pose(tcp):
        T=np.asarray(tcp,float)
        if (T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or
            not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(T[:3,:3])<.999999):
            raise ValueError('finite proper TCP transform required')
        return T

    def feedback(self,status='state'):
        return dict(attached=self._name is not None,generation=self._generation,status=status)

    def update(self,tcp):
        T=self._pose(tcp)
        if self._name is None:return
        part=self._stage.GetPrimAtPath('/World/Parts/'+self._name)
        ops=UsdGeom.Xformable(part).GetOrderedXformOps()
        if len(ops)!=1 or ops[0].GetOpType()!=UsdGeom.XformOp.TypeTransform:
            raise RuntimeError('kinematic part requires one transform op')
        ops[0].Set(Gf.Matrix4d((T@self._relative).T.tolist()))

    def command(self,action,tcp,target=None):
        T=self._pose(tcp)
        if action=='capture':
            if self._name is not None:return self.feedback('already_attached')
            if not isinstance(target,str) or not target or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in target):
                raise ValueError('invalid semantic target identifier')
            part=self._stage.GetPrimAtPath('/World/Parts/'+target)
            if not part or not part.GetAttribute('magnet:ferromagnetic').Get():return self.feedback('capture_failed')
            pose=np.array(world(part)).T;local=np.array(part.GetAttribute('magnet:contactLocal').Get())
            contact=(pose@np.r_[local,1])[:3]
            if np.linalg.norm(T[:3,3]-contact)>self._distance or np.dot(T[:3,2],pose[:3,2])>-.7:
                return self.feedback('capture_failed')
            joint=UsdPhysics.FixedJoint.Define(self._stage,self._joint)
            joint.CreateBody0Rel().SetTargets(['/World/RightMagnet']);joint.CreateBody1Rel().SetTargets([part.GetPath()])
            magnet=np.array(world(self._stage.GetPrimAtPath('/World/RightMagnet'))).T
            rel=Gf.Matrix4d((np.linalg.inv(magnet)@pose).T.tolist())
            joint.CreateLocalPos0Attr(Gf.Vec3f(rel.ExtractTranslation()));joint.CreateLocalRot0Attr(Gf.Quatf(rel.ExtractRotationQuat()))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0));joint.CreateLocalRot1Attr(Gf.Quatf(1))
            joint.CreateJointEnabledAttr(False);joint.CreateCollisionEnabledAttr(False)
            self._name=target;self._relative=np.linalg.inv(T)@pose;self._generation+=1
            return self.feedback('captured_kinematically')
        if action=='release':
            if self._name is None:return self.feedback('already_empty')
            self.update(T);self._stage.RemovePrim(self._joint)
            self._name=None;self._relative=None;self._generation+=1
            return self.feedback('released_without_pose_jump')
        raise ValueError('unsupported magnetic command')
