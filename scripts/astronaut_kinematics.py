"""Exact first-order joint-graph FK for the astronaut USD (no dynamic solver).

In Isaac Sim Script Editor::
    import sys; sys.path.insert(0, r'D:\work_space\lunwen2\scripts')
    from astronaut_kinematics import DualArmKinematics
    import omni.usd
    robot = DualArmKinematics(omni.usd.get_context().get_stage())
    robot.set_positions([0.1]*8, [0.0]*8)  # radians, absolute positions
"""
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from build_astronaut_dual_arm import frame, transform, world, SAGITTAL_MIRROR, LOCAL_MIRROR

class DualArmKinematics:
    def __init__(self,stage):
        self.stage=stage
        self.edges=[]
        for p in stage.Traverse():
            if not p.IsA(UsdPhysics.Joint) or not p.GetAttribute('kinematics:enabled').Get():
                continue
            j=UsdPhysics.Joint(p)
            b0,b1=j.GetBody0Rel().GetTargets(),j.GetBody1Rel().GetTargets()
            self.edges.append((p,j,str(b0[0]) if b0 else '',str(b1[0]),frame(j,0),frame(j,1)))
        self.q=np.zeros((2,8),dtype=float)
        self.set_positions(self.q[0],self.q[1])

    @staticmethod
    def canonical_to_native(side,q):
        """Shared-planner coordinates -> native USD joint angles (also inverse)."""
        if side not in ('left','right'):
            raise ValueError('side must be left or right')
        q=np.asarray(q,dtype=float)
        if q.shape[-1:]!=(8,) or not np.isfinite(q).all():
            raise ValueError('Expected finite (...,8) joint angles')
        return q.copy() if side=='left' else -q

    def set_canonical_positions(self,left,right):
        """Equal canonical angles produce sagittal-mirrored arm configurations."""
        return self.set_positions(self.canonical_to_native('left',left),self.canonical_to_native('right',right))

    @staticmethod
    def mirror_world_points(points):
        """Mirror world targets AND obstacle geometry through the sagittal Y=0 plane."""
        p=np.asarray(points,dtype=float)
        if p.shape[-1:]!=(3,) or not np.isfinite(p).all():
            raise ValueError('Expected finite (...,3) points')
        return p@SAGITTAL_MIRROR[:3,:3]

    @staticmethod
    def mirror_world_pose(pose):
        """Row-vector USD RH pose reflection, compensating the local Y axis.

        This maps corresponding arm-link frames, NOT unlike camera/magnet TCPs.
        For column-vector robotics libraries pass pose.T and transpose back.
        """
        p=np.asarray(pose,dtype=float)
        if p.shape!=(4,4) or not np.isfinite(p).all():
            raise ValueError('Expected a finite 4x4 USD row-vector pose')
        return LOCAL_MIRROR@p@SAGITTAL_MIRROR

    def set_positions(self,left,right,time=Usd.TimeCode.Default()):
        q=np.asarray([left,right],dtype=float)
        if q.shape!=(2,8) or not np.isfinite(q).all():
            raise ValueError('Expected finite radians with shape (2,8).')
        # Conservative software bound, NOT a manufacturer-certified joint limit.
        q=np.clip(q,-np.pi,np.pi)
        poses={'':Gf.Matrix4d(1)}
        pending=list(self.edges)
        while pending:
            progress=False
            for edge in pending[:]:
                p,j,b0,b1,f0,f1=edge
                if b0 not in poses:
                    continue
                motion=Gf.Matrix4d(1)
                if p.IsA(UsdPhysics.RevoluteJoint):
                    side=0 if '/LeftArm/' in str(p.GetPath()) else 1
                    index=int(p.GetName().replace('joint',''))-1
                    angle=q[side,index]
                    rev=UsdPhysics.RevoluteJoint(p)
                    low,high=rev.GetLowerLimitAttr().Get(),rev.GetUpperLimitAttr().Get()
                    angle=float(np.clip(angle,np.deg2rad(low),np.deg2rad(high)))
                    q[side,index]=angle
                    axis={'X':Gf.Vec3d(1,0,0),'Y':Gf.Vec3d(0,1,0),'Z':Gf.Vec3d(0,0,1)}[rev.GetAxisAttr().Get()]
                    motion.SetRotate(Gf.Rotation(axis,float(np.rad2deg(angle))))
                poses[b1]=f1.GetInverse()*motion*f0*poses[b0]
                prim=self.stage.GetPrimAtPath(b1)
                parent=world(prim.GetParent())
                local=poses[b1]*parent.GetInverse()
                xf=UsdGeom.Xformable(prim)
                if not prim.HasAttribute('xformOp:transform'):
                    transform(prim,local)
                prim.GetAttribute('xformOp:transform').Set(local,time)
                pending.remove(edge)
                progress=True
            if not progress:
                raise RuntimeError('Joint graph is disconnected or cyclic.')
        self.q=q.copy()
        return poses

    def step_towards(self,left,right,dt=1/60,max_speed=.25):
        """Optional joint rate clipping (rad/s); collision avoidance is separate."""
        if not np.isfinite(dt) or dt<=0 or not np.isfinite(max_speed) or max_speed<0:
            raise ValueError('Invalid control timestep/speed.')
        target=np.asarray([left,right],dtype=float)
        if target.shape!=(2,8) or not np.isfinite(target).all():
            raise ValueError('Expected finite radians with shape (2,8).')
        q=self.q+np.clip(target-self.q,-max_speed*dt,max_speed*dt)
        return self.set_positions(*q)
