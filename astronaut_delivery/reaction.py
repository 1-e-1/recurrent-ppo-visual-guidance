"""Fixed-C-frame arm reaction estimator, separate from kinematic actuation.

Module masses are user-confirmed (2026-09-08). COM/principal inertias are imported
from USD and NOT independently measured. Tool, camera and payload inertia are
excluded by design; their collision geometry is not excluded.
"""
from dataclasses import dataclass
import numpy as np
from pxr import Gf,UsdPhysics,UsdGeom
from .motion import rot,world

def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

class DifferentialArm:
    def __init__(self,arm):self.arm=arm

    def evaluate(self,q):
        poses={'base/a1':self.arm.base};chains={'base/a1':[]};axes={};origins={}
        for parent,name,f0,f1,j,axis in self.arm.edges:
            joint=poses[parent]@f0
            if j>=0:
                axes[j]=joint[:3,:3]@axis;origins[j]=joint[:3,3]
            poses[name]=joint@(rot(axis,q[j]) if j>=0 else np.eye(4))@f1
            chains[name]=chains[parent]+([j] if j>=0 else [])
        return DifferentialState(poses,chains,axes,origins)

@dataclass
class DifferentialState:
    poses:dict
    chains:dict
    axes:dict
    origins:dict

    def point(self,name,local):
        t=self.poses[name];p=t[:3,:3]@local+t[:3,3]
        Jv=np.zeros((3,8));Jw=np.zeros((3,8))
        ids=self.chains[name]
        if ids:
            axes=np.array([self.axes[j] for j in ids]);origins=np.array([self.origins[j] for j in ids])
            Jw[:,ids]=axes.T;Jv[:,ids]=np.cross(axes,p-origins).T
        return p,Jv,Jw

class ArmReaction:
    def __init__(self,stage,arm,origin,rotation):
        self.arm=arm;self.tree=DifferentialArm(arm);self.origin=origin;self.rotation=rotation;self.bodies=[]
        for name in arm.names:
            prim=stage.GetPrimAtPath(arm.prefix+'/'+name)
            if not prim.HasAPI(UsdPhysics.MassAPI):raise ValueError('Missing module inertia: '+str(prim.GetPath()))
            api=UsdPhysics.MassAPI(prim);mass=float(api.GetMassAttr().Get());com=np.array(api.GetCenterOfMassAttr().Get(),float)
            diag=np.array(api.GetDiagonalInertiaAttr().Get(),float)
            P=np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(api.GetPrincipalAxesAttr().Get())))).T
            if not np.isfinite(np.r_[mass,com,diag]).all() or mass<=0 or min(diag)<=0 or 2*max(diag)>sum(diag)+1e-10:
                raise ValueError('Invalid physical inertia: '+name)
            self.bodies.append((name,mass,com,P@np.diag(diag)@P.T))
        # Fixed pairs share a screw chain. Aggregate them exactly with the
        # parallel-axis theorem, retaining raw bodies for the independent audit.
        zero=self.tree.evaluate(np.zeros(8));groups={}
        for name,m,c,I in self.bodies:groups.setdefault(tuple(zero.chains[name]),[]).append((name,m,c,I))
        self.composites=[]
        for chain,bodies in groups.items():
            anchor=bodies[0][0];inverse=np.linalg.inv(zero.poses[anchor]);items=[]
            for name,m,c,I in bodies:
                relative=inverse@zero.poses[name];R=relative[:3,:3]
                items.append((m,R@c+relative[:3,3],R@I@R.T))
            mass=sum(x[0] for x in items);center=sum(m*c for m,c,I in items)/mass
            inertia=sum(I+m*(np.dot(c-center,c-center)*np.eye(3)-np.outer(c-center,c-center)) for m,c,I in items)
            self.composites.append((anchor,mass,center,inertia))

    def matrix(self,q,state=None):
        state=state or self.tree.evaluate(q);A=np.zeros((6,8))
        for name,m,c,I in self.composites:
            p,Jv,Jw=state.point(name,c);R=state.poses[name][:3,:3]
            A[:3]+=m*Jv;A[3:]+=R@I@R.T@Jw+skew(p-self.origin)@(m*Jv)
        A[:3]=self.rotation.T@A[:3];A[3:]=self.rotation.T@A[3:]
        return A

    def newton_euler_reference(self,q,v,a,h=1e-4):
        """Independent Cartesian second-difference audit (not the online path)."""
        states=[self.tree.evaluate(q+v*t+.5*a*t*t) for t in (-h,0,h)]
        wrench=np.zeros(6)
        for name,m,c,Ilocal in self.bodies:
            transforms=[s.poses[name] for s in states];R=transforms[1][:3,:3]
            p=[t[:3,:3]@c+t[:3,3] for t in transforms]
            acceleration=(p[2]-2*p[1]+p[0])/(h*h)
            dR=(transforms[2][:3,:3]-transforms[0][:3,:3])/(2*h)
            ddR=(transforms[2][:3,:3]-2*R+transforms[0][:3,:3])/(h*h)
            W=dR@R.T;Alpha=ddR@R.T+dR@dR.T
            vee=lambda M:np.array([M[2,1]-M[1,2],M[0,2]-M[2,0],M[1,0]-M[0,1]])*.5
            omega=vee(W);alpha=vee(Alpha);I=R@Ilocal@R.T;force=m*acceleration
            wrench[:3]-=force;wrench[3:]-=I@alpha+np.cross(omega,I@omega)+np.cross(p[1]-self.origin,force)
        wrench[:3]=self.rotation.T@wrench[:3];wrench[3:]=self.rotation.T@wrench[3:]
        return wrench

class CombinedReaction:
    def __init__(self,stage,left,right):
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage),1.):raise ValueError('Expected metre stage')
        base=np.array(world(stage.GetPrimAtPath('/World/C_structure'))).T
        R=base[:3,:3]
        if not np.allclose(R.T@R,np.eye(3),atol=1e-6):raise ValueError('Scaled reference frame')
        self.models=[ArmReaction(stage,arm,base[:3,3],R) for arm in (left,right)]
        from .reaction_batch import BatchedArmReaction
        self.batch_models=[BatchedArmReaction(m) for m in self.models]

    def wrench_batch(self,q,v,a):
        q,v,a=(np.asarray(x,dtype=float) for x in (q,v,a))
        if q.ndim!=2 or q.shape[1]!=16 or v.shape!=q.shape or a.shape!=q.shape or not all(np.isfinite(x).all() for x in (q,v,a)):
            raise ValueError('expected finite q, v, a arrays [B,16]')
        return sum(m.wrench(q[:,i*8:(i+1)*8],v[:,i*8:(i+1)*8],a[:,i*8:(i+1)*8]) for i,m in enumerate(self.batch_models))

    def matrix(self,q):
        q=np.asarray(q).reshape(2,8)
        return np.concatenate([m.matrix(x) for m,x in zip(self.models,q)],axis=1)

    def coefficients(self,q,v):
        q=np.asarray(q).reshape(16);v=np.asarray(v).reshape(16)
        A=self.matrix(q);h=1e-5/max(1.,np.linalg.norm(v))
        bias=((self.matrix(q+h*v)-self.matrix(q-h*v))/(2*h))@v if np.any(v) else np.zeros(6)
        return A,bias

    def wrench(self,q,v,a):
        A,b=self.coefficients(q,v);return -(A@np.asarray(a).reshape(16)+b)

    def reference_wrench(self,q,v,a):
        return sum(m.newton_euler_reference(x,y,z) for m,x,y,z in zip(self.models,np.reshape(q,(2,8)),np.reshape(v,(2,8)),np.reshape(a,(2,8))))

    def manifest(self):
        return dict(reference='/World/C_structure',axes='C-local, fixed in an inertial frame',force_units='N',torque_units='N*m',
                    module_masses_user_confirmed=True,com_inertia_status='USD_IMPORT_NOT_INDEPENDENTLY_MEASURED',
                    module_mass_kg=[sum(b[1] for b in m.bodies) for m in self.models],
                    excluded_inertia=['camera','camera_adapter','magnetic_tool','payload'],
                    exclusions=['gravity','external contact impulse','astronaut motion','base flexibility','rotor dynamics not in CAD'])

def reaction_metrics(wrenches,dt,window_s=.25):
    w=np.asarray(wrenches,float).reshape(-1,6);dt=np.broadcast_to(np.asarray(dt,float),(len(w),))
    if not len(w) or not np.isfinite(w).all() or np.any(dt<=0):raise ValueError('Invalid wrench time series')
    result={}
    for name,x in [('force',w[:,:3]),('torque',w[:,3:])]:
        norm=np.linalg.norm(x,axis=1);times=np.r_[0,np.cumsum(dt)];integral=np.vstack([np.zeros(3),np.cumsum(x*dt[:,None],axis=0)])
        # Integrate exact piecewise-constant samples over each trailing window.
        beginning=np.maximum(0,times[1:]-window_s)
        initial=np.column_stack([np.interp(beginning,times,integral[:,j]) for j in range(3)])
        result[name]=dict(peak=float(norm.max()),rms=float(np.sqrt(np.sum(norm*norm*dt)/dt.sum())),
                          accumulated_magnitude=float(norm@dt),net_impulse=integral[-1].tolist(),
                          maximum_window_impulse=float(np.linalg.norm(integral[1:]-initial,axis=1).max()))
    result['window_s']=window_s
    return result
