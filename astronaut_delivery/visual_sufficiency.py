"""Eight-variable visual-set momentum allocation; no trained components.

Visibility constraints are geometric/measurement surrogates, not an occlusion
certificate. The caller must independently certify the joint path and maintain
fresh image feedback. This module never reads ground-truth object poses.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
import osqp
from .reaction import DifferentialArm,skew


@dataclass(frozen=True)
class SufficiencyConfig:
    joint_step:float=.12
    minimum_pixels:int=12
    maximum_sigma_m:float=.005
    maximum_age_s:float=.25
    minimum_range_m:float=.45
    maximum_range_m:float=1.2
    minimum_elevation:float=.30
    image_half_width:float=.60
    image_half_height:float=.33
    cap_radius_m:float=.020
    sigma_multiplier:float=3.
    regularization:float=.001
    iterations:int=400
    objective:str='momentum'
    accept_feasible_iterate:bool=False
    endpoint_reserve:float=0.


def reaction_authority(A_left,b,task_jacobian=None):
    """Unbounded linear least-squares LOWER BOUND, not feasible minimum wrench.

    An active inequality is one-sided; treating it as an equality only describes
    its tangent face, not the entire feasible cone. Never call rank a guarantee.
    """
    N=np.eye(8) if task_jacobian is None else np.eye(8)-np.linalg.pinv(task_jacobian,rcond=1e-6)@task_jacobian
    B=A_left@N;residual=(np.eye(6)-B@np.linalg.pinv(B,rcond=1e-6))@b
    return dict(reaction_rank=int(np.linalg.matrix_rank(B,tol=1e-6)),
                unbounded_linear_residual=float(np.linalg.norm(residual)),
                available_tangent_dof=int(round(np.trace(N))))


def projected_support_proxy(camera,target):
    """Uncalibrated horizontal-patch pixels proxy e/r^2, not visibility."""
    delta=np.asarray(camera,float)-np.asarray(target,float);distance=np.linalg.norm(delta)
    if distance<=.05:return 0.
    return max(0.,float(delta[2]))/distance**3


def feasible_ray(x,C,lower,upper):
    """Contract a candidate toward a FEASIBLE zero increment in O(m*n).

    For a convex linear feasible set containing zero, every point on the
    retained ray is feasible. This is not a projection in Euclidean distance.
    Returns zero if no positive feasible fraction exists; never relaxes limits.
    """
    x=np.asarray(x,float);C=np.asarray(C,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    if not np.isfinite(np.r_[x,C.ravel()]).all() or np.any(np.isnan(lower)) or np.any(np.isnan(upper)):raise ValueError('invalid ray data')
    if np.any(lower>0) or np.any(upper<0):raise ValueError('zero increment is not feasible')
    value=C@x;ratios=[1.]
    positive=value>1e-14;negative=value<-1e-14
    if positive.any():ratios.extend((upper[positive]/value[positive]).tolist())
    if negative.any():ratios.extend((lower[negative]/value[negative]).tolist())
    factor=float(np.clip(min(ratios),0,1))
    if factor<1:factor*=1-1e-8
    return x*factor,factor


class VisualSetAllocator:
    def __init__(self,left,reaction,cfg=None):
        self.left=left;self.reaction=reaction;self.cfg=cfg or SufficiencyConfig();self.tree=DifferentialArm(left)
        if not 0<=self.cfg.endpoint_reserve<=.05:raise ValueError('invalid visual endpoint reserve')

    def endpoint_valid(self,start,end,context):
        reserve=np.minimum(self.margins(start,context),self.cfg.endpoint_reserve)
        return bool(np.all(self.margins(end,context)>=reserve-1e-6))

    def features(self,q,target):
        s=self.tree.evaluate(q);cp,Jv,Jw=s.point('b8',self.left.tool_local[:3,3])
        R=(s.poses['b8']@self.left.tool_local)[:3,:3]@np.diag([1.,-1.,-1.])
        p=R.T@(target-cp);z=-p[2];d=np.linalg.norm(target-cp)
        if z<=.05 or d<=.05:raise ValueError('target_behind_camera')
        dp=-R.T@Jv+R.T@skew(target-cp)@Jw
        project=np.array([[1/z,0,p[0]/z**2],[0,-1/z,-p[1]/z**2]])
        J=np.vstack([project@dp,(cp-target)/d@Jv,
                     (np.array([0,0,1])/d-(cp[2]-target[2])*(cp-target)/d**3)@Jv])
        return np.array([p[0]/z,-p[1]/z,d,(cp[2]-target[2])/d]),J,np.vstack([Jv,Jw]),z

    def context(self,q,target,covariance,pixels,age,*,require_feasible=True):
        cfg=self.cfg;target=np.asarray(target,float);cov=np.asarray(covariance,float)
        if target.shape!=(3,) or cov.shape!=(3,3) or not np.isfinite(np.r_[target,cov.ravel(),pixels,age]).all():raise ValueError('invalid_measurement')
        if not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min()<-1e-12:raise ValueError('invalid_covariance')
        sigma=float(np.sqrt(max(0,np.linalg.eigvalsh(cov).max())))
        if not 0<=age<=cfg.maximum_age_s or pixels<cfg.minimum_pixels or sigma>cfg.maximum_sigma_m:raise ValueError('insufficient_measured_quality')
        f,J,Jpose,z=self.features(q,target)
        # Predict distance-sensitive depth sigma and projected disk support.
        # These are explicitly uncalibrated local surrogates; not confidence bounds.
        dmax=min(cfg.maximum_range_m,f[2]*np.sqrt(cfg.maximum_sigma_m/max(sigma,1e-8)))
        context=dict(target=target,sigma=sigma,pixels=int(pixels),initial=f.copy(),maximum_range=dmax)
        if require_feasible and np.min(self.margins(q,context))<0:raise ValueError('initial_view_outside_visual_set')
        return context

    def restore_view(self,q,target,covariance,pixels,age):
        """Smallest local increment restoring the UNCHANGED visual set.

        Sensing-only proposal with execution arm stationary. Does not certify
        collision or whole-path operational visibility. The caller rechecks
        actual endpoint quality and all environment/robot geometry.
        """
        q=np.asarray(q,float);context=self.context(q,target,covariance,pixels,age,require_feasible=False)
        base=self.margins(q,context);h=1e-5
        G=np.column_stack([(self.margins(q+np.eye(8)[j]*h,context)-self.margins(q-np.eye(8)[j]*h,context))/(2*h) for j in range(8)])
        C=np.vstack([np.eye(8),G]);lo=np.r_[np.maximum(-self.cfg.joint_step,-np.pi-q),.005-base]
        hi=np.r_[np.minimum(self.cfg.joint_step,np.pi-q),np.full(len(base),np.inf)]
        solver=osqp.OSQP();solver.setup(P=sparse.eye(8,format='csc')*2,q=np.zeros(8),A=sparse.csc_matrix(C),l=lo,u=hi,
            verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=self.cfg.iterations,polish=False)
        result=solver.solve()
        if result.info.status_val not in (1,2,osqp.constant('OSQP_MAX_ITER_REACHED')):return None
        if result.x is None or not np.isfinite(result.x).all():return None
        value=C@result.x
        if max(0.,float(np.max(lo-value)),float(np.max(value-hi)))>1e-6:return None
        candidate=q+result.x
        try:self.context(candidate,target,covariance,pixels,age)
        except ValueError:return None
        return candidate

    def margins(self,q,context):
        cfg=self.cfg;f,_,_,z=self.features(q,context['target']);d=f[2]
        support=(cfg.cap_radius_m+cfg.sigma_multiplier*context['sigma'])/z
        pixels=context['pixels']*(context['initial'][2]/d)**2*f[3]/max(context['initial'][3],1e-8)
        return np.array([cfg.image_half_width-support-f[0],cfg.image_half_width-support+f[0],
                         cfg.image_half_height-support-f[1],cfg.image_half_height-support+f[1],
                         d-cfg.minimum_range_m,context['maximum_range']-d,
                         f[3]-cfg.minimum_elevation,(pixels-cfg.minimum_pixels)/max(context['pixels'],cfg.minimum_pixels)])

    def path_valid(self,start,end,context,samples=9):
        try:return all(np.min(self.margins(start+u*(end-start),context))>=-1e-6 for u in np.linspace(0,1,samples))
        except ValueError:return False

    def propose(self,q,right_goal,context,mode='set'):
        if mode not in ('set','pose'):raise ValueError('unknown allocation mode')
        start=time.perf_counter();q=np.asarray(q,float).reshape(2,8);cfg=self.cfg
        A=self.reaction.matrix(q.ravel());D=np.diag([1.,1.,1.,10.,10.,10.]);B=D@A[:,:8];b=D@A[:,8:]@(right_goal-q[1])
        base=self.margins(q[0],context);h=1e-5
        G=np.column_stack([(self.margins(q[0]+np.eye(8)[j]*h,context)-self.margins(q[0]-np.eye(8)[j]*h,context))/(2*h) for j in range(8)])
        reserve=np.minimum(base,cfg.endpoint_reserve)
        rows=[np.eye(8),G];lo=[np.maximum(-cfg.joint_step,-np.pi-q[0]),reserve-base];hi=[np.minimum(cfg.joint_step,np.pi-q[0]),np.full(len(base),np.inf)]
        _,_,Jpose,_=self.features(q[0],context['target'])
        if mode=='pose':
            rows.append(Jpose);lo.append(np.zeros(6));hi.append(np.zeros(6))
        M,y=B,b
        if cfg.objective=='wrench':
            # Gauss-Newton model of the actual six-axis reaction on a unit-time
            # quintic. Common time scaling multiplies every candidate by 1/T^2.
            # This is a bounded, LOW-frequency planning computation, not 60 Hz.
            # All 51 states share one vectorized analytic tree traversal.
            step=1e-4;delta=np.vstack([np.zeros(8),np.eye(8)*step,-np.eye(8)*step])
            delta=np.column_stack([delta,np.broadcast_to(right_goal-q[1],(17,8))])
            u=np.array([.2113248654,.5,.7886751346])
            s=10*u**3-15*u**4+6*u**5;ds=30*u**2-60*u**3+30*u**4;dds=60*u-180*u**2+120*u**3
            states=[(q.ravel()[None,None,:]+delta[:,None,:]*s[None,:,None]).reshape(-1,16),
                    (delta[:,None,:]*ds[None,:,None]).reshape(-1,16),
                    (delta[:,None,:]*dds[None,:,None]).reshape(-1,16)]
            values=self.reaction.wrench_batch(*states).reshape(17,3,6)*np.array([1/.35]*3+[1/.15]*3)
            residuals=values.reshape(17,18);y=residuals[0];M=((residuals[1:9]-residuals[9:17])/(2*step)).T
        elif cfg.objective!='momentum':raise ValueError('unknown reaction objective')
        H=2*(M.T@M+cfg.regularization*np.eye(8));g=2*M.T@y
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(np.triu(H)),q=g,A=sparse.csc_matrix(np.vstack(rows)),l=np.concatenate(lo),u=np.concatenate(hi),
                                       verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=cfg.iterations,polish=False)
        result=solver.solve();delta=np.zeros(8);accepted=False;violation=None;ray_scale=1.
        statuses=(1,2,osqp.constant('OSQP_MAX_ITER_REACHED')) if cfg.accept_feasible_iterate else (1,2)
        if result.info.status_val in statuses and result.x is not None and np.isfinite(result.x).all():
            x=result.x
            if cfg.accept_feasible_iterate and mode=='set':
                x,ray_scale=feasible_ray(x,np.vstack(rows),np.concatenate(lo),np.concatenate(hi))
            constraints=np.vstack(rows)@x
            violation=float(max(0,np.max(np.concatenate(lo)-constraints),np.max(constraints-np.concatenate(hi))))
            merit=float(np.linalg.norm(M@x+y)**2+cfg.regularization*np.dot(x,x))
            if violation<=1e-6 and merit<=float(np.dot(y,y))+1e-8:
                for scale in (1.,.5,.25,.125):
                    trial=x*scale
                    if self.path_valid(q[0],q[0]+trial,context) and self.endpoint_valid(q[0],q[0]+trial,context):delta=trial;accepted=True;break
        authority=reaction_authority(B,b,Jpose if mode=='pose' else None)
        active=G[self.margins(q[0]+delta,context)<.002]
        return q[0]+delta,dict(mode=mode,objective=cfg.objective,qp_status=result.info.status,qp_iterations=int(result.info.iter),accepted=accepted,
                              accepted_suboptimal=bool(accepted and result.info.status_val not in (1,2)),linear_constraint_violation=violation,
                              feasible_ray_scale=ray_scale,
                              endpoint_reserve=cfg.endpoint_reserve,
                              optimizer_wall_s=time.perf_counter()-start,initial_proxy=float(np.linalg.norm(b)),
                              final_proxy=float(np.linalg.norm(B@delta+b)),delta_rad=delta.tolist(),
                              initial_objective=float(np.linalg.norm(y)),linearized_final_objective=float(np.linalg.norm(M@delta+y)),
                              minimum_sampled_visual_margin=float(min(self.margins(q[0]+u*delta,context).min() for u in np.linspace(0,1,9))),
                              active_visual_rows=len(active),active_visual_rank=int(np.linalg.matrix_rank(active,tol=1e-5)) if len(active) else 0,
                              authority=authority,scope='Local reaction allocation; sampled geometric visual surrogates, not occlusion/recursive-feasibility certificate')
