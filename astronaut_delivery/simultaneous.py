"""Continuous joint-box certificates for TWO moving arms, not static-other tests."""
import numpy as np
from .motion import Arm
from .reaction import DifferentialArm

class TightBoundArm(Arm):
    """Configuration-dependent, conservative bounds; no linear-only heuristic.

    Apply perturbations distal-to-proximal. Each joint's current lever radius
    can grow by at most the OLD proven downstream displacement bound. Summing
    delta_j * (current_radius_j + downstream_bound_j) therefore bounds the whole
    box, including nonlinear cross terms. Module-cover spheres bound all points.
    """
    def __init__(self,stage,side):
        super().__init__(stage,side);self._bound_key=None;self._lever_cache={};self._tree=DifferentialArm(self)
    def prepare_motion_bounds(self,q):
        q=np.asarray(q,float);key=q.tobytes()
        if key==self._bound_key:return
        self._bound_key=key;self._lever_cache={};self._state=self._tree.evaluate(q)
        self._centers=Arm.sphere_positions(self,q)
    def sphere_positions(self,q):
        self.prepare_motion_bounds(q);return self._centers
    def relative_motion_bound(self,lower_group,higher_name,delta,payload=None):
        original=super().relative_motion_bound(lower_group,higher_name,delta,payload)
        if self._bound_key is None or payload is not None:return original
        high=self.body_groups[higher_name]
        if higher_name not in self._lever_cache:
            mask=np.array(self.sphere_names)==higher_name;p=self._centers[mask];r=self.radii[mask]
            radii=[]
            for j in range(high):
                lever=np.linalg.norm(np.cross(p-self._state.origins[j],self._state.axes[j]),axis=1)+r
                radii.append(float(lever.max()))
            self._lever_cache[higher_name]=radii
        tight=0.
        for j in range(lower_group,high):
            downstream=super().relative_motion_bound(j+1,higher_name,delta)
            tight+=delta[j]*(self._lever_cache[higher_name][j]+downstream)
        return min(original,float(tight))

class SimultaneousGuard:
    def __init__(self,guard):self.guard=guard;self.certificates=0;self.box_checks=0

    def box(self,lo,hi,held=None,contact_part=None,support_contact=False):
        lo=np.asarray(lo,float).reshape(2,8);hi=np.asarray(hi,float).reshape(2,8)
        if not np.isfinite(np.r_[lo.ravel(),hi.ravel()]).all() or np.any(lo>hi) or np.any(lo<-np.pi) or np.any(hi>np.pi):return False,'joint_box'
        q=(lo+hi)/2;delta=(hi-lo)/2
        for arm,value in zip((self.guard.left,self.guard.right),q):
            if hasattr(arm,'prepare_motion_bounds'):arm.prepare_motion_bounds(value)
        left_bound=max(self.guard.left.absolute_motion_bound(n,delta[0]) for n in self.guard.left.local_hulls)
        self.box_checks+=1
        ok,why=self.guard.check('left',q[0],q[1],motion_delta=delta[0],moving_part=held.name if held else None)
        if not ok:return False,'left/'+why
        return self.guard.check('right',q[1],q[0],motion_delta=delta[1],other_inflate=left_bound,
                                held=held,contact_part=contact_part,support_contact=support_contact)

    def segment(self,start,end,max_depth=12,**kwargs):
        a=np.asarray(start).reshape(16);b=np.asarray(end).reshape(16)
        def visit(x,y,depth):
            if self.box(np.minimum(x,y),np.maximum(x,y),**kwargs)[0]:return True
            mid=(x+y)/2
            if depth>=max_depth or not self.box(mid,mid,**kwargs)[0]:return False
            return visit(x,mid,depth+1) and visit(mid,y,depth+1)
        ok=visit(a,b,0);self.certificates+=int(ok);return ok

    def acceleration(self,q,v,a,dt,max_depth=8,**kwargs):
        q,v,a=[np.asarray(x,float).reshape(16) for x in (q,v,a)]
        if not np.isfinite(np.r_[q,v,a,dt]).all() or dt<=0:return False
        def position(t):return q+t*v+.5*t*t*a
        def visit(t0,t1,depth):
            p0,p1=position(t0),position(t1);lo=np.minimum(p0,p1);hi=np.maximum(p0,p1)
            extremum=np.divide(-v,a,out=np.zeros(16),where=abs(a)>1e-15)
            inside=(abs(a)>1e-15)&(extremum>t0)&(extremum<t1)
            p=position(np.clip(extremum,t0,t1));lo=np.where(inside,np.minimum(lo,p),lo);hi=np.where(inside,np.maximum(hi,p),hi)
            if self.box(lo,hi,**kwargs)[0]:return True
            if depth>=max_depth:return False
            return visit(t0,(t0+t1)/2,depth+1) and visit((t0+t1)/2,t1,depth+1)
        ok=visit(0.,dt,0);self.certificates+=int(ok);return ok

    def brake(self,q,v,deceleration=.5,**kwargs):
        """A common stop duration makes the geometric brake path a line segment.

        Finite emergency deceleration, NOT a certified hardware jerk-limited stop.
        Returned (duration, acceleration) is executable, not an instantaneous hold.
        """
        if not np.isfinite(deceleration) or deceleration<=0:raise ValueError('Invalid deceleration')
        q=np.asarray(q).reshape(16);v=np.asarray(v).reshape(16)
        duration=float(np.max(abs(v))/deceleration)
        if duration<1e-12:return (0.,np.zeros(16)) if self.box(q,q,**kwargs)[0] else None
        if self.segment(q,q+.5*duration*v,**kwargs):return duration,-v/duration
        return None
