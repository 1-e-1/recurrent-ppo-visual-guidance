"""Scalar quintic clock on a continuously certified joint-space line.

Geometry and timing are separate: re-timing and emergency braking cannot leave
the certified segment. Exact polynomial extrema, not discrete limit sampling.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import polynomial as poly

def candidates(c,T,derivative=0):
    c=poly.polyder(c,derivative);roots=poly.polyroots(poly.polyder(c)) if len(c)>1 else []
    return np.array([0.,T]+[float(r.real) for r in roots if abs(r.imag)<1e-8 and 0<r.real<T])

@dataclass
class SmoothPath:
    start:np.ndarray
    end:np.ndarray
    duration:float
    coefficients:np.ndarray

    @classmethod
    def make(cls,start,end,vmax=.3,amax=.5,jmax=5.,minimum=.1):
        start,end=np.asarray(start,float).reshape(16),np.asarray(end,float).reshape(16)
        d=float(np.max(abs(end-start)))
        T=max(minimum,1.875*d/vmax,np.sqrt(5.773502692*d/amax),np.cbrt(60*d/jmax))
        return cls(start.copy(),end.copy(),T,np.array([0,0,0,10/T**3,-15/T**4,6/T**5]))

    def scalar(self,t,derivative=0):return poly.polyval(np.clip(t,0,self.duration),poly.polyder(self.coefficients,derivative))
    def state(self,t):
        delta=self.end-self.start
        return self.start+delta*self.scalar(t),delta*self.scalar(t,1),delta*self.scalar(t,2),delta*self.scalar(t,3)
    def bounds(self):
        d=float(np.max(abs(self.end-self.start)))
        return [d*float(np.max(abs(self.scalar(candidates(self.coefficients,self.duration,k),k)))) for k in (1,2,3)]
    def retime(self,factor):
        if not np.isfinite(factor) or factor<1:raise ValueError('Only slowing down is allowed')
        return SmoothPath(self.start.copy(),self.end.copy(),self.duration*factor,self.coefficients/factor**np.arange(6))
    def stop(self,t,vmax=.3,amax=.5,jmax=5.):
        """Choose a monotone C2 brake on the same line; original suffix is fallback.

        No acceleration jump. Geometric safety is inherited ONLY if the original
        line's certificate is still valid for the unchanged scene/payload.
        """
        t=float(np.clip(t,0,self.duration));s,v,a=[self.scalar(t,k) for k in (0,1,2)]
        remaining=self.duration-t
        if remaining<1e-9:return None
        for T in np.linspace(min(.15,remaining),remaining,24):
            dest=s+v*T*.5+a*T*T/12
            if not s-1e-10<=dest<=1+1e-10:continue
            c=np.zeros(6);c[:3]=[s,v,a/2]
            B=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
            c[3:]=np.linalg.solve(B,[dest-s-v*T-a*T*T/2,-v-a*T,-a])
            path=SmoothPath(self.start,self.end,T,c)
            velocity=path.scalar(candidates(c,T,1),1)
            if velocity.min()<-1e-9:continue
            if np.all(np.array(path.bounds())<=np.array([vmax,amax,jmax])+1e-8):return path
        # Translate the original polynomial to t=0, retaining its complete
        # certified suffix. This is a bounded stop, not an instantaneous hold.
        c=np.array([poly.polyval(t,poly.polyder(self.coefficients,k))/__import__('math').factorial(k) for k in range(6)])
        return SmoothPath(self.start,self.end,remaining,c)

def reaction_scale(path,model,force=.35,torque=.15,samples=21):
    """Sample-based design target, NOT a hard continuous wrench certificate."""
    times=np.linspace(0,path.duration,samples)
    W=np.array([model.wrench(*path.state(t)[:3]) for t in times])
    factor=max(1.,np.sqrt(np.linalg.norm(W[:,:3],axis=1).max()/force),np.sqrt(np.linalg.norm(W[:,3:],axis=1).max()/torque))
    return path.retime(factor*1.02),W
