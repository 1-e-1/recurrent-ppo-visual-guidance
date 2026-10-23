"""Batched exact rigid-body inertial reaction for a FIXED base, no gravity.

Analytic spatial recursion avoids differencing momentum matrices in each
Gauss-Newton perturbation. Kept separate from the independent audit path.
"""
import numpy as np


def mv(R,v):return np.einsum('bij,bj->bi',R,v)


class BatchedArmReaction:
    def __init__(self,model):
        self.model=model;self.edges=[]
        for parent,name,f0,f1,j,axis in model.arm.edges:
            x,y,z=axis
            K=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            self.edges.append((parent,name,f0,f1,j,axis,K,K@K))

    def wrench(self,q,v,a):
        n=len(q);base=self.model.arm.base
        zero=np.zeros((n,3));R0=np.broadcast_to(base[:3,:3],(n,3,3))
        # pose R,p, angular velocity, angular acceleration, origin acceleration
        states={'base/a1':(R0,np.broadcast_to(base[:3,3],(n,3)),zero,zero,zero)}
        for parent,name,f0,f1,j,axis,K,KK in self.edges:
            R,p,w,alpha,acc=states[parent]
            r0=np.einsum('bij,j->bi',R,f0[:3,3]);Rj=R@f0[:3,:3]
            pj=p+r0;aj=acc+np.cross(alpha,r0)+np.cross(w,np.cross(w,r0))
            if j>=0:
                axis_world=np.einsum('bij,j->bi',Rj,axis)
                wj=w+axis_world*v[:,j,None]
                alphaj=alpha+axis_world*a[:,j,None]+np.cross(w,axis_world)*v[:,j,None]
                angle=q[:,j,None,None]
                Rj=Rj@(np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*KK)
            else:wj,alphaj=w,alpha
            r1=np.einsum('bij,j->bi',Rj,f1[:3,3])
            ac=aj+np.cross(alphaj,r1)+np.cross(wj,np.cross(wj,r1))
            states[name]=(Rj@f1[:3,:3],pj+r1,wj,alphaj,ac)
        result=np.zeros((n,6))
        for name,mass,com,inertia in self.model.composites:
            R,p,w,alpha,acc=states[name]
            r=np.einsum('bij,j->bi',R,com);c=p+r
            force=mass*(acc+np.cross(alpha,r)+np.cross(w,np.cross(w,r)))
            I=R@inertia@R.transpose(0,2,1)
            torque=mv(I,alpha)+np.cross(w,mv(I,w))+np.cross(c-self.model.origin,force)
            result[:,:3]-=force;result[:,3:]-=torque
        result[:,:3]=result[:,:3]@self.model.rotation
        result[:,3:]=result[:,3:]@self.model.rotation
        return result
