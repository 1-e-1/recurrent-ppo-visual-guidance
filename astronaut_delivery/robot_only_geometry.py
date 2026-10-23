"""Known robot self geometry only. NEVER load astronaut/environment objects.

The C mounting interface belongs to the robot; the astronaut remains part of
the measured environment. Masked self returns become missing, not FREE.
"""
from types import MappingProxyType
import numpy as np
from pxr import UsdGeom
from .motion import CollisionGuard,Hull,vertices,transformed,world
from classical_fusion.self_filter import sphere_return_mask
from classical_fusion.rgbd_guard import robot_hulls


class RobotOnlyGuard(CollisionGuard):
    def __init__(self,stage,left,right,margin=.003):
        self.left,self.right,self.margin=left,right,margin
        self.parts=MappingProxyType({});self.checks=0;self.separation_axes={}
        p=stage.GetPrimAtPath('/World/C_structure/Geometry/mesh')
        v=transformed(vertices(p),np.array(world(p)))
        triangles=np.asarray(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        y=v[triangles].mean(1)[:,1];fixed=[]
        for name,mask in [('front',abs(y)<=.26),('left',y>.26),('right',y<-.26)]:
            if mask.any():fixed.append(Hull.make('/World/C_structure/'+name,v[triangles[mask]].reshape(-1,3)))
        self.fixed=fixed

    def mask_self_returns(self,depth,K,T,q):
        """Conservative FK-cover mask, with no sensor instance labels."""
        d=np.asarray(depth).copy();v,u=np.indices(d.shape)
        valid=np.isfinite(d)&(d>0);ids=np.flatnonzero(valid);z=d.flat[ids]
        p=np.column_stack([(u.flat[ids]-K[0,2])*z/K[0,0],-(v.flat[ids]-K[1,2])*z/K[1,1],-z])
        p=p@T[:3,:3].T+T[:3,3];mask=np.zeros(len(ids),bool)
        centers=[];radii=[]
        for arm,angles in zip((self.left,self.right),np.asarray(q).reshape(2,8)):
            centers.extend(arm.sphere_positions(angles));radii.extend(arm.radii+.012)
        broad=sphere_return_mask(p,centers,radii)
        # A cover sphere is sufficient for robot collision broad phase, but is
        # too large for sensor self removal near a grasp. Refine with known
        # ROBOT convex hulls so nearby environment measurements survive.
        if all(hasattr(arm,'hulls_world') for arm in (self.left,self.right)):
            candidates=np.flatnonzero(broad)
            for arm,angles in zip((self.left,self.right),np.asarray(q).reshape(2,8)):
                for _,(points,normals) in robot_hulls(arm,angles):
                    normals=normals/np.linalg.norm(normals,axis=1,keepdims=True)
                    select=candidates[((p[candidates]>=points.min(0)-.012)&(p[candidates]<=points.max(0)+.012)).all(1)]
                    support=(points@normals.T).max(0)+.012
                    for first in range(0,len(normals),24):
                        if not len(select):break
                        select=select[(p[select]@normals[first:first+24].T<=support[first:first+24]).all(1)]
                    mask[select]=True
        else:mask|=broad
        for hull in self.fixed:
            lo,hi=hull.points.min(0)-.012,hull.points.max(0)+.012
            inside=np.flatnonzero(((p>=lo)&(p<=hi)).all(1))
            if len(inside):mask[inside]|=(p[inside]@hull.equations[:,:3].T+hull.equations[:,3]<=.012).all(1)
        d.flat[ids[mask]]=0;return d
