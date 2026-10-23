"""USD-derived FK, covering-sphere collision guards and joint-space planning.

Known CAD geometry is a simulator safety model, NOT inferred RGB-D geometry.
Never treat successful sampled planning as continuous collision certification.
"""
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.cluster.vq import kmeans2
from pxr import Gf, Usd, UsdGeom, UsdPhysics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_astronaut_dual_arm import frame, world

def vertices(prim):
    if prim.IsA(UsdGeom.Mesh):return np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
    if prim.IsA(UsdGeom.Cube):
        size=UsdGeom.Cube(prim).GetSizeAttr().Get()
        return np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])*size
    if prim.IsA(UsdGeom.Cylinder):
        c=UsdGeom.Cylinder(prim);r,h=c.GetRadiusAttr().Get(),c.GetHeightAttr().Get()
        a=np.arange(32)*2*np.pi/32
        # Circumscribed polygon, so it contains the analytic cylinder.
        r/=np.cos(np.pi/32)
        p=np.array([[r*np.cos(t),r*np.sin(t),z] for z in (-h/2,h/2) for t in a])
        if c.GetAxisAttr().Get()=='X':p=p[:,[2,0,1]]
        if c.GetAxisAttr().Get()=='Y':p=p[:,[0,2,1]]
        return p
    return None

def transformed(points,row_matrix):
    return (np.c_[points,np.ones(len(points))]@row_matrix)[:,:3]

def rot(axis,q):
    c,s=np.cos(q),np.sin(q)
    u=np.array(axis);K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    m=np.eye(4);m[:3,:3]=np.eye(3)+s*K+(1-c)*(K@K)
    return m

def separated(va,na,vb,nb,margin):
    """Same face-axis SAT result, with an early small-subset rejection.

    No new separating axes or geometry approximation: all early axes are a
    subset of the original complete face-axis test. Magnet CAD has 3,639 faces.
    """
    for axes in (np.vstack([na[:3],nb[:3]]),np.vstack([na,nb])):
        a,b=va@axes.T,vb@axes.T
        if np.maximum(a.min(0)-b.max(0),b.min(0)-a.max(0)).max()>margin:return True
    return False

class Arm:
    def __init__(self,stage,side):
        self.side=side;self.prefix='/World/'+('LeftArm' if side=='left' else 'RightArm')
        self.base=np.array(world(stage.GetPrimAtPath(self.prefix+'/base/a1'))).T
        self.names=['base/a1']
        self.edges=[]
        known={'base/a1'};pending=[]
        for p in Usd.PrimRange(stage.GetPrimAtPath(self.prefix)):
            if p.IsA(UsdPhysics.Joint):
                j=UsdPhysics.Joint(p)
                a=str(j.GetBody0Rel().GetTargets()[0]).removeprefix(self.prefix+'/')
                b=str(j.GetBody1Rel().GetTargets()[0]).removeprefix(self.prefix+'/')
                index=int(p.GetName().replace('joint',''))-1 if p.IsA(UsdPhysics.RevoluteJoint) else -1
                axis=np.eye(3)['XYZ'.index(UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get())] if index>=0 else np.array([0,0,1])
                pending.append((a,b,np.array(frame(j,0)).T,np.linalg.inv(np.array(frame(j,1)).T),index,axis))
        while pending:
            ready=[e for e in pending if e[0] in known]
            if not ready:raise ValueError('Disconnected arm')
            for e in ready:self.edges.append(e);self.names.append(e[1]);known.add(e[1]);pending.remove(e)
        self.spheres=[]
        self.local_hulls={}
        for name in self.names:
            body=stage.GetPrimAtPath(self.prefix+'/'+name)
            inverse=np.linalg.inv(np.array(world(body)))
            group=0 if name=='base/a1' else int(name[1:])-(1 if name.startswith('a') else 0)
            for p in Usd.PrimRange(body):
                if not p.IsA(UsdGeom.Mesh) or p.GetName()!='visuals':continue
                mesh=UsdGeom.Mesh(p);points=transformed(vertices(p),np.array(world(p))@inverse)
                self.local_hulls[name]=Hull.make(name,points)
                faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
                centers,labels=kmeans2(points[faces].mean(1),6,minit='++',seed=17)
                for i,center in enumerate(centers):
                    assigned=points[faces[labels==i]].reshape(-1,3)
                    if not len(assigned):continue
                    radius=np.linalg.norm(assigned-center,axis=1).max()+1e-6
                    self.spheres.append((name,center,radius,group))
        self.tool_local=np.eye(4)
        self.tool_local[2,3]=.05288
        if side=='left':
            inv_tip=np.linalg.inv(np.array(world(stage.GetPrimAtPath(self.prefix+'/b8'))))
            camera=stage.GetPrimAtPath(self.prefix+'/b8/Camera')
            self.tool_local=(np.array(world(camera))@inv_tip).T@np.diag([1.,-1.,-1.,1.])
            for name in ('CameraHousing','CameraMount'):
                proxy=stage.GetPrimAtPath(self.prefix+'/b8/D435i/'+name)
                if not proxy:continue
                points=transformed(vertices(proxy),np.array(world(proxy))@inv_tip)
                center=(points.min(0)+points.max(0))/2
                self.local_hulls[name]=Hull.make(name,points)
                self.spheres.append((name,center,np.linalg.norm(points-center,axis=1).max()+1e-6,8))
        if side=='right':
            magnet=stage.GetPrimAtPath('/World/RightMagnet')
            inv_tip=np.linalg.inv(np.array(world(stage.GetPrimAtPath(self.prefix+'/b8'))))
            points=[];tool_faces=[];tool_offset=0
            for p in Usd.PrimRange(magnet):
                v=vertices(p)
                if v is not None:
                    points.append(transformed(v,np.array(world(p))@inv_tip))
                    if p.IsA(UsdGeom.Mesh):
                        mesh=UsdGeom.Mesh(p);indices=np.array(mesh.GetFaceVertexIndicesAttr().Get());offset=0;triangles=[]
                        for count in mesh.GetFaceVertexCountsAttr().Get():
                            polygon=indices[offset:offset+count];offset+=count
                            triangles.extend([[polygon[0],polygon[j],polygon[j+1]] for j in range(1,count-1)])
                        tool_faces.append(np.array(triangles,int)+tool_offset)
                    else:tool_faces.append(ConvexHull(v).simplices+tool_offset)
                    tool_offset+=len(v)
            points=np.concatenate(points);center=(points.min(0)+points.max(0))/2
            self.local_hulls['magnet']=Hull.make('magnet',points)
            from .robot_hull_cover import axial_cover
            self.tool_hull_cover=axial_cover(points,np.concatenate(tool_faces))
            self.spheres.append(('magnet',center,np.linalg.norm(points-center,axis=1).max(),8))
            self.tool_local[2,3]+=.0565
        self.group=np.array([x[3] for x in self.spheres]);self.radii=np.array([x[2] for x in self.spheres])
        self.sphere_names=[x[0] for x in self.spheres]
        self.sphere_local_centers=np.array([x[1] for x in self.spheres])
        self.self_pairs=np.array([(i,j) for i in range(len(self.spheres)) for j in range(i) if abs(self.group[i]-self.group[j])>1],int)
        self.chain_length=sum(np.linalg.norm(e[2][:3,3])+np.linalg.norm(e[3][:3,3]) for e in self.edges)
        self.motion_radius=self.chain_length+max(np.linalg.norm(h.points,axis=1).max() for h in self.local_hulls.values())
        self.prefix_length={'base/a1':0.};self.joint_prefix=np.zeros(8)
        for a,b,f0,f1,i,axis in self.edges:
            origin=self.prefix_length[a]+np.linalg.norm(f0[:3,3])
            if i>=0:self.joint_prefix[i]=origin
            self.prefix_length[b]=origin+np.linalg.norm(f1[:3,3])
        self.body_groups={name:group for name,center,radius,group in self.spheres}

    def relative_motion_bound(self,lower_group,higher_name,delta,payload=None):
        name='b8' if higher_name in ('magnet','payload','CameraHousing','CameraMount') else higher_name
        if payload is None:
            extent=np.linalg.norm(self.local_hulls[higher_name].points,axis=1).max()
            higher_group=self.body_groups[higher_name]
        else:
            extent=np.linalg.norm(self.tool_local[:3,3])+np.linalg.norm(payload.local_points,axis=1).max()
            higher_group=8
        reach=np.maximum(0.,self.prefix_length[name]-self.joint_prefix[lower_group:higher_group]+extent)
        return float(reach@delta[lower_group:higher_group])

    def absolute_motion_bound(self,name,delta,payload=None):
        if name=='base/a1':return 0.
        return self.relative_motion_bound(0,name,delta,payload)

    def hulls_world(self,q):
        poses=self.fk(q)
        result={}
        for name,h in self.local_hulls.items():
            t=poses['b8' if name in ('magnet','CameraHousing','CameraMount') else name]
            result[name]=(h.points@t[:3,:3].T+t[:3,3],h.equations[:,:3]@t[:3,:3].T)
        return result

    def collision_hulls_world(self,q):
        """Known robot convex components; repeated family names are deliberate."""
        hulls=self.hulls_world(q)
        if not hasattr(self,'tool_hull_cover'):return list(hulls.items())
        pose=self.fk(q)['b8'];R=pose[:3,:3];t=pose[:3,3]
        result=[(name,hull) for name,hull in hulls.items() if name!='magnet']
        result.extend(('magnet',(p@R.T+t,e[:,:3]@R.T)) for p,e in self.tool_hull_cover)
        return result

    def fk(self,q):
        q=np.asarray(q,float)
        if q.shape!=(8,) or not np.isfinite(q).all():raise ValueError('Expected finite q[8]')
        poses={'base/a1':self.base}
        for a,b,f0,f1,i,axis in self.edges:
            poses[b]=poses[a]@f0@(rot(axis,q[i]) if i>=0 else np.eye(4))@f1
        return poses

    def tcp(self,q):return self.fk(q)['b8']@self.tool_local

    def camera(self,q):
        m=self.tcp(q).copy();m[:3,:3]=m[:3,:3]@np.diag([1,-1,-1]);return m

    def sphere_positions(self,q):
        poses=self.fk(q)
        transforms=np.array([poses['b8' if n in ('magnet','CameraHousing','CameraMount') else n] for n in self.sphere_names])
        return np.einsum('nij,nj->ni',transforms[:,:3,:3],self.sphere_local_centers)+transforms[:,:3,3]

    def solve(self,target,axis=None,seed=None,tries=20,valid=None,rng=None,offset=None,rotation=None,max_delta=None,extra_residual=None,vertical_row=None,initial_points=None,orientation_deadband=False):
        rng=rng or np.random.default_rng(0);target=np.asarray(target,float)
        lower=np.full(8,-np.pi);upper=np.full(8,np.pi)
        if max_delta is not None:
            if seed is None or not np.isfinite(max_delta) or not 0<max_delta<=np.pi:raise ValueError('bounded IK requires seed and positive radius')
            center=np.asarray(seed,float)
            if center.shape!=(8,) or not np.isfinite(center).all() or np.any(abs(center)>np.pi):raise ValueError('invalid bounded IK seed')
            lower=np.maximum(lower,center-max_delta);upper=np.minimum(upper,center+max_delta)
        if initial_points is not None:
            initial_points=np.asarray(initial_points,float)
            if (initial_points.ndim!=2 or initial_points.shape[1]!=8 or len(initial_points)==0
                or not np.isfinite(initial_points).all() or np.any(initial_points<lower) or np.any(initial_points>upper)):
                raise ValueError('initial IK points must lie inside original joint trust region')
            tries=len(initial_points)
        if rotation is not None:
            rotation=np.asarray(rotation,float)
            if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<.99999:
                raise ValueError('proper requested tool rotation required')
        if vertical_row is not None:
            vertical_row=np.asarray(vertical_row,float)
            if (vertical_row.shape!=(3,) or not np.isfinite(vertical_row).all()
                or abs(np.linalg.norm(vertical_row)-1)>1e-6 or axis is not None or rotation is not None):
                raise ValueError('unit vertical row requires no competing axis or full rotation constraint')
        if np.linalg.norm(target-self.base[:3,3])>1.1:return None
        best=None
        for i in range(tries):
            initial=np.zeros(8) if seed is None else np.asarray(seed)
            if initial_points is not None:
                initial=np.clip(initial_points[i],lower+1e-9,upper-1e-9)
            elif i:
                scale=(.8 if i<tries/2 else 1.7) if max_delta is None else max_delta*(.2 if i<tries/2 else .5)
                initial=np.clip(initial+rng.normal(0,scale,8),lower+1e-9,upper-1e-9)
            def residual(q):
                t=self.tcp(q)@(np.eye(4) if offset is None else offset)
                r=[(t[:3,3]-target)*12]
                def orientation_error(error,band):
                    if not orientation_deadband:return error*2
                    norm=np.linalg.norm(error)
                    return error*(2*max(0.,norm-band)/max(norm,1e-12))
                if axis is not None:r.append(orientation_error(t[:3,2]-axis,.020))
                if rotation is not None:r.append(orientation_error((t[:3,:3]-rotation).ravel(),.008))
                if vertical_row is not None:r.append(orientation_error(t[2,:3]-vertical_row,.0008))
                if extra_residual is not None:r.append(np.asarray(extra_residual(q),float))
                r.append((q-initial)*.0005)
                return np.concatenate(r)
            fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=160,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            t=self.tcp(fit.x)@(np.eye(4) if offset is None else offset)
            if np.linalg.norm(t[:3,3]-target)>.003:continue
            if axis is not None and np.linalg.norm(t[:3,2]-axis)>.025:continue
            if rotation is not None and np.linalg.norm(t[:3,:3]-rotation)>.01:continue
            if vertical_row is not None and np.linalg.norm(t[2,:3]-vertical_row)>.001:continue
            if valid is not None and not valid(fit.x):continue
            score=np.linalg.norm(fit.x-(seed if seed is not None else 0))
            if best is None or score<best[0]:best=score,fit.x
            if best is not None and i>=3:return best[1]
        return None if best is None else best[1]

@dataclass
class Hull:
    name:str
    points:np.ndarray
    equations:np.ndarray

    @classmethod
    def make(cls,name,points):
        points=np.unique(points,axis=0)
        hull=ConvexHull(points)
        _,ids=np.unique(np.round(hull.equations,8),axis=0,return_index=True)
        equations=hull.equations[ids]
        first=np.argmax(np.abs(equations[:,:3]),axis=0)
        order=list(dict.fromkeys(first.tolist()+list(range(len(equations)))))
        return cls(name,points[hull.vertices],equations[order])

@dataclass
class Payload:
    name:str
    local_points:np.ndarray
    local_pose:np.ndarray

    @property
    def center(self):return (self.local_points.min(0)+self.local_points.max(0))/2
    @property
    def radius(self):return float(np.linalg.norm(self.local_points-self.center,axis=1).max())

class CollisionGuard:
    def __init__(self,stage,left,right,margin=.003):
        self.left,self.right,self.margin=left,right,margin
        self.fixed=[];self.parts={};self.checks=0;self.separation_axes={}
        for p in stage.Traverse():
            path=str(p.GetPath());v=vertices(p)
            if v is None or not p.HasAPI(UsdPhysics.CollisionAPI):continue
            if '/WorkPoseCollision/' in path or path.startswith('/World/Station/'):
                self.fixed.append(Hull.make(path,transformed(v,np.array(world(p)))))
        p=stage.GetPrimAtPath('/World/C_structure/Geometry/mesh')
        v=transformed(vertices(p),np.array(world(p)))
        triangles=np.asarray(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        center_y=v[triangles].mean(1)[:,1]
        for name,mask in [('front',abs(center_y)<=.26),('left',center_y>.26),('right',center_y<-.26)]:
            # Every complete triangle belongs to a hull, including boundary faces.
            self.fixed.append(Hull.make('/World/C_structure/'+name,v[triangles[mask]].reshape(-1,3)))
        for body in stage.GetPrimAtPath('/World/Parts').GetChildren():
            points=[]
            for p in Usd.PrimRange(body):
                v=vertices(p)
                if v is not None:points.append(transformed(v,np.array(world(p))))
            self.parts[body.GetName()]=Hull.make('part/'+body.GetName(),np.concatenate(points))

    def separated(self,key,va,na,vb,nb,margin):
        # A remembered axis is only a proposal: reproject ALL current vertices.
        # Any unit separating direction certifies clearance, even if it is no
        # longer a current face normal. No cached collision answer is trusted.
        axis=self.separation_axes.get(key)
        if axis is not None:
            a,b=va@axis,vb@axis
            if max(a.min()-b.max(),b.min()-a.max())>margin:return True
        for axes in (np.vstack([na[:3],nb[:3]]),np.vstack([na,nb])):
            a,b=va@axes.T,vb@axes.T
            gaps=np.maximum(a.min(0)-b.max(0),b.min(0)-a.max(0));i=int(np.argmax(gaps))
            if gaps[i]>margin:
                self.separation_axes[key]=axes[i].copy();return True
        return False

    def check(self,side,q,other_q,*,inflate=0.,contact_part=None,held=None,support_contact=False,motion_delta=None,other_inflate=0.,moving_part=None):
        self.checks+=1
        arm=self.left if side=='left' else self.right
        if np.asarray(q).shape!=(8,) or not np.isfinite(q).all() or (abs(np.asarray(q))>np.pi).any():return False,'joint_limit'
        p=arm.sphere_positions(q)
        body_bounds={n:inflate if motion_delta is None else arm.absolute_motion_bound(n,motion_delta) for n in set(arm.sphere_names)}
        r=arm.radii+self.margin+np.array([body_bounds[n] for n in arm.sphere_names])
        hulls=None
        pairs=arm.self_pairs
        if len(pairs):
            i,j=pairs.T
            bad=np.linalg.norm(p[i]-p[j],axis=1)<r[i]+r[j]
            if bad.any():
                hulls=arm.hulls_world(q)
                candidate={(arm.sphere_names[i[k]],arm.sphere_names[j[k]]) for k in np.flatnonzero(bad)}
                for a,b in candidate:
                    va,na=hulls[a];vb,nb=hulls[b]
                    bound=2*inflate
                    if motion_delta is not None:
                        ga,gb=arm.body_groups[a],arm.body_groups[b]
                        bound=arm.relative_motion_bound(min(ga,gb),a if ga>gb else b,motion_delta)
                    if not self.separated((side,a,b),va,na,vb,nb,2*self.margin+bound):return False,f'self/{a}/{b}'
        # Only the right arm takes responsibility for inter-arm avoidance.
        if side=='right':
            other=self.left.sphere_positions(other_q)
            if (np.linalg.norm(p[:,None]-other[None],axis=2)<r[:,None]+self.left.radii[None]+self.margin+other_inflate).any():return False,'inter_arm'
        names=arm.sphere_names
        if held is not None:
            local_center,radius,part_name=held.center,held.radius,held.name
            t=arm.tcp(q);center=(t@np.r_[local_center,1])[:3]
            payload_bound=inflate if motion_delta is None else arm.absolute_motion_bound('payload',motion_delta,held)
            # Carried object is part of the moving assembly, including self/inter-arm checks.
            distant=arm.group<7
            relative=np.full(int(distant.sum()),inflate)
            if motion_delta is not None:
                relative=np.array([arm.relative_motion_bound(int(g),'payload',motion_delta,held) for g in arm.group[distant]])
            payload_candidates=np.linalg.norm(p[distant]-center,axis=1)<arm.radii[distant]+radius+2*self.margin+relative
            if payload_candidates.any():
                # Spheres are only a broad phase: a long measured payload can
                # have a large covering sphere without touching a wrist link.
                # Refine with the same uncertainty-inclusive payload hull and
                # known robot hulls, retaining the relative interval bound.
                if hulls is None:hulls=arm.hulls_world(q)
                vp=held.local_points@t[:3,:3].T+t[:3,3]
                payload_normals=ConvexHull(vp).equations[:,:3]
                indices=np.flatnonzero(distant)[payload_candidates]
                for name in set(arm.sphere_names[i] for i in indices):
                    bound=inflate if motion_delta is None else arm.relative_motion_bound(arm.body_groups[name],'payload',motion_delta,held)
                    va,na=hulls[name]
                    if not self.separated((side,'payload',name),vp,payload_normals,va,na,2*self.margin+bound):
                        return False,'payload_self/'+name
            if side=='right' and (np.linalg.norm(other-center,axis=1)<self.left.radii+radius+self.margin+payload_bound+other_inflate).any():return False,'payload_inter_arm'
            p=np.vstack([p,center]);r=np.r_[r,radius+self.margin+payload_bound];names=names+['payload'];body_bounds['payload']=payload_bound
        else:part_name=None
        for hull in self.fixed+list(self.parts.values()):
            if hull.name=='part/'+str(part_name):continue
            if moving_part is not None and hull.name=='part/'+str(moving_part):continue
            mask=np.ones(len(p),bool)
            if '/C_structure/' in hull.name:
                # Only the actual installation-adjacent groups have allowed contact.
                mask[:len(arm.group)]&=arm.group>1
            if hull.name=='part/'+str(contact_part):
                mask&=np.array([n!='magnet' for n in names])
            if not mask.any():continue
            pts=p[mask];radii=r[mask]
            lo,hi=hull.points.min(0),hull.points.max(0)
            broad=((pts+radii[:,None]>=lo)&(pts-radii[:,None]<=hi)).all(1)
            if not broad.any():continue
            distance=(pts[broad]@hull.equations[:,:3].T+hull.equations[:,3]).max(1)
            potential=distance<radii[broad]
            if potential.any():
                if hulls is None:hulls=arm.hulls_world(q)
                indices=np.flatnonzero(mask)[np.flatnonzero(broad)[potential]]
                for name in set(names[i] for i in indices):
                    if name=='payload':
                        t=arm.tcp(q);vp=held.local_points@t[:3,:3].T+t[:3,3]
                        np_=ConvexHull(vp).equations[:,:3]
                        if support_contact and ('/Storage/Top' in hull.name or '/Storage/Pad' in hull.name or '/Operation/Top' in hull.name or '/Operation/Pad' in hull.name):
                            if vp[:,2].min()-body_bounds[name]>=hull.points[:,2].max()-.0025:continue
                    else:vp,np_=hulls[name]
                    if not self.separated((side,name,hull.name),vp,np_,hull.points,hull.equations[:,:3],2*self.margin+body_bounds[name]):return False,hull.name+'/'+name
        return True,'clear'

    def edge(self,side,a,b,other_q,*,continuous=False,resolution=.04,**kwargs):
        a,b=np.asarray(a),np.asarray(b)
        def safe(q,inflate=0.,delta=None):
            if not self.check(side,q,other_q,inflate=inflate,motion_delta=delta,**kwargs)[0]:return False
            # Left-arm planner itself has no mutual-collision objective. The
            # right safety supervisor still protects the stationary right arm
            # against the proposed left motion. Initial survey is empty-handed.
            if side=='left' and not self.check('right',other_q,q,other_inflate=inflate)[0]:return False
            return True
        if not continuous:
            steps=max(1,int(np.ceil(np.max(abs(b-a))/resolution)))
            return all(safe(a+(b-a)*t) for t in np.linspace(0,1,steps+1))
        # Triangle inequality on the entire USD joint chain bounds distance from
        # ANY joint to ANY moving point, independently of joint configuration.
        arm=self.left if side=='left' else self.right
        lever=arm.motion_radius
        held=kwargs.get('held')
        if held is not None:
            lever=max(lever,arm.chain_length+np.linalg.norm(arm.tool_local[:3,3])+np.linalg.norm(held.local_points,axis=1).max())
        # Nearest-endpoint displacement <= lever * ||dq||_1 / 2.
        def certify(x,y,depth):
            bound=lever*.5*float(np.abs(y-x).sum())
            delta=np.abs(y-x)*.5
            if safe(x,bound,delta) and safe(y,bound,delta):return True
            mid=(x+y)/2
            if depth>=16 or not safe(mid):return False
            return certify(x,mid,depth+1) and certify(mid,y,depth+1)
        return certify(a,b,0)

    def plan(self,side,start,goal,other_q,*,seed=0,iterations=1800,resolution=.04,**kwargs):
        start,goal=np.array(start),np.array(goal)
        if not self.check(side,start,other_q,**kwargs)[0] or not self.check(side,goal,other_q,**kwargs)[0]:return None
        if self.edge(side,start,goal,other_q,resolution=resolution,**kwargs):return np.array([start,goal])
        rng=np.random.default_rng(seed);nodes=[start];parents=[-1]
        for _ in range(iterations):
            sample=goal if rng.random()<.25 else rng.uniform(-np.pi,np.pi,8)
            i=int(np.argmin(np.linalg.norm(np.array(nodes)-sample,axis=1)))
            delta=sample-nodes[i];q=nodes[i]+delta*min(1,.35/max(np.linalg.norm(delta),1e-9))
            if not self.edge(side,nodes[i],q,other_q,resolution=resolution,**kwargs):continue
            nodes.append(q);parents.append(i)
            if not self.edge(side,q,goal,other_q,resolution=resolution,**kwargs):continue
            path=[goal];j=len(nodes)-1
            while j>=0:path.append(nodes[j]);j=parents[j]
            return np.array(path[::-1])
        return None
