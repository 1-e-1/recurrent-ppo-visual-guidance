"""RGB-D-only detection of visible white square placement markings.

Known marking appearance/scale is a task-interface prior. No USD paths,
destination coordinates, instance IDs, or object count enter the estimator.
"""
import numpy as np
from scipy.spatial import cKDTree
from .vision import TargetEstimate


def fit_marking_edges(points,initial,half_size=.045):
    """Fit the declared axis-aligned square MARKING, not object geometry.

    At least three observed bar identities and both coordinate directions are
    required. Covariance describes estimated center precision, not the spatial
    spread of a 90 mm square divided by pixel count. Sensor/common-mode floors
    are added separately; this local model is not hardware calibrated.
    """
    xy=np.asarray(points,float)[:,:2];initial=np.asarray(initial,float)[:2];center=initial.copy()
    if len(xy)<30 or not np.isfinite(xy).all():return None
    for _ in range(4):
        offset=xy-center;axis=np.argmin(abs(abs(offset)-half_size),axis=1)
        sign=np.where(offset[np.arange(len(xy)),axis]>=0,1.,-1.)
        residual=offset[np.arange(len(xy)),axis]-sign*half_size
        valid=abs(residual)<.008
        labels=axis*2+(sign>0)
        if valid.mean()<.8 or sum(np.sum(valid&(labels==k))>=4 for k in range(4))<3:return None
        updated=center.copy();variance=np.zeros(2)
        for j in range(2):
            keep=valid&(axis==j)
            if keep.sum()<8:return None
            samples=xy[keep,j]-sign[keep]*half_size
            updated[j]=np.median(samples)
            sigma=max(.004/np.sqrt(12),1.4826*np.median(abs(samples-updated[j])))
            variance[j]=(np.pi/2)*sigma*sigma/len(samples)
        if np.max(abs(updated-initial))>.015:return None
        center=updated
    return center,np.diag(variance)


def placement_measurements(obs,diagnostics=None,priors=()):
    d=obs.depth;v,u=np.indices(d.shape);valid=np.isfinite(d)&(d>.1)
    if obs.robot_mask is not None:valid &= ~obs.robot_mask
    if obs.color_valid is not None:valid &= obs.color_valid
    color=np.asarray(obs.rgb);paint=np.array([.75,.85,.9])
    mask=valid&(np.linalg.norm(color-paint,axis=2)<.09)
    yy,xx=np.where(mask)
    if diagnostics is not None:diagnostics['white_pixels']=len(xx)
    if len(xx)<30:return []
    def world_points(y,x):
        z=d[y,x];camera=np.c_[(x-obs.K[0,2])*z/obs.K[0,0],-(y-obs.K[1,2])*z/obs.K[1,1],-z]
        return camera@obs.camera_world[:3,:3].T+obs.camera_world[:3,3]
    points=world_points(yy,xx);tree=cKDTree(points);visited=set();groups=[]
    for i in range(len(points)):
        if i in visited:continue
        todo=[i];visited.add(i);group=[]
        while todo:
            j=todo.pop();group.append(j)
            for k in tree.query_ball_point(points[j],.013):
                if k not in visited:visited.add(k);todo.append(k)
        if len(group)>=30:groups.append(points[group])
    # Association crops come ONLY from previous RGB-D estimates. Aggregate
    # still-visible edges even when occlusion fragments or merges components.
    # Two observed extents must remain at least 80 mm: no fresh pose from a
    # short isolated edge or from an empty image.
    for previous in priors:
        group=points[(np.max(abs(points[:,:2]-previous.position[:2]),axis=1)<.057)&(abs(points[:,2]-previous.position[2])<.018)]
        if len(group)>=30:
            extent=np.ptp(group,axis=0)
            if (extent[:2]>=.080).all():groups.append(group)
    y,x=np.where(valid);all_points=world_points(y,x)
    all_points=all_points[np.linalg.norm(color[y,x]-paint,axis=1)>=.09]
    measurements=[]
    squares=[]
    for group in groups:
        lo,hi=np.quantile(group,[.01,.99],axis=0);extent=hi-lo;center=(lo+hi)/2
        if .065<extent[0]<.115 and extent[1]>.115 and extent[2]<.016:
            # Adjacent square markings can merge through depth noise. Infer
            # horizontal bar pairs at the known 90 mm marking separation,
            # not a predeclared number of slots or their world positions.
            rows=np.sort(group[abs(group[:,0]-center[0])<.03,1])
            clusters=np.split(rows,np.flatnonzero(np.diff(rows)>.007)+1)
            bars=[float(np.median(c)) for c in clusters if len(c)>=4]
            for i,lower in enumerate(bars):
                for upper in bars[i+1:]:
                    if .080<upper-lower<.100:
                        candidate=group[(group[:,1]>lower-.006)&(group[:,1]<upper+.006)]
                        if len(candidate)>=30:squares.append(candidate)
        else:squares.append(group)
    for group in squares:
        lo,hi=np.quantile(group,[.01,.99],axis=0);extent=hi-lo
        if diagnostics is not None:diagnostics.setdefault('groups',[]).append(dict(points=len(group),extent=extent.tolist(),center=((lo+hi)/2).tolist()))
        if not ((extent[:2]>.065)&(extent[:2]<.115)).all() or extent[2]>.016:continue
        center=(lo+hi)/2;fit=fit_marking_edges(group,center)
        if fit is None:continue
        center[:2],center_covariance=fit
        radius=np.max(abs(all_points[:,:2]-center[:2]),axis=1)
        # Surface below the marking is measured from its neighbouring pixels;
        # marker height is not silently assumed equal to the support plane.
        neighbours=all_points[(radius>.014)&(radius<.065)&(all_points[:,2]<center[2]+.002)&(all_points[:,2]>center[2]-.014)]
        if len(neighbours)<20:continue
        center[2]=np.median(neighbours[:,2])
        covariance=np.eye(3)*(.002**2);covariance[:2,:2]+=center_covariance
        spread=max(.001,1.4826*np.median(abs(neighbours[:,2]-center[2])))
        covariance[2,2]+=(np.pi/2)*spread**2/len(neighbours)
        if obs.depth_sigma is not None:covariance+=np.eye(3)*float(np.median(obs.depth_sigma[mask])**2)
        measurements.append(TargetEstimate('',center,covariance,len(group),obs.stamp,kind='placement_marking'))
    unique=[]
    for measurement in sorted(measurements,key=lambda e:-e.pixels):
        if all(np.linalg.norm(measurement.position-other.position)>.025 for other in unique):unique.append(measurement)
    return sorted(unique,key=lambda e:(e.position[1],e.position[0]))


class PlacementTracker:
    def __init__(self,*,partial_marking=False):
        self.estimates={};self.next_id=0;self.partial_marking=bool(partial_marking)
    def update(self,obs):
        assigned=set()
        priors=list(self.estimates.values())
        measurements=placement_measurements(obs,priors=priors)
        if self.partial_marking:
            from .partial_marking import partial_marking_measurements
            measurements.extend(partial_marking_measurements(obs,priors,measurements))
        for measurement in measurements:
            candidates=[(np.linalg.norm(old.position-measurement.position),name) for name,old in self.estimates.items() if name not in assigned]
            distance,name=min(candidates) if candidates else (np.inf,None)
            if distance>.025:name=f'place_{self.next_id}';self.next_id+=1
            else:measurement.position=.5*measurement.position+.5*self.estimates[name].position
            measurement.name=name;self.estimates[name]=measurement;assigned.add(name)
        return self.estimates
