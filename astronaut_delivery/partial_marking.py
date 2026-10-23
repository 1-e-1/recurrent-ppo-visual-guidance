"""Partial-marking measurements; never environment collision authority.

Inputs must be CURRENT RGB-D white-marker points and a previously measured
identity association. The prior labels edges only, never contributes samples
or refreshes a timestamp. Sensor/common-mode covariance floors remain the
caller's responsibility, exactly as for the original edge fit.
"""
import numpy as np


def fit_orthogonal_edges(points, prior_center):
    points=np.asarray(points,float);prior=np.asarray(prior_center,float)
    if (points.ndim!=2 or points.shape[1]!=3 or prior.shape!=(3,)
            or len(points)<30 or not np.isfinite(points).all() or not np.isfinite(prior).all()):
        return None
    xy=points[:,:2];center=prior[:2].copy();variance=np.zeros(2)
    # Only nearby observations can be associated to this known 90 mm marking.
    if np.max(abs(xy-prior[:2]))>.057 or np.max(abs(points[:,2]-prior[2]))>.018:
        return None
    for _ in range(4):
        offset=xy-center;distance=abs(abs(offset)-.045)
        axis=np.argmin(distance,axis=1)
        sign=np.where(offset[np.arange(len(xy)),axis]>=0,1.,-1.)
        residual=offset[np.arange(len(xy)),axis]-sign*.045
        # Corner-only evidence cannot determine which observed normal applies.
        unambiguous=abs(distance[:,0]-distance[:,1])>.004
        valid=(abs(residual)<.008)&unambiguous
        if valid.mean()<.8:return None
        updated=center.copy();counts=[];spans=[]
        for j in range(2):
            keep=valid&(axis==j)
            if keep.sum()<8:return None
            tangent=xy[keep,1-j]
            span=float(np.quantile(tangent,.95)-np.quantile(tangent,.05))
            samples=xy[keep,j]-sign[keep]*.045
            updated[j]=np.median(samples)
            sigma=max(.004/np.sqrt(12),1.4826*np.median(abs(samples-updated[j])))
            if sigma>.003:return None
            variance[j]=(np.pi/2)*sigma*sigma/len(samples)
            counts.append(int(keep.sum()));spans.append(span)
        if np.max(abs(updated-prior[:2]))>.015:return None
        center=updated
    # Include uncertain in-plane yaw as a nuisance variable, scaled to metres.
    # This replaces an arbitrary minimum visible tangent span, not the final
    # position-quality threshold or any geometric safety certificate.
    offset=xy-center
    used=np.flatnonzero(valid);directions=axis[used]
    J=np.zeros((len(used),3));J[np.arange(len(used)),directions]=-1.
    J[:,2]=np.where(directions==0,offset[used,1],-offset[used,0])/.045
    scales=np.zeros(2)
    for j in range(2):
        keep=valid&(axis==j);samples=xy[keep,j]-sign[keep]*.045
        scales[j]=max(.004/np.sqrt(12),1.4826*np.median(abs(samples-np.median(samples))))
    information=(J/scales[directions,None]).T@(J/scales[directions,None])
    if np.linalg.matrix_rank(information,tol=1e-6)!=3:return None
    covariance=(np.pi/2)*np.linalg.inv(information)[:2,:2]
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).max()>.005**2:return None
    return dict(center=center,covariance=covariance,axis_counts=counts,
        tangent_spans_m=spans,scope='Current orthogonal edge samples only; association prior is not a new measurement')


def complete_partial_measurement(group, all_points, prior_center, stamp, *, depth_sigma=None, wall_stamp=None):
    """Same measured support neighbourhood and covariance floors as baseline.

    all_points contains current valid hardware-masked RGB-D returns, not map
    samples. Neither support height nor measurement timestamp comes from prior.
    """
    from astronaut_delivery.vision import TargetEstimate
    fitted=fit_orthogonal_edges(group,prior_center)
    p=np.asarray(all_points,float)
    if (fitted is None or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
            or not np.isfinite(stamp) or (wall_stamp is not None and not np.isfinite(wall_stamp))):
        return None
    if depth_sigma is not None and (not np.isfinite(depth_sigma) or depth_sigma<0):return None
    center=np.r_[fitted['center'],np.median(np.asarray(group)[:,2])]
    radius=np.max(abs(p[:,:2]-center[:2]),axis=1)
    neighbours=p[(radius>.014)&(radius<.065)&(p[:,2]<center[2]+.002)&(p[:,2]>center[2]-.014)]
    if len(neighbours)<20:return None
    center[2]=np.median(neighbours[:,2])
    covariance=np.eye(3)*(.002**2);covariance[:2,:2]+=fitted['covariance']
    spread=max(.001,1.4826*np.median(abs(neighbours[:,2]-center[2])))
    covariance[2,2]+=(np.pi/2)*spread**2/len(neighbours)
    if depth_sigma is not None:covariance+=np.eye(3)*float(depth_sigma**2)
    return TargetEstimate('',center,covariance,len(group),stamp,kind='placement_marking',wall_stamp=wall_stamp,
        measurement_method='rgbd_information_edges_with_current_support')


def partial_marking_measurements(obs, priors, existing=()):
    """Additional current-frame observations only; preserve original fits.

    No prior means no new identity. This function never modifies input images,
    self masks, depth, mapping state, stored estimates or timestamps.
    """
    if not priors:return []
    for prior in priors:
        position=np.asarray(prior.position);cov=np.asarray(prior.covariance)
        if (position.shape!=(3,) or cov.shape!=(3,3) or not np.isfinite(position).all()
                or not np.isfinite(cov).all() or not np.allclose(cov,cov.T)
                or np.linalg.eigvalsh(cov).min()<0):return []
    d=np.asarray(obs.depth);valid=np.isfinite(d)&(d>.1)
    if obs.robot_mask is not None:valid&=~obs.robot_mask
    if obs.color_valid is not None:valid&=obs.color_valid
    white=valid&(np.linalg.norm(np.asarray(obs.rgb)-[.75,.85,.9],axis=2)<.09)
    if white.sum()<30:return []
    def points(mask):
        y,x=np.where(mask);z=d[y,x]
        local=np.c_[(x-obs.K[0,2])*z/obs.K[0,0],-(y-obs.K[1,2])*z/obs.K[1,1],-z]
        return local@obs.camera_world[:3,:3].T+obs.camera_world[:3,3]
    painted=points(white);all_points=points(valid);measurements=[]
    sigma=None if obs.depth_sigma is None else float(np.median(obs.depth_sigma[white]))
    for prior in priors:
        if (np.asarray(prior.position).shape!=(3,) or not np.isfinite(prior.position).all()
                or np.asarray(prior.covariance).shape!=(3,3) or not np.isfinite(prior.covariance).all()
                or np.linalg.eigvalsh(prior.covariance).max()>.005**2):continue
        if any(np.linalg.norm(e.position-prior.position)<.025 for e in [*existing,*measurements]):continue
        group=painted[(np.max(abs(painted[:,:2]-prior.position[:2]),axis=1)<.057)
                      &(abs(painted[:,2]-prior.position[2])<.018)]
        estimate=complete_partial_measurement(group,all_points,prior.position,obs.stamp,
            depth_sigma=sigma,wall_stamp=(getattr(obs,'metadata',None) or {}).get('exposure_wall_s'))
        if estimate is None:continue
        if np.linalg.eigvalsh(estimate.covariance).max()>.005**2:continue
        distances=sorted(np.linalg.norm(estimate.position-p.position) for p in priors)
        if not distances or distances[0]>.025 or (len(distances)>1 and distances[1]<.05):continue
        measurements.append(estimate)
    return measurements
