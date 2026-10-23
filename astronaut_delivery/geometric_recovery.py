"""Bounded geometric proposals; never collision or visibility permission."""
import numpy as np


def raised_transport_waypoint(tcp,initial_height,accepted_count,*,step=.012,maximum_count=6):
    p=np.asarray(tcp,float)
    if (p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(initial_height)
        or not isinstance(accepted_count,int) or accepted_count<0
        or step!=.012 or maximum_count!=6):
        raise ValueError('finite position and fixed bounded recovery budget required')
    if accepted_count>=maximum_count:return None
    ceiling=float(initial_height)+step*maximum_count
    height=min(float(p[2]+step),ceiling)
    if height<=p[2]+.001:return None
    return np.r_[p[:2],height]


def extraction_waypoints(source,current,destination,support_z,payload_below_tcp,transport_height):
    """Proposal after a failed vertical lift at rest, with unchanged pose row.

    Contact lift remains at source XY. The lateral/second lift use transport
    guards, including the original 25 mm measured source clearance certificate.
    """
    source=np.asarray(source,float);current=np.asarray(current,float);destination=np.asarray(destination,float)
    if (any(p.shape!=(3,) for p in (source,current,destination))
        or not np.isfinite(np.r_[source,current,destination,support_z,payload_below_tcp,transport_height]).all()
        or payload_below_tcp<=0):raise ValueError('finite measured extraction geometry required')
    if np.linalg.norm(current[:2]-source[:2])>.003:return None
    low=max(float(current[2]),float(support_z+payload_below_tcp+.030))
    if transport_height<=low+.004:return None
    direction=destination[:2]-source[:2];distance=np.linalg.norm(direction)
    if distance<.025:return None
    inward=source[:2]+.020*direction/distance
    return [('lift',np.r_[source[:2],low]),('transport',np.r_[inward,low]),
            ('transport',np.r_[inward,transport_height])]


def far_first_placements(placements,camera):
    """Back-to-front heuristic over observed available markings only.

    Does not manufacture targets, know scene count, or ensure view/reachability.
    Exact newly measured destination remains required before manipulation.
    """
    camera=np.asarray(camera,float)
    if camera.shape!=(3,) or not np.isfinite(camera).all():raise ValueError('finite camera required')
    values=list(placements)
    if any(np.asarray(e.position).shape!=(3,) or not np.isfinite(e.position).all() for e in values):
        raise ValueError('finite measured placement positions required')
    return sorted(values,key=lambda e:(-float(np.linalg.norm(e.position-camera)),e.name))


def capacity_aware_placements(placements,camera,source,maximum_cells):
    """Filter destination proposals by map capacity, NEVER test scenes.

    Exact measured_roi bounds and resolution are retained. A rejected slot is
    not reserved and may remain usable by a later object with a smaller ROI.
    """
    from astronaut_delivery.transfer_map_capacity import measured_roi
    admitted=[];diagnostics=[]
    for candidate in far_first_placements(placements,camera):
        try:
            lower,upper,cells=measured_roi(source,candidate.position,maximum_cells)
        except RuntimeError as exc:
            if str(exc)!='local_transfer_map_budget_exceeded':raise
            diagnostics.append(dict(name=candidate.name,admitted=False,reason=str(exc)))
        else:
            admitted.append(candidate);diagnostics.append(dict(name=candidate.name,admitted=True,cells=cells))
    return admitted,diagnostics
