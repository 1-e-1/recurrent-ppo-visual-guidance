"""Height proposal from observed geometry and locally retired evidence.

This only proposes a waypoint. It never labels unknown voxels free and cannot
replace any full-arm/payload certificate. The ownership-change volume must be
cleared before disabling the conditional source-contact permission.
"""
import numpy as np
from scipy.spatial import ConvexHull


def lower_within_measured_band(current,floor,step=.006):
    if not np.isfinite([current,floor,step]).all() or step<=0:raise ValueError('finite positive height proposal required')
    proposed=max(float(floor),float(current-step))
    return proposed if proposed<current-.001 else None


def propose_height(tcp,envelope,surface,destination,spacing,planning_margin=.005,retirement_padding=.008,*,owned_source_departure=False):
    t=np.asarray(tcp);v=np.asarray(envelope.vertices);s=np.asarray(surface);d=np.asarray(destination)
    if s.ndim!=2 or s.shape[1]!=3 or not np.isfinite(s).all():raise ValueError('finite measured surface required')
    corridor=np.vstack([v[:,:2],v[:,:2]+d[:2]-t[:2,3]])
    e=ConvexHull(corridor).equations
    selected=(s[:,:2]@e[:,:2].T+e[:,2]<=retirement_padding).all(1)
    if not selected.any():raise ValueError('unobserved_transport_height')
    below=float(t[2,3]-v[:,2].min())
    top=max(float(s[selected,2].max()),float(v[:,2].max()-envelope.margin_m))
    # Region invalidation selects complete intersecting voxels. Add a whole
    # cell diagonal plus collision margin above its explicit upper bound.
    retired_upper=float(v[:,2].max()+retirement_padding)
    height=max(float(t[2,3]+.04),top+below+.025,
        retired_upper+below+planning_margin+float(np.linalg.norm(spacing)))
    if owned_source_departure:
        # This changes a waypoint heuristic, not a voxel label. Excluding the
        # held object's old measured surfaces is conditional on the SAME
        # isolated rigid-object ownership used for the source contact column.
        equations=np.asarray(envelope.equations,float)
        own=(s@equations[:,:3].T+equations[:,3]<=retirement_padding).all(1)
        static=s[selected&~own]
        top=max(float(envelope.support_z),float(d[2]),float(static[:,2].max()) if len(static) else -np.inf)
        height=max(float(t[2,3]+.04),top+below+.035)
    return height,dict(highest_observed_surface_m=top,payload_below_tcp_m=below,
        retired_source_upper_m=retired_upper,tcp_height_m=height,
        owned_source_departure=bool(owned_source_departure),
        note='Waypoint heuristic only; all unknown-space and whole-path checks remain mandatory')
