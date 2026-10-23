"""Export the existing measured TSDF surface and its differential geometry.

No convex boxing, completion, artificial hole closing or collision certificate.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np


def export_surface(mapper,folder):
    folder=Path(folder)
    with mapper._lock:
        vertices,triangles=mapper.mesh()
        gradient,hessian,valid=mapper.differentials(vertices)
        v,f,g,H,ok=[x.cpu().numpy() for x in (vertices,triangles,gradient,hessian,valid)]
        length=np.linalg.norm(g,axis=1);ok &= length>1e-6
        normal=np.full_like(g,np.nan);normal[ok]=g[ok]/length[ok,None]
        curvature=np.full(len(v),np.nan,dtype=np.float32)
        # div(grad(phi)/|grad(phi)|)/2; signed mean curvature. Invalid stencils
        # and unknown surfaces are not turned into plausible-looking numbers.
        curvature[ok]=(np.trace(H[ok],axis1=1,axis2=2)-np.einsum('bi,bij,bj->b',normal[ok],H[ok],normal[ok]))/(2*length[ok])
        g[~ok]=np.nan;H[~ok]=np.nan
        np.savez_compressed(folder/'measured_surface.npz',vertices=v,triangles=f,gradient=g,hessian=H,
                            differential_valid=ok,normals=normal,mean_curvature=curvature)
        np.savez_compressed(folder/'planning_map_state.npz',distance=mapper.distance.cpu().numpy(),weight=mapper.weight.cpu().numpy(),
                            free_views=mapper.free_views.cpu().numpy(),surface_veto=mapper.surface_veto.cpu().numpy(),
                            lower=mapper.lower.cpu().numpy(),spacing=mapper.spacing.cpu().numpy())
        lines=['# Measured open surface; unknown is not free, units metre']
        lines.extend('v '+' '.join(f'{x:.9g}' for x in p) for p in v)
        lines.extend('f '+' '.join(str(int(x)+1) for x in face) for face in f)
        (folder/'measured_surface.obj').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        metadata=dict(vertices=len(v),triangles=len(f),valid_differentials=int(ok.sum()),config=asdict(mapper.cfg),
                      independent_evidence_views=len(mapper.evidence_poses) if hasattr(mapper,'evidence_poses') else None,
                      maximum_evidence_views=getattr(mapper,'maximum_evidence_views',None),
                      revision=mapper.revision,scope='Observed open TSDF surface, not completed objects or watertight collision solid',
                      differential_units=dict(gradient='dimensionless',hessian='1/m',mean_curvature='1/m'))
        (folder/'measured_surface.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        return metadata
