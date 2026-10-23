"""Retain a measured object submap under known rigid attachment motion.

No invented underside, closed hull, environment CAD, or free-space evidence.
This is a measurement layer alongside the conservative planning volume.
"""
import json
from pathlib import Path
import numpy as np


def extract_layer(vertices,triangles,indices,tcp,gradient,hessian,valid):
    v=np.asarray(vertices);f=np.asarray(triangles);ids=np.asarray(indices,int);T=np.asarray(tcp,float)
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(v)):
        raise ValueError('unique measured surface indices required')
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(T[:3,:3])<.999:
        raise ValueError('proper rigid measured transform required')
    lookup=np.full(len(v),-1,int);lookup[ids]=np.arange(len(ids));kept=f[(lookup[f]>=0).all(1)]
    R=T[:3,:3]
    return dict(local_vertices=(v[ids]-T[:3,3])@R,triangles=lookup[kept],
        local_gradient=np.asarray(gradient)@R,local_hessian=np.einsum('ji,njk,kl->nil',R,np.asarray(hessian),R),
        differential_valid=np.asarray(valid,bool),source_surface_indices=ids,capture_tcp=T)


def transform_layer(layer,tcp):
    T=np.asarray(tcp);R=T[:3,:3]
    g=layer['local_gradient']@R.T;H=np.einsum('ij,njk,lk->nil',R,layer['local_hessian'],R)
    valid=layer['differential_valid'].copy();length=np.linalg.norm(g,axis=1);valid &= length>1e-6
    n=np.full_like(g,np.nan);n[valid]=g[valid]/length[valid,None]
    curvature=np.full(len(g),np.nan)
    curvature[valid]=(np.trace(H[valid],axis1=1,axis2=2)-np.einsum('bi,bij,bj->b',n[valid],H[valid],n[valid]))/(2*length[valid])
    g[~valid]=np.nan;H[~valid]=np.nan
    return dict(vertices=layer['local_vertices']@R.T+T[:3,3],triangles=layer['triangles'],gradient=g,hessian=H,
        differential_valid=valid,normals=n,mean_curvature=curvature)


def export_composite(folder,layers):
    """Union of open measured surfaces, NOT a new collision-free certificate."""
    folder=Path(folder)
    with np.load(folder/'measured_surface.npz') as z:parts=[{k:z[k] for k in z.files}]
    parts.extend(layers);offset=0;faces=[]
    for part in parts:faces.append(part['triangles']+offset);offset+=len(part['vertices'])
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0] if k!='triangles'}
    data['triangles']=np.concatenate(faces)
    cfg=json.loads((folder/'measured_surface.json').read_text())['config']
    inside=((data['vertices']>=cfg['workspace_min'])&(data['vertices']<=cfg['workspace_max'])).all(1)
    lookup=np.full(len(inside),-1,int);lookup[inside]=np.arange(int(inside.sum()))
    kept=data['triangles'][(inside[data['triangles']]).all(1)]
    data={k:v[inside] for k,v in data.items() if k!='triangles'};data['triangles']=lookup[kept]
    np.savez_compressed(folder/'persistent_measured_surface.npz',**data)
    (folder/'persistent_measured_surface.json').write_text(json.dumps(dict(vertices=len(data['vertices']),triangles=len(data['triangles']),
        retained_object_layers=len(layers),new_free_space_votes=0,
        scope='Open observed TSDF surface plus rigidly transported observed object surfaces; no hole filling or invented underside',
        assumptions=['Rigid attachment and known robot TCP transforms','Measured component ownership is correct','Layer is not a watertight collision solid']),indent=2),encoding='utf-8')
