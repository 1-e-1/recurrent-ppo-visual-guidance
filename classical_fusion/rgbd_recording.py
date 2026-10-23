"""Replay raw calibrated measurements, never a serialized fabricated free grid.

Hashes establish file identity, not truthful sensor provenance or scene
immutability. Those are explicit acquisition/commissioning assumptions.
"""
import hashlib,json
from io import BytesIO
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

# One authenticated, immutable decoded recording per process. RGB is validated
# from the raw bytes but is not needed by geometric integration. The cache is
# NEVER a free-space grid and raw file hashes are checked again on every replay.
REPLAY_CACHE_LIMIT_BYTES=2*1024**3
_decoded_replay_cache=None


def validate_frame(rgb,depth,K,camera_world):
    rgb,depth,K,T=[np.asarray(x) for x in (rgb,depth,K,camera_world)]
    if depth.ndim!=2 or min(depth.shape)<4 or rgb.shape!=depth.shape+(3,):raise ValueError('invalid RGB-D image dimensions')
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]) or abs(K[0,1])+abs(K[1,0])>1e-9:
        raise ValueError('expected rectified camera intrinsics')
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):raise ValueError('invalid calibrated pose')
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:raise ValueError('camera orientation must be a proper rotation')
    if not np.isfinite(rgb).all():raise ValueError('invalid RGB values')
    # Bad depth returns stay invalid; never complete them from RGB or CAD.
    return rgb,np.where(np.isfinite(depth)&(depth>0),depth,0).astype(np.float32),K,T


def integrate_gl_frame(mapper,rgb,depth,K,camera_world):
    _,depth,K,T=validate_frame(rgb,depth,K,camera_world)
    R=T[:3,:3]@np.array([[0,-1,0],[0,0,1],[-1,0,0.]])
    xyzw=Rotation.from_matrix(R).as_quat();pose=np.r_[T[:3,3],xyzw[3],xyzw[:3]]
    return mapper.integrate(depth,K,pose)


def replay_initial_scan(mapper,manifest_path,*,expected_scene_token,expected_calibration_id,assume_unchanged_scene=False):
    global _decoded_replay_cache
    if not assume_unchanged_scene:raise ValueError('stored free evidence requires an explicitly unchanged scene assumption')
    if mapper.frames or mapper.weight.any() or mapper.free_views.any():raise ValueError('initial replay requires an empty map')
    path=Path(manifest_path).resolve();root=path.parent;manifest_bytes=path.read_bytes();meta=json.loads(manifest_bytes.decode('utf-8'))
    if meta.get('format')!='rgbd-initial-map-v1':raise ValueError('raw RGB-D recording manifest required; map grids are not accepted')
    if not expected_scene_token or meta.get('scene_token')!=expected_scene_token:raise ValueError('scene token mismatch')
    if not expected_calibration_id or meta.get('calibration_id')!=expected_calibration_id:raise ValueError('calibration identity mismatch')
    records=meta.get('records',[])
    if not records:raise ValueError('empty initial scan')
    verified=[];last=-np.inf;decoded=[];decoded_bytes=0
    cache_key=(str(path),hashlib.sha256(manifest_bytes).hexdigest(),expected_scene_token,expected_calibration_id)
    cache_hit=bool(_decoded_replay_cache is not None and _decoded_replay_cache[0]==cache_key
        and _decoded_replay_cache[2]<=REPLAY_CACHE_LIMIT_BYTES)
    retain=not cache_hit
    if not cache_hit:_decoded_replay_cache=None
    def immutable(value):
        a=np.ascontiguousarray(value)
        return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
    def authenticate(target,digest):
        payload=target.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('raw RGB-D hash mismatch')
        return payload
    def load(target,digest):
        return np.load(BytesIO(authenticate(target,digest)),allow_pickle=False)
    # Authenticate and validate the complete recording before changing the map.
    for record in records:
        target=(root/record['file']).resolve()
        if not target.is_relative_to(root):raise ValueError('record path escapes recording directory')
        if cache_hit:
            authenticate(target,record['sha256'])
            verified.append((target,record['sha256']));continue
        with load(target,record['sha256']) as data:
            if not {'rgb','depth','K','camera_world','exposure_s'}<=set(data.files):raise ValueError('missing raw measurement fields')
            if set(data.files)-{'rgb','depth','K','camera_world','exposure_s','q'}:raise ValueError('unexpected nonmeasurement fields')
            _,depth,K,T=validate_frame(data['rgb'],data['depth'],data['K'],data['camera_world'])
            stamp=float(data['exposure_s'])
            if not np.isfinite(stamp) or stamp<=last:raise ValueError('exposures must be strictly ordered')
            last=stamp
            if retain:
                R=T[:3,:3]@np.array([[0,-1,0],[0,0,1],[-1,0,0.]])
                xyzw=Rotation.from_matrix(R).as_quat();pose=np.r_[T[:3,3],xyzw[3],xyzw[:3]]
                amount=sum(a.nbytes for a in (depth,K,pose))
                if decoded_bytes+amount>REPLAY_CACHE_LIMIT_BYTES:
                    retain=False;decoded=[];decoded_bytes=0
                else:
                    decoded.append(tuple(immutable(a) for a in (depth,K,pose)));decoded_bytes+=amount
        verified.append((target,record['sha256']))
    if cache_hit:decoded=_decoded_replay_cache[1]
    elif retain:_decoded_replay_cache=(cache_key,tuple(decoded),decoded_bytes)
    try:
        if cache_hit or retain:
            for frame in decoded:
                # A mapper cannot mutate retained measurements; use working
                # copies, also avoiding writable-tensor views of read-only RAM.
                mapper.integrate(*(a.copy() for a in frame))
        else:
            for target,digest in verified:
                with load(target,digest) as data:
                    integrate_gl_frame(mapper,data['rgb'],data['depth'],data['K'],data['camera_world'])
    except Exception:
        mapper.reset()  # incomplete startup is never accepted as an initial map
        raise
    return dict(frames=len(verified),scene_token=expected_scene_token,calibration_id=expected_calibration_id,
        decoded_measurement_cache_hit=cache_hit,decoded_measurement_cache_retained=bool(cache_hit or retain),
        environment_source='RAW_RGBD_ONLY',scene_unchanged_assumed=True,
        provenance_scope='File integrity verified; acquisition authenticity/static-scene assumption requires independent commissioning evidence.')
