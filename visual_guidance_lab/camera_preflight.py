"""Actual IsaacLab tiled RGB-D smoke test on existing cabin geometry.

This tests cameras/rendering only, not cloned task environments, robot control,
collision correctness, PPO training, or task success.
"""
import argparse
import json
from pathlib import Path
import time
import os
import faulthandler


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--num-cameras',type=int,default=1)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if not 1<=args.num_cameras<=16:raise ValueError('bounded preflight camera count')
    args.enable_cameras=True
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    runtime=output/'kit_runtime';runtime.mkdir()
    args.kit_args=(args.kit_args or '')+' --portable-root '+runtime.as_posix()+' --/app/settings/persistent=false'
    os.environ['OPTIX_CACHE_PATH']=str(runtime/'optix')
    stack=(output/'stacks.log').open('w',encoding='utf-8')
    faulthandler.enable(file=stack)
    faulthandler.dump_traceback_later(60,repeat=True,file=stack)
    def mark(phase):
        (output/'progress.json').write_text(json.dumps(dict(phase=phase,wall_time=time.time())),encoding='utf-8')
        print('PREFLIGHT '+phase,flush=True)
    launcher=None
    sim=None
    try:
        mark('app_start')
        launcher=AppLauncher(args)
        mark('app_ready')
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCamera,TiledCameraCfg
        from isaacsim.core.utils.prims import create_prim
        scene=Path(__file__).resolve().parents[1]/'isaacsim_model/astronaut_delivery_rgbd_cabin.usda'
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/60,device=args.device,gravity=(0.,0.,0.)))
        # Existing overlay USD has /World but no defaultPrim; an implicit
        # reference can silently produce an empty rendered scene.
        from pxr import Usd
        mark('source_open')
        source=Usd.Stage.Open(str(scene))
        if not source.GetPrimAtPath('/World').IsValid():
            raise ValueError('source scene lacks /World')
        cabin=create_prim('/World/Cabin','Xform')
        mark('scene_reference')
        cabin.GetReferences().AddReference(str(scene),'/World')
        mark('scene_referenced')
        for index in range(args.num_cameras):create_prim('/World/Probe_'+str(index),'Xform')
        light=sim_utils.DomeLightCfg(intensity=1000.)
        light.func('/World/ProbeLight',light)
        camera=TiledCamera(TiledCameraCfg(prim_path='/World/Probe_.*/Camera',
            height=120,width=160,data_types=['rgb','distance_to_image_plane'],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.,horizontal_aperture=20.955,
                                            clipping_range=(.05,5.))))
        mark('simulation_reset')
        sim.reset()
        mark('simulation_ready')
        eyes=torch.tensor([[1.3,-.3+i*.02,1.5] for i in range(args.num_cameras)],device=args.device)
        targets=torch.tensor([[.5,-.3,.8]]*args.num_cameras,device=args.device)
        camera.set_world_poses_from_view(eyes,targets)
        began=time.perf_counter()
        for _ in range(12):sim.render();camera.update(1/60,force_recompute=True)
        depth=camera.data.output['distance_to_image_plane']
        rgb=camera.data.output['rgb']
        fractions=((torch.isfinite(depth))&(depth>0)&(depth<5)).float().flatten(1).mean(1).cpu().tolist()
        import numpy as np
        from PIL import Image
        pixels=rgb.cpu().numpy();depth_array=depth.cpu().numpy()
        np.savez_compressed(output/'frames.npz',rgb=pixels,depth=depth_array)
        Image.fromarray(pixels[0,:,:,:3]).save(output/'camera_0.png')
        report=dict(status='rendered',scope=__doc__,camera_count=args.num_cameras,
            rgb_shape=list(rgb.shape),depth_shape=list(depth.shape),valid_depth_fraction=fractions,
            wall_s=time.perf_counter()-began,passed=all(v>0 for v in fractions),
            task_training_started=False,d435i_calibration_verified=False)
        (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
        mark('render_complete')
    except Exception as exc:
        (output/'error.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc),
            task_training_started=False),indent=2),encoding='utf-8')
        raise
    finally:
        mark('closing')
        # IsaacLab 2.3.2's STOP callback renders until play resumes. During
        # Replicator shutdown that can loop forever. Public clear_instance()
        # unsubscribes the callback before SimulationApp stops the timeline.
        if sim is not None:type(sim).clear_instance()
        if launcher is not None:launcher.app.close()
        faulthandler.cancel_dump_traceback_later()
        stack.close()


if __name__=='__main__':main()
