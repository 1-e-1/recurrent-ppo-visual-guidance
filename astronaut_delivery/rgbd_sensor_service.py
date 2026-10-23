"""Isolated RGB-D acquisition with latest-request semantics, no object oracle.

Only robot joints, camera pose, exposure time and an image-derived ROI focus
are sent to the worker. Static environment geometry stays sensor-side.
"""
import multiprocessing as mp
import queue,time
import numpy as np


def _worker(requests,replies,scene,fault_config=None,fault_output=None,partial_marking=False,interface_palette=None):
    from pxr import Usd
    from threadpoolctl import threadpool_limits
    from .simultaneous import TightBoundArm
    from .robot_only_geometry import RobotOnlyGuard
    from .run import DualArmKinematics
    from .d435i import D435iCamera
    from .vision import VisualGuide
    from .rgbd_magnet_backend import KinematicMagnetBackend
    from .placement_vision import PlacementTracker
    from .interface_mask_contract import known_hardware_colour_contract
    from .measured_payload import mask_payload_returns
    from .measurement_clock import refreshed_wall_stamps
    from .rgbd_sensor_faults import RenderedSensorFault
    with threadpool_limits(1):
        fault=None if fault_config is None else RenderedSensorFault(fault_config,fault_output)
        stage=Usd.Stage.Open(scene);stage.SetEditTarget(stage.GetSessionLayer())
        robot=DualArmKinematics(stage);arms=[TightBoundArm(stage,s) for s in ('left','right')]
        mask=RobotOnlyGuard(stage,*arms);camera=D435iCamera();count=0
        magnet=KinematicMagnetBackend(stage)
        placements=PlacementTracker(partial_marking=True) if partial_marking else PlacementTracker()
        guide=VisualGuide(interface_palette or dict(bolt=[.85,.12,.10],washer=[.10,.75,.18],bracket=[.12,.25,.90]),sensor_profile=camera.profile)
        guide.fit_circular_cap=True;guide.reacquire_stale=True
        guide.registered_interface_override=known_hardware_colour_contract(stage,guide.palette)
        while True:
            request=requests.get()
            if request is None:break
            try:
                start=time.perf_counter();q=np.array(request['q']);robot.set_positions(*q)
                magnet.update(arms[1].tcp(q[1]));feedback=magnet.feedback()
                if request.get('actuation') is not None:
                    command=request['actuation']
                    feedback=magnet.command(command['action'],arms[1].tcp(q[1]),command.get('target'))
                T=arms[0].camera(q[0]);focus=request.get('focus');stamp=request['stamp']
                if focus is None:obs=camera.capture(stage,T,stamp,seed=17000+count)
                else:obs=camera.capture_roi(stage,T,stamp,np.array(focus),seed=17000+count)
                fault_info=None;delay=0.
                if fault is not None:fault_info,clean,delay=fault.prepare(obs,stamp,time.perf_counter())
                obs.metadata=dict(obs.metadata or {},exposure_wall_s=request['requested_wall_s'])
                previous=dict(guide.estimates)
                previous_clocks={k:v.wall_stamp for k,v in previous.items() if v.wall_stamp is not None}
                depth=mask.mask_self_returns(obs.depth,obs.K,T,q)
                obs.robot_mask=(obs.depth>0)&(depth==0);guide.update(obs,integrate_map=False)
                guide.estimates.update(placements.update(obs))
                clocks=refreshed_wall_stamps(previous,guide.estimates,previous_clocks,request['requested_wall_s'])
                for name,estimate in guide.estimates.items():estimate.wall_stamp=clocks.get(name)
                if feedback['attached'] and request.get('payload_local_points') is not None:
                    depth=mask_payload_returns(depth,obs.K,T,request['payload_local_points'],arms[1].tcp(q[1]))
                # Do not serialize the renderer-provided instance mask.
                packet=dict(id=request['id'],q=q,rgb=obs.rgb,depth=obs.depth,K=obs.K,camera_world=T,stamp=stamp,
                    self_mask=(obs.depth>0)&(depth==0),color_valid=obs.color_valid,depth_sigma=obs.depth_sigma,
                    requested_wall_s=request['requested_wall_s'],
                    estimates={k:dict(name=v.name,position=v.position.copy(),covariance=v.covariance.copy(),pixels=v.pixels,stamp=v.stamp,kind=v.kind,wall_stamp=v.wall_stamp,measurement_method=v.measurement_method) for k,v in guide.estimates.items()},
                    sensor_wall_s=time.perf_counter()-start,seed=17000+count,actuator=feedback)
                if fault is not None:
                    if delay:time.sleep(delay)
                    fault.record(obs,clean,request,fault_info,delay,time.perf_counter())
                    packet['sensor_fault']=fault_info;packet['sensor_wall_s']=time.perf_counter()-start
                count+=1
            except Exception as exc:packet=dict(id=request['id'],error=repr(exc))
            try:replies.put(packet,timeout=2.)
            except queue.Full:
                try:replies.get_nowait()
                except queue.Empty:pass
                replies.put(packet,timeout=2.)


class RGBDSensorService:
    def __init__(self,scene,*,fault_config=None,fault_output=None,partial_marking=False,interface_palette=None):
        from .rgbd_sensor_faults import validate_fault
        fault_config=validate_fault(fault_config)
        if fault_config is not None and fault_output is None:raise ValueError('fault evidence output required')
        ctx=mp.get_context('spawn');self.requests=ctx.Queue(maxsize=1);self.replies=ctx.Queue(maxsize=2)
        self.process=ctx.Process(target=_worker,args=(self.requests,self.replies,str(scene),fault_config,fault_output,bool(partial_marking),interface_palette),daemon=True)
        self.process.start();self.sequence=0;self.closed=False
        self.payload_local_points=None

    def submit(self,q,stamp,focus=None,actuation=None):
        if self.closed or not self.process.is_alive():raise RuntimeError('RGB-D worker unavailable')
        self.sequence+=1;request=dict(id=self.sequence,q=np.asarray(q).tolist(),stamp=float(stamp),focus=None if focus is None else np.asarray(focus).tolist())
        request['requested_wall_s']=time.perf_counter()
        if actuation is not None:request['actuation']=actuation
        if self.payload_local_points is not None:request['payload_local_points']=np.asarray(self.payload_local_points).tolist()
        try:self.requests.put_nowait(request)
        except queue.Full:
            try:self.requests.get_nowait()
            except queue.Empty:return None
            try:self.requests.put_nowait(request)
            except queue.Full:return None
        return self.sequence

    def poll(self):
        latest=None
        while True:
            try:latest=self.replies.get_nowait()
            except queue.Empty:break
        if latest and 'error' in latest:raise RuntimeError(latest['error'])
        if not self.process.is_alive():raise RuntimeError('RGB-D worker exited unexpectedly')
        return latest

    def acquire(self,q,stamp,timeout=30.,focus=None,actuation=None):
        # Called at rest only; drain pending requests before the full frame.
        deadline=time.monotonic()+timeout;identifier=None
        while identifier is None:
            identifier=self.submit(q,stamp,focus,actuation=actuation)
            if time.monotonic()>deadline:raise TimeoutError('RGB-D submission timeout')
            if identifier is None:time.sleep(.005)
        while time.monotonic()<deadline:
            packet=self.poll()
            if packet and packet['id']==identifier:return packet
            time.sleep(.005)
        raise TimeoutError('RGB-D full-frame acquisition timeout')

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            self.requests.put(None,timeout=2.)
        except queue.Full:pass
        self.process.join(timeout=5.)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=2.)
        self.requests.close();self.replies.close()
