"""Measured view memory and map-certified camera-only reacquisition.

Old targets propose gaze only. Execution-arm motion cannot use an old target;
new target feedback is required by the caller after this sensing-only phase.
"""
from dataclasses import asdict
import time
import numpy as np
from .smooth_path import SmoothPath


class HistoryReacquisitionMixin:
    def remember_measured_views(self, packet, row):
        if not getattr(self, 'history_reacquisition', False):return
        history=self.__dict__.setdefault('measured_view_history', {})
        q=np.asarray(packet['q'],float)
        if q.shape!=(2,8) or not np.isfinite(q).all():return
        row['q']=q.tolist()
        for name, e in packet['estimates'].items():
            # A new packet must not refresh a retained detector estimate.
            if e.get('wall_stamp')!=packet['requested_wall_s']:continue
            if e['pixels']<self.allocator.cfg.minimum_pixels:continue
            covariance=np.asarray(e['covariance'],float)
            if covariance.shape!=(3,3) or not np.isfinite(covariance).all():continue
            if np.linalg.eigvalsh(covariance).max()>self.allocator.cfg.maximum_sigma_m**2:continue
            value=dict(q=q[0].copy(),position=np.asarray(e['position'],float).copy(),
                file=row['file'],stamp=e['stamp'],wall_stamp=e['wall_stamp'])
            views=history.setdefault(name,[])
            if views and np.linalg.norm(views[-1]['q']-q[0])<.06:
                # Keep the earliest view as well as recent local evidence.
                if len(views)>1:views[-1]=value
            else:
                views.append(value)
                if len(views)>12:del views[1]

    def reacquire_from_measured_history(self):
        if not self.at_known_rest() or not self.scan_source[3]:return False
        views=getattr(self,'measured_view_history',{}).get(self.active,[])
        if not views:return False
        owner=getattr(getattr(self,'held',None),'name',None) or self.active
        budget=self.__dict__.setdefault('history_reacquisition_counts',{})
        if budget.get(owner,0)>=6:return False
        historical=self.guide.estimates.get(self.active)
        if historical is None:return False
        start=self.q.ravel().copy();began=time.perf_counter();planning_recorded=False
        gap=float(self.right.tcp(self.q[1])[2,3]-historical.position[2])
        try:
            env,snapshot=self.make_environment(historical,gap)
            tried=self.__dict__.setdefault('history_reacquisition_tried',set())
            ordered=list(reversed(views))
            if getattr(self,'_history_stall_ranking',False):
                from .visual_stall_gate import rank_views
                ordered=rank_views(self.left,ordered)
            proposals=[(v['q'],v,'recorded_camera_pose') for v in ordered]
            # Local gaze alternatives use a historical point only as a proposal.
            latest=views[-1];focus=latest['position'];camera=self.left.camera(self.q[0])[:3,3]
            for offset in ([0,0,.04],[.04,0,0],[-.04,0,0],[0,.04,0],[0,-.04,0]):
                position=camera+offset;axis=focus-position;axis/=max(np.linalg.norm(axis),1e-12)
                q=self.left.solve(position,axis,seed=self.q[0],max_delta=.30,tries=4,
                    rng=np.random.default_rng(62019))
                if q is not None:proposals.append((q,latest,'historical_gaze_offset'))
            for q,view,kind in proposals:
                if np.linalg.norm(q-self.q[0])<.015:continue
                key=(owner,self.active,*np.round(self.q[1],3),*np.round(q,3))
                if key in tried:continue
                distance=np.linalg.norm(self.left.camera(q)[:3,3]-view['position'])
                if not self.allocator.cfg.minimum_range_m<=distance<=self.allocator.cfg.maximum_range_m:continue
                end=np.r_[q,self.q[1]]
                certificate=env.segment(start,end)
                robot=certificate.allowed and self.self_guard.segment(start,end,**self.robot_guard_arguments())
                if not robot:continue
                tried.add(key);budget[owner]=budget.get(owner,0)+1
                motion=SmoothPath.make(start,end,minimum=1.2)
                row=dict(start=start.tolist(),end=end.tolist(),duration_s=motion.duration,time_s=self.time,
                    target=historical.position.tolist(),standoff_m=gap,map_revision=snapshot.revision,
                    environment_certificate=asdict(certificate),robot_certificate=True,
                    phase='certified_history_reacquisition_right_rest',operational_visual_sufficiency_required=False,
                    visual_target_name=self.active,visual_target_position=historical.position.tolist(),
                    visual_certificate=dict(allowed=None,scope='Camera-only sensing with raw-frame watchdog, not operational target visibility'),
                    history_source_file=view['file'],history_source_stamp_s=view['stamp'],proposal_kind=kind,
                    scope='Historical RGB-D proposes gaze, current map certifies camera-only motion; fresh target required before any right-arm motion')
                held=getattr(self,'held',None)
                if held is not None:
                    row.update(payload_name=held.name,payload_local_points=held.local_points.tolist(),
                               payload_contact_phase=self.transfer_phase)
                if getattr(self,'transfer_phase',None) in ('lift','place'):
                    column=self.source_column if self.transfer_phase=='lift' else self.place_column
                    row['contact_column']=dict(equations=column.equations.tolist(),lower=column.lower,
                                               upper=column.upper,links=sorted(column.links))
                if self.paths and 'released_target_contact' in self.paths[-1]:
                    row['released_target_contact']=self.paths[-1]['released_target_contact']
                self.paths.append(row)
                self.decisions.append(dict(status='HISTORY_SENSING_REACQUISITION',time_s=self.time,
                    owner=owner,attempt=budget[owner],source_file=view['file'],proposal_kind=kind))
                self.planning_wall.append(time.perf_counter()-began);planning_recorded=True
                first=len(self.steps);self._view_rollback_active=True
                try:self.execute(motion)
                except RuntimeError as exc:
                    if str(exc) not in ('rgbd_loss_smooth_brake','visual_loss_smooth_brake','pre_execution_measurement_rejected'):raise
                    if not self.at_known_rest():raise
                    self.observe(integrate=True)
                finally:
                    self._view_rollback_active=False
                    for item in self.steps[first:]:
                        if item['phase']!='braking':item['phase']=row['phase']
                return True
            return False
        finally:
            if not planning_recorded:self.planning_wall.append(time.perf_counter()-began)
