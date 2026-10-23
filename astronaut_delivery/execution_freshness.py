"""A fresh exposure gates execution, not historical planning proposals.

Motion and monotonic wall clocks stay distinct. ROI refresh never commits a map
or blesses a revoked geometric certificate. Failure leaves both arms at rest.
"""
import time
import numpy as np
from .measurement_clock import observation_age


class FreshExecutionMixin:
    def measurement_age(self,target):
        if not getattr(self,'wall_clock_guidance',False):return self.time-target.stamp
        return observation_age(target.stamp,getattr(target,'wall_stamp',None),
            motion_now=self.time,wall_now=time.perf_counter())

    def raw_measurement_age(self):
        if not getattr(self,'observations',None):return np.inf
        row=self.observations[-1]
        if not getattr(self,'wall_clock_guidance',False):return self.time-row['time_s']
        return observation_age(row['time_s'],row.get('requested_wall_s'),
            motion_now=self.time,wall_now=time.perf_counter())

    def at_known_rest(self):
        if self.steps:
            values=np.r_[self.steps[-1]['v'],self.steps[-1]['a']]
            return bool(np.isfinite(values).all() and np.linalg.norm(values)<=1e-6)
        return bool(self.time==0 and np.array_equal(self.q,self.start_joints))

    def evidence_expired(self):
        now=time.monotonic()
        maps=[self.mapper,getattr(self,'local_mapper',None)]
        block=getattr(self,'robot_refinement',None)
        if block is not None:maps.append(block['mapper'])
        def expired(m):
            if m is None:return False
            try:stamp=float(m.latest_observation_time)
            except (TypeError,ValueError):return True
            return not np.isfinite(stamp) or not 0<=now-stamp<=m.cfg.snapshot_max_age_s
        return any(expired(m) for m in maps)

    def refresh_expired_evidence(self):
        if not getattr(self,'wall_clock_guidance',False) or not self.evidence_expired():return False
        if not self.at_known_rest() or getattr(self,'expiry_refresh_count',0)>=3:return False
        self.observe(integrate=True);self.expiry_refresh_count=getattr(self,'expiry_refresh_count',0)+1
        self.decisions.append(dict(status='EXPIRED_EVIDENCE_REOBSERVED_AT_REST',time_s=self.time,
            attempt=self.expiry_refresh_count,scope='New measurement and revision, not a timestamp-only map refresh'))
        return True

    def prepare_execution_measurement(self,motion):
        if not getattr(self,'wall_clock_guidance',False):return
        from .continuous_visual import certify
        began=time.perf_counter();row=self.paths[-1];failure='no_fresh_measured_target'
        if not self.at_known_rest() or not np.allclose(motion.state(0)[0],self.q.ravel(),atol=1e-10,rtol=0):
            raise RuntimeError('execution_refresh_requires_certified_rest_start')
        certificate=row['environment_certificate']
        def map_valid():
            local=getattr(self,'local_mapper',None)
            block=getattr(self,'robot_refinement',None)
            refined_revision=certificate.get('refinement_revision')
            refinement_ok=(refined_revision is None or block is not None
                and block['mapper'].revision==refined_revision and block['generation']==certificate.get('refinement_generation'))
            return (not self.evidence_expired() and certificate.get('allowed')
                and refinement_ok
                and certificate['revision']==self.mapper.revision
                and ('local_revision' not in certificate or local is not None and certificate['local_revision']==local.revision))
        try:
            for attempt in range(3):
                if not map_valid():failure='expired_or_revised_geometry';break
                estimate=self.guide.estimates.get(self.active)
                try:
                    if self.sensor_service is None:self.observe(integrate=False)
                    else:
                        packet=self.sensor_service.acquire(self.q,self.time,focus=None if estimate is None else estimate.position)
                        if self.accept_packet(packet,integrate=False) is None:
                            failure='fresh_packet_not_accepted';continue
                except Exception as exc:
                    failure='sensor_exception_at_rest/'+type(exc).__name__+'/'+str(exc);break
                failure=self.feedback_stop_reason()
                if failure is not None:continue
                target=self.guide.estimates.get(self.active)
                visual=None
                if row.get('visual_certificate',{}).get('allowed') is True:
                    try:
                        context=self.allocator.context(self.q[0],target.position,target.covariance,target.pixels,self.measurement_age(target))
                        visual=self.certify_visual_path(self.q[0],motion.end[:8],context)
                        if not visual.allowed or not self.allocator.endpoint_valid(self.q[0],motion.end[:8],context):
                            failure='new_measurement_rejects_visual_segment';continue
                    except ValueError as exc:
                        failure='new_measurement_visual_set/'+str(exc);continue
                elif row.get('endpoint_operational_visual_set_passed'):
                    try:self.allocator.context(motion.end[:8],target.position,target.covariance,target.pixels,self.measurement_age(target))
                    except ValueError as exc:
                        failure='new_measurement_rejects_acquisition_endpoint/'+str(exc);continue
                failure=self.feedback_stop_reason()
                if failure is not None:continue
                if not map_valid():failure='expired_or_revised_geometry';break
                # Begin the next exposure BEFORE moving. Waiting four motion
                # ticks to submit it can consume the remaining image lifetime.
                next_request=None
                if self.sensor_service is not None:
                    try:next_request=self.sensor_service.submit(self.q,self.time,focus=None if target is None else target.position)
                    except Exception as exc:
                        failure='stream_prime_failed_at_rest/'+str(exc);break
                    if next_request is None:failure='stream_prime_not_submitted';continue
                row['execution_refresh']=dict(attempt=attempt+1,requested_wall_s=self.observations[-1]['requested_wall_s'],
                    execution_wall_s=time.perf_counter(),target=self.active,
                    next_sensor_request=next_request,
                    target_age_s=None if getattr(self,'_view_rollback_active',False) else self.measurement_age(target),
                    raw_age_s=self.raw_measurement_age(),visual=None if visual is None else visual.dictionary(),
                    scope='Fresh ROI plus unchanged geometry revisions; no map integration during execution refresh')
                if visual is not None:row['execution_visual_target_position']=context['target'].tolist()
                return
            # A rejected proposal must not appear as an executed/audited path.
            if self.paths[-1] is not row:raise RuntimeError('execution proposal changed during refresh')
            self.paths.pop()
            self.decisions.append(dict(status='PRE_EXECUTION_MEASUREMENT_REJECTED',time_s=self.time,
                reason=failure,proposal=row,motion_commanded=False))
            raise RuntimeError('pre_execution_measurement_rejected')
        finally:
            self.timing.setdefault('pre_execution_refresh_s',[]).append(time.perf_counter()-began)
