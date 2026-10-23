"""One bounded, raw-RGB-D-replayed block for a rejected robot component pair.

This is the same TSDF evidence and uncertainty model at higher resolution,
not an object model, free-space interpolation or CAD environment completion.
"""
import time
import numpy as np


class RobotEvidenceRefinementMixin:
    def mapping_layers(self):
        layers=[('broad',self.mapper),('local',getattr(self,'local_mapper',None))]
        block=getattr(self,'robot_refinement',None)
        if block is not None:layers.append(('robot_refinement',block['mapper']))
        return [(name,m) for name,m in layers if m is not None]

    def refined_environment(self,env):
        block=getattr(self,'robot_refinement',None)
        if block is not None and hasattr(env,'broad') and hasattr(env,'local_snapshot'):
            from classical_fusion.static_interval_snapshot import StaticIntervalSnapshot
            from classical_fusion.refined_robot_guard import RefinedRobotGuard
            snapshot=StaticIntervalSnapshot(block['mapper'],assume_static_epoch=True)
            env=RefinedRobotGuard(env,snapshot,block['names'],block['generation'])
        if getattr(self,'partitioned_geometry_proof',False):
            from classical_fusion.partitioned_guard import PartitionedGuard
            env=PartitionedGuard(env)
        return env

    def prepare_robot_refinement(self):
        if (not self.scan_source[3] or not self.static_interval_fusion or not self.at_known_rest()
            or getattr(self,'refinement_count',0)>=3):return False
        if getattr(self,'held',None) is None and (not getattr(self,'precontact_refinement',False)
            or getattr(self,'local_mapper',None) is None):return False
        witness=next((d.get('rejection') for d in reversed(self.decisions)
            if d.get('status')=='IK_REJECTED' and d.get('rejection')),None)
        if not witness or witness.get('arm_index')!=1:return False
        name=witness.get('link','')
        if len(name)!=2 or name[0] not in 'ab' or name[1] not in '12345678':return False
        names=[n for n in ('a'+name[1],'b'+name[1]) if n in self.right.local_hulls]
        old=getattr(self,'robot_refinement',None)
        # Replaying identical raw evidence into the same block is not a new view.
        if old is not None and old['names']==names:return False
        hulls=self.right.hulls_world(self.q[1]);vertices=np.concatenate([hulls[n][0] for n in names])
        lower=vertices.min(0)-.04;upper=vertices.max(0)+.04
        if np.prod(np.ceil((upper-lower)/.002))>2_000_000:return False
        from classical_fusion.planning_map import PlanningMapConfig
        from classical_fusion.rgbd_recording import replay_initial_scan
        from .temporal_evidence_map import TemporalEvidenceTSDF
        from .measured_task_memory import replay_changes
        from .measured_surface import export_surface
        began=time.perf_counter()
        cfg=PlanningMapConfig(workspace_min=tuple(lower),workspace_max=tuple(upper),voxel_size_m=.002,
            truncation_m=.006,surface_min_weight=1.,snapshot_max_age_s=30.)
        mapper=TemporalEvidenceTSDF(cfg,device=self.mapper.device);mapper.patch_radii=(4,6,8,10,12,16,24,32)
        scan,scene,calibration,unchanged=self.scan_source
        provenance=replay_initial_scan(mapper,scan,expected_scene_token=scene,expected_calibration_id=calibration,
            assume_unchanged_scene=unchanged)
        changes=replay_changes(mapper,self.transfer_events)
        generation=getattr(self,'refinement_count',0)+1
        if old is not None:
            folder=self.output/f'robot_refinement_{old["generation"]:02d}'
            folder.mkdir();export_surface(old['mapper'],folder)
        self.robot_refinement=dict(mapper=mapper,names=names,generation=generation)
        self.refinement_count=generation
        self.observe(integrate=True)
        self.decisions.append(dict(status='ROBOT_COMPONENT_RAW_RGBD_REFINEMENT',time_s=self.time,names=names,
            generation=generation,cells=int(np.prod(mapper.shape)),lower=lower.tolist(),upper=upper.tolist(),
            raw_provenance=provenance,replayed_changes=changes,wall_s=time.perf_counter()-began,
            scope='Complete component assignment by containment; unknown fine cells remain blocked; maximum three blocks built, one active'))
        return True
