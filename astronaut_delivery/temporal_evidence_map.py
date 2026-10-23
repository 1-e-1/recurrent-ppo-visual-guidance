"""Conservative exact-view pixel evidence accumulation, separate from TSDF.

Within an unchanged static-scene epoch, a valid measured ray remains evidence.
Keep the MINIMUM of valid depths; absent rays remain absent. Never blend poses,
intrinsics, or crop sizes, and discard this small cache on any scene change.
"""
from collections import OrderedDict
import torch
from .adaptive_planning_map import AdaptivePlanningTSDF


class TemporalEvidenceTSDF(AdaptivePlanningTSDF):
    maximum_cached_images=8

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pixel_evidence=OrderedDict()

    def reset(self):
        super().reset()
        if hasattr(self,'pixel_evidence'):self.pixel_evidence.clear()

    def invalidate_region(self,lower,upper,**kwargs):
        with self._lock:
            result=super().invalidate_region(lower,upper,**kwargs)
            # Clearing all image evidence is conservative; the unchanged
            # spatial map and its per-cell independent votes remain intact.
            self.pixel_evidence.clear()
            return result

    def _on_depth_projection(self,depth,intrinsics,pose,u,v,z,ids):
        key=(tuple(depth.shape),intrinsics.detach().cpu().numpy().tobytes(),
             pose.detach().cpu().numpy().tobytes(),self.change_epoch)
        valid=torch.isfinite(depth)&(depth>=self.cfg.min_depth_m)&(depth<=self.cfg.evidence_max_depth_m)
        current=torch.where(valid,depth,torch.inf)
        previous=self.pixel_evidence.pop(key,None)
        evidence=current if previous is None else torch.minimum(previous,current)
        self.pixel_evidence[key]=evidence
        while len(self.pixel_evidence)>self.maximum_cached_images:
            self.pixel_evidence.popitem(last=False)
        # Original reconstruction still integrates the current raw frame.
        # Only conservative free-ray / obstacle-veto evidence uses this cache.
        accumulated=torch.where(torch.isfinite(evidence),evidence,0.)
        super()._on_depth_projection(accumulated,intrinsics,pose,u,v,z,ids)
