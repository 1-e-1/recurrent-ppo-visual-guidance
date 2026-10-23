"""One measured map, smallest sufficient image patch for free-space evidence.

Preserves the original error bounds and independent-view/local-epoch rules.
Invalid pixels stay invalid; no hole filling or probability-based free carving.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from classical_fusion.planning_map import PlanningTSDF


class AdaptivePlanningTSDF(PlanningTSDF):
    patch_radii=(4,6,8,10,12,16)
    maximum_evidence_views=252

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # One bit per exact camera view, per voxel. Repeated static-camera
        # frames may fill missing pixels without counting as another view.
        self.view_bits=torch.zeros(self.shape,dtype=torch.int64,device=self.device)
        self.extra_view_bits={}

    def _view_word(self,index):
        word=index//63
        if word==0:return self.view_bits
        if word not in self.extra_view_bits:
            # Projection runs under inference_mode. Allocate ordinary mutable
            # evidence so later scene-change invalidation outside it is legal.
            with torch.inference_mode(False):
                self.extra_view_bits[word]=torch.zeros(self.shape,dtype=torch.int64,device=self.device)
        return self.extra_view_bits[word]

    def reset(self):
        super().reset()
        if hasattr(self,'view_bits'):self.view_bits.zero_()
        if hasattr(self,'extra_view_bits'):self.extra_view_bits.clear()

    def invalidate_region(self,lower,upper,**kwargs):
        with self._lock:
            result=super().invalidate_region(lower,upper,**kwargs)
            self.view_bits[self.cell_change_epoch==self.change_epoch]=0
            for word in self.extra_view_bits.values():word[self.cell_change_epoch==self.change_epoch]=0
            return result

    def _on_depth_projection(self,depth,intrinsics,pose,u,v,z,ids):
        cfg=self.cfg;self.latest_observation_time=time.monotonic()
        ui,vi=u[ids].round().long(),v[ids].round().long();measured=depth[vi,ui]
        valid=torch.isfinite(measured)&(measured>=cfg.min_depth_m)&(measured<=cfg.evidence_max_depth_m)
        ray_scale=torch.sqrt(1+((u[ids]-intrinsics[0,2])/intrinsics[0,0])**2+((v[ids]-intrinsics[1,2])/intrinsics[1,1])**2)
        displacement=self.spacing.norm()*.5+cfg.extrinsic_translation_bound_m+2*z[ids]*ray_scale*np.sin(cfg.extrinsic_rotation_bound_rad*.5)
        guard=displacement+cfg.depth_error_bound_m
        self.surface_veto.view(-1)[ids[valid&((measured-z[ids]).abs()<=guard)]]=True
        matched=[];view_index=None
        if self.evidence_poses:
            past=torch.stack(self.evidence_poses);translation=(past[:,:3]-pose[:3]).norm(dim=1)
            angle=2*torch.acos((past[:,3:]*pose[3:]).sum(1).abs().clamp(0,1))
            matched=torch.where((translation<cfg.independent_view_translation_m)&(angle<cfg.independent_view_rotation_rad))[0].tolist()
            if matched:
                # Only numerically identical extrinsics may fill this view's
                # holes. Nearby, nonidentical poses neither create independent
                # votes nor borrow an old anchor's claimed camera position.
                exact=torch.where((translation<1e-7)&((past[:,3:]-pose[3:]).abs().amax(1)<1e-7))[0].tolist()
                if not exact:return
                view_index=exact[0]
        if view_index is None and len(self.evidence_poses)>=self.maximum_evidence_views:return
        all_valid=torch.isfinite(depth)&(depth>=cfg.min_depth_m)&(depth<=cfg.evidence_max_depth_m)
        safe=torch.where(all_valid,depth,0.)
        offx=(u[ids]-intrinsics[0,2]).abs()/intrinsics[0,0];offy=(v[ids]-intrinsics[1,2]).abs()/intrinsics[1,1]
        spread=torch.maximum(intrinsics[0,0]*(1+offx),intrinsics[1,1]*(1+offy))*displacement/(z[ids]-displacement).clamp_min(1e-6)+cfg.intrinsic_pixel_bound
        free=torch.zeros_like(valid);assigned=torch.zeros_like(valid)
        for radius in self.patch_radii:
            selected=(~assigned)&(spread<=radius-.5)&(z[ids]>displacement)
            if not selected.any():continue
            minimum=-F.max_pool2d(-F.pad(safe[None,None],(radius,)*4,value=0.),2*radius+1,1)[0,0]
            free[selected]=minimum[vi[selected],ui[selected]]>z[ids[selected]]+guard[selected]
            assigned|=selected
        if view_index is not None:
            free &= (self._view_word(view_index).view(-1)[ids] & (1<<(view_index%63)))==0
        if free.any():
            if view_index is None:
                view_index=len(self.evidence_poses)
                self.evidence_poses.append(pose.clone());self.evidence_epochs.append(self.change_epoch)
            selected=ids[free];flat=self.free_views.view(-1)
            flat[selected]=(flat[selected].to(torch.int16)+1).clamp_max(255).to(torch.uint8)
            self._view_word(view_index).view(-1)[selected] |= (1<<(view_index%63))
