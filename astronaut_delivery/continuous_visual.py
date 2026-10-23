"""Whole-segment geometric visual-set certificate by interval subdivision.

Unlike sampled margins, each accepted leaf covers every joint configuration in
its box. This certifies the stated geometric proxy under a fixed target and
bounded calibration, NOT occlusion, Gaussian confidence, or measured pixels.
"""
from dataclasses import dataclass,asdict
import time
import numpy as np


@dataclass
class VisualCertificate:
    allowed:bool
    reason:str
    boxes:int
    leaves:int
    minimum_margin:float
    wall_s:float
    scope:str='continuous_geometric_visual_proxy_fixed_target_not_occlusion'

    def dictionary(self):
        result=asdict(self)
        if not np.isfinite(result['minimum_margin']):result['minimum_margin']=None
        return result


def box_margins(allocator,lower,upper,context):
    """Conservative interval lower bounds for all eight visual margins."""
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    if lower.shape!=(8,) or upper.shape!=(8,) or not np.isfinite([lower,upper]).all() or np.any(lower>upper):
        raise ValueError('invalid joint box')
    q=(lower+upper)/2;delta=(upper-lower)/2;arm=allocator.left;cfg=allocator.cfg
    T=arm.camera(q);offset=np.asarray(context['target'])-T[:3,3];p=T[:3,:3].T@offset
    distance=float(np.linalg.norm(offset))
    # Serial-chain path lengths bound the lever arm of the optical centre,
    # including its calibrated local offset, at ALL configurations in the box.
    reach=np.maximum(0.,arm.prefix_length['b8']-arm.joint_prefix+np.linalg.norm(arm.tool_local[:3,3]))
    translation=float(reach@delta)
    angle=min(np.pi,float(delta.sum()))
    camera_coordinate_error=translation+2*distance*np.sin(angle/2)
    zlo=-p[2]-camera_coordinate_error;zhi=-p[2]+camera_coordinate_error
    dlo=max(0.,distance-translation);dhi=distance+translation
    if zlo<=.05 or dlo<=.05:return np.full(8,-np.inf)
    def quotient(lo,hi,denlo,denhi):
        values=[lo/denlo,lo/denhi,hi/denlo,hi/denhi]
        return min(values),max(values)
    ulo,uhi=quotient(p[0]-camera_coordinate_error,p[0]+camera_coordinate_error,zlo,zhi)
    vlo,vhi=quotient(-p[1]-camera_coordinate_error,-p[1]+camera_coordinate_error,zlo,zhi)
    height=-offset[2]
    elo,ehi=quotient(height-translation,height+translation,dlo,dhi)
    support=(cfg.cap_radius_m+cfg.sigma_multiplier*context['sigma'])/zlo
    pixel_factor=context['pixels']*context['initial'][2]**2/max(context['initial'][3],1e-8)
    pixel_lower=min(pixel_factor*elo/dlo**2,pixel_factor*elo/dhi**2)
    return np.array([cfg.image_half_width-support-uhi,cfg.image_half_width-support+ulo,
                     cfg.image_half_height-support-vhi,cfg.image_half_height-support+vlo,
                     dlo-cfg.minimum_range_m,context['maximum_range']-dhi,
                     elo-cfg.minimum_elevation,(pixel_lower-cfg.minimum_pixels)/max(context['pixels'],cfg.minimum_pixels)])


def certify(allocator,start,end,context,max_depth=16,max_boxes=4095):
    began=time.perf_counter();start,end=np.asarray(start,float),np.asarray(end,float)
    boxes=0;leaves=0;minimum=np.inf;stack=[(start,end,0)]
    if max_depth<0 or max_boxes<1:raise ValueError('invalid certification budget')
    while stack:
        a,b,depth=stack.pop();boxes+=1
        if boxes>max_boxes:
            return VisualCertificate(False,'box_budget_exhausted',boxes-1,leaves,float(minimum),time.perf_counter()-began)
        margins=box_margins(allocator,np.minimum(a,b),np.maximum(a,b),context)
        if np.min(margins)>=0:
            leaves+=1;minimum=min(minimum,float(np.min(margins)));continue
        middle=(a+b)/2
        try:invalid=np.min(allocator.margins(middle,context))<0
        except ValueError:invalid=True
        if invalid or depth>=max_depth:
            return VisualCertificate(False,'midpoint_infeasible' if invalid else 'depth_budget_exhausted',boxes,leaves,float(np.min(margins)),time.perf_counter()-began)
        stack.extend([(middle,b,depth+1),(a,middle,depth+1)])
    return VisualCertificate(True,'whole_segment_geometric_proxy',boxes,leaves,float(minimum),time.perf_counter()-began)


def certify_refined(allocator,start,end,context,maximum_boxes=4095):
    """Bounded proof refinement, NOT sampled acceptance or relaxed margins."""
    if maximum_boxes not in (127,511,2047,4095):raise ValueError('unsupported visual proof budget')
    attempts=[]
    for budget in (127,511,2047,4095):
        if budget>maximum_boxes:break
        result=certify(allocator,start,end,context,max_boxes=budget)
        attempts.append(dict(budget=budget,**result.dictionary()))
        if result.allowed or result.reason!='box_budget_exhausted':break
    return result,attempts
