"""Validate a task-colour/self-geometry disambiguation contract.

Only known robot/C/tool materials are inspected. Environment geometry and
target identities are not consulted. This assumption must be revalidated
under real lighting; a match fails closed and leaves conservative masks on.
"""
import numpy as np


def colours_separated(palette,hardware_colours,margin=.12):
    paints=np.asarray(list(palette.values()),float)
    hardware=np.asarray(hardware_colours,float)
    if not len(hardware) or not np.isfinite(hardware).all() or not np.isfinite(paints).all():return False
    paints=paints/np.maximum(paints.sum(1,keepdims=True),1e-6)
    hardware=hardware/np.maximum(hardware.sum(1,keepdims=True),1e-6)
    return bool((np.linalg.norm(paints[:,None,:]-hardware[None,:,:],axis=2)>margin).all())


def known_hardware_colour_contract(stage,palette):
    from pxr import UsdGeom
    from .vision import color
    roots=('/World/LeftArm/','/World/RightArm/','/World/RightMagnet/','/World/C_structure/')
    colours=[color(p) for p in stage.Traverse() if str(p.GetPath()).startswith(roots) and p.IsA(UsdGeom.Gprim)]
    return colours_separated(palette,colours)
