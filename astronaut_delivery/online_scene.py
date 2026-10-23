"""Separate online-coordination scene; preserve historical delivery layers."""
from pxr import Usd,UsdGeom
from .scene import ASSETS

ONLINE_SCENE=ASSETS/'astronaut_delivery_online.usd'

def build():
    if ONLINE_SCENE.exists():raise RuntimeError('Online scene already exists; do not overwrite user edits')
    s=Usd.Stage.CreateNew(str(ONLINE_SCENE));s.GetRootLayer().subLayerPaths=['astronaut_delivery_d435i.usd']
    UsdGeom.SetStageMetersPerUnit(s,1.);UsdGeom.SetStageUpAxis(s,'Z');s.SetDefaultPrim(s.GetPrimAtPath('/World'))
    s.GetRootLayer().Save();print(ONLINE_SCENE)

if __name__=='__main__':build()
