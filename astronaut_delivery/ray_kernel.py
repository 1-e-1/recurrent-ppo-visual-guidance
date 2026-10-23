"""Warp sensor kernel: only RGB and measured axial depth leave the renderer."""
import warp as wp

@wp.kernel
def ray_image(mesh:wp.uint64,colors:wp.array(dtype=wp.vec3),robot_faces:wp.array(dtype=wp.int32),width:int,height:int,fx:float,fy:float,cx:float,cy:float,origin:wp.vec3,R:wp.mat33,shaded:int,near_plane:float,depth:wp.array(dtype=float),rgb:wp.array(dtype=wp.vec3),robot_mask:wp.array(dtype=wp.int32)):
    i=wp.tid();u=i%width;v=i//width
    local=wp.vec3((float(u)-cx)/fx,-(float(v)-cy)/fy,-1.)
    length=wp.length(local);direction=R*(local/length)
    # Match the authored USD camera's 1 cm axial near plane. Its optical centre
    # is 0.12 mm behind the model's flange surface, just as in the template.
    near=near_plane*length
    hit=wp.mesh_query_ray(mesh,origin+direction*near,direction,4.)
    if hit.result:
        depth[i]=(hit.t+near)/length
        rgb[i]=colors[hit.face]
        robot_mask[i]=robot_faces[hit.face]
        if shaded==1:
            illumination=.35+.65*wp.abs(wp.dot(hit.normal,wp.normalize(wp.vec3(.5,-.6,.8))))
            rgb[i]=rgb[i]*illumination
    else:
        depth[i]=0.
        rgb[i]=wp.vec3(.025,.03,.04)
