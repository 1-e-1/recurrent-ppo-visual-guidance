"""One confidence-weighted geometric map for reconstruction and active vision.

Camera poses: xyz + wxyz, Isaac world-camera axes (+X forward, +Y left,
+Z up). Depth is axial metres. All grids use explicit x,y,z tensor order.
No semantic labels, object metadata, generated points or learned weights enter
this module. The robot provides calibrated K and reachable (q,T_camera) pairs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as functional

from codex_zbot.perception.pointcloud import quaternion_to_matrix


@dataclass(frozen=True)
class ActiveTSDFConfig:
    workspace_min: tuple[float, float, float] = (0.45, -0.65, 0.02)
    workspace_max: tuple[float, float, float] = (1.05, 0.65, 0.75)
    voxel_size_m: float = 0.006
    truncation_m: float = 0.018
    min_depth_m: float = 0.08
    max_depth_m: float = 2.0
    max_weight: float = 32.0
    depth_edge_m: float = 0.08
    surface_min_weight: float = 1.0
    view_grid_stride: int = 3
    view_image_size: int = 64
    motion_cost: float = 0.01
    camera_clearance_m: float = 0.40
    unknown_exploration_weight: float = 0.0
    robust_delta_m: float = 0.0

    def validate(self):
        if self.voxel_size_m <= 0 or self.truncation_m < self.voxel_size_m:
            raise ValueError("require truncation >= voxel size > 0")
        if any(a >= b for a, b in zip(self.workspace_min, self.workspace_max)):
            raise ValueError("invalid workspace")
        if self.max_weight < 1 or self.view_grid_stride < 1 or self.view_image_size < 4:
            raise ValueError("invalid confidence/view settings")
        if not 0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("invalid depth range")

    def to_dict(self):
        return asdict(self)


@dataclass
class MapUpdate:
    new_known_voxels: int
    information_gain: float
    observed_voxels: int


class ActiveTSDF:
    def __init__(self, config: ActiveTSDFConfig | None = None, device="cpu"):
        self.cfg = config or ActiveTSDFConfig()
        self.cfg.validate()
        self.device = torch.device(device)
        self.lower = torch.tensor(self.cfg.workspace_min, device=self.device)
        upper = torch.tensor(self.cfg.workspace_max, device=self.device)
        self.shape = tuple(torch.ceil((upper - self.lower) / self.cfg.voxel_size_m).long().tolist())
        # Exact workspace boundaries, with every axis spacing <= voxel_size_m.
        self.spacing = (upper - self.lower) / torch.tensor(self.shape, device=self.device)
        axes = [self.lower[i] + (torch.arange(n, device=self.device) + .5) * self.spacing[i]
                for i, n in enumerate(self.shape)]
        self.centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        self.distance = torch.full(self.shape, self.cfg.truncation_m, device=self.device)
        self.weight = torch.zeros(self.shape, device=self.device)
        self.frames = 0
        self.revision = 0

    def reset(self):
        self.distance.fill_(self.cfg.truncation_m)
        self.weight.zero_()
        self.frames = 0
        self.revision += 1

    def _on_depth_projection(self, depth, intrinsics, pose, u, v, z, ids):
        """Optional planning evidence, using this same projection/map."""
        pass

    @staticmethod
    def _project(points, pose, intrinsics):
        rotation = quaternion_to_matrix(pose[3:])
        local = (points - pose[:3]) @ rotation
        z = local[:, 0]
        denom = z.clamp_min(1e-6)
        u = intrinsics[0, 2] - intrinsics[0, 0] * local[:, 1] / denom
        v = intrinsics[1, 2] - intrinsics[1, 1] * local[:, 2] / denom
        return u, v, z

    @torch.inference_mode()
    def integrate(self, depth, intrinsics, pose) -> MapUpdate:
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device).squeeze()
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device)
        pose = torch.as_tensor(pose, dtype=torch.float32, device=self.device)
        if depth.ndim != 2 or intrinsics.shape != (3, 3) or pose.shape != (7,):
            raise ValueError("expected depth [H,W], K [3,3], pose [7]")
        h, w = depth.shape
        if (min(h,w) < 4 or not torch.isfinite(intrinsics).all()
                or intrinsics[0,0] <= 0 or intrinsics[1,1] <= 0
                or not torch.isfinite(pose).all()
                or abs(float(pose[3:].norm())-1) > .005):
            raise ValueError("invalid image, calibrated intrinsics or unit-quaternion camera pose")
        valid = torch.isfinite(depth) & (depth >= self.cfg.min_depth_m) & (depth <= self.cfg.max_depth_m)
        safe_depth = torch.where(valid, depth, 0)
        # Bilinear axial depth is accurate inside a surface. Its local range
        # rejects mixed foreground/background measurements at silhouettes.
        maximum = functional.max_pool2d(safe_depth[None, None], 3, 1, 1)[0, 0]
        minimum = -functional.max_pool2d(-safe_depth[None, None], 3, 1, 1)[0, 0]
        reliable = valid & ((maximum - minimum) <= self.cfg.depth_edge_m)
        u, v, z = self._project(self.centers, pose, intrinsics)
        inside = (z > 0) & (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
        ids = torch.where(inside)[0]
        if ids.numel() == 0:
            return MapUpdate(0, 0.0, 0)
        self._on_depth_projection(depth,intrinsics,pose,u,v,z,ids)
        pixels = torch.stack([2 * u[ids] / (w - 1) - 1, 2 * v[ids] / (h - 1) - 1], -1)
        sampled = functional.grid_sample(
            torch.stack([safe_depth, reliable.float()])[None], pixels[None, None],
            mode="bilinear", align_corners=True,
        )[0, :, 0]
        # On a discontinuity, bilinear interpolation would blend two objects.
        # Keep the actual nearest pixel at reduced confidence instead of
        # discarding silhouettes (which erased much of small objects).
        ui, vi = u[ids].round().long(), v[ids].round().long()
        smooth = sampled[1] > .999
        measured = torch.where(smooth, sampled[0], safe_depth[vi, ui])
        measurement_weight = torch.where(smooth, 1., .5)
        residual = measured - z[ids]
        usable = valid[vi, ui] & (residual >= -self.cfg.truncation_m)
        measurement_weight = measurement_weight[usable]
        ids, residual = ids[usable], residual[usable]
        distance, weights = self.distance.view(-1), self.weight.view(-1)
        old_weight = weights[ids]
        new_known = int((old_weight == 0).sum())
        measurement = residual.clamp(-self.cfg.truncation_m, self.cfg.truncation_m)
        if self.cfg.robust_delta_m>0:
            error=(measurement-distance[ids]).abs().clamp_min(1e-8)
            robust=(self.cfg.robust_delta_m/error).clamp_max(1)
            measurement_weight*=torch.where(old_weight>=2,robust,1.)
        # Confidence is the inverse variance in units of one depth sample.
        # Weighted averaging solves the per-voxel squared depth residual loss.
        increment = torch.minimum((self.cfg.max_weight - old_weight).clamp_min(0), measurement_weight)
        denominator = (old_weight + increment).clamp_min(1e-8)
        distance[ids] = (old_weight * distance[ids] + increment * measurement) / denominator
        weights[ids] = old_weight + increment
        gain = torch.log1p(increment / (1 + old_weight)).sum() / self.weight.numel()
        self.frames += 1
        self.revision += 1
        return MapUpdate(new_known, float(gain), len(ids))

    @torch.inference_mode()
    def surface(self, min_weight: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Subvoxel zero crossings, with confidence. Unknown never closes a shell."""
        threshold = self.cfg.surface_min_weight if min_weight is None else min_weight
        parts, confidence = [], []
        for axis in range(3):
            first, second = [slice(None)] * 3, [slice(None)] * 3
            first[axis], second[axis] = slice(None, -1), slice(1, None)
            a, b = self.distance[tuple(first)], self.distance[tuple(second)]
            wa, wb = self.weight[tuple(first)], self.weight[tuple(second)]
            crossing = ((a < 0) != (b < 0)) & (wa >= threshold) & (wb >= threshold)
            index = torch.nonzero(crossing).float()
            if len(index) == 0:
                continue
            fraction = a[crossing] / (a[crossing] - b[crossing])
            index[:, axis] += fraction
            parts.append(self.lower + (index + .5) * self.spacing)
            confidence.append(torch.minimum(wa[crossing], wb[crossing]))
        if not parts:
            return self.centers[:0], self.weight.view(-1)[:0]
        return torch.cat(parts), torch.cat(confidence)

    @torch.inference_mode()
    def differentials(self, points):
        """First/second partials of the SAME TSDF, evaluated only at queries.

        Returns gradient [N,3], symmetric Hessian [N,3,3], and a validity mask.
        Derivatives touching unobserved cells are explicitly invalid.
        """
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        index = torch.round((points-self.lower)/self.spacing-.5).long()
        dims = torch.tensor(self.shape, device=self.device)
        valid = ((index >= 1) & (index < dims-1)).all(1)
        index = torch.maximum(torch.minimum(index, dims-2), torch.ones_like(index))
        def sample(offset):
            at = index + offset
            observed = self.weight[at[:,0],at[:,1],at[:,2]] >= self.cfg.surface_min_weight
            return self.distance[at[:,0],at[:,1],at[:,2]], observed
        zero = torch.zeros(3, dtype=torch.long, device=self.device)
        center, known = sample(zero)
        valid &= known
        gradient = torch.zeros(len(points),3,device=self.device)
        hessian = torch.zeros(len(points),3,3,device=self.device)
        basis = torch.eye(3,dtype=torch.long,device=self.device)
        for i in range(3):
            plus,kp=sample(basis[i]); minus,km=sample(-basis[i]); valid &= kp & km
            gradient[:,i]=(plus-minus)/(2*self.spacing[i])
            hessian[:,i,i]=(plus-2*center+minus)/self.spacing[i]**2
            for j in range(i):
                pp,k1=sample(basis[i]+basis[j]); pm,k2=sample(basis[i]-basis[j])
                mp,k3=sample(-basis[i]+basis[j]); mm,k4=sample(-basis[i]-basis[j])
                value=(pp-pm-mp+mm)/(4*self.spacing[i]*self.spacing[j])
                hessian[:,i,j]=value; hessian[:,j,i]=value
                valid &= k1 & k2 & k3 & k4
        return gradient,hessian,valid

    @torch.inference_mode()
    def mesh(self):
        """Vectorized surface nets, directly from observed TSDF zero crossings.

        One vertex per crossing cell and two triangles per dual quad retain
        concavities. Unobserved boundaries remain open: this is measured
        geometry, not proof of a closed solid or a traversable unknown region.
        """
        corners = torch.tensor([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)],
                               device=self.device,dtype=torch.long)
        cell_shape=tuple(n-1 for n in self.shape)
        values=[]; weights=[]
        for c in corners.tolist():
            at=tuple(slice(c[i],c[i]+cell_shape[i]) for i in range(3))
            values.append(self.distance[at]); weights.append(self.weight[at])
        low=values[0].clone(); high=low.clone(); known=weights[0]>=self.cfg.surface_min_weight
        for d,w in zip(values[1:],weights[1:]):
            low=torch.minimum(low,d); high=torch.maximum(high,d); known &= w>=self.cfg.surface_min_weight
        active=(low<0)&(high>=0)&known
        index=active.nonzero()
        vertex_grid=torch.full(cell_shape,-1,dtype=torch.long,device=self.device)
        vertex_grid[active]=torch.arange(len(index),device=self.device)
        samples=torch.stack([d[active] for d in values],1)
        total=torch.zeros(len(index),3,device=self.device); count=torch.zeros(len(index),device=self.device)
        edge_pairs=[(a,b) for a in range(8) for b in range(a+1,8) if (a^b) in (1,2,4)]
        for a,b in edge_pairs:
            crossing=(samples[:,a]<0)!=(samples[:,b]<0)
            ids=crossing.nonzero().flatten()
            alpha=samples[ids,a]/(samples[ids,a]-samples[ids,b])
            total[ids]+=corners[a]+alpha[:,None]*(corners[b]-corners[a])
            count[ids]+=1
        vertices=self.lower+(index+total/count[:,None].clamp_min(1)+.5)*self.spacing
        faces=[]
        for axis in range(3):
            a,b=[slice(None)]*3,[slice(None)]*3
            a[axis],b[axis]=slice(None,-1),slice(1,None)
            crossing=((self.distance[tuple(a)]<0)!=(self.distance[tuple(b)]<0))
            origin=crossing.nonzero()
            other=[j for j in range(3) if j!=axis]
            inside=(origin[:,other[0]]>0)&(origin[:,other[1]]>0)
            inside &= (origin[:,other[0]]<cell_shape[other[0]]) & (origin[:,other[1]]<cell_shape[other[1]])
            origin=origin[inside]
            quad=[]
            for s,t in ((0,0),(-1,0),(-1,-1),(0,-1)):
                at=origin.clone(); at[:,other[0]]+=s; at[:,other[1]]+=t
                quad.append(vertex_grid[at[:,0],at[:,1],at[:,2]])
            quad=torch.stack(quad,1)
            good=(quad>=0).all(1); quad=quad[good]
            faces.extend([quad[:,[0,1,2]],quad[:,[0,2,3]]])
        triangles=torch.cat(faces) if faces else torch.empty(0,3,dtype=torch.long,device=self.device)
        if len(triangles):
            midpoint=vertices[triangles].mean(1)
            grad,_,_=self.differentials(midpoint)
            normal=torch.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],
                               vertices[triangles[:,2]]-vertices[triangles[:,0]],dim=1)
            flip=(normal*grad).sum(1)<0
            triangles[flip]=triangles[flip][:,[0,2,1]]
        return vertices,triangles

    @torch.inference_mode()
    def view_scores(self, camera_poses, intrinsics, image_shape, *, joints=None,
                    current_joints=None, reachable=None, surface_cache=None) -> torch.Tensor:
        """Score reachable views using ONLY the current map and candidate FK.

        Future depth/RGB is not an argument. A z-buffer of measured surfaces
        occludes candidate information. Unknown cells near surfaces and weakly
        observed surfaces carry the remaining mapping uncertainty.
        """
        poses = torch.as_tensor(camera_poses, dtype=torch.float32, device=self.device)
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError("expected candidate camera poses [C,7]")
        k = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device).clone()
        h, w = image_shape
        r = self.cfg.view_image_size
        k[0] *= r / w
        k[1] *= r / h
        stride = self.cfg.view_grid_stride
        q_weight = self.weight[::stride, ::stride, ::stride]
        q_sdf = self.distance[::stride, ::stride, ::stride]
        query = self.centers.reshape(*self.shape, 3)[::stride, ::stride, ::stride].reshape(-1, 3)
        surface_band = (q_weight > 0) & (q_sdf.abs() <= self.cfg.voxel_size_m * 2)
        near_surface = functional.max_pool3d(surface_band[None, None].float(), 5, 1, 2)[0, 0] > 0
        frontier = (q_weight == 0) & near_surface
        utility = torch.where(frontier, .693147, torch.zeros_like(q_weight))
        utility += (q_weight==0)*(self.cfg.unknown_exploration_weight*.693147)
        utility += surface_band * torch.log1p(1 / (1 + q_weight))
        positive = utility.flatten() > 0
        query, utility = query[positive], utility.flatten()[positive]
        surface, _ = self.surface() if surface_cache is None else surface_cache
        scores = torch.full((len(poses),), -torch.inf, device=self.device)
        feasible = torch.ones(len(poses), dtype=torch.bool, device=self.device) if reachable is None else torch.as_tensor(reachable, device=self.device).bool()
        if feasible.shape != (len(poses),):
            raise ValueError("expected reachable mask [C]")
        feasible = feasible & torch.isfinite(poses).all(1) & ((poses[:,3:].norm(dim=1)-1).abs() <= .005)
        for i, pose in enumerate(poses):
            if not feasible[i]:
                continue
            if len(surface) and float((surface - pose[:3]).norm(dim=1).min()) < self.cfg.camera_clearance_m:
                continue
            zbuffer = torch.full((r * r,), torch.inf, device=self.device)
            if len(surface):
                u, v, z = self._project(surface, pose, k)
                valid = (z > 0) & (u >= 0) & (u < r) & (v >= 0) & (v < r)
                pixel = v[valid].long() * r + u[valid].long()
                zbuffer.scatter_reduce_(0, pixel, z[valid], reduce="amin", include_self=True)
            u, v, z = self._project(query, pose, k)
            inside = (z > 0) & (z < self.cfg.max_depth_m) & (u >= 0) & (u < r) & (v >= 0) & (v < r)
            ids = torch.where(inside)[0]
            pixel = v[ids].long() * r + u[ids].long()
            visible = z[ids] <= zbuffer[pixel] + self.cfg.voxel_size_m * 2
            # Per-pixel maximum avoids rewarding a dense stack behind one ray.
            pixel_gain = torch.zeros(r * r, device=self.device)
            pixel_gain.scatter_reduce_(0, pixel[visible], utility[ids][visible], reduce="amax", include_self=True)
            scores[i] = pixel_gain.mean()
        if joints is not None and current_joints is not None:
            q = torch.as_tensor(joints, dtype=torch.float32, device=self.device)
            current = torch.as_tensor(current_joints, dtype=torch.float32, device=self.device)
            scores -= self.cfg.motion_cost * (q - current).square().mean(dim=1)
        return scores

    @staticmethod
    def joint_step(current, target, lower, upper, max_increment_rad=.03):
        """Kinematic command; caller must validate the swept path and e-stop."""
        if max_increment_rad <= 0:
            raise ValueError("joint increment must be positive")
        bounded = torch.maximum(torch.minimum(target, upper), lower)
        return current + (bounded - current).clamp(-max_increment_rad, max_increment_rad)
