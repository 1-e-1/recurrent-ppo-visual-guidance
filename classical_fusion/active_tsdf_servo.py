"""Streaming joint-control interface for the single-map active vision route."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .active_tsdf import ActiveTSDF


@dataclass
class ServoDecision:
    joint_command: torch.Tensor
    selected_candidate: int
    information_score: float
    status: str


class ActiveTSDFServo:
    """Fuse synchronized frames; occasionally choose a reachable next view.

    PPO/world-model or a deterministic joint-space proposal bank can provide
    candidates. ``forward_kinematics(q_batch)`` returns [C,7] camera poses.
    ``path_is_safe(current, targets)`` must check the entire swept arm path,
    joint limits and other arms; it returns [C] booleans. Endpoint camera
    clearance in the map does not replace this callback.
    """
    def __init__(self, mapper: ActiveTSDF, joint_lower, joint_upper, *,
                 forward_kinematics, path_is_safe, max_increment_rad=.03,
                 selection_interval_frames=15, target_tolerance_rad=.01):
        if max_increment_rad <= 0 or selection_interval_frames < 1:
            raise ValueError("invalid servo timing")
        self.mapper = mapper
        self.lower = torch.as_tensor(joint_lower, dtype=torch.float32, device=mapper.device)
        self.upper = torch.as_tensor(joint_upper, dtype=torch.float32, device=mapper.device)
        if (self.lower.ndim != 1 or self.lower.numel() == 0
                or self.lower.shape != self.upper.shape
                or not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all()
                or (self.lower >= self.upper).any()):
            raise ValueError("invalid joint limits")
        self.fk = forward_kinematics
        self.path_is_safe = path_is_safe
        self.increment = max_increment_rad
        self.interval = selection_interval_frames
        self.tolerance = target_tolerance_rad
        self.target = None
        self.target_index = -1
        self.last_score = 0.
        self.since_selection = self.interval
        self.last_frame_timestamp = None

    def _safe_paths(self, current, targets):
        safe = torch.as_tensor(self.path_is_safe(current, targets), device=current.device)
        if safe.shape != (len(targets),) or safe.dtype != torch.bool:
            raise ValueError("path checker must return booleans [C]")
        return safe

    def reset(self):
        self.mapper.reset()
        self.target = None
        self.target_index = -1
        self.since_selection = self.interval
        self.last_score = 0.
        self.last_frame_timestamp = None

    @torch.inference_mode()
    def step(self, depth, intrinsics, camera_pose, joints, candidate_joints, *,
             frame_timestamp_s, pose_timestamp_s, emergency_stop=False,
             max_timestamp_skew_s=.01) -> ServoDecision:
        q = torch.as_tensor(joints, dtype=torch.float32, device=self.mapper.device)
        if q.shape != self.lower.shape or not torch.isfinite(q).all():
            raise ValueError("joint readback is invalid")
        if emergency_stop:
            self.target = None
            return ServoDecision(q.clone(), -1, 0., "emergency_stop_hold")
        if ((q < self.lower) | (q > self.upper)).any():
            self.target = None
            return ServoDecision(q.clone(), -1, 0., "joint_limit_hold")
        if (not all(math.isfinite(t) for t in (frame_timestamp_s, pose_timestamp_s, max_timestamp_skew_s))
                or max_timestamp_skew_s < 0
                or abs(frame_timestamp_s-pose_timestamp_s) > max_timestamp_skew_s):
            self.target = None
            return ServoDecision(q.clone(), -1, 0., "unsynchronized_frame_hold")
        if self.last_frame_timestamp is not None and frame_timestamp_s <= self.last_frame_timestamp:
            self.target = None
            return ServoDecision(q.clone(), -1, 0., "stale_frame_hold")
        update = self.mapper.integrate(depth, intrinsics, camera_pose)
        self.last_frame_timestamp = frame_timestamp_s
        if update.observed_voxels == 0:
            self.target = None
            return ServoDecision(q.clone(), -1, 0., "no_valid_depth_hold")
        self.since_selection += 1
        arrived = self.target is None or bool((self.target-q).abs().max() <= self.tolerance)
        if arrived and self.since_selection >= self.interval:
            candidates = torch.as_tensor(candidate_joints, dtype=torch.float32, device=self.mapper.device)
            if candidates.ndim != 2 or candidates.shape[1:] != q.shape:
                raise ValueError("expected candidate joints [C,J]")
            if len(candidates) == 0:
                self.target = None
                return ServoDecision(q.clone(), -1, 0., "no_safe_candidate_hold")
            within_limits = ((candidates >= self.lower) & (candidates <= self.upper)).all(1)
            within_limits &= torch.isfinite(candidates).all(1)
            # Only pass limit-valid proposals into FK / collision geometry.
            candidates_safe_input = torch.where(within_limits[:,None], candidates, q[None])
            path_safe = self._safe_paths(q, candidates_safe_input)
            poses = self.fk(candidates_safe_input)
            scores = self.mapper.view_scores(
                poses, intrinsics, tuple(torch.as_tensor(depth).squeeze().shape),
                joints=candidates_safe_input, current_joints=q, reachable=within_limits & path_safe,
            )
            if not torch.isfinite(scores).any():
                self.target = None
                return ServoDecision(q.clone(), -1, 0., "no_safe_candidate_hold")
            self.target_index = int(scores.argmax())
            self.target = candidates[self.target_index].clone()
            self.last_score = float(scores[self.target_index])
            self.since_selection = 0
        if self.target is None:
            return ServoDecision(q.clone(), -1, 0., "integrating_hold")
        command = self.mapper.joint_step(q, self.target, self.lower, self.upper, self.increment)
        # Recheck on every step because other arms may have moved since the
        # view was selected. The callback is intentionally fail-closed.
        if not bool(self._safe_paths(q, command[None])[0]):
            self.target = None
            return ServoDecision(q.clone(), -1, 0., "path_changed_hold")
        at_target = bool((self.target-q).abs().max() <= self.tolerance)
        return ServoDecision(command, self.target_index, self.last_score, "observing" if at_target else "moving")
