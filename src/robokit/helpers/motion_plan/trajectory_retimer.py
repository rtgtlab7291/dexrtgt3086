# pyright: reportArgumentType=false, reportIndexIssue=false
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import warp as wp

from robokit.robo.robot import Robot


# --- configuration ---------------------------------------------------------
@dataclass
class TrajectoryRetimerConfig:
    """Limits used to retime a completed trajectory."""

    dt_scale: float = 1.05
    minimum_dt: float = 0.02
    accel_limit: float = 15.0
    jerk_limit: float = 7500.0
    max_dt: float = 0.0


# --- kernels ---------------------------------------------------------------
@wp.func
def _compute_retimed_kinematics(
    q: wp.array3d(dtype=wp.float32),
    batch: int,
    velocity_limits: wp.array1d(dtype=wp.float32),
    num_frames: int,
    num_dofs: int,
    dt: float,
    dt_scale: float,
    minimum_dt: float,
    max_dt: float,
    accel_limit: float,
    jerk_limit: float,
    enabled: int,
) -> wp.vec2:
    velocity_dt = float(0.0)
    acceleration = float(0.0)
    jerk = float(0.0)
    for frame in range(num_frames - 1):
        for dof in range(num_dofs):
            diff = wp.abs(q[batch, frame + 1, dof] - q[batch, frame, dof])
            velocity_dt = wp.max(velocity_dt, diff / wp.max(velocity_limits[dof], 1.0e-8))
    for frame in range(num_frames - 2):
        for dof in range(num_dofs):
            diff = (q[batch, frame + 2, dof] - q[batch, frame + 1, dof]) - (
                q[batch, frame + 1, dof] - q[batch, frame, dof]
            )
            acceleration = wp.max(acceleration, wp.abs(diff))
    for frame in range(num_frames - 3):
        for dof in range(num_dofs):
            d0 = q[batch, frame + 1, dof] - q[batch, frame, dof]
            d1 = q[batch, frame + 2, dof] - q[batch, frame + 1, dof]
            d2 = q[batch, frame + 3, dof] - q[batch, frame + 2, dof]
            jerk = wp.max(jerk, wp.abs(d2 - 2.0 * d1 + d0))
    final_dt = dt
    if enabled != 0:
        acceleration_dt = wp.sqrt(acceleration / (accel_limit + 1.0e-8))
        jerk_dt = wp.pow(jerk / (jerk_limit + 1.0e-8), 1.0 / 3.0)
        ceiling = dt
        if max_dt > 0.0:
            ceiling = max_dt
        final_dt = wp.clamp(wp.max(wp.max(velocity_dt, acceleration_dt), jerk_dt) * dt_scale, minimum_dt, ceiling)
    return wp.vec2(final_dt, jerk / (final_dt * final_dt * final_dt + 1.0e-30))


@wp.kernel
def _compute_motion_time_kernel(
    q: wp.array3d(dtype=wp.float32),
    velocity_limits: wp.array1d(dtype=wp.float32),
    num_frames: int,
    num_dofs: int,
    dt: float,
    dt_scale: float,
    minimum_dt: float,
    max_dt: float,
    accel_limit: float,
    jerk_limit: float,
    enabled: int,
    motion_time: wp.array1d(dtype=wp.float32),
):
    batch = wp.tid()
    kinematics = _compute_retimed_kinematics(
        q,
        batch,
        velocity_limits,
        num_frames,
        num_dofs,
        dt,
        dt_scale,
        minimum_dt,
        max_dt,
        accel_limit,
        jerk_limit,
        enabled,
    )
    motion_time[batch] = wp.float32(num_frames - 1) * kinematics[0]


# --- retimer ---------------------------------------------------------------
class TrajectoryRetimer:
    """Compute execution duration after trajectory postprocessing."""

    def __init__(
        self,
        robot: Robot,
        num_frames: int,
        dt: float,
        config: Optional[TrajectoryRetimerConfig],
        device: str,
    ):
        self.num_frames = num_frames
        self.dt = dt
        self.config = config
        self.device = wp.get_device(device)
        self._velocity_limits = wp.from_numpy(
            robot.spec.actuated_joint_velocity_limits.astype(np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self._num_dofs = robot.num_actuated_joints
        self._motion_time: Optional[wp.array] = None

    def warmup(self, batch_size: int):
        """Allocate the borrowed output buffer once."""
        if self._motion_time is None:
            self._motion_time = wp.empty(batch_size, dtype=wp.float32, device=self.device)
        elif self._motion_time.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: expected {self._motion_time.shape[0]}, got {batch_size}")

    def compute_motion_time(self, q: wp.array) -> wp.array:
        """Compute borrowed per-trajectory execution durations.

        Args:
            q: Batched trajectories with shape `(batch, frames, dofs)`.

        Returns:
            Borrowed execution durations with shape `(batch,)`.
        """
        self.warmup(q.shape[0])
        config = self.config
        wp.launch(
            _compute_motion_time_kernel,
            dim=q.shape[0],
            inputs=[
                q,
                self._velocity_limits,
                self.num_frames,
                self._num_dofs,
                self.dt,
                config.dt_scale if config is not None else 1.0,
                config.minimum_dt if config is not None else self.dt,
                config.max_dt if config is not None else 0.0,
                config.accel_limit if config is not None else 1.0,
                config.jerk_limit if config is not None else 1.0,
                int(config is not None),
                self._motion_time,
            ],
            device=self.device,
        )
        return cast(wp.array, self._motion_time)


__all__ = ["TrajectoryRetimer", "TrajectoryRetimerConfig"]
