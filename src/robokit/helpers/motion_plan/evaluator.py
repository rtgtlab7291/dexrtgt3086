# pyright: reportArgumentType=false, reportIndexIssue=false
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import warp as wp

from robokit.robo.robot import Robot
from robokit.utils.warp_utils import wp_vec7


ArrayT = TypeVar("ArrayT")


# --- results ---------------------------------------------------------------
@dataclass
class MotionPlanMetrics(Generic[ArrayT]):
    """Endpoint success metrics whose arrays are borrowed from the evaluator."""

    success: ArrayT
    position_error_m: ArrayT
    orientation_error_rad: ArrayT


# --- kernels ---------------------------------------------------------------
@wp.func
def _compute_ee_pose_error(ee: wp_vec7, goal: wp_vec7) -> wp.vec2:
    dx = ee[0] - goal[0]
    dy = ee[1] - goal[1]
    dz = ee[2] - goal[2]
    position_error = wp.sqrt(dx * dx + dy * dy + dz * dz)
    rx = ee[3] * goal[4] - ee[4] * goal[3] - ee[5] * goal[6] + ee[6] * goal[5]
    ry = ee[3] * goal[5] - ee[5] * goal[3] - ee[6] * goal[4] + ee[4] * goal[6]
    rz = ee[3] * goal[6] - ee[6] * goal[3] - ee[4] * goal[5] + ee[5] * goal[4]
    orientation_error = 2.0 * wp.asin(wp.min(wp.sqrt(rx * rx + ry * ry + rz * rz), 1.0))
    return wp.vec2(position_error, orientation_error)


@wp.kernel
def _compute_pose_goal_metrics_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    T_world_target: wp.array1d(dtype=wp_vec7),
    ee_link_index: int,
    last_frame: int,
    position_threshold_m: float,
    orientation_threshold_rad: float,
    success: wp.array1d(dtype=wp.bool),
    position_error_m: wp.array1d(dtype=wp.float32),
    orientation_error_rad: wp.array1d(dtype=wp.float32),
):
    batch = wp.tid()
    error = _compute_ee_pose_error(T_world_link[batch, last_frame, ee_link_index], T_world_target[batch])
    position_error_m[batch] = error[0]
    orientation_error_rad[batch] = error[1]
    success[batch] = error[0] < position_threshold_m and error[1] < orientation_threshold_rad


@wp.kernel
def _compute_joint_goal_metrics_kernel(
    q: wp.array3d(dtype=wp.float32),
    target_q: wp.array2d(dtype=wp.float32),
    last_frame: int,
    num_dofs: int,
    q_threshold_rad: float,
    success: wp.array1d(dtype=wp.bool),
    q_error_rad: wp.array1d(dtype=wp.float32),
    orientation_error_rad: wp.array1d(dtype=wp.float32),
):
    batch = wp.tid()
    error_sq = float(0.0)
    for dof in range(num_dofs):
        error = q[batch, last_frame, dof] - target_q[batch, dof]
        error_sq += error * error
    error = wp.sqrt(error_sq)
    q_error_rad[batch] = error
    orientation_error_rad[batch] = 0.0
    success[batch] = error < q_threshold_rad


# --- evaluator -------------------------------------------------------------
class MotionPlanEvaluator:
    """Compute endpoint success independently from trajectory optimization and benchmarking."""

    def __init__(
        self,
        robot: Robot,
        ee_link_index: int,
        num_frames: int,
        position_threshold_m: float,
        orientation_threshold_rad: float,
        q_threshold_rad: float,
        device: str,
    ):
        self.robot = robot
        self.ee_link_index = ee_link_index
        self.num_frames = num_frames
        self.position_threshold_m = position_threshold_m
        self.orientation_threshold_rad = orientation_threshold_rad
        self.q_threshold_rad = q_threshold_rad
        self.device = wp.get_device(device)
        self._batch_size: Optional[int] = None

    def warmup(self, batch_size: int):
        """Allocate batch-shaped evaluation buffers once."""
        if self._batch_size is not None:
            if batch_size != self._batch_size:
                raise ValueError(f"Batch size mismatch: expected {self._batch_size}, got {batch_size}")
            return
        self._state = self.robot.state(
            q=wp.empty(
                (batch_size, self.num_frames, self.robot.num_actuated_joints),
                dtype=wp.float32,
                device=self.device,
            )
        )
        self._batch_size = batch_size

    def build_metrics(self, batch_size: int) -> MotionPlanMetrics[wp.array]:
        """Build borrowed metric buffers for `batch_size` trajectories."""
        return MotionPlanMetrics(
            success=wp.empty(batch_size, dtype=wp.bool, device=self.device),
            position_error_m=wp.empty(batch_size, dtype=wp.float32, device=self.device),
            orientation_error_rad=wp.empty(batch_size, dtype=wp.float32, device=self.device),
        )

    def evaluate(
        self,
        q: wp.array,
        T_world_target: Optional[wp.array] = None,
        target_q: Optional[wp.array] = None,
        out: Optional[MotionPlanMetrics[wp.array]] = None,
    ) -> MotionPlanMetrics[wp.array]:
        """Compute metrics for exactly one pose or joint-space goal.

        Args:
            q: Batched trajectories with shape `(batch, frames, dofs)`.
            T_world_target: Optional batched pose goals.
            target_q: Optional batched joint-space goals.
            out: Optional borrowed output buffers.

        Returns:
            Borrowed endpoint metrics.

        """
        if (T_world_target is None) == (target_q is None):
            raise ValueError("Provide exactly one of T_world_target or target_q.")
        self.warmup(q.shape[0])
        metrics = out if out is not None else self.build_metrics(q.shape[0])
        if target_q is not None:
            wp.launch(
                _compute_joint_goal_metrics_kernel,
                dim=q.shape[0],
                inputs=[
                    q,
                    target_q,
                    self.num_frames - 1,
                    self.robot.num_actuated_joints,
                    self.q_threshold_rad,
                    metrics.success,
                    metrics.position_error_m,
                    metrics.orientation_error_rad,
                ],
                device=self.device,
            )
            return metrics
        wp.copy(self._state.q, q)
        self._state.invalidate()
        self.robot.forward_kinematics(self._state)
        wp.launch(
            _compute_pose_goal_metrics_kernel,
            dim=q.shape[0],
            inputs=[
                self._state.T_world_link,
                T_world_target,
                self.ee_link_index,
                self.num_frames - 1,
                self.position_threshold_m,
                self.orientation_threshold_rad,
                metrics.success,
                metrics.position_error_m,
                metrics.orientation_error_rad,
            ],
            device=self.device,
        )
        return metrics


__all__ = ["MotionPlanEvaluator", "MotionPlanMetrics"]
