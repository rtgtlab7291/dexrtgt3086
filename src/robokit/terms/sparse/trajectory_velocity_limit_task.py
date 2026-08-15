# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportUndefinedVariable=false
"""Sparse trajectory joint-velocity limits."""

from typing import TYPE_CHECKING, Optional

import numpy as np
import warp as wp

from robokit.terms.task import GradientTask, SparseTask, SparsityPattern, sparse_cost_and_gradient
from robokit.utils.warp_utils import wp_device_type


# --- task -------------------------------------------------------------------
class TrajectoryVelocityLimitTask(SparseTask, GradientTask):
    """Penalize joint velocity limit violations between consecutive trajectory frames."""

    compute_weighted_cost_and_gradient = sparse_cost_and_gradient

    def __init__(
        self,
        robot: "Robot",  # noqa: F821
        num_frames: int,
        dt: float,
        velocity_limits: Optional[np.ndarray] = None,
        weight: float = 1.0,
    ):
        self.robot = robot
        self.num_frames = num_frames
        self.dt = dt
        self.weight = weight

        velocity_limits_np = (
            velocity_limits if velocity_limits is not None else robot.spec.actuated_joint_velocity_limits
        )
        self._velocity_limits_np = velocity_limits_np.astype(np.float32)

        num_dofs = robot.spec.num_actuated_joints
        residual_weight_np = np.full(num_dofs, weight, dtype=np.float32)
        self._residual_weight_np = residual_weight_np

        self.device: Optional[wp_device_type] = None
        self.velocity_limits = None
        self.residual_weight = None

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.velocity_limits = wp.from_numpy(self._velocity_limits_np, dtype=wp.float32, device=device)
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return (self.num_frames - 1) * self.robot.spec.num_actuated_joints

    def compute_weighted_residual(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get("robot")
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = out_residual.device if out_residual is not None else self.device
        num_dofs = self.robot.spec.num_actuated_joints

        if out_residual is None:
            out_residual = wp.empty((robot_state.q.shape[0], self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_trajectory_velocity_limit_residual_kernel,
            dim=(robot_state.q.shape[0], self.num_frames - 1, num_dofs),
            inputs=[
                robot_state.q,
                self.velocity_limits,
                self.residual_weight,
                self.dt,
                row_offset,
                num_dofs,
                self.use_mask,
                self._get_frame_mask(),
            ],
            outputs=[out_residual],
            device=kernel_device,
        )

        return out_residual

    def compute_sparse_jacobian_pattern(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> SparsityPattern:
        robot_state = var_values.get("robot")
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = robot_state.q.device
        num_dofs = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        single_tangent_dim = base_dim + num_dofs

        nnz = (self.num_frames - 1) * num_dofs * 2

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=compute_trajectory_velocity_limit_jacobian_pattern_kernel,
            dim=(self.num_frames - 1, num_dofs),
            inputs=[
                num_dofs,
                single_tangent_dim,
                base_dim,
                offset,
                nnz_offset,
            ],
            outputs=[row_indices, col_indices],
            device=kernel_device,
        )

        pattern = SparsityPattern()
        pattern.row_indices = row_indices
        pattern.col_indices = col_indices
        return pattern

    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get("robot")
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = robot_state.q.device
        num_dofs = self.robot.spec.num_actuated_joints
        nnz = (self.num_frames - 1) * num_dofs * 2

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((robot_state.q.shape[0], nnz), dtype=wp.float32, device=kernel_device)

        wp.launch(
            kernel=compute_trajectory_velocity_limit_jacobian_values_kernel,
            dim=(robot_state.q.shape[0], self.num_frames - 1, num_dofs),
            inputs=[
                robot_state.q,
                self.velocity_limits,
                self.residual_weight,
                self.dt,
                num_dofs,
                nnz_offset,
                self.use_mask,
                self._get_frame_mask(),
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )

        return out_jacobian_values


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_trajectory_velocity_limit_residual_kernel(
    q_traj: wp.array3d(dtype=wp.float32),
    velocity_limits: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    dt: float,
    row_offset: int,
    num_dofs: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, pair_idx, dof_idx = wp.tid()

    frame_curr = pair_idx + 1
    frame_prev = pair_idx

    q_curr = q_traj[batch_idx, frame_curr, dof_idx]
    q_prev = q_traj[batch_idx, frame_prev, dof_idx]
    q_dot = (q_curr - q_prev) / dt

    v_limit = velocity_limits[dof_idx]
    violation = wp.max(0.0, wp.abs(q_dot) - v_limit)
    mask = wp.float32(1.0)
    if use_mask:
        mask = wp.float32(frame_mask[batch_idx, pair_idx]) * wp.float32(frame_mask[batch_idx, pair_idx + 1])

    residual_idx = pair_idx * num_dofs + dof_idx
    out_residual[batch_idx, row_offset + residual_idx] = residual_weight[dof_idx] * violation * mask


@wp.kernel
def compute_trajectory_velocity_limit_jacobian_pattern_kernel(
    num_dofs: int,
    single_tangent_dim: int,
    base_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    pair_idx, dof_idx = wp.tid()

    residual_idx = row_offset + pair_idx * num_dofs + dof_idx

    frame_curr = pair_idx + 1
    frame_prev = pair_idx

    col_idx_curr = frame_curr * single_tangent_dim + base_dim + dof_idx
    nnz_idx_curr = nnz_offset + (pair_idx * num_dofs + dof_idx) * 2 + 0

    row_indices[nnz_idx_curr] = residual_idx
    col_indices[nnz_idx_curr] = col_idx_curr

    col_idx_prev = frame_prev * single_tangent_dim + base_dim + dof_idx
    nnz_idx_prev = nnz_offset + (pair_idx * num_dofs + dof_idx) * 2 + 1

    row_indices[nnz_idx_prev] = residual_idx
    col_indices[nnz_idx_prev] = col_idx_prev


@wp.kernel
def compute_trajectory_velocity_limit_jacobian_values_kernel(
    q_traj: wp.array3d(dtype=wp.float32),
    velocity_limits: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    dt: float,
    num_dofs: int,
    nnz_offset: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    jacobian_values: wp.array2d(dtype=wp.float32),
):
    batch_idx, pair_idx, dof_idx = wp.tid()

    frame_curr = pair_idx + 1
    frame_prev = pair_idx

    q_curr = q_traj[batch_idx, frame_curr, dof_idx]
    q_prev = q_traj[batch_idx, frame_prev, dof_idx]
    q_dot = (q_curr - q_prev) / dt

    v_limit = velocity_limits[dof_idx]
    is_violating = wp.abs(q_dot) > v_limit
    mask = wp.float32(1.0)
    if use_mask:
        mask = wp.float32(frame_mask[batch_idx, pair_idx]) * wp.float32(frame_mask[batch_idx, pair_idx + 1])

    if is_violating:
        sign = 1.0 if q_dot > 0.0 else -1.0
        jacobian_value = residual_weight[dof_idx] * sign / dt * mask

        nnz_idx_curr = nnz_offset + (pair_idx * num_dofs + dof_idx) * 2 + 0
        jacobian_values[batch_idx, nnz_idx_curr] = jacobian_value

        nnz_idx_prev = nnz_offset + (pair_idx * num_dofs + dof_idx) * 2 + 1
        jacobian_values[batch_idx, nnz_idx_prev] = -jacobian_value
    else:
        nnz_idx_curr = nnz_offset + (pair_idx * num_dofs + dof_idx) * 2 + 0
        jacobian_values[batch_idx, nnz_idx_curr] = 0.0

        nnz_idx_prev = nnz_offset + (pair_idx * num_dofs + dof_idx) * 2 + 1
        jacobian_values[batch_idx, nnz_idx_prev] = 0.0


if TYPE_CHECKING:
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot
