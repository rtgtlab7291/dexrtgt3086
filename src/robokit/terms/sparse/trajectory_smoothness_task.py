# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportUndefinedVariable=false
"""Sparse finite-difference trajectory smoothness."""

from math import comb
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import (
    se3_adjoint_func,
    se3_compose_func,
    se3_inverse_func,
    se3_jlog_func,
    se3_log_map_func,
)
from robokit.terms.task import GradientTask, SparseTask, SparsityPattern, sparse_cost_and_gradient
from robokit.utils import warp_utils
from robokit.utils.warp_utils import wp_device_type, wp_vec7


_HISTORY_FRAMES = 3  # executed-state ring; index 2 aligns with trajectory frame 0


# --- task -------------------------------------------------------------------
class TrajectorySmoothnessTask(SparseTask, GradientTask):
    """Penalize joint differences and optional floating-base motion."""

    compute_weighted_cost_and_gradient = sparse_cost_and_gradient

    def __init__(
        self,
        robot: "Robot",  # noqa: F821
        num_frames: int,
        weight: Union[float, np.ndarray] = 1.0,
        order: int = 1,
        dt: float = 1.0,
        reference_q: Optional[np.ndarray] = None,
        base_weight: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ):
        if order < 1 or order >= num_frames:
            raise ValueError(f"order must be in [1, {num_frames - 1}], got {order}")
        if base_weight is not None and order != 1:
            raise ValueError("base_weight is only supported for order=1.")
        self.robot = robot
        self.num_frames = num_frames
        self.weight = weight
        self.order = int(order)
        self.dt = float(dt)
        self.reference_q_input = reference_q
        self.base_weight = base_weight
        self.device: Optional[wp_device_type] = None
        self.residual_weight: Optional[wp.array] = None
        self.base_residual_weight: Optional[wp.array] = None
        self.boundary_weight: Optional[wp.array] = None
        self.coefficients: Optional[wp.array] = None
        self.q_history: Optional[wp.array] = None
        self.history_valid: Optional[wp.array] = None
        self.reference_q: Optional[wp.array] = None

    def init_buffers(self, device: wp_device_type):
        """Build finite-difference weights and history buffers.

        Lifecycle:
            1. Build joint and optional base weights.
            2. Build the requested difference stencil.
            3. Allocate history and reference trajectories.
        """
        self.device = device
        num_dofs = self.robot.spec.num_actuated_joints
        num_interior = self.num_frames - self.order
        dt_scale = np.float32(1.0 / self.dt**self.order)

        if isinstance(self.weight, np.ndarray):
            flat = self.weight.astype(np.float32).flatten()
            if flat.size == num_interior * num_dofs:
                residual_weight = flat
            else:
                residual_weight = np.tile(self.weight.astype(np.float32), num_interior)
            dof_weight = self.weight.astype(np.float32)
        else:
            residual_weight = np.full(num_interior * num_dofs, self.weight, dtype=np.float32)
            dof_weight = np.full(num_dofs, self.weight, dtype=np.float32)

        self.residual_weight = wp.from_numpy(residual_weight * dt_scale, dtype=wp.float32, device=device)  # pyright: ignore[reportIncompatibleVariableOverride]
        self.boundary_weight = wp.from_numpy(dof_weight * dt_scale, dtype=wp.float32, device=device)
        coefficients = np.array(
            [((-1) ** (self.order - j)) * comb(self.order, j) for j in range(self.order + 1)], dtype=np.float32
        )
        self.coefficients = wp.from_numpy(coefficients, dtype=wp.float32, device=device)
        if self.q_history is None:
            self.q_history = wp.zeros((1, _HISTORY_FRAMES, num_dofs), dtype=wp.float32, device=device)
            self.history_valid = wp.zeros(1, dtype=wp.float32, device=device)

        if self.base_weight is not None:
            base_weight = np.broadcast_to(np.asarray(self.base_weight, dtype=np.float32), (6,))
            self.base_residual_weight = wp.from_numpy(
                np.tile(base_weight, self.num_frames - 1) / self.dt,
                dtype=wp.float32,
                device=device,
            )

        if self.reference_q is None:
            reference_q = (
                np.zeros((1, self.num_frames, num_dofs), dtype=np.float32)
                if self.reference_q_input is None
                else self.reference_q_input.astype(np.float32)
            )
            if reference_q.shape == (self.num_frames, num_dofs):
                reference_q = reference_q[None]
            if reference_q.shape[1:] != (self.num_frames, num_dofs):
                raise ValueError(f"reference_q must have shape [T, D] or [B, {self.num_frames}, {num_dofs}].")
            self.reference_q = wp.from_numpy(reference_q, dtype=wp.float32, device=device)

    def set_history(self, q_history: Optional[wp.array], num_seeds: int, history_valid: bool):
        """Set the fixed executed prefix. No-op at `order=1`, which never reads it."""
        if self.device is None:
            if q_history is None:
                return
            self.init_buffers(q_history.device)
        if q_history is not None:
            rows = q_history.shape[0] * num_seeds
            if self.q_history.shape[0] != rows:
                self.q_history = wp.zeros(
                    (rows,) + tuple(self.q_history.shape[1:]), dtype=wp.float32, device=self.device
                )
                self.history_valid = wp.zeros(rows, dtype=wp.float32, device=self.device)
            warp_utils.repeat(q_history, num_seeds, out=self.q_history)
        self.history_valid.fill_(1.0 if history_valid else 0.0)

    @property
    def residual_dim(self) -> int:
        channels = self.robot.spec.num_actuated_joints + (6 if self.base_weight is not None else 0)
        return (self.num_frames - 1) * channels

    def _nnz(self, num_dofs: int) -> int:
        joint_nnz = (self.order + 1) * (self.num_frames - 1) * num_dofs
        base_nnz = (self.num_frames - 1) * 6 * 12 if self.base_weight is not None else 0
        return joint_nnz + base_nnz

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
        batch = robot_state.q.shape[0]

        if out_residual is None:
            out_residual = wp.empty((batch, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_trajectory_smoothness_residual_kernel,
            dim=(batch, self.num_frames - 1, num_dofs),
            inputs=[
                robot_state.q,
                self.q_history,
                self.history_valid,
                self.reference_q,
                self.residual_weight,
                self.boundary_weight,
                self.coefficients,
                self.order,
                row_offset,
                num_dofs,
                self.use_mask,
                self._get_frame_mask(),
            ],
            outputs=[out_residual],
            device=kernel_device,
        )
        if self.base_weight is not None:
            if not robot_state.has_floating_base:
                raise ValueError("base_weight requires a floating-base robot state.")
            wp.launch(
                kernel=_compute_base_smoothness_residual_kernel,
                dim=(batch, self.num_frames - 1),
                inputs=[
                    robot_state.T_world_base,
                    self.base_residual_weight,
                    row_offset + (self.num_frames - 1) * num_dofs,
                    wp.float32(1e-4),
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
    ) -> SparsityPattern:  # pyright: ignore[reportGeneralTypeIssues]
        robot_state = var_values.get("robot")
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = robot_state.q.device
        num_dofs = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        nnz = self._nnz(num_dofs)
        joint_residual_dim = (self.num_frames - 1) * num_dofs
        joint_nnz = (self.order + 1) * joint_residual_dim

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=compute_trajectory_smoothness_jacobian_sparse_pattern_kernel,
            dim=(self.num_frames - 1, num_dofs),
            inputs=[num_dofs, base_dim + num_dofs, base_dim, self.order, offset, nnz_offset],
            outputs=[row_indices, col_indices],
            device=kernel_device,
        )
        if self.base_weight is not None:
            if base_dim == 0:
                raise ValueError("base_weight requires a floating-base robot state.")
            wp.launch(
                kernel=_compute_base_smoothness_jacobian_sparse_pattern_kernel,
                dim=(self.num_frames - 1, 6),
                inputs=[base_dim + num_dofs, offset + joint_residual_dim, nnz_offset + joint_nnz],
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
        batch = robot_state.q.shape[0]
        joint_nnz = (self.order + 1) * (self.num_frames - 1) * num_dofs

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((batch, self._nnz(num_dofs)), dtype=wp.float32, device=kernel_device)

        wp.launch(
            kernel=compute_trajectory_smoothness_jacobian_sparse_values_kernel,
            dim=(batch, self.num_frames - 1, num_dofs),
            inputs=[
                self.residual_weight,
                self.boundary_weight,
                self.history_valid,
                self.coefficients,
                self.order,
                num_dofs,
                nnz_offset,
                max(1, batch // self.history_valid.shape[0]),
                self.use_mask,
                self._get_frame_mask(),
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )
        if self.base_weight is not None:
            if not robot_state.has_floating_base:
                raise ValueError("base_weight requires a floating-base robot state.")
            wp.launch(
                kernel=_compute_base_smoothness_jacobian_sparse_values_kernel,
                dim=(batch, self.num_frames - 1),
                inputs=[
                    robot_state.T_world_base,
                    self.base_residual_weight,
                    nnz_offset + joint_nnz,
                    wp.float32(1e-4),
                    self.use_mask,
                    self._get_frame_mask(),
                ],
                outputs=[out_jacobian_values],
                device=kernel_device,
            )
        return out_jacobian_values


# --- device code ------------------------------------------------------------
@wp.func
def _compute_window_mask_func(
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    batch_idx: int,
    first_frame: int,
    last_frame: int,
) -> float:
    if not use_mask:
        return float(1.0)
    return float(frame_mask[batch_idx, first_frame]) * float(frame_mask[batch_idx, last_frame])


@wp.kernel
def compute_trajectory_smoothness_residual_kernel(
    q_traj: wp.array3d(dtype=wp.float32),
    q_history: wp.array3d(dtype=wp.float32),
    history_valid: wp.array1d(dtype=wp.float32),
    reference_q: wp.array3d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    boundary_weight: wp.array1d(dtype=wp.float32),
    coefficients: wp.array1d(dtype=wp.float32),
    order: int,
    row_offset: int,
    num_dofs: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, row_idx, dof_idx = wp.tid()  # type: ignore
    residual_idx = row_idx * num_dofs + dof_idx
    interior_start = row_idx - (order - 1)
    is_interior = interior_start >= 0
    # per-batch buffers broadcast over the launch batch, each by its own row count
    ref_idx = batch_idx // wp.max(1, q_traj.shape[0] // reference_q.shape[0])
    hist_idx = batch_idx // wp.max(1, q_traj.shape[0] // q_history.shape[0])

    diff = wp.float32(0.0)
    for j in range(order + 1):
        frame = interior_start + j
        if frame < 0:
            # history index 2 aligns with trajectory frame 0
            value = q_history[hist_idx, 2 + frame, dof_idx]
        elif is_interior:
            value = q_traj[batch_idx, frame, dof_idx] - reference_q[ref_idx, frame, dof_idx]
        else:
            value = q_traj[batch_idx, frame, dof_idx]
        diff += coefficients[j] * value

    if is_interior:
        weight = residual_weight[interior_start * num_dofs + dof_idx]
        mask = _compute_window_mask_func(use_mask, frame_mask, batch_idx, interior_start, interior_start + order)
    else:
        weight = history_valid[hist_idx] * boundary_weight[dof_idx]
        mask = wp.float32(1.0)

    out_residual[batch_idx, row_offset + residual_idx] = weight * diff * mask


@wp.kernel
def compute_trajectory_smoothness_jacobian_sparse_pattern_kernel(
    num_dofs: int,
    single_tangent_dim: int,
    base_dim: int,
    order: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    row_idx, dof_idx = wp.tid()  # type: ignore
    residual_idx = row_idx * num_dofs + dof_idx
    base = nnz_offset + (order + 1) * residual_idx
    interior_start = row_idx - (order - 1)

    for j in range(order + 1):
        # map history slots onto zero-valued leading trajectory columns
        frame = interior_start + j
        if interior_start < 0:
            frame = j
        row_indices[base + j] = row_offset + residual_idx
        col_indices[base + j] = frame * single_tangent_dim + base_dim + dof_idx


@wp.kernel
def compute_trajectory_smoothness_jacobian_sparse_values_kernel(
    residual_weight: wp.array1d(dtype=wp.float32),
    boundary_weight: wp.array1d(dtype=wp.float32),
    history_valid: wp.array1d(dtype=wp.float32),
    coefficients: wp.array1d(dtype=wp.float32),
    order: int,
    num_dofs: int,
    nnz_offset: int,
    # rows launched per history_valid row; this kernel reads no state array to derive it from
    batch_expansion: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, row_idx, dof_idx = wp.tid()  # type: ignore
    residual_idx = row_idx * num_dofs + dof_idx
    base = nnz_offset + (order + 1) * residual_idx
    interior_start = row_idx - (order - 1)

    if interior_start >= 0:
        weight = residual_weight[interior_start * num_dofs + dof_idx]
        mask = _compute_window_mask_func(use_mask, frame_mask, batch_idx, interior_start, interior_start + order)
        for j in range(order + 1):
            values[batch_idx, base + j] = coefficients[j] * weight * mask
        return

    weight = history_valid[batch_idx // batch_expansion] * boundary_weight[dof_idx]
    for j in range(order + 1):
        values[batch_idx, base + j] = wp.float32(0.0)
    for j in range(order + 1):
        frame = interior_start + j
        if frame >= 0:
            values[batch_idx, base + frame] = coefficients[j] * weight


@wp.kernel
def _compute_base_smoothness_residual_kernel(
    T_world_base: wp.array2d(dtype=wp_vec7),
    residual_weight: wp.array1d(dtype=wp.float32),
    row_offset: int,
    eps: wp.float32,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx = wp.tid()
    T_rel = se3_compose_func(
        se3_inverse_func(T_world_base[batch_idx, frame_idx]),
        T_world_base[batch_idx, frame_idx + 1],
    )
    error = se3_log_map_func(T_rel, eps)
    mask = _compute_window_mask_func(use_mask, frame_mask, batch_idx, frame_idx, frame_idx + 1)
    for i in range(6):
        residual_idx = frame_idx * 6 + i
        out_residual[batch_idx, row_offset + residual_idx] = residual_weight[residual_idx] * error[i] * mask


@wp.kernel
def _compute_base_smoothness_jacobian_sparse_pattern_kernel(
    single_tangent_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    frame_idx, residual_dof_idx = wp.tid()
    residual_idx = frame_idx * 6 + residual_dof_idx
    nnz_base = nnz_offset + 12 * residual_idx
    for col_dof in range(6):
        row_indices[nnz_base + col_dof] = row_offset + residual_idx
        col_indices[nnz_base + col_dof] = frame_idx * single_tangent_dim + col_dof
        row_indices[nnz_base + 6 + col_dof] = row_offset + residual_idx
        col_indices[nnz_base + 6 + col_dof] = (frame_idx + 1) * single_tangent_dim + col_dof


@wp.kernel
def _compute_base_smoothness_jacobian_sparse_values_kernel(
    T_world_base: wp.array2d(dtype=wp_vec7),
    residual_weight: wp.array1d(dtype=wp.float32),
    nnz_offset: int,
    eps: wp.float32,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx = wp.tid()
    T_rel = se3_compose_func(
        se3_inverse_func(T_world_base[batch_idx, frame_idx]),
        T_world_base[batch_idx, frame_idx + 1],
    )
    jlog_R = se3_jlog_func(T_rel, eps)
    jlog_L = jlog_R * se3_adjoint_func(se3_inverse_func(T_rel))
    mask = _compute_window_mask_func(use_mask, frame_mask, batch_idx, frame_idx, frame_idx + 1)
    for residual_dof_idx in range(6):
        residual_idx = frame_idx * 6 + residual_dof_idx
        weight = residual_weight[residual_idx] * mask
        nnz_base = nnz_offset + 12 * residual_idx
        for col_dof in range(6):
            values[batch_idx, nnz_base + col_dof] = -weight * jlog_L[residual_dof_idx, col_dof]
            values[batch_idx, nnz_base + 6 + col_dof] = weight * jlog_R[residual_dof_idx, col_dof]


if TYPE_CHECKING:
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot
