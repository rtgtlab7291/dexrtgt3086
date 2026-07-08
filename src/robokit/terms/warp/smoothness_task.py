# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.warp_se3_kernels import se3_inverse_func, se3_jlog_func, se3_log_map_func, se3_multiply_func
from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


class WarpSmoothnessTask(WarpTask):
    """
    Warp-based smoothness task that penalizes velocity changes.

    Residual: r = q_current - q_previous
    Jacobian: J = I (identity matrix for joints)

    For floating base, includes base twist difference.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="warp")
        >>> state_prev = robot.state(q=robot.spec.zero_q)
        >>> state_curr = robot.state(q=robot.spec.zero_q + 0.1)
        >>> smooth_task = WarpSmoothnessTask(robot=robot, weight=0.5, batch_size=1)
    """

    def __init__(
        self,
        robot: WarpRobot,
        prev_var: Optional[WarpRobotState] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        base_weight: Optional[Union[float, Sequence[float]]] = None,
        batch_size: int = 1,
        num_seeds: int = 1,
    ):
        """
        Args:
            robot: Robot instance
            prev_var: Previous state (can be set later)
            weight: Weight for joint smoothness
            base_weight: Weight for base smoothness (scalar or 6D)
            batch_size: Batch size
            num_seeds: Number of seeds per batch element
        """
        self.robot = robot
        self.batch_size = batch_size
        self.num_seeds = num_seeds
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be >= 1.")
        if self.batch_size % self.num_seeds != 0:
            raise ValueError("batch_size must be divisible by num_seeds.")
        self._prev_batch_size = self.batch_size // self.num_seeds
        self.prev_var = prev_var.clone() if prev_var is not None else None

        num_joints = robot.num_actuated_joints
        self._include_base_residual = base_weight is not None or (prev_var is not None and prev_var.has_floating_base)

        residual_weight = np.ones(num_joints, dtype=np.float32)
        if weight is not None:
            if isinstance(weight, (float, int)):
                residual_weight[:] = weight
            else:
                residual_weight[:] = weight

        self._joint_weight_np = residual_weight.copy()
        self._base_weight = base_weight
        if self._include_base_residual:
            self._residual_dim = num_joints + 6
            residual_weight_full = np.ones(self._residual_dim, dtype=np.float32)
            residual_weight_full[:num_joints] = self._joint_weight_np
            if self._base_weight is not None:
                if isinstance(self._base_weight, (float, int)):
                    residual_weight_full[num_joints:] = self._base_weight
                else:
                    residual_weight_full[num_joints:] = self._base_weight
            else:
                residual_weight_full[num_joints:] = self._joint_weight_np[0] if len(self._joint_weight_np) > 0 else 1.0
            self._residual_weight_np = residual_weight_full
        else:
            self._residual_dim = num_joints
            self._residual_weight_np = residual_weight
        self.device = None
        self.residual_weight = None

    def set_prev_state(self, prev_state: WarpRobotState) -> None:
        if prev_state.batch_size != self._prev_batch_size:
            raise ValueError(
                f"prev_state batch size mismatch: expected {self._prev_batch_size}, got {prev_state.batch_size}"
            )
        if self.prev_var is None:
            self.prev_var = prev_state.clone()
            return
        if self.prev_var.q.shape != prev_state.q.shape:
            self.prev_var = prev_state.clone()
            return
        wp.copy(self.prev_var.q, prev_state.q)
        if self.prev_var.has_floating_base and prev_state.has_floating_base:
            wp.copy(self.prev_var.T_world_base.xyz_wxyz, prev_state.T_world_base.xyz_wxyz)

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        prev_var: Optional[WarpRobotState] = None,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if prev_var is None:
            prev_var = self.prev_var

        if prev_var is None:
            # Return zeros if no previous state
            if residual_buffer is None:
                return wp.zeros((self.batch_size, self.residual_dim), dtype=wp.float32, device=var.q.device)
            else:
                return residual_buffer
        if prev_var.batch_size != self._prev_batch_size:
            raise ValueError(
                f"prev_state batch size mismatch: expected {self._prev_batch_size}, got {prev_var.batch_size}"
            )
        if self._include_base_residual and (not var.has_floating_base or not prev_var.has_floating_base):
            raise ValueError(
                "WarpSmoothnessTask is configured with base residuals but var/prev_var has no floating base."
            )

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = residual_buffer.device if residual_buffer is not None else self.device
        if residual_buffer is None:
            residual_buffer = wp.empty((self.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self._include_base_residual:
            wp.launch(
                kernel=compute_smoothness_task_weighted_residual_with_base_kernel,
                dim=self.batch_size,
                inputs=[
                    var.q,
                    prev_var.q,
                    var.T_world_base.xyz_wxyz,
                    prev_var.T_world_base.xyz_wxyz,
                    self.num_seeds,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[residual_buffer],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_smoothness_task_weighted_residual_kernel,
                dim=(self.batch_size, self.robot.num_actuated_joints),
                inputs=[
                    var.q,
                    prev_var.q,
                    self.num_seeds,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[residual_buffer],
                device=kernel_device,
            )

        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: WarpRobotState,
        prev_var: Optional[WarpRobotState] = None,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> wp.array:
        if prev_var is None:
            prev_var = self.prev_var

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else self.device
        total_dofs = var.tangent_dim

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (self.batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device
            )
            row_offset = 0
            col_offset = 0

        if prev_var is None:
            return jacobian_buffer
        if prev_var.batch_size != self._prev_batch_size:
            raise ValueError(
                f"prev_state batch size mismatch: expected {self._prev_batch_size}, got {prev_var.batch_size}"
            )
        if self._include_base_residual and (not var.has_floating_base or not prev_var.has_floating_base):
            raise ValueError(
                "WarpSmoothnessTask is configured with base residuals but var/prev_var has no floating base."
            )

        if self._include_base_residual:
            wp.launch(
                kernel=compute_smoothness_task_weighted_jacobian_with_base_kernel,
                dim=(self.batch_size, total_dofs),
                inputs=[
                    var.T_world_base.xyz_wxyz,
                    prev_var.T_world_base.xyz_wxyz,
                    self.num_seeds,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                    self.robot.num_actuated_joints,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_smoothness_task_weighted_jacobian_kernel,
                dim=(self.batch_size, self.robot.num_actuated_joints),
                inputs=[
                    var.has_floating_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )

        return jacobian_buffer


@wp.kernel
def compute_smoothness_task_weighted_residual_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    num_seeds: int,
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore
    prev_idx = batch_idx // num_seeds
    diff = q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]
    residual_buffer[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * diff


@wp.kernel
def compute_smoothness_task_weighted_residual_with_base_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    T_world_base_curr: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_prev: wp.array1d(dtype=wp_vec7),  # [batch]
    num_seeds: int,
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx = wp.tid()  # type: ignore
    num_joints = q_curr.shape[1]
    prev_idx = batch_idx // num_seeds

    # Joint residuals
    for joint_idx in range(num_joints):
        diff = q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]
        residual_buffer[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * diff

    # Base residuals
    T_prev_inv = se3_inverse_func(T_world_base_prev[prev_idx])
    T_error = se3_multiply_func(T_world_base_curr[batch_idx], T_prev_inv)
    r_base = se3_log_map_func(T_error, wp.float32(1e-4))

    residual_buffer[batch_idx, row_offset + num_joints + 0] = residual_weight[num_joints + 0] * r_base[0]
    residual_buffer[batch_idx, row_offset + num_joints + 1] = residual_weight[num_joints + 1] * r_base[1]
    residual_buffer[batch_idx, row_offset + num_joints + 2] = residual_weight[num_joints + 2] * r_base[2]
    residual_buffer[batch_idx, row_offset + num_joints + 3] = residual_weight[num_joints + 3] * r_base[3]
    residual_buffer[batch_idx, row_offset + num_joints + 4] = residual_weight[num_joints + 4] * r_base[4]
    residual_buffer[batch_idx, row_offset + num_joints + 5] = residual_weight[num_joints + 5] * r_base[5]


@wp.kernel
def compute_smoothness_task_weighted_jacobian_kernel(
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    row_offset: int,
    col_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore

    base_col_offset = 6 if has_floating_base else 0

    # Identity Jacobian for joints
    jacobian_buffer[batch_idx, row_offset + joint_idx, col_offset + base_col_offset + joint_idx] = residual_weight[
        joint_idx
    ]


@wp.kernel
def compute_smoothness_task_weighted_jacobian_with_base_kernel(
    T_world_base_curr: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_prev: wp.array1d(dtype=wp_vec7),  # [batch]
    num_seeds: int,
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    row_offset: int,
    col_offset: int,
    num_joints: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, col_idx = wp.tid()  # type: ignore
    prev_idx = batch_idx // num_seeds

    # Joint block: identity for joint columns (starting at column 6)
    if col_idx >= 6 and col_idx < (6 + num_joints):
        joint_idx = col_idx - 6
        jacobian_buffer[batch_idx, row_offset + joint_idx, col_offset + col_idx] = residual_weight[joint_idx]

    # Base block: Jlog @ Ad for base columns (first 6)
    if col_idx < 6:
        T_prev_inv = se3_inverse_func(T_world_base_prev[prev_idx])
        T_error = se3_multiply_func(T_world_base_curr[batch_idx], T_prev_inv)

        # Compute Jlog @ Ad_T column by column
        jlog = se3_jlog_func(T_error, wp.float32(1e-4))

        # Ad_T column for unit vector e_col_idx
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = 1.0
        ad_col = se3_adjoint_multiply_vec6_func(T_world_base_prev[prev_idx], unit_vec)

        # Jlog @ ad_col
        result_col = wp.mul(jlog, ad_col)

        jacobian_buffer[batch_idx, row_offset + num_joints + 0, col_offset + col_idx] = (
            residual_weight[num_joints + 0] * result_col[0]
        )
        jacobian_buffer[batch_idx, row_offset + num_joints + 1, col_offset + col_idx] = (
            residual_weight[num_joints + 1] * result_col[1]
        )
        jacobian_buffer[batch_idx, row_offset + num_joints + 2, col_offset + col_idx] = (
            residual_weight[num_joints + 2] * result_col[2]
        )
        jacobian_buffer[batch_idx, row_offset + num_joints + 3, col_offset + col_idx] = (
            residual_weight[num_joints + 3] * result_col[3]
        )
        jacobian_buffer[batch_idx, row_offset + num_joints + 4, col_offset + col_idx] = (
            residual_weight[num_joints + 4] * result_col[4]
        )
        jacobian_buffer[batch_idx, row_offset + num_joints + 5, col_offset + col_idx] = (
            residual_weight[num_joints + 5] * result_col[5]
        )
