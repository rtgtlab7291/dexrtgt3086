# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.warp_se3 import WarpSE3
from robokit.lie.warp_se3_kernels import se3_inverse_func, se3_jlog_func, se3_log_map_func, se3_multiply_func
from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


class WarpRestTask(WarpTask):
    """
    Warp-based regularization task that biases joints towards a rest pose.

    Residual: r = q - q_rest
    Jacobian: J = I (identity matrix for joints)

    For floating base robots, can optionally include base regularization.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="warp")
        >>> rest_q = robot.spec.midrange_q
        >>> rest_task = WarpRestTask(robot=robot, rest_q=rest_q, weight=0.1, batch_size=1)
        >>> q_init = wp.from_numpy(rest_q + 0.1, dtype=wp.float32)
        >>> state = robot.state(q=q_init)
        >>> residual = rest_task.compute_weighted_residual(state)
        >>> # Residual should be ~0.1 * weight
    """

    def __init__(
        self,
        robot: WarpRobot,
        rest_q: Optional[np.ndarray] = None,
        T_world_base_rest: Optional[WarpSE3] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        base_weight: Optional[Union[float, Sequence[float]]] = None,
        batch_size: int = 1,
    ):
        """
        Args:
            robot: Robot instance
            rest_q: Rest joint configuration (NumPy). If None, uses robot.spec.zero_q
            T_world_base_rest: Rest base pose (for floating base). If None, uses identity
            weight: Weight for joint regularization (scalar or per-joint vector)
            base_weight: Weight for base regularization (scalar or 6D vector)
            batch_size: Batch size
        """
        self.robot = robot
        self.rest_q_np = rest_q if rest_q is not None else robot.spec.zero_q
        self.T_world_base_rest = T_world_base_rest
        self.batch_size = batch_size

        num_joints = robot.num_actuated_joints
        has_base = T_world_base_rest is not None

        if has_base:
            residual_dim = num_joints + 6
            residual_weight = np.ones(residual_dim, dtype=np.float32)
            if weight is not None:
                if isinstance(weight, (float, int)):
                    residual_weight[:num_joints] = weight
                else:
                    residual_weight[:num_joints] = weight
            if base_weight is not None:
                if isinstance(base_weight, (float, int)):
                    residual_weight[num_joints:] = base_weight
                else:
                    residual_weight[num_joints:] = base_weight
        else:
            residual_dim = num_joints
            residual_weight = np.ones(residual_dim, dtype=np.float32)
            if weight is not None:
                if isinstance(weight, (float, int)):
                    residual_weight[:] = weight
                else:
                    residual_weight[:] = weight

        self._residual_weight_np = residual_weight
        self._residual_dim = residual_dim
        self.device = None
        self.rest_q = None
        self.residual_weight = None

    def init_buffers(self, device: wp_device_type):
        self.device = device
        rest_q_tiled = np.tile(self.rest_q_np.reshape(1, -1), (self.batch_size, 1))
        self.rest_q = wp.from_numpy(rest_q_tiled.astype(np.float32), dtype=wp.float32, device=device)
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

    def set_rest_state(self, rest_q: wp.array, T_world_base_rest: Optional[WarpSE3] = None) -> None:
        wp.copy(self.rest_q, rest_q)
        if T_world_base_rest is not None and self.T_world_base_rest is not None:
            wp.copy(self.T_world_base_rest.xyz_wxyz, T_world_base_rest.xyz_wxyz)

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = residual_buffer.device if residual_buffer is not None else self.device
        if residual_buffer is None:
            residual_buffer = wp.empty((self.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self.T_world_base_rest is not None and var.has_floating_base:
            wp.launch(
                kernel=compute_rest_task_weighted_residual_with_base_kernel,
                dim=self.batch_size,
                inputs=[
                    var.q,
                    var.T_world_base.xyz_wxyz,
                    self.rest_q,
                    self.T_world_base_rest.xyz_wxyz,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[residual_buffer],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_rest_task_weighted_residual_kernel,
                dim=(self.batch_size, self.robot.num_actuated_joints),
                inputs=[
                    var.q,
                    self.rest_q,
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
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> wp.array:
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

        if self.T_world_base_rest is not None and var.has_floating_base:
            wp.launch(
                kernel=compute_rest_task_weighted_jacobian_with_base_kernel,
                dim=(self.batch_size, total_dofs),
                inputs=[
                    var.T_world_base.xyz_wxyz,
                    self.T_world_base_rest.xyz_wxyz,
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
                kernel=compute_rest_task_weighted_jacobian_kernel,
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
def compute_rest_task_weighted_residual_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    rest_q: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore
    diff = q[batch_idx, joint_idx] - rest_q[batch_idx, joint_idx]
    residual_buffer[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * diff


@wp.kernel
def compute_rest_task_weighted_residual_with_base_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    rest_q: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    T_world_base_rest: wp.array1d(dtype=wp_vec7),  # [batch] or [1]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx = wp.tid()  # type: ignore
    num_joints = q.shape[1]

    # Joint residuals
    for joint_idx in range(num_joints):
        diff = q[batch_idx, joint_idx] - rest_q[batch_idx, joint_idx]
        residual_buffer[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * diff

    # Base residuals
    T_ref_inv = se3_inverse_func(
        T_world_base_rest[0] if T_world_base_rest.shape[0] == 1 else T_world_base_rest[batch_idx]
    )
    T_error = se3_multiply_func(T_world_base[batch_idx], T_ref_inv)
    r_base = se3_log_map_func(T_error, wp.float32(1e-4))

    residual_buffer[batch_idx, row_offset + num_joints + 0] = residual_weight[num_joints + 0] * r_base[0]
    residual_buffer[batch_idx, row_offset + num_joints + 1] = residual_weight[num_joints + 1] * r_base[1]
    residual_buffer[batch_idx, row_offset + num_joints + 2] = residual_weight[num_joints + 2] * r_base[2]
    residual_buffer[batch_idx, row_offset + num_joints + 3] = residual_weight[num_joints + 3] * r_base[3]
    residual_buffer[batch_idx, row_offset + num_joints + 4] = residual_weight[num_joints + 4] * r_base[4]
    residual_buffer[batch_idx, row_offset + num_joints + 5] = residual_weight[num_joints + 5] * r_base[5]


@wp.kernel
def compute_rest_task_weighted_jacobian_kernel(
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
def compute_rest_task_weighted_jacobian_with_base_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_rest: wp.array1d(dtype=wp_vec7),  # [batch] or [1]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    row_offset: int,
    col_offset: int,
    num_joints: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, col_idx = wp.tid()  # type: ignore

    # Joint block: identity for joint columns (starting at column 6)
    if col_idx >= 6 and col_idx < (6 + num_joints):
        joint_idx = col_idx - 6
        jacobian_buffer[batch_idx, row_offset + joint_idx, col_offset + col_idx] = residual_weight[joint_idx]

    # Base block: Jlog @ Ad for base columns (first 6)
    if col_idx < 6:
        T_ref_inv = se3_inverse_func(
            T_world_base_rest[0] if T_world_base_rest.shape[0] == 1 else T_world_base_rest[batch_idx]
        )
        T_error = se3_multiply_func(T_world_base[batch_idx], T_ref_inv)

        # Compute Jlog @ Ad_T column by column
        jlog = se3_jlog_func(T_error, wp.float32(1e-4))

        # Ad_T column for unit vector e_col_idx
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = 1.0
        T_rest = T_world_base_rest[0] if T_world_base_rest.shape[0] == 1 else T_world_base_rest[batch_idx]
        ad_col = se3_adjoint_multiply_vec6_func(T_rest, unit_vec)

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
