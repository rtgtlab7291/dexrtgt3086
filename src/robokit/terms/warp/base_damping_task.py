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


class WarpBaseDampingTask(WarpTask):
    """
    Warp-based damping task for floating base that penalizes base motion.

    Residual: r = log(T_world_base @ T_ref^-1) (base twist)
    Jacobian: J = [Jlog @ Ad, 0] (base columns only, joints are zero)

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> from robokit.lie.warp_se3 import WarpSE3
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="warp")
        >>> T_base = WarpSE3(wp.from_numpy(np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=wp_vec7))
        >>> state = robot.state(q=robot.spec.zero_q, T_world_base=T_base)
        >>> damping_task = WarpBaseDampingTask(robot=robot, weight=0.1, batch_size=1)
        >>> # Residual should be small twist from identity
    """

    def __init__(
        self,
        robot: WarpRobot,
        T_world_base_ref: Optional[WarpSE3] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        batch_size: int = 1,
    ):
        """
        Args:
            robot: Robot instance
            T_world_base_ref: Reference base pose. If None, uses identity
            weight: Weight for base damping (scalar or 6D vector)
            batch_size: Batch size
        """
        self.robot = robot
        self.T_world_base_ref = T_world_base_ref
        self.batch_size = batch_size

        residual_weight = np.ones(6, dtype=np.float32)
        if weight is not None:
            if isinstance(weight, (float, int)):
                residual_weight[:] = weight
            else:
                residual_weight[:] = weight

        self._residual_weight_np = residual_weight
        self.device = None
        self.residual_weight = None

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return 6

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
            residual_buffer = wp.empty((self.batch_size, 6), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if not var.has_floating_base:
            # Return zeros if no floating base
            return residual_buffer

        if self.T_world_base_ref is not None:
            wp.launch(
                kernel=compute_base_damping_weighted_residual_with_ref_kernel,
                dim=self.batch_size,
                inputs=[
                    var.T_world_base.xyz_wxyz,
                    self.T_world_base_ref.xyz_wxyz,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[residual_buffer],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_base_damping_weighted_residual_kernel,
                dim=self.batch_size,
                inputs=[
                    var.T_world_base.xyz_wxyz,
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
    ) -> wp.array:
        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else self.device
        total_dofs = var.tangent_dim

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros((self.batch_size, 6, total_dofs), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if not var.has_floating_base:
            return jacobian_buffer

        if self.T_world_base_ref is not None:
            wp.launch(
                kernel=compute_base_damping_weighted_jacobian_with_ref_kernel,
                dim=(self.batch_size, 6),  # 6 base dofs
                inputs=[
                    var.T_world_base.xyz_wxyz,
                    self.T_world_base_ref.xyz_wxyz,
                    self.residual_weight,
                    row_offset,
                    self.robot.num_actuated_joints,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_base_damping_weighted_jacobian_kernel,
                dim=(self.batch_size, 6),  # 6 base dofs
                inputs=[
                    var.T_world_base.xyz_wxyz,
                    self.residual_weight,
                    row_offset,
                    self.robot.num_actuated_joints,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )

        return jacobian_buffer


@wp.kernel
def compute_base_damping_weighted_residual_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    residual_weight: wp.array1d(dtype=wp.float32),  # [6]
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx = wp.tid()  # type: ignore

    T_error = T_world_base[batch_idx]
    r_base = se3_log_map_func(T_error, wp.float32(1e-4))

    residual_buffer[batch_idx, row_offset + 0] = residual_weight[0] * r_base[0]
    residual_buffer[batch_idx, row_offset + 1] = residual_weight[1] * r_base[1]
    residual_buffer[batch_idx, row_offset + 2] = residual_weight[2] * r_base[2]
    residual_buffer[batch_idx, row_offset + 3] = residual_weight[3] * r_base[3]
    residual_buffer[batch_idx, row_offset + 4] = residual_weight[4] * r_base[4]
    residual_buffer[batch_idx, row_offset + 5] = residual_weight[5] * r_base[5]


@wp.kernel
def compute_base_damping_weighted_residual_with_ref_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_ref: wp.array1d(dtype=wp_vec7),  # [batch] or [1]
    residual_weight: wp.array1d(dtype=wp.float32),  # [6]
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx = wp.tid()  # type: ignore

    T_ref_inv = se3_inverse_func(T_world_base_ref[0] if T_world_base_ref.shape[0] == 1 else T_world_base_ref[batch_idx])
    T_error = se3_multiply_func(T_world_base[batch_idx], T_ref_inv)

    r_base = se3_log_map_func(T_error, wp.float32(1e-4))

    residual_buffer[batch_idx, row_offset + 0] = residual_weight[0] * r_base[0]
    residual_buffer[batch_idx, row_offset + 1] = residual_weight[1] * r_base[1]
    residual_buffer[batch_idx, row_offset + 2] = residual_weight[2] * r_base[2]
    residual_buffer[batch_idx, row_offset + 3] = residual_weight[3] * r_base[3]
    residual_buffer[batch_idx, row_offset + 4] = residual_weight[4] * r_base[4]
    residual_buffer[batch_idx, row_offset + 5] = residual_weight[5] * r_base[5]


@wp.kernel
def compute_base_damping_weighted_jacobian_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    residual_weight: wp.array1d(dtype=wp.float32),  # [6]
    row_offset: int,
    num_joints: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, 6, total_dofs]
):
    batch_idx, col_idx = wp.tid()  # type: ignore

    # Only process base columns (first 6)
    if col_idx < 6:
        T_error = T_world_base[batch_idx]

        # Compute Jlog @ Ad_T column by column
        jlog = se3_jlog_func(T_error, wp.float32(1e-4))

        # Ad(I) = I when ref is identity, so just apply jlog to unit_vec
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = 1.0
        result_col = wp.mul(jlog, unit_vec)

        jacobian_buffer[batch_idx, row_offset + 0, col_idx] = residual_weight[0] * result_col[0]
        jacobian_buffer[batch_idx, row_offset + 1, col_idx] = residual_weight[1] * result_col[1]
        jacobian_buffer[batch_idx, row_offset + 2, col_idx] = residual_weight[2] * result_col[2]
        jacobian_buffer[batch_idx, row_offset + 3, col_idx] = residual_weight[3] * result_col[3]
        jacobian_buffer[batch_idx, row_offset + 4, col_idx] = residual_weight[4] * result_col[4]
        jacobian_buffer[batch_idx, row_offset + 5, col_idx] = residual_weight[5] * result_col[5]


@wp.kernel
def compute_base_damping_weighted_jacobian_with_ref_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_ref: wp.array1d(dtype=wp_vec7),  # [batch] or [1]
    residual_weight: wp.array1d(dtype=wp.float32),  # [6]
    row_offset: int,
    num_joints: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, 6, total_dofs]
):
    batch_idx, col_idx = wp.tid()  # type: ignore

    # Only process base columns (first 6)
    if col_idx < 6:
        T_ref_inv = se3_inverse_func(
            T_world_base_ref[0] if T_world_base_ref.shape[0] == 1 else T_world_base_ref[batch_idx]
        )
        T_error = se3_multiply_func(T_world_base[batch_idx], T_ref_inv)

        # Compute Jlog @ Ad_T column by column
        jlog = se3_jlog_func(T_error, wp.float32(1e-4))

        # Ad_T column for unit vector e_col_idx
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = 1.0
        T_ref = T_world_base_ref[0] if T_world_base_ref.shape[0] == 1 else T_world_base_ref[batch_idx]
        ad_col = se3_adjoint_multiply_vec6_func(T_ref, unit_vec)

        result_col = wp.mul(jlog, ad_col)

        jacobian_buffer[batch_idx, row_offset + 0, col_idx] = residual_weight[0] * result_col[0]
        jacobian_buffer[batch_idx, row_offset + 1, col_idx] = residual_weight[1] * result_col[1]
        jacobian_buffer[batch_idx, row_offset + 2, col_idx] = residual_weight[2] * result_col[2]
        jacobian_buffer[batch_idx, row_offset + 3, col_idx] = residual_weight[3] * result_col[3]
        jacobian_buffer[batch_idx, row_offset + 4, col_idx] = residual_weight[4] * result_col[4]
        jacobian_buffer[batch_idx, row_offset + 5, col_idx] = residual_weight[5] * result_col[5]
