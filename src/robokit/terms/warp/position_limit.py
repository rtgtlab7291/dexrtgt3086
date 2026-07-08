# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
from typing import Literal, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type


class WarpPositionLimit(WarpTask):
    """
    Warp-based position limit constraint for joint positions.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
        >>> joint_var = robot.actuated_joint_limits[:, 0] - 0.1
        >>> state = robot.state()
        >>> state.set_configuration(q=wp.from_numpy(joint_var[None].astype(np.float32), dtype=wp.float32))
        >>> position_limit = WarpPositionLimit(robot=robot)
        >>> weighted_residual = position_limit.compute_weighted_residual(state)
        >>> np.allclose(weighted_residual.numpy()[0], np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32))
        True
        >>> weighted_jacobian = position_limit.compute_weighted_jacobian(state)
        >>> np.allclose(np.diag(weighted_jacobian.numpy()[0]), np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32))
        True
    """

    def __init__(
        self,
        robot: WarpRobot,
        weight: Optional[Union[float, Sequence[float]]] = None,
        batch_size: int = 1,
        residual_mode: Literal["abs", "sqrt_abs"] = "abs",
        residual_eps: float = 1e-6,
    ):
        self.robot = robot
        self.batch_size = batch_size
        if residual_mode not in {"abs", "sqrt_abs"}:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")
        self.residual_mode = residual_mode
        self.residual_eps = residual_eps
        self.n_dofs = robot.num_actuated_joints

        residual_weight = np.zeros(robot.num_actuated_joints, dtype=np.float32)
        residual_weight[:] = weight if weight is not None else 1.0
        self._residual_weight_np = residual_weight
        self.residual_weight = wp.from_numpy(residual_weight, dtype=wp.float32)
        self.device = None

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return self.n_dofs

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        residual_buffer: wp.array = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self.device is None:
            self.init_buffers(var.q.device)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if residual_buffer is None:
            residual_buffer = wp.empty((self.batch_size, self.n_dofs), dtype=wp.float32, device=self.device)
            row_offset = 0
        wp.launch(
            kernel=compute_position_limit_weighted_residual_kernel,
            dim=(self.batch_size, self.n_dofs),
            inputs=[
                var.q,
                var.spec_tensors.actuated_joint_limits,
                self.residual_weight,
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                row_offset,
            ],
            outputs=[residual_buffer],
            device=self.device,
        )
        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: WarpRobotState,
        jacobian_buffer: wp.array = None,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> wp.array:
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        if self.device is None:
            self.init_buffers(var.q.device)
        total_dofs = var.tangent_dim
        base_col_offset = 6 if var.has_floating_base else 0
        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros((self.batch_size, self.n_dofs, total_dofs), dtype=wp.float32, device=self.device)
            row_offset = 0
            col_offset = 0
        wp.launch(
            kernel=compute_position_limit_weighted_jacobian_kernel,
            dim=(self.batch_size, self.n_dofs),
            inputs=[
                var.q,
                var.spec_tensors.actuated_joint_limits,
                self.residual_weight,
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                row_offset,
                col_offset,
                base_col_offset,
            ],
            outputs=[jacobian_buffer],
            device=self.device,
        )
        return jacobian_buffer


@wp.kernel
def compute_position_limit_weighted_residual_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, n_dofs]
    joint_limits: wp.array2d(dtype=wp.float32),  # [n_dofs, 2]
    residual_weight: wp.array1d(dtype=wp.float32),  # [n_dofs]
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, dof_idx = wp.tid()  # pyright: ignore

    joint_pos = q[batch_idx, dof_idx]
    lower_limit = joint_limits[dof_idx, 0]
    upper_limit = joint_limits[dof_idx, 1]

    error_upper = wp.max(0.0, joint_pos - upper_limit)
    error_lower = wp.max(0.0, lower_limit - joint_pos)

    error = error_upper + error_lower
    residual = error
    if use_sqrt:
        residual = wp.where(error > wp.float32(0.0), wp.sqrt(error + eps), wp.float32(0.0))
    residual_buffer[batch_idx, row_offset + dof_idx] = residual_weight[dof_idx] * residual


@wp.kernel
def compute_position_limit_weighted_jacobian_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, n_dofs]
    joint_limits: wp.array2d(dtype=wp.float32),  # [n_dofs, 2]
    residual_weight: wp.array1d(dtype=wp.float32),  # [n_dofs]
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    col_offset: int,
    base_col_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, dof_idx = wp.tid()  # pyright: ignore

    joint_pos = q[batch_idx, dof_idx]
    lower_limit = joint_limits[dof_idx, 0]
    upper_limit = joint_limits[dof_idx, 1]

    sign = 0.0
    if joint_pos > upper_limit:
        sign = 1.0
    elif joint_pos < lower_limit:
        sign = -1.0

    error_upper = wp.max(0.0, joint_pos - upper_limit)
    error_lower = wp.max(0.0, lower_limit - joint_pos)
    error = error_upper + error_lower
    scale = wp.float32(1.0)
    if use_sqrt and sign != 0.0:
        scale = wp.float32(0.5) / wp.sqrt(error + eps)

    jacobian_buffer[batch_idx, row_offset + dof_idx, col_offset + base_col_offset + dof_idx] = (
        residual_weight[dof_idx] * sign * scale
    )
