# pyright: reportArgumentType=false
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type


class WarpVelocityLimitTask(WarpTask):
    """
    Warp-based task that penalizes joint velocity limit violations.

    Residual: r = max(0, |q_dot| - q_dot_limit) where q_dot = (q_curr - q_prev) / dt
    Jacobian: J = sign(q_dot) / dt for joints exceeding limits

    Note: Warp backend uses soft constraints only (no QP/Limit interface).

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="warp")
        >>> state_prev = robot.state(q=robot.spec.zero_q)
        >>> state_curr = robot.state(q=robot.spec.zero_q + 0.5)
        >>> vel_task = WarpVelocityLimitTask(robot=robot, dt=0.1, weight=1.0, batch_size=1)
        >>> # Residual should be positive if velocity exceeds limits
    """

    def __init__(
        self,
        robot: WarpRobot,
        dt: float,
        prev_state_var: Optional[WarpRobotState] = None,
        velocity_limits: Optional[np.ndarray] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        batch_size: int = 1,
        num_seeds: int = 1,
    ):
        """
        Args:
            robot: Robot instance
            dt: Time step between states
            prev_state_var: Previous state (optional, can be set later)
            velocity_limits: Velocity limits (rad/s or m/s). If None, uses robot.spec.actuated_joint_velocity_limits
            weight: Weight for soft penalty (scalar or per-joint vector)
            batch_size: Batch size
            num_seeds: Number of seeds per batch element
        """
        assert dt > 0, "dt must be positive"
        self.robot = robot
        self.dt = dt
        self.batch_size = batch_size
        self.num_seeds = num_seeds
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be >= 1.")
        if self.batch_size % self.num_seeds != 0:
            raise ValueError("batch_size must be divisible by num_seeds.")
        self._prev_batch_size = self.batch_size // self.num_seeds
        self.prev_state_var = prev_state_var.clone() if prev_state_var is not None else None

        velocity_limits_np = (
            velocity_limits if velocity_limits is not None else robot.spec.actuated_joint_velocity_limits
        )
        self._velocity_limits_np = velocity_limits_np.astype(np.float32)

        num_joints = robot.num_actuated_joints
        residual_weight = np.ones(num_joints, dtype=np.float32)
        if weight is not None:
            if isinstance(weight, (float, int)):
                residual_weight[:] = weight
            else:
                residual_weight[:] = weight

        self._residual_weight_np = residual_weight
        self.device = None
        self.velocity_limits = None
        self.residual_weight = None

    def set_prev_state(self, prev_state: WarpRobotState) -> None:
        if prev_state.batch_size != self._prev_batch_size:
            raise ValueError(
                f"prev_state batch size mismatch: expected {self._prev_batch_size}, got {prev_state.batch_size}"
            )
        if self.prev_state_var is None:
            self.prev_state_var = prev_state.clone()
            return
        if self.prev_state_var.q.shape != prev_state.q.shape:
            self.prev_state_var = prev_state.clone()
            return
        wp.copy(self.prev_state_var.q, prev_state.q)

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.velocity_limits = wp.from_numpy(self._velocity_limits_np, dtype=wp.float32, device=device)
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return self.robot.num_actuated_joints

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        prev_state_var: Optional[WarpRobotState] = None,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if prev_state_var is None:
            prev_state_var = self.prev_state_var

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = residual_buffer.device if residual_buffer is not None else self.device
        if residual_buffer is None:
            residual_buffer = wp.empty((self.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if prev_state_var is None:
            return residual_buffer
        if prev_state_var.batch_size != self._prev_batch_size:
            raise ValueError(
                f"prev_state batch size mismatch: expected {self._prev_batch_size}, got {prev_state_var.batch_size}"
            )

        wp.launch(
            kernel=compute_velocity_limit_weighted_residual_kernel,
            dim=(self.batch_size, self.robot.num_actuated_joints),
            inputs=[
                var.q,
                prev_state_var.q,
                self.num_seeds,
                self.velocity_limits,
                self.residual_weight,
                self.dt,
                row_offset,
            ],
            outputs=[residual_buffer],
            device=kernel_device,
        )

        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: WarpRobotState,
        prev_state_var: Optional[WarpRobotState] = None,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if prev_state_var is None:
            prev_state_var = self.prev_state_var

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else self.device
        total_dofs = var.tangent_dim

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (self.batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device
            )
            row_offset = 0

        if prev_state_var is None:
            return jacobian_buffer
        if prev_state_var.batch_size != self._prev_batch_size:
            raise ValueError(
                f"prev_state batch size mismatch: expected {self._prev_batch_size}, got {prev_state_var.batch_size}"
            )

        base_col_offset = 6 if var.has_floating_base else 0

        wp.launch(
            kernel=compute_velocity_limit_weighted_jacobian_kernel,
            dim=(self.batch_size, self.robot.num_actuated_joints),
            inputs=[
                var.q,
                prev_state_var.q,
                self.num_seeds,
                self.velocity_limits,
                self.residual_weight,
                self.dt,
                row_offset,
                base_col_offset,
            ],
            outputs=[jacobian_buffer],
            device=kernel_device,
        )

        return jacobian_buffer


@wp.kernel
def compute_velocity_limit_weighted_residual_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    num_seeds: int,
    velocity_limits: wp.array1d(dtype=wp.float32),  # [num_joints]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    dt: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore

    prev_idx = batch_idx // num_seeds
    q_dot = (q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]) / dt
    abs_q_dot = wp.abs(q_dot)
    v_limit = velocity_limits[joint_idx]

    violation = wp.max(0.0, abs_q_dot - v_limit)
    residual_buffer[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * violation


@wp.kernel
def compute_velocity_limit_weighted_jacobian_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    num_seeds: int,
    velocity_limits: wp.array1d(dtype=wp.float32),  # [num_joints]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    dt: float,
    row_offset: int,
    base_col_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore

    prev_idx = batch_idx // num_seeds
    q_dot = (q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]) / dt
    abs_q_dot = wp.abs(q_dot)
    v_limit = velocity_limits[joint_idx]

    # Only set Jacobian if violating
    if abs_q_dot > v_limit:
        sign = 1.0 if q_dot > 0.0 else -1.0
        J_value = residual_weight[joint_idx] * sign / dt
        jacobian_buffer[batch_idx, row_offset + joint_idx, base_col_offset + joint_idx] = J_value
