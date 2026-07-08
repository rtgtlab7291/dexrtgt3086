# pyright: reportArgumentType=false
from typing import Optional, Sequence, Union

import numpy as np
import torch
from jaxtyping import Float

from robokit.robo.torch_robot import TorchRobot, TorchRobotState
from robokit.terms.terms import TorchTask


class TorchVelocityLimitTask(TorchTask):
    """
    Torch-based task that penalizes joint velocity limit violations.

    Residual: r = max(0, |q_dot| - q_dot_limit) where q_dot = (q_curr - q_prev) / dt
    Jacobian: J = sign(q_dot) / dt for joints exceeding limits

    Note: Torch backend uses soft constraints only (no QP/Limit interface).

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="torch")
        >>> state_prev = robot.state(q=robot.zero_q)
        >>> state_curr = robot.state(q=robot.zero_q + 0.5)  # Large change
        >>> vel_task = TorchVelocityLimitTask(robot=robot, dt=0.1, prev_state_var=state_prev, weight=1.0)
        >>> residual = vel_task.compute_residual(state_curr)
        >>> # Should be positive if velocity exceeds limits
    """

    def __init__(
        self,
        robot: TorchRobot,
        dt: float,
        prev_state_var: Optional[TorchRobotState] = None,
        velocity_limits: Optional[Float[torch.Tensor, "num_actuated_joints"]] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
    ):
        assert dt > 0, "dt must be positive"
        self.robot = robot
        self.dt = dt
        self.prev_state_var = prev_state_var
        self.velocity_limits = (
            velocity_limits if velocity_limits is not None else robot.spec.actuated_joint_velocity_limits
        )
        if not isinstance(self.velocity_limits, torch.Tensor):
            self.velocity_limits = torch.from_numpy(self.velocity_limits).float()

        num_joints = robot.num_actuated_joints
        residual_weight = np.ones(num_joints, dtype=float)
        if weight is not None:
            if isinstance(weight, (float, int)):
                residual_weight[:] = weight
            else:
                residual_weight[:] = weight

        self.residual_weight = residual_weight

    def compute_residual(self, var: TorchRobotState, prev_state_var: Optional[TorchRobotState] = None) -> torch.Tensor:
        if prev_state_var is None:
            prev_state_var = self.prev_state_var

        if prev_state_var is None:
            return torch.zeros(self.robot.num_actuated_joints, dtype=var.q.dtype, device=var.q.device)

        q_curr = var.q
        q_prev = prev_state_var.q
        if q_prev.device != q_curr.device:
            q_prev = q_prev.to(q_curr.device)
        q_dot = (q_curr - q_prev) / self.dt

        v_limits = self.velocity_limits.to(q_dot.device)
        violation = torch.clamp(torch.abs(q_dot) - v_limits, min=0.0)
        return violation

    def compute_jacobian_analytic(
        self, var: TorchRobotState, prev_state_var: Optional[TorchRobotState] = None
    ) -> torch.Tensor:
        if prev_state_var is None:
            prev_state_var = self.prev_state_var

        num_joints = self.robot.num_actuated_joints
        base_col_offset = 6 if var.has_floating_base else 0
        total_dofs = base_col_offset + num_joints

        device = var.q.device
        dtype = var.q.dtype
        batch_shape = var.q.shape[:-1]

        if prev_state_var is None:
            return torch.zeros(batch_shape + (num_joints, total_dofs), device=device, dtype=dtype)

        q_curr = var.q
        q_prev = prev_state_var.q
        if q_prev.device != q_curr.device:
            q_prev = q_prev.to(q_curr.device)
        q_dot = (q_curr - q_prev) / self.dt

        v_limits = self.velocity_limits.to(device)
        violation_mask = torch.abs(q_dot) > v_limits
        sign = torch.sign(q_dot)
        J_values = sign / self.dt

        J = torch.zeros(batch_shape + (num_joints, total_dofs), device=device, dtype=dtype)

        for i in range(num_joints):
            J[..., i, base_col_offset + i] = torch.where(
                violation_mask[..., i], J_values[..., i], torch.zeros_like(J_values[..., i])
            )

        return J
