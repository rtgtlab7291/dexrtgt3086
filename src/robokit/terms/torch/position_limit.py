# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import torch

from robokit.robo.torch_robot import TorchRobot, TorchRobotState
from robokit.terms.terms import TorchTask


class TorchPositionLimit(TorchTask):
    """
    Torch-based position limit constraint for joint positions.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"), backend="torch")
        >>> joint_var = robot.actuated_joint_limits[:, 0] - 0.1
        >>> state = robot.state()
        >>> state.set_configuration(q=torch.tensor(joint_var, dtype=torch.float32))
        >>> position_limit = TorchPositionLimit(robot=robot)
        >>> residual = position_limit.compute_residual(state)
        >>> torch.allclose(residual, torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32))
        True
        >>> jacobian = position_limit.compute_jacobian(state)
        >>> torch.allclose(torch.diag(jacobian), torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=torch.float32))
        True
    """

    def __init__(
        self,
        robot: TorchRobot,
        weight: Optional[Union[float, Sequence[float]]] = None,
    ):
        self.robot = robot
        self.residual_weight = np.zeros(robot.num_actuated_joints, dtype=float)
        self.residual_weight[:] = weight if weight is not None else 1.0
        self.projection_matrix = torch.eye(robot.num_actuated_joints)

    def compute_residual(self, var: TorchRobotState) -> "torch.Tensor":
        joint_positions = var.q
        joint_limits = var.spec_tensors.actuated_joint_limits
        error_upper = torch.clamp(joint_positions - joint_limits[..., 1], min=0)
        error_lower = torch.clamp(joint_limits[..., 0] - joint_positions, min=0)
        return error_upper + error_lower

    def compute_jacobian_analytic(self, var: TorchRobotState) -> "torch.Tensor":
        joint_positions = var.q
        joint_limits = var.spec_tensors.actuated_joint_limits
        sign = torch.zeros_like(joint_positions)
        sign[joint_positions > joint_limits[..., 1]] = 1.0
        sign[joint_positions < joint_limits[..., 0]] = -1.0

        base_col_offset = 6 if var.has_floating_base else 0
        num_joints = var.spec.num_actuated_joints
        total_dofs = base_col_offset + num_joints

        device = joint_positions.device
        dtype = joint_positions.dtype
        batch_shape = sign.shape[:-1]

        J = torch.zeros(batch_shape + (num_joints, total_dofs), device=device, dtype=dtype)
        for i in range(num_joints):
            J[..., i, base_col_offset + i] = sign[..., i]

        return J
