# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import torch

from robokit.lie.torch_se3 import TorchSE3
from robokit.robo.torch_robot import TorchRobot, TorchRobotState
from robokit.terms.terms import TorchTask


class TorchBaseDampingTask(TorchTask):
    """
    Torch-based damping task for floating base that penalizes base motion.

    Residual: r = log(T_world_base @ T_ref^-1) (base twist)
    Jacobian: J = [Jlog @ Ad, 0] (identity for base, zero for joints)

    This encourages the optimizer to prefer joint motion over base motion
    by penalizing base twists directly.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> from robokit.lie.torch_se3 import TorchSE3
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="torch")
        >>> T_base = TorchSE3(torch.tensor([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        >>> state = robot.state(q=robot.zero_q, T_world_base=T_base)
        >>> damping_task = TorchBaseDampingTask(robot=robot, weight=0.1)
        >>> residual = damping_task.compute_residual(state)
        >>> # Residual should be small twist from identity
    """

    def __init__(
        self,
        robot: TorchRobot,
        T_world_base_ref: Optional[TorchSE3] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
    ):
        self.robot = robot
        self.T_world_base_ref = T_world_base_ref

        residual_weight = np.ones(6, dtype=float)
        if weight is not None:
            if isinstance(weight, (float, int)):
                residual_weight[:] = weight
            else:
                residual_weight[:] = weight

        self.residual_weight = residual_weight

    def compute_residual(self, var: TorchRobotState) -> torch.Tensor:
        if not var.has_floating_base:
            return torch.zeros(6, dtype=var.q.dtype, device=var.q.device)

        T_world_base = var.T_world_base
        if self.T_world_base_ref is not None:
            T_error = T_world_base @ self.T_world_base_ref.inverse()
        else:
            T_error = T_world_base

        return T_error.log()

    def compute_jacobian_analytic(self, var: TorchRobotState) -> torch.Tensor:
        device = var.q.device
        dtype = var.q.dtype
        batch_shape = var.q.shape[:-1]

        if not var.has_floating_base:
            total_dofs = var.spec.num_actuated_joints
            return torch.zeros(batch_shape + (6, total_dofs), device=device, dtype=dtype)

        num_joints = self.robot.num_actuated_joints
        total_dofs = 6 + num_joints

        T_world_base = var.T_world_base
        if self.T_world_base_ref is not None:
            T_error = T_world_base @ self.T_world_base_ref.inverse()
        else:
            T_error = T_world_base

        J_base = T_error.jlog() @ T_world_base.adjoint()

        J = torch.zeros(batch_shape + (6, total_dofs), device=device, dtype=dtype)
        J[..., :, :6] = J_base

        return J
