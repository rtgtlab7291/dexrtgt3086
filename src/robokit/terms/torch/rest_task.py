# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import torch
from jaxtyping import Float

from robokit.lie.torch_se3 import TorchSE3
from robokit.robo.torch_robot import TorchRobot, TorchRobotState
from robokit.terms.terms import TorchTask


class TorchRestTask(TorchTask):
    """
    Torch-based regularization task that biases joints towards a rest pose.

    Residual: r = q - q_rest
    Jacobian: J = I (identity matrix for joints)

    For floating base robots, can optionally include base regularization
    where residual includes base twist: r_base = log(T_world_base @ T_rest^-1)

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="torch")
        >>> rest_q = robot.midrange_q
        >>> rest_task = TorchRestTask(robot=robot, rest_q=rest_q, weight=0.1)
        >>> state = robot.state(q=rest_q + 0.1)
        >>> residual = rest_task.compute_residual(state)
        >>> torch.allclose(residual, torch.tensor(0.1), atol=1e-6)
        True
    """

    def __init__(
        self,
        robot: TorchRobot,
        rest_q: Optional[Float[torch.Tensor, "num_actuated_joints"]] = None,
        T_world_base_rest: Optional[TorchSE3] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        base_weight: Optional[Union[float, Sequence[float]]] = None,
    ):
        self.robot = robot
        self.rest_q = rest_q if rest_q is not None else robot.zero_q
        self.T_world_base_rest = T_world_base_rest

        num_joints = robot.num_actuated_joints
        has_base = T_world_base_rest is not None

        if has_base:
            residual_dim = num_joints + 6
            residual_weight = np.ones(residual_dim, dtype=float)
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
            residual_weight = np.ones(residual_dim, dtype=float)
            if weight is not None:
                if isinstance(weight, (float, int)):
                    residual_weight[:] = weight
                else:
                    residual_weight[:] = weight

        self.residual_weight = residual_weight

    def compute_residual(self, var: TorchRobotState) -> torch.Tensor:
        q = var.q
        rest_q = self.rest_q.to(q.device) if self.rest_q.device != q.device else self.rest_q
        r_joints = q - rest_q

        if self.T_world_base_rest is not None and var.has_floating_base:
            T_world_base = var.T_world_base
            T_error = T_world_base @ self.T_world_base_rest.inverse()
            r_base = T_error.log()

            while r_base.ndim > r_joints.ndim:
                r_base = r_base.squeeze(0)

            return torch.cat([r_joints, r_base], dim=-1)
        else:
            return r_joints

    def compute_jacobian_analytic(self, var: TorchRobotState) -> torch.Tensor:
        num_joints = self.robot.num_actuated_joints
        base_col_offset = 6 if var.has_floating_base else 0
        total_dofs = base_col_offset + num_joints

        device = var.q.device
        dtype = var.q.dtype
        batch_shape = var.q.shape[:-1]

        if self.T_world_base_rest is not None and var.has_floating_base:
            J = torch.zeros(batch_shape + (num_joints + 6, total_dofs), device=device, dtype=dtype)
            for i in range(num_joints):
                J[..., i, base_col_offset + i] = 1.0

            T_world_base = var.T_world_base
            T_error = T_world_base @ self.T_world_base_rest.inverse()
            J_base = T_error.jlog() @ self.T_world_base_rest.adjoint()
            J[..., num_joints:, :6] = J_base
        else:
            J = torch.zeros(batch_shape + (num_joints, total_dofs), device=device, dtype=dtype)
            for i in range(num_joints):
                J[..., i, base_col_offset + i] = 1.0

        return J
