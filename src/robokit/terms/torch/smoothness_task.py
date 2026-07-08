# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import torch

from robokit.robo.torch_robot import TorchRobot, TorchRobotState
from robokit.terms.terms import TorchTask


class TorchSmoothnessTask(TorchTask):
    """
    Torch-based smoothness task that penalizes joint velocity (difference between consecutive configurations).

    Residual: r = q_current - q_previous
    Jacobian: J = I (identity matrix for joints)

    This encourages smooth trajectories by penalizing large configuration changes.

    Note: Requires previous state to be passed as second argument or stored in task.
    For single-state optimization, set prev_var=None to disable.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="torch")
        >>> state_prev = robot.state(q=robot.zero_q)
        >>> state_curr = robot.state(q=robot.zero_q + 0.1)
        >>> smooth_task = TorchSmoothnessTask(robot=robot, weight=0.5)
        >>> residual = smooth_task.compute_residual(state_curr, state_prev)
        >>> torch.allclose(residual, torch.tensor(0.1), atol=1e-6)
        True
    """

    def __init__(
        self,
        robot: TorchRobot,
        prev_var: Optional[TorchRobotState] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        base_weight: Optional[Union[float, Sequence[float]]] = None,
    ):
        self.robot = robot
        self.prev_var = prev_var

        num_joints = robot.num_actuated_joints

        residual_weight = np.ones(num_joints, dtype=float)
        if weight is not None:
            if isinstance(weight, (float, int)):
                residual_weight[:] = weight
            else:
                residual_weight[:] = weight

        self._joint_weight = residual_weight.copy()
        self._base_weight = base_weight
        self.residual_weight = residual_weight

    def compute_residual(self, var: TorchRobotState, prev_var: Optional[TorchRobotState] = None) -> torch.Tensor:
        if prev_var is None:
            prev_var = self.prev_var

        if prev_var is None:
            num_joints = self.robot.num_actuated_joints
            if var.has_floating_base:
                return torch.zeros(num_joints + 6, dtype=var.q.dtype, device=var.q.device)
            return torch.zeros(num_joints, dtype=var.q.dtype, device=var.q.device)

        q_curr = var.q
        q_prev = prev_var.q
        if q_prev.device != q_curr.device:
            q_prev = q_prev.to(q_curr.device)
        r_joints = q_curr - q_prev

        if var.has_floating_base and prev_var.has_floating_base:
            T_curr = var.T_world_base
            T_prev = prev_var.T_world_base
            T_error = T_curr @ T_prev.inverse()
            r_base = T_error.log()

            while r_base.ndim > r_joints.ndim:
                r_base = r_base.squeeze(0)

            if len(self.residual_weight) == self.robot.num_actuated_joints:
                num_joints = self.robot.num_actuated_joints
                residual_weight = np.ones(num_joints + 6, dtype=float)
                residual_weight[:num_joints] = self._joint_weight
                if self._base_weight is not None:
                    if isinstance(self._base_weight, (float, int)):
                        residual_weight[num_joints:] = self._base_weight
                    else:
                        residual_weight[num_joints:] = self._base_weight
                else:
                    residual_weight[num_joints:] = self._joint_weight[0] if len(self._joint_weight) > 0 else 1.0
                self.residual_weight = residual_weight

            return torch.cat([r_joints, r_base], dim=-1)
        else:
            if len(self.residual_weight) != self.robot.num_actuated_joints:
                self.residual_weight = self._joint_weight.copy()
            return r_joints

    def compute_jacobian_analytic(
        self, var: TorchRobotState, prev_var: Optional[TorchRobotState] = None
    ) -> torch.Tensor:
        if prev_var is None:
            prev_var = self.prev_var

        num_joints = self.robot.num_actuated_joints
        base_col_offset = 6 if var.has_floating_base else 0
        total_dofs = base_col_offset + num_joints

        device = var.q.device
        dtype = var.q.dtype
        batch_shape = var.q.shape[:-1]

        if prev_var is None:
            if var.has_floating_base:
                return torch.zeros(batch_shape + (num_joints + 6, total_dofs), device=device, dtype=dtype)
            return torch.zeros(batch_shape + (num_joints, total_dofs), device=device, dtype=dtype)

        if var.has_floating_base and prev_var.has_floating_base:
            J = torch.zeros(batch_shape + (num_joints + 6, total_dofs), device=device, dtype=dtype)
            for i in range(num_joints):
                J[..., i, base_col_offset + i] = 1.0

            T_curr = var.T_world_base
            T_prev = prev_var.T_world_base
            T_error = T_curr @ T_prev.inverse()
            J_base = T_error.jlog() @ T_curr.adjoint()
            J[..., num_joints:, :6] = J_base
        else:
            J = torch.zeros(batch_shape + (num_joints, total_dofs), device=device, dtype=dtype)
            for i in range(num_joints):
                J[..., i, base_col_offset + i] = 1.0

        return J
