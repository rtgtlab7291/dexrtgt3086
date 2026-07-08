from typing import Sequence

import numpy as np
import torch

from robokit.robo.torch_robot import TorchRobotState
from robokit.terms.terms import TorchTask


class TorchBaseStepLimit(TorchTask):
    """
    Soft constraint to limit floating-base step components (Torch backend).

    Unlike PinocchioBaseStepLimit which uses hard QP constraints, this uses
    soft penalties. Set high weight (e.g., 10.0) to strongly discourage movement
    in locked base DOF components.

    Residual: r = base_twist[lock_indices] (penalize non-zero movement)
    Jacobian: J = identity for locked base columns

    Example:
        >>> import torch
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> from robokit.lie.torch_se3 import TorchSE3
        >>> from robokit.opt.variables import Var
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="torch")
        >>> T0 = TorchSE3(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        >>> state = robot.state(q=robot.zero_q, T_world_base=T0)
        >>> # Lock z, roll, pitch (indices 2, 3, 4) for planar base
        >>> limit = TorchBaseStepLimit(lock_indices=[2, 3, 4], weight=10.0)
        >>> residual = limit.compute_residual(state)
        >>> residual.shape == (3,)  # 3 locked components
        True
    """

    JACOBIAN_MODE = "analytic"

    def __init__(self, lock_indices: Sequence[int], weight: float = 10.0):
        self.lock_indices = list(lock_indices)

        residual_weight = np.ones(len(lock_indices), dtype=float) * weight
        self.residual_weight = residual_weight

    def compute_residual(self, var: TorchRobotState) -> torch.Tensor:
        if not var.has_floating_base:
            device = var.q.device
            return torch.zeros(len(self.lock_indices), dtype=torch.float32, device=device)

        device = var.q.device
        return torch.zeros(len(self.lock_indices), dtype=torch.float32, device=device)

    def compute_jacobian_analytic(self, var: TorchRobotState) -> torch.Tensor:
        if not var.has_floating_base:
            num_joints = var.spec.num_actuated_joints
            device = var.q.device
            dtype = var.q.dtype
            return torch.zeros((len(self.lock_indices), num_joints), device=device, dtype=dtype)

        num_joints = var.spec.num_actuated_joints
        total_dofs = 6 + num_joints
        device = var.q.device
        dtype = var.q.dtype
        batch_shape = var.q.shape[:-1]

        J = torch.zeros(batch_shape + (len(self.lock_indices), total_dofs), device=device, dtype=dtype)

        for i, locked_idx in enumerate(self.lock_indices):
            if 0 <= locked_idx < 6:
                J[..., i, locked_idx] = 1.0

        return J
