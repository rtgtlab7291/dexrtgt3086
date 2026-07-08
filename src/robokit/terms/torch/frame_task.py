# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Sequence, Union

import numpy as np
import torch
from jaxtyping import Float

from robokit.lie.torch_se3 import TorchSE3
from robokit.robo.torch_robot import TorchRobot, TorchRobotState
from robokit.terms.terms import TorchTask


class TorchFrameTask(TorchTask):
    """
    Torch-based frame task for computing SE(3) residuals and Jacobians.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"), backend="torch")
        >>> target_link_index = robot.link_names.index("ee_link")
        >>> state = robot.state(q=robot.zero_q)
        >>> target_link_pose = robot.forward_kinematics(state).get_T_world_link(target_link_index)
        >>> frame_task = TorchFrameTask(
        ...     robot=robot,
        ...     frame_index=target_link_index,
        ...     T_world_target=target_link_pose,
        ...     position_weight=1.0,
        ...     orientation_weight=1.0)
        >>> random_q = torch.tensor([-0.5, 0.3, -0.2, 0.4, -0.1, 0.25], dtype=torch.float32)
        >>> state = robot.state()
        >>> state.set_configuration(q=random_q)
        >>> error = frame_task.compute_residual(state)
        >>> jacobian = frame_task.compute_jacobian(state)
        >>> torch.allclose(error[:3], torch.tensor([0.5839, -0.0795, -0.1784], dtype=torch.float32), atol=1e-3)
        True
        >>> torch.allclose(jacobian[:, 0], torch.tensor([-1.1628, 0.1626, -0.167, -0.0765, 0.3698, 0.9521], dtype=torch.float32), atol=1e-3)
        True
    """

    def __init__(
        self,
        robot: TorchRobot,
        frame_index: int,
        T_world_target: TorchSE3,
        position_weight: Union[float, Sequence[float]],
        orientation_weight: Union[float, Sequence[float]],
    ):
        self.robot = robot
        self.frame_index = frame_index
        self.T_world_target = T_world_target
        self.residual_weight = np.ones(6)
        self.residual_weight[0:3] = position_weight
        self.residual_weight[3:6] = orientation_weight

    def compute_residual(self, var: TorchRobotState) -> Float[torch.Tensor, "... 6"]:
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        T_world_frame = var.get_T_world_link(self.frame_index)
        T_frame_target = T_world_frame.inverse() @ self.T_world_target
        error_in_frame = T_frame_target.log()
        return error_in_frame

    def compute_jacobian_analytic(self, var: TorchRobotState) -> Float[torch.Tensor, "... 6 num_dofs"]:
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        T_world_frame = var.get_T_world_link(self.frame_index)
        T_target_frame = self.T_world_target.inverse() @ T_world_frame
        jacobian_in_frame = var.get_link_jacobian(self.frame_index, reference_frame="body")
        J = -T_target_frame.jlog() @ jacobian_in_frame
        return J
