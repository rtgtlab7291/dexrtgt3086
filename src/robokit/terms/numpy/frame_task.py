# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Sequence, Union

import numpy as np
import pinocchio as pin

from robokit.lie.pinocchio_se3 import PinocchioSE3
from robokit.robo.numpy_robot import NumpyRobot, NumpyRobotState
from robokit.terms.terms import NumpyTask


class PinocchioFrameTask(NumpyTask):
    """
    Frame task for computing SE(3) residuals and Jacobians.
    Supports both single-frame and multi-frame targets.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"), backend="numpy")
        >>> target_link_index = robot.link_names.index("ee_link")
        >>> state = robot.state(q=robot.zero_q)
        >>> target_link_pose = robot.forward_kinematics(state).get_T_world_link(target_link_index)
        >>> frame_task = PinocchioFrameTask(
        ...     robot=robot,
        ...     frame_index=target_link_index,
        ...     T_world_target=target_link_pose,
        ...     position_weight=1.0,
        ...     orientation_weight=1.0)
        >>> random_q = np.array([-0.5, 0.3, -0.2, 0.4, -0.1, 0.25])
        >>> state = robot.state()
        >>> state.set_configuration(q=random_q)
        >>> error = frame_task.compute_residual(state)
        >>> jacobian = frame_task.compute_jacobian(state)
        >>> np.allclose(error[:3], [0.5839, -0.0795, -0.1784], atol=1e-3)
        True
        >>> np.allclose(jacobian[:, 0], [-1.1628, 0.1626, -0.167 , -0.0765, 0.3698, 0.9521], atol=1e-3)
        True
    """

    def __init__(
        self,
        robot: NumpyRobot,
        frame_index: Union[int, Sequence[int]],
        T_world_target: Union[PinocchioSE3, Sequence[PinocchioSE3]],
        position_weight: Union[float, Sequence[float]],
        orientation_weight: Union[float, Sequence[float]],
    ):
        self.robot = robot

        if isinstance(frame_index, int):
            self.frame_indices = [frame_index]
        else:
            self.frame_indices = list(frame_index)

        if isinstance(T_world_target, PinocchioSE3):
            self.T_world_targets = [T_world_target]
        else:
            self.T_world_targets = list(T_world_target)

        if len(self.frame_indices) != len(self.T_world_targets):
            raise ValueError(
                f"Number of frame indices ({len(self.frame_indices)}) must match "
                f"number of targets ({len(self.T_world_targets)})"
            )

        self.num_frames = len(self.frame_indices)
        self.residual_weight = np.tile(np.ones(6), self.num_frames)
        for i in range(self.num_frames):
            self.residual_weight[i * 6 : i * 6 + 3] = position_weight
            self.residual_weight[i * 6 + 3 : i * 6 + 6] = orientation_weight

    def set_targets(self, targets: Sequence[PinocchioSE3]) -> None:
        if len(targets) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} targets, got {len(targets)}")
        for i, target in enumerate(targets):
            self.T_world_targets[i] = target

    def compute_residual(self, var: NumpyRobotState) -> np.ndarray:
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)

        residuals = []
        for frame_index, T_world_target in zip(self.frame_indices, self.T_world_targets):
            T_world_frame = var.get_T_world_link(frame_index)
            T_frame_target = T_world_frame.pin_se3.actInv(T_world_target.pin_se3)
            error_in_frame = pin.log(T_frame_target).vector
            residuals.append(error_in_frame)
        return np.concatenate(residuals)

    def compute_jacobian(self, var: NumpyRobotState) -> np.ndarray:
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)

        jacobians = []
        for frame_index, T_world_target in zip(self.frame_indices, self.T_world_targets):
            T_world_frame = var.get_T_world_link(frame_index)
            T_target_frame = T_world_target.pin_se3.actInv(T_world_frame.pin_se3)
            jacobian_in_frame = var.get_link_jacobian(frame_index, reference_frame="body")
            J = -pin.Jlog6(T_target_frame) @ jacobian_in_frame
            jacobians.append(J)
        return np.vstack(jacobians)
