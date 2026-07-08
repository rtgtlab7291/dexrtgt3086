# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from robokit.robo.numpy_robot import NumpyRobot, NumpyRobotState
from robokit.terms.terms import NumpyTask, QPLimit


class PinocchioPositionLimit(NumpyTask, QPLimit):
    """
    Position limit constraint for joint positions.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"), backend="numpy")
        >>> joint_var = robot.actuated_joint_limits[:, 0] - 0.1
        >>> state = robot.state()
        >>> state.set_configuration(q=joint_var)
        >>> position_limit = PinocchioPositionLimit(robot=robot)
        >>> residual = position_limit.compute_residual(state)
        >>> np.allclose(residual, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        True
        >>> jacobian = position_limit.compute_jacobian(state)
        >>> np.allclose(np.diag(jacobian), [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        True
    """

    def __init__(
        self,
        robot: NumpyRobot,
        weight: Optional[Union[float, Sequence[float]]] = None,
    ):
        self.robot = robot
        self.residual_weight = np.zeros(robot.num_actuated_joints, dtype=float)
        self.residual_weight[:] = weight if weight is not None else 1.0
        self.projection_matrix = np.eye(robot.num_actuated_joints, dtype=float)

    def compute_residual(self, var: NumpyRobotState) -> np.ndarray:
        joint_positions = var._q
        actuated_joint_limits = var.spec.actuated_joint_limits
        error_upper = np.maximum(0, joint_positions - actuated_joint_limits[..., 1])
        error_lower = np.maximum(0, actuated_joint_limits[..., 0] - joint_positions)
        return error_upper + error_lower

    def compute_jacobian(self, var: NumpyRobotState) -> np.ndarray:
        joint_positions = var._q
        actuated_joint_limits = var.spec.actuated_joint_limits
        sign = np.zeros_like(joint_positions, dtype=float)
        sign[joint_positions > actuated_joint_limits[..., 1]] = 1.0
        sign[joint_positions < actuated_joint_limits[..., 0]] = -1.0

        base_col_offset = 6 if var.has_floating_base else 0
        total_dofs = base_col_offset + var.spec.num_actuated_joints
        J = np.zeros((var.spec.num_actuated_joints, total_dofs), dtype=float)
        J[:, base_col_offset : base_col_offset + var.spec.num_actuated_joints] = np.diag(sign)
        return J

    def compute_qp_inequalities(self, var: NumpyRobotState) -> Tuple[np.ndarray, np.ndarray]:
        joint_positions = var._q
        actuated_joint_limits = var.spec.actuated_joint_limits
        delta_q_max = actuated_joint_limits[..., 1] - joint_positions
        delta_q_min = actuated_joint_limits[..., 0] - joint_positions
        p_max = delta_q_max
        p_min = delta_q_min

        base_col_offset = 6 if var.has_floating_base else 0
        num_j = var.spec.num_actuated_joints
        total_dofs = base_col_offset + num_j
        P = np.zeros((num_j, total_dofs), dtype=float)
        P[:, base_col_offset : base_col_offset + num_j] = np.eye(num_j, dtype=float)

        G = np.vstack([P, -P])
        h = np.hstack([p_max, -p_min])
        return G, h
