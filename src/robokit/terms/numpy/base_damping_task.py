# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from robokit.lie.pinocchio_se3 import PinocchioSE3
from robokit.robo.numpy_robot import NumpyRobot, NumpyRobotState
from robokit.terms.terms import NumpyTask


class PinocchioBaseDampingTask(NumpyTask):
    """
    Damping task for floating base that penalizes base motion.

    Residual: r = log(T_world_base) (base twist)
    Jacobian: J = [I_6, 0] (identity for base, zero for joints)

    This encourages the optimizer to prefer joint motion over base motion
    by penalizing base twists directly.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> from robokit.lie.pinocchio_se3 import PinocchioSE3
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="numpy")
        >>> T_base = PinocchioSE3(np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        >>> state = robot.state(q=robot.zero_q, T_world_base=T_base)
        >>> damping_task = PinocchioBaseDampingTask(robot=robot, weight=0.1)
        >>> residual = damping_task.compute_residual(state)
        >>> # Residual should be small twist from identity
    """

    def __init__(
        self,
        robot: NumpyRobot,
        T_world_base_ref: Optional[PinocchioSE3] = None,
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

    def compute_residual(self, var: NumpyRobotState) -> np.ndarray:
        if not var.has_floating_base:
            return np.zeros(6, dtype=float)

        T_world_base = var._T_world_base
        assert T_world_base is not None
        if self.T_world_base_ref is not None:
            assert self.T_world_base_ref is not None
            T_error = T_world_base @ self.T_world_base_ref.inverse()
        else:
            T_error = T_world_base

        return T_error.log()

    def compute_jacobian(self, var: NumpyRobotState) -> np.ndarray:
        if not var.has_floating_base:
            total_dofs = var.spec.num_actuated_joints
            return np.zeros((6, total_dofs), dtype=float)

        num_joints = self.robot.num_actuated_joints
        total_dofs = 6 + num_joints

        T_world_base = var._T_world_base
        assert T_world_base is not None
        if self.T_world_base_ref is not None:
            assert self.T_world_base_ref is not None
            T_error = T_world_base @ self.T_world_base_ref.inverse()
        else:
            T_error = T_world_base

        J_base = pin.Jlog6(T_error.pin_se3) @ T_world_base.adjoint()

        J = np.zeros((6, total_dofs), dtype=float)
        J[:, :6] = J_base

        return J
