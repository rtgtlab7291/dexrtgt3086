# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin
from jaxtyping import Float

from robokit.lie.pinocchio_se3 import PinocchioSE3
from robokit.robo.numpy_robot import NumpyRobot, NumpyRobotState
from robokit.terms.terms import NumpyTask


class PinocchioRestTask(NumpyTask):
    """
    Regularization task that biases joints towards a rest pose.

    Residual: r = q - q_rest
    Jacobian: J = I (identity matrix for joints)

    For floating base robots, can optionally include base regularization
    where residual includes base twist: r_base = log(T_world_base)

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="numpy")
        >>> rest_q = robot.midrange_q
        >>> rest_task = PinocchioRestTask(robot=robot, rest_q=rest_q, weight=0.1)
        >>> state = robot.state(q=rest_q + 0.1)
        >>> residual = rest_task.compute_residual(state)
        >>> np.allclose(residual, 0.1, atol=1e-6)
        True
    """

    def __init__(
        self,
        robot: NumpyRobot,
        rest_q: Optional[Float[np.ndarray, "num_actuated_joints"]] = None,
        T_world_base_rest: Optional[PinocchioSE3] = None,
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

    def compute_residual(self, var: NumpyRobotState) -> np.ndarray:
        q = var._q
        r_joints = q - self.rest_q

        if self.T_world_base_rest is not None and var.has_floating_base:
            T_world_base = var._T_world_base
            assert T_world_base is not None
            T_error = T_world_base @ self.T_world_base_rest.inverse()
            r_base = T_error.log()
            return np.concatenate([r_joints, r_base])
        else:
            return r_joints

    def compute_jacobian(self, var: NumpyRobotState) -> np.ndarray:
        num_joints = self.robot.num_actuated_joints
        base_col_offset = 6 if var.has_floating_base else 0
        total_dofs = base_col_offset + num_joints

        if self.T_world_base_rest is not None and var.has_floating_base:
            J = np.zeros((num_joints + 6, total_dofs), dtype=float)
            J[:num_joints, base_col_offset:] = np.eye(num_joints, dtype=float)
            T_world_base = var._T_world_base
            assert T_world_base is not None
            T_error = T_world_base @ self.T_world_base_rest.inverse()
            J_base = pin.Jlog6(T_error.pin_se3) @ T_world_base.adjoint()
            J[num_joints:, :6] = J_base
        else:
            J = np.zeros((num_joints, total_dofs), dtype=float)
            J[:, base_col_offset:] = np.eye(num_joints, dtype=float)

        return J
