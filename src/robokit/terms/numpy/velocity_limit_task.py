# pyright: reportArgumentType=false
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from jaxtyping import Float

from robokit.robo.numpy_robot import NumpyRobot, NumpyRobotState
from robokit.terms.terms import NumpyTask, QPLimit


class PinocchioVelocityLimitTask(NumpyTask, QPLimit):
    """
    Task that penalizes joint velocity limit violations.

    As a Task: penalizes velocities exceeding limits
    Residual: r = max(0, |q_dot| - q_dot_limit) where q_dot = (q_curr - q_prev) / dt
    Jacobian: J = sign(q_dot) / dt for joints exceeding limits

    As a Limit: enforces hard velocity constraints via QP
    QP inequalities: |q_dot| <= q_dot_limit

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="numpy")
        >>> state_prev = robot.state(q=robot.zero_q)
        >>> state_curr = robot.state(q=robot.zero_q + 0.5)  # Large change
        >>> vel_task = PinocchioVelocityLimitTask(robot=robot, dt=0.1, prev_state_var=state_prev, weight=1.0)
        >>> residual = vel_task.compute_residual(state_curr)
        >>> # Should be positive if velocity exceeds limits
    """

    def __init__(
        self,
        robot: NumpyRobot,
        dt: float,
        prev_state_var: Optional[NumpyRobotState] = None,
        velocity_limits: Optional[Float[np.ndarray, "num_actuated_joints"]] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        use_hard_constraints: bool = False,
    ):
        assert dt > 0, "dt must be positive"
        self.robot = robot
        self.dt = dt
        self.prev_state_var = prev_state_var
        self.velocity_limits = (
            velocity_limits if velocity_limits is not None else robot.spec.actuated_joint_velocity_limits
        )
        self.use_hard_constraints = use_hard_constraints

        num_joints = robot.num_actuated_joints
        residual_weight = np.ones(num_joints, dtype=float)
        if weight is not None:
            if isinstance(weight, (float, int)):
                residual_weight[:] = weight
            else:
                residual_weight[:] = weight

        self.residual_weight = residual_weight

    def compute_residual(self, var: NumpyRobotState, prev_state_var: Optional[NumpyRobotState] = None) -> np.ndarray:
        if prev_state_var is None:
            prev_state_var = self.prev_state_var
        if prev_state_var is None:
            return np.zeros(self.robot.num_actuated_joints, dtype=float)

        q_curr = var._q
        q_prev = prev_state_var._q
        assert q_curr is not None and q_prev is not None  # Type narrowing
        q_dot = (q_curr - q_prev) / self.dt

        violation = np.maximum(0.0, np.abs(q_dot) - self.velocity_limits)
        return violation

    def compute_jacobian(self, var: NumpyRobotState, prev_state_var: Optional[NumpyRobotState] = None) -> np.ndarray:
        if prev_state_var is None:
            prev_state_var = self.prev_state_var

        if prev_state_var is None:
            num_joints = self.robot.num_actuated_joints
            base_col_offset = 6 if var.has_floating_base else 0
            total_dofs = base_col_offset + num_joints
            return np.zeros((num_joints, total_dofs), dtype=float)

        q_curr = var._q
        q_prev = prev_state_var._q
        assert q_curr is not None and q_prev is not None  # Type narrowing
        q_dot = (q_curr - q_prev) / self.dt

        num_joints = self.robot.num_actuated_joints
        base_col_offset = 6 if var.has_floating_base else 0
        total_dofs = base_col_offset + num_joints

        J = np.zeros((num_joints, total_dofs), dtype=float)

        violation_mask = np.abs(q_dot) > self.velocity_limits
        sign = np.sign(q_dot)
        J_values = sign / self.dt

        violating_indices = np.where(violation_mask)[0]
        if len(violating_indices) > 0:
            J[violating_indices, base_col_offset + violating_indices] = J_values[violating_indices]

        return J

    def compute_qp_inequalities(
        self, var: NumpyRobotState, prev_state_var: Optional[NumpyRobotState] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_hard_constraints:
            return np.zeros((0, var.tangent_dim), dtype=float), np.zeros((0,), dtype=float)

        if prev_state_var is None:
            prev_state_var = self.prev_state_var

        if prev_state_var is None:
            return np.zeros((0, var.tangent_dim), dtype=float), np.zeros((0,), dtype=float)

        q_prev = prev_state_var._q
        q_curr = var._q

        num_joints = self.robot.num_actuated_joints
        base_col_offset = 6 if var.has_floating_base else 0
        total_dofs = base_col_offset + num_joints

        assert q_curr is not None and q_prev is not None  # Type narrowing
        current_delta_q = q_curr - q_prev

        max_delta = self.velocity_limits * self.dt
        delta_q_max = max_delta - current_delta_q
        delta_q_min = -max_delta - current_delta_q

        P = np.zeros((num_joints, total_dofs), dtype=float)
        P[:, base_col_offset : base_col_offset + num_joints] = np.eye(num_joints, dtype=float)

        G = np.vstack([P, -P])
        h = np.hstack([delta_q_max, -delta_q_min])

        return G, h
