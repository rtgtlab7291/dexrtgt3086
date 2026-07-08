from typing import Sequence, Tuple

import numpy as np

from robokit.robo.numpy_robot import NumpyRobotState
from robokit.terms.terms import QPLimit


class PinocchioBaseStepLimit(QPLimit):
    """
    Limit to constrain floating-base step components.

    This enforces selected base twist components (in the 6D base block) to be zero.
    lock_indices is [x, y, z, roll, pitch, yaw] indices in the base twist.

    Example:
        >>> import numpy as np
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> from robokit.lie.pinocchio_se3 import PinocchioSE3
        >>> from robokit.opt.variables import Var
        >>> from robokit.terms.numpy.base_step_limit import PinocchioBaseStepLimit
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="numpy")
        >>> T0 = PinocchioSE3(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        >>> state = robot.state(q=robot.zero_q, T_world_base=T0)
        >>> limit = PinocchioBaseStepLimit(lock_indices=[2, 3, 4])
        >>> G, h = limit.compute_qp_inequalities(state)
        >>> G.shape[0] == 2 * len([2, 3, 4])
        True
        >>> G.shape[1] == 6 + robot.num_actuated_joints
        True
        >>> np.allclose(h, 0.0)
        True

    """

    def __init__(self, lock_indices: Sequence[int], tolerance: float = 0.0):
        self.lock_indices = list(lock_indices)
        self.tolerance = float(tolerance)

    def compute_qp_inequalities(self, var: NumpyRobotState) -> Tuple[np.ndarray, np.ndarray]:
        if not var.has_floating_base:
            return np.zeros((0, var.spec.num_actuated_joints), dtype=float), np.zeros((0,), dtype=float)

        num_j = var.spec.num_actuated_joints
        total_dofs = 6 + num_j  # base(6) + joints

        P = np.zeros((len(self.lock_indices), total_dofs), dtype=float)
        for r, col in enumerate(self.lock_indices):
            if not (0 <= col < 6):
                raise ValueError(f"Base lock index out of range [0,5]: {col}")
            P[r, col] = 1.0

        G = np.vstack([P, -P])
        h = np.full(G.shape[0], self.tolerance, dtype=float)
        return G, h
