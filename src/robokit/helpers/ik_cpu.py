from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, cast

import numpy as np

from robokit.lie.pinocchio_se3 import PinocchioSE3
from robokit.opt.numpy_optimizer import NumpyOptimizerConfig
from robokit.robo.numpy_robot import NumpyRobot


@dataclass
class CPUIKHelperConfig:
    position_weight: float = 10.0
    orientation_weight: float = 5.0
    limit_weight: float = 50.0
    optimizer_config: Optional["NumpyOptimizerConfig"] = None


class CPUIKHelper:
    def __init__(
        self,
        robot: "NumpyRobot",
        target_frame: Union[str, int, Sequence[Union[str, int]]],
        config: Optional[CPUIKHelperConfig] = None,
    ):
        from robokit.lie.pinocchio_se3 import PinocchioSE3
        from robokit.opt.numpy_optimizer import NumpyOptimizer, NumpyOptimizerConfig
        from robokit.terms.numpy.frame_task import PinocchioFrameTask
        from robokit.terms.numpy.position_limit import PinocchioPositionLimit

        self.robot = robot
        self.config = config or CPUIKHelperConfig()

        if isinstance(target_frame, (str, int)):
            target_frames_list: List[Union[str, int]] = [target_frame]
        else:
            target_frames_list = list(target_frame)

        self.target_link_indices: List[int] = []
        for frame in target_frames_list:
            if isinstance(frame, str):
                self.target_link_indices.append(self.robot.link_names.index(frame))
            else:
                self.target_link_indices.append(frame)

        self.num_frames = len(self.target_link_indices)

        dummy_targets = [PinocchioSE3(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])) for _ in range(self.num_frames)]

        frame_task = PinocchioFrameTask(
            robot=self.robot,
            frame_index=self.target_link_indices,
            T_world_target=dummy_targets,
            position_weight=self.config.position_weight,
            orientation_weight=self.config.orientation_weight,
        )
        position_limit = PinocchioPositionLimit(robot=self.robot, weight=self.config.limit_weight)

        if self.config.optimizer_config is None:
            optimizer_config = NumpyOptimizerConfig(
                use_qpsolver=False,
                lm_lambda=1.0,
                use_early_stopping=True,
                max_iter=64,
            )
        else:
            optimizer_config = self.config.optimizer_config

        self._optimizer = NumpyOptimizer(terms=[frame_task, position_limit], config=optimizer_config)
        self._frame_task = frame_task
        self._initial_q = self.robot.midrange_q
        self._state = self.robot.state(q=self._initial_q)

    def solve(self, target_se3: Union["PinocchioSE3", Sequence["PinocchioSE3"]]) -> np.ndarray:
        """Solve IK for the given target pose(s).

        Args:
            target_se3: Target pose(s) as PinocchioSE3 or list of PinocchioSE3

        Returns:
            Solved joint configuration as numpy array
        """
        from robokit.lie.pinocchio_se3 import PinocchioSE3

        if isinstance(target_se3, PinocchioSE3):
            targets_list = [target_se3]
        else:
            targets_list = list(target_se3)

        if len(targets_list) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} targets, got {len(targets_list)}")

        self._frame_task.set_targets(targets_list)
        self._state.set_configuration(q=self._initial_q)
        solved_state = self._optimizer.solve(self._state)
        return solved_state.q

    def solve_numpy(
        self,
        target_pos: Union[np.ndarray, List[np.ndarray]],
        target_quat_wxyz: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """Solve IK for the given target position(s) and orientation(s).

        Args:
            target_pos: Target position(s) as numpy array(s) of shape (3,)
            target_quat_wxyz: Target orientation(s) as quaternion(s) (w, x, y, z) of shape (4,)

        Returns:
            Solved joint configuration as numpy array
        """
        from robokit.lie.pinocchio_se3 import PinocchioSE3

        if isinstance(target_pos, list):
            quat_list = cast(List[np.ndarray], target_quat_wxyz)
            targets = [PinocchioSE3(np.concatenate([pos, quat])) for pos, quat in zip(target_pos, quat_list)]
            return self.solve(targets)
        else:
            quat_array = cast(np.ndarray, target_quat_wxyz)
            target_pose = np.concatenate([target_pos, quat_array])
            target_se3 = PinocchioSE3(target_pose)
            return self.solve(target_se3)
