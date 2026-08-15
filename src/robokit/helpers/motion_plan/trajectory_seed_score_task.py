# pyright: reportArgumentType=false, reportIndexIssue=false
from typing import Callable, Optional, Sequence

import warp as wp

from robokit.helpers.motion_plan.evaluator import _compute_ee_pose_error
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec7


# --- kernel ----------------------------------------------------------------
@wp.kernel
def _compute_trajectory_seed_score_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    goal_vec7: wp.array1d(dtype=wp_vec7),
    cost: wp.array1d(dtype=wp.float32),
    penetration: wp.array1d(dtype=wp.float32),
    ee_link_index: int,
    last_frame: int,
    row_offset: int,
    num_seeds: int,
    keep_seed_zero: int,
    feasibility_first: int,
    residual: wp.array2d(dtype=wp.float32),
):
    batch = wp.tid()
    error = _compute_ee_pose_error(T_world_link[batch, last_frame, ee_link_index], goal_vec7[batch])
    score = 1000.0 * error[0] + 500.0 * error[1] + 0.01 * cost[batch] + 1.0e5 * penetration[batch]
    if feasibility_first != 0:
        if penetration[batch] > 0.001:
            score += 1.0e6
    elif error[0] >= 0.005 or error[1] >= 0.05:
        score += 1.0e6
    seed_zero = batch - batch % num_seeds
    if keep_seed_zero != 0 and batch != seed_zero:
        if feasibility_first != 0:
            if penetration[seed_zero] <= 0.001:
                score += 1.0e6
        else:
            seed_zero_error = _compute_ee_pose_error(
                T_world_link[seed_zero, last_frame, ee_link_index], goal_vec7[seed_zero]
            )
            if seed_zero_error[0] < 0.005 and seed_zero_error[1] < 0.05 and penetration[seed_zero] <= 1.0e-5:
                score += 1.0e6
    residual[batch, row_offset] = wp.sqrt(2.0 * score)


# --- task ------------------------------------------------------------------
class TrajectorySeedScoreTask(ResidualTask):
    """Adapt the motion planner's seed ranking to the population solver score API."""

    def __init__(
        self,
        robot: Robot,
        ee_link_index: int,
        last_frame: int,
        goal_vec7: wp.array,
        cost_input: Callable[[], wp.array],
        collision_tasks: Sequence[TrajectoryCollisionTask],
        num_seeds: int,
        keep_seed_zero: bool,
        feasibility_first: bool,
    ):
        self.robot = robot
        self.ee_link_index = ee_link_index
        self.last_frame = last_frame
        self.goal_vec7 = goal_vec7
        self.cost_input = cost_input
        self.collision_tasks = tuple(collision_tasks)
        self.num_seeds = num_seeds
        self.keep_seed_zero = keep_seed_zero
        self.feasibility_first = feasibility_first
        self._penetrations = wp.zeros(goal_vec7.shape[0], dtype=wp.float32, device=goal_vec7.device)
        self._collision_jacobians = [
            wp.empty(
                (goal_vec7.shape[0], task.residual_dim * robot.num_actuated_joints),
                dtype=wp.float32,
                device=goal_vec7.device,
            )
            for task in collision_tasks
        ]

    @property
    def residual_dim(self) -> int:
        return 1

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        state: RobotState = var_values.get("robot")  # type: ignore[assignment]
        if not state.is_fk_computed:
            self.robot.forward_kinematics(state)
        if out_residual is None:
            out_residual = wp.empty((state.q.shape[0], 1), dtype=wp.float32, device=state.q.device)
            row_offset = 0
        self._penetrations.zero_()
        for task, jacobian in zip(self.collision_tasks, self._collision_jacobians):
            task.compute_weighted_sparse_jacobian_values(var_values, out_jacobian_values=jacobian)
            task.accumulate_max_penetration(self._penetrations)
        wp.launch(
            _compute_trajectory_seed_score_residual_kernel,
            dim=state.q.shape[0],
            inputs=[
                state.T_world_link,
                self.goal_vec7,
                self.cost_input(),
                self._penetrations,
                self.ee_link_index,
                self.last_frame,
                row_offset,
                self.num_seeds,
                int(self.keep_seed_zero),
                int(self.feasibility_first),
            ],
            outputs=[out_residual],
            device=state.q.device,
        )
        return out_residual
