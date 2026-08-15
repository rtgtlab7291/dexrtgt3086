# pyright: reportArgumentType=false
import dataclasses
import math
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.helpers.motion_plan.trajectory_seed_score_task import TrajectorySeedScoreTask
from robokit.opt.gd_optimizer import GDOptimizerConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
from robokit.opt.multi_seed_solver import MultiSeedSolver, MultiSeedSolverConfig, StageConfig
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.sparse.trajectory_velocity_limit_task import TrajectoryVelocityLimitTask
from robokit.terms.task import ResidualTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils import warp_utils
from robokit.utils.warp_utils import wp_vec7


if TYPE_CHECKING:
    from robokit.geom import WarpScene


@wp.kernel
def _compute_collision_margins_kernel(
    start_q: wp.array2d(dtype=wp.float32),
    goal_q: wp.array2d(dtype=wp.float32),
    margin: float,
    goal_radius: float,
    out: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    distance_sq = float(0.0)
    for joint_idx in range(start_q.shape[1]):
        delta = goal_q[batch_idx, joint_idx] - start_q[batch_idx, joint_idx]
        distance_sq += delta * delta
    out[batch_idx] = margin * wp.min(wp.sqrt(distance_sq) / goal_radius, 1.0)


# --- task construction -----------------------------------------------------
def _build_trajectory_tasks(
    config: Any,
    robot: Robot,
    ee_link_index: int,
    num_frames: int,
    batch_size: int,
    dt: float,
    scene: Optional["WarpScene"],
    scene_indices: wp.array,
    device: str,
) -> Tuple[
    List[ResidualTask],
    Optional[TrajectoryTask],
    List[TrajectoryCollisionTask],
    Optional[TrajectoryTask],
    Optional[TrajectoryTask],
]:
    """Build the tasks shared by gradient and MPPI trajectory optimizers."""
    target_np = np.tile(np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (batch_size, 1))
    T_world_goal = wp.from_numpy(target_np[:, None], dtype=wp_vec7, device=device)
    tasks: List[ResidualTask] = []
    start_task: Optional[TrajectoryTask] = None
    collision_tasks: List[TrajectoryCollisionTask] = []
    goal_task: Optional[TrajectoryTask] = None
    frame_task: Optional[TrajectoryTask] = None

    if config.start_config_weight > 0:
        start_task = TrajectoryTask(
            RestTask(
                robot=robot,
                rest_q=np.zeros((batch_size, robot.num_actuated_joints), dtype=np.float32),
                weight=config.start_config_weight,
            ),
            num_frames=num_frames,
            frame_indices=[0],
        )
        # allocate now: _set_inputs runs inside CUDA graph capture, where allocation is illegal
        start_task.init_buffers(device)
    if config.position_weight > 0 or config.orientation_weight > 0:
        frame_task = TrajectoryTask(
            FrameTask(
                robot=robot,
                frame_index=ee_link_index,
                T_world_target=cast(wp.array, T_world_goal[:, 0]).contiguous(),
                position_weight=config.position_weight,
                orientation_weight=config.orientation_weight,
            ),
            num_frames=num_frames,
            frame_indices=[-1],
        )
        frame_task.init_buffers(device)
        tasks.append(frame_task)
    if start_task is not None:
        tasks.append(start_task)
    if config.smoothness_weight > 0:
        weights = np.full(num_frames - 1, config.smoothness_weight, dtype=np.float32)
        if not config.smooth_boundary:
            weights[0] = 0.0
            weights[-1] = 0.0
        tasks.append(
            TrajectorySmoothnessTask(
                robot=robot,
                num_frames=num_frames,
                weight=np.repeat(weights, robot.num_actuated_joints),
            )
        )
    if config.accel_smoothness_weight > 0:
        tasks.append(
            TrajectorySmoothnessTask(
                robot=robot,
                num_frames=num_frames,
                weight=config.accel_smoothness_weight,
                order=3,
                dt=dt,
            )
        )
    if config.accel2_smoothness_weight > 0:
        tasks.append(
            TrajectorySmoothnessTask(
                robot=robot,
                num_frames=num_frames,
                weight=config.accel2_smoothness_weight,
                order=2,
                dt=dt,
            )
        )
    if config.velocity_limit_weight > 0:
        tasks.append(
            TrajectoryVelocityLimitTask(
                robot=robot,
                num_frames=num_frames,
                dt=dt,
                weight=config.velocity_limit_weight,
            )
        )
    if config.position_limit_weight > 0:
        tasks.append(
            TrajectoryTask(
                PositionLimit(robot=robot, weight=config.position_limit_weight),
                num_frames=num_frames,
            )
        )
    if config.rest_weight > 0:
        tasks.append(
            TrajectoryTask(
                RestTask(
                    robot=robot,
                    rest_q=robot.spec.midrange_q.astype(np.float32),
                    weight=config.rest_weight,
                ),
                num_frames=num_frames,
            )
        )
    if config.self_collision_weight > 0:
        tasks.append(
            TrajectorySelfCollisionTask(
                robot=robot,
                num_frames=num_frames,
                weight=config.self_collision_weight,
                margin=config.self_collision_margin,
            )
        )
    if config.collision_weight > 0 and scene is not None:
        for geometry in scene.geoms or (None,):
            task = TrajectoryCollisionTask(
                robot=robot,
                scene=scene,
                geometry=geometry,
                scene_indices=scene_indices,
                num_frames=num_frames,
                weight=config.collision_weight,
                margin=config.collision_margin,
                free_goal_frame=config.free_goal_frame,
                sweep_steps=config.collision_sweep_steps,
                penalty=config.collision_penalty,
            )
            collision_tasks.append(task)
            tasks.append(task)
    if config.goal_config_weight > 0:
        goal_task = TrajectoryTask(
            RestTask(
                robot=robot,
                rest_q=np.zeros((batch_size, robot.num_actuated_joints), dtype=np.float32),
                weight=config.goal_config_weight,
            ),
            num_frames=num_frames,
            frame_indices=[-1],
        )
        goal_task.init_buffers(device)
        tasks.append(goal_task)
    if not tasks:
        raise ValueError("Trajectory optimizer requires at least one enabled task.")
    return tasks, start_task, collision_tasks, goal_task, frame_task


# --- configuration ---------------------------------------------------------
@dataclasses.dataclass
class GradientTrajectoryOptimizerConfig:
    """Configuration for gradient trajectory optimization."""

    position_weight: float
    orientation_weight: float
    smoothness_weight: float
    collision_weight: float
    start_config_weight: float = 100.0
    goal_config_weight: float = 0.0
    velocity_limit_weight: float = 1.0
    position_limit_weight: float = 100.0
    rest_weight: float = 0.01
    self_collision_weight: float = 10.0
    self_collision_margin: float = 0.02
    collision_margin: float = 0.01
    collision_goal_radius: float = 0.0  # Taper collision margin inside this joint-space goal radius.
    max_iter: int = 50
    num_seeds: int = 1
    init_sample_range: float = 0.2
    keep_init_seed: bool = True
    use_cuda_graph: bool = True
    accel_smoothness_weight: float = 0.0
    accel2_smoothness_weight: float = 0.0
    smooth_boundary: bool = False
    lock_start: bool = False
    lock_endpoints: bool = False
    free_goal_frame: bool = True
    collision_sweep_steps: int = 0
    collision_penalty: Literal["smooth", "plain", "surface_distance"] = "plain"
    lm_lambda: float = 0.5
    solver: Literal["lm", "lbfgs", "gd", "adam", "adamw"] = "lm"
    lambda_factor: float = 2.0
    lambda_min: float = 1e-6
    lambda_max: float = 1e6
    rho_min: float = 1e-4
    gain_ratio_epsilon: float = 1e-8
    lbfgs_line_search_alphas: tuple = (0.001, 0.01, 0.05, 0.1)
    lbfgs_history_len: int = 15
    lbfgs_normalize_direction: bool = True
    lbfgs_h0_scale: float = 1.0
    lbfgs_bb_h0: bool = False
    gd_learning_rate: float = 0.02
    gd_eta_min: float = 3e-3
    gd_weight_decay: float = 1e-4
    cg_max_iter: Optional[int] = None


def build_default_gradient_trajectory_optimizer_config() -> GradientTrajectoryOptimizerConfig:
    """Build the default gradient trajectory optimizer configuration."""
    return GradientTrajectoryOptimizerConfig(
        position_weight=400.0,
        orientation_weight=200.0,
        smoothness_weight=5.0,
        collision_weight=10.0,
        lm_lambda=0.5,
        max_iter=50,
    )


# --- optimizer -------------------------------------------------------------
class GradientTrajectoryOptimizer:
    """Gradient trajectory optimization initialized by the caller on every solve.

    Lifecycle:
        1. `__init__` stores configuration and creates the active-DOF mask.
        2. `warmup(batch_size)` builds batch-shaped tasks, buffers, and the solver once. The first
           `solve` calls it automatically when needed.
        3. `solve(...)` performs an independent optimization:
           a. `_set_inputs(...)` updates the tasks and initializes the trajectory variable.
           b. `_solver.solve(...)` returns the best robot state and costs.
    """

    def __init__(
        self,
        config: GradientTrajectoryOptimizerConfig,
        robot: Robot,
        ee_link_name_or_index: Union[str, int],
        num_frames: int,
        dt: float,
        scene: Optional["WarpScene"] = None,
        device: str = "cuda:0",
    ):
        if config.collision_weight > 0 and scene is None:
            raise ValueError("A collision scene is required when collision_weight > 0.")
        self.robot = robot
        self.ee_link_index = (
            robot.link_names.index(ee_link_name_or_index)
            if isinstance(ee_link_name_or_index, str)
            else int(ee_link_name_or_index)
        )
        self.num_frames = num_frames
        self._scene = scene
        self.device = device
        self.num_dofs = robot.num_actuated_joints
        self._num_seeds = config.num_seeds
        self._config = config
        self._dt = dt
        self.batch_size = 0
        self._initialized = False
        lock_mask = np.ones(self.num_dofs * num_frames, dtype=np.float32)
        if config.lock_start or config.lock_endpoints:
            lock_mask[: self.num_dofs] = 0.0
        if config.lock_endpoints and num_frames >= 2:
            lock_mask[-self.num_dofs :] = 0.0
        self._lock_endpoints_mask_np = lock_mask
        self._active_dof_mask_wp = wp.from_numpy(lock_mask, dtype=wp.float32, device=device)
        self._solver: Optional[MultiSeedSolver] = None

    def set_active_joint_mask(self, joint_mask: Sequence[float]):
        """Lock or unlock actuated joints without rebuilding the optimizer."""
        joint_mask_np = np.asarray(joint_mask, dtype=np.float32)
        self._active_dof_mask_wp.assign(np.tile(joint_mask_np, self.num_frames) * self._lock_endpoints_mask_np)

    def _build_problem(self, batch_size: int):
        """Build batch-shaped tasks and optimization variables."""
        self.batch_size = batch_size
        actual_batch = batch_size * self._num_seeds
        scene_indices = wp.zeros(actual_batch, dtype=wp.int32, device=self.device)
        (
            self._tasks,
            self._start_task,
            self._collision_tasks,
            self._goal_task,
            self._frame_task,
        ) = _build_trajectory_tasks(
            self._config,
            self.robot,
            self.ee_link_index,
            self.num_frames,
            actual_batch,
            self._dt,
            self._scene,
            scene_indices,
            self.device,
        )
        self._history_tasks = [
            task for task in self._tasks if isinstance(task, TrajectorySmoothnessTask) and task.order > 1
        ]
        q = wp.zeros((actual_batch, self.num_frames, self.num_dofs), dtype=wp.float32, device=self.device)
        self._state = self.robot.state(q=q)
        self._var = VarValues(robot=self._state)
        self._goal_vec7_buf = wp.zeros(actual_batch, dtype=wp_vec7, device=self.device)
        self._sq_buf = wp.zeros((actual_batch, self.num_dofs), dtype=wp.float32, device=self.device)
        self._target_q_per_seed_buf = wp.zeros((actual_batch, self.num_dofs), dtype=wp.float32, device=self.device)
        self._scene_indices_buf = wp.zeros(actual_batch, dtype=wp.int32, device=self.device)
        self._collision_margin_buf = wp.zeros(actual_batch, dtype=wp.float32, device=self.device)

    def _set_inputs(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        traj_q_init: wp.array,
        target_q_per_seed: wp.array,
        scene_indices: wp.array,
        q_history: Optional[wp.array],
        history_count: int,
    ):
        """Set task inputs and copy the initial trajectory into the solver variable."""
        target = self._goal_vec7_buf
        goal_q = target_q_per_seed
        sq = warp_utils.repeat(start_q, self._num_seeds, out=self._sq_buf)
        if T_world_target is not None:
            warp_utils.repeat(T_world_target, self._num_seeds, out=target)
        if target_q is not None:
            goal_q = warp_utils.repeat(target_q, self._num_seeds, out=self._target_q_per_seed_buf)
        si = warp_utils.repeat(scene_indices, self._num_seeds, out=self._scene_indices_buf)
        if self._config.collision_goal_radius > 0.0:
            wp.launch(
                kernel=_compute_collision_margins_kernel,
                dim=goal_q.shape[0],
                inputs=[
                    sq,
                    goal_q,
                    self._config.collision_margin,
                    self._config.collision_goal_radius,
                ],
                outputs=[self._collision_margin_buf],
                device=self.device,
            )
        for task in self._collision_tasks:
            task.set_scene_indices(si)
            if self._config.collision_goal_radius > 0.0:
                task.set_margin(self._collision_margin_buf)
        if self._start_task is not None:
            self._start_task.dense_task.set_rest_state(sq)
        if self._goal_task is not None:
            self._goal_task.dense_task.set_rest_state(goal_q)
        if self._frame_task is not None:
            self._frame_task.dense_task.set_target(target)
        for task in self._history_tasks:
            task.set_history(q_history, self._num_seeds, history_count >= 3)
        wp.copy(self._state.q, traj_q_init)
        self._var.invalidate()

    def warmup(self, batch_size: int):
        """Build the batch-shaped tasks, buffers, and solver."""
        if self._initialized:
            return
        self._build_problem(batch_size)
        config = cast(GradientTrajectoryOptimizerConfig, self._config)
        stages = [StageConfig(config.num_seeds, config.max_iter, config.lm_lambda)]
        mode = "full" if config.use_cuda_graph else "none"
        score_task = (
            TrajectorySeedScoreTask(
                self.robot,
                self.ee_link_index,
                self.num_frames - 1,
                self._goal_vec7_buf,
                lambda: cast(MultiSeedSolver, self._solver).candidate_scores,
                self._collision_tasks,
                self._num_seeds,
                config.keep_init_seed,
                not config.free_goal_frame,
            )
            if self._num_seeds > 1
            else None
        )
        score_terms = [score_task] if score_task is not None else None

        if config.solver == "lbfgs":
            solver_config = MultiSeedSolverConfig(
                stages=stages,
                cuda_graph_mode=mode,
                active_dof_mask=self._active_dof_mask_wp,
                lbfgs=LBFGSOptimizerConfig(
                    gradient_mode="analytic_jacobian",
                    line_search_alphas=config.lbfgs_line_search_alphas,
                    history_len=config.lbfgs_history_len,
                    normalize_direction=config.lbfgs_normalize_direction,
                    h0_scale=config.lbfgs_h0_scale,
                    bb_h0=config.lbfgs_bb_h0,
                    fold_line_search_gradient=True,
                ),
            )
        elif config.solver in ("gd", "adam", "adamw"):

            def compute_cosine_learning_rate(it: int, lr: float) -> float:
                n = max(config.max_iter, 1)
                return config.gd_eta_min + 0.5 * (config.gd_learning_rate - config.gd_eta_min) * (
                    1.0 + math.cos(math.pi * min(it, n) / n)
                )

            solver_config = MultiSeedSolverConfig(
                stages=stages,
                cuda_graph_mode=mode,
                active_dof_mask=self._active_dof_mask_wp,
                gd=GDOptimizerConfig(
                    learning_rate=config.gd_learning_rate,
                    lr_schedule=compute_cosine_learning_rate,
                    optimizer_type=config.solver,
                    weight_decay=config.gd_weight_decay,
                    gradient_mode="analytic_jacobian",
                    use_early_stopping=False,
                ),
            )
        else:
            solver_config = MultiSeedSolverConfig(
                stages=stages,
                optimizer_type="sparse",
                cuda_graph_mode=mode,
                active_dof_mask=self._active_dof_mask_wp,
                cg_max_iter_override=config.cg_max_iter,
                lambda_factor=config.lambda_factor,
                rho_min=config.rho_min,
                gain_ratio_epsilon=config.gain_ratio_epsilon,
            )
        self._solver = MultiSeedSolver(
            terms=[self._tasks],
            config=solver_config,
            device=self.device,
            score_terms=score_terms,
        )
        self._initialized = True

    def solve(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        traj_q_init: wp.array,
        target_q_per_seed: wp.array,
        scene_indices: wp.array,
        q_history: Optional[wp.array] = None,
        history_count: int = 0,
    ) -> Tuple[RobotState, wp.array]:
        """Solve one independent trajectory optimization problem.

        Args:
            start_q: Batched starting configurations.
            T_world_target: Optional batched pose goals.
            target_q: Optional batched joint-space goals.
            traj_q_init: Initial trajectories for every seed.
            target_q_per_seed: Goal configurations for every seed.
            scene_indices: Scene index for each query.
            q_history: Optional last three executed configurations.
            history_count: Number of valid configurations in `q_history`.

        Returns:
            The selected robot state and final ranking costs.

        """
        batch_size = start_q.shape[0]
        if not self._initialized:
            self.warmup(batch_size)
        elif batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: expected {self.batch_size}, got {batch_size}")
        self._set_inputs(
            start_q,
            T_world_target,
            target_q,
            traj_q_init,
            target_q_per_seed,
            scene_indices,
            q_history,
            history_count,
        )
        best_var, costs = cast(MultiSeedSolver, self._solver).solve(self._var)
        state = cast(RobotState, best_var.get("robot"))
        self.robot.forward_kinematics(state)
        return state, costs
