# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
import dataclasses
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.helpers.motion_plan.gradient_trajectory_optimizer import _build_trajectory_tasks
from robokit.helpers.motion_plan.trajectory_seed_score_task import TrajectorySeedScoreTask
from robokit.opt.multi_seed_solver import StageConfig
from robokit.opt.particle_solver import (
    ParticleSolver,
    ParticleSolverConfig,
    _mppi_add_inplace_kernel,
    _mppi_broadcast_add_kernel,
    _MppiStage,
)
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryPositionTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils import warp_utils
from robokit.utils.warp_utils import wp_vec7


if TYPE_CHECKING:
    from robokit.geom import WarpScene
    from robokit.helpers.motion_plan.motion_planner import OnlineState


# --- kernels ---------------------------------------------------------------
@wp.kernel
def _set_link_position_targets_kernel(
    T_world_target: wp.array1d(dtype=wp_vec7),
    out: wp.array3d(dtype=wp.float32),
):
    particle, frame = wp.tid()  # type: ignore
    target = T_world_target[0]
    out[particle, frame, 0] = target[0]
    out[particle, frame, 1] = target[1]
    out[particle, frame, 2] = target[2]


@wp.kernel
def _compute_exploration_scale_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    target: wp.array1d(dtype=wp_vec7),
    ee_link_index: int,
    settle_radius: float,
    explore_scale: wp.array1d(dtype=wp.float32),
):
    batch = wp.tid()
    ee = T_world_link[batch, ee_link_index]
    goal = target[batch]
    distance = wp.sqrt(
        (ee[0] - goal[0]) * (ee[0] - goal[0])
        + (ee[1] - goal[1]) * (ee[1] - goal[1])
        + (ee[2] - goal[2]) * (ee[2] - goal[2])
    )
    explore_scale[batch] = wp.min(1.0, distance / settle_radius) if settle_radius > 0.0 else 1.0


@wp.kernel
def _set_acceleration_start_state_kernel(
    q: wp.array3d(dtype=wp.float32),
    current_q: wp.array2d(dtype=wp.float32),
    warm_start: wp.array1d(dtype=wp.int32),
    dt: float,
    q0: wp.array2d(dtype=wp.float32),
    prev_q: wp.array2d(dtype=wp.float32),
    v0: wp.array2d(dtype=wp.float32),
):
    batch_idx, joint_idx = wp.tid()
    if warm_start[0] != 0:
        current = current_q[batch_idx, joint_idx]
        v0[batch_idx, joint_idx] = (current - prev_q[batch_idx, joint_idx]) / dt
        q0[batch_idx, joint_idx] = current
        prev_q[batch_idx, joint_idx] = current
    else:
        current = q[batch_idx, 0, joint_idx]
        q0[batch_idx, joint_idx] = current
        prev_q[batch_idx, joint_idx] = current
        v0[batch_idx, joint_idx] = 0.0


@wp.kernel
def _set_shifted_pinned_trajectory_kernel(
    src: wp.array3d(dtype=wp.float32),
    current: wp.array2d(dtype=wp.float32),
    num_frames: int,
    dst: wp.array3d(dtype=wp.float32),
):
    batch_idx, frame_idx, joint_idx = wp.tid()
    if frame_idx == 0:
        dst[batch_idx, 0, joint_idx] = current[batch_idx, joint_idx]
    elif frame_idx < num_frames - 1:
        dst[batch_idx, frame_idx, joint_idx] = src[batch_idx, frame_idx + 1, joint_idx]
    else:
        dst[batch_idx, frame_idx, joint_idx] = src[batch_idx, num_frames - 1, joint_idx]


@wp.kernel
def _compute_acceleration_rollout_kernel(
    q0: wp.array2d(dtype=wp.float32),
    v0: wp.array2d(dtype=wp.float32),
    action: wp.array2d(dtype=wp.float32),
    dt_schedule: wp.array1d(dtype=wp.float32),
    accel_limit: float,
    velocity_limit: wp.array1d(dtype=wp.float32),
    q_lower: wp.array1d(dtype=wp.float32),
    q_upper: wp.array1d(dtype=wp.float32),
    particles_per_instance: int,
    num_frames: int,
    num_dofs: int,
    q: wp.array3d(dtype=wp.float32),
):
    row_idx, joint_idx = wp.tid()
    batch_idx = row_idx / particles_per_instance
    position = q0[batch_idx, joint_idx]
    velocity = v0[batch_idx, joint_idx]
    q[row_idx, 0, joint_idx] = position
    max_velocity = velocity_limit[joint_idx]
    for frame_idx in range(1, num_frames):
        dt = dt_schedule[frame_idx]
        accel = wp.clamp(action[row_idx, frame_idx * num_dofs + joint_idx], -accel_limit, accel_limit)
        velocity = wp.clamp(velocity + accel * dt, -max_velocity, max_velocity)
        position = wp.clamp(position + velocity * dt, q_lower[joint_idx], q_upper[joint_idx])
        q[row_idx, frame_idx, joint_idx] = position


@wp.kernel
def _compute_smoothed_noise_kernel(
    noise: wp.array2d(dtype=wp.float32),
    b0: float,
    b1: float,
    b2: float,
    num_frames: int,
    num_dofs: int,
):
    row_idx, joint_idx = wp.tid()
    for frame_idx in range(2, num_frames):
        noise[row_idx, frame_idx * num_dofs + joint_idx] = (
            b0 * noise[row_idx, frame_idx * num_dofs + joint_idx]
            + b1 * noise[row_idx, (frame_idx - 1) * num_dofs + joint_idx]
            + b2 * noise[row_idx, (frame_idx - 2) * num_dofs + joint_idx]
        )


@wp.kernel
def _compute_knot_smoothed_noise_kernel(
    knot_frames: wp.array1d(dtype=wp.int32),
    num_knots: int,
    num_particles: int,
    knot_count: int,
    num_dofs: int,
    noise: wp.array2d(dtype=wp.float32),
):
    row_idx, frame_idx, joint_idx = wp.tid()
    if row_idx % num_particles >= knot_count or num_knots < 2:
        return
    interval = int(0)
    for knot_idx in range(num_knots - 1):
        if frame_idx > knot_frames[knot_idx]:
            interval = knot_idx
    first = knot_frames[interval]
    second = knot_frames[interval + 1]
    if frame_idx == first or frame_idx == second:  # noqa: PLR1714
        return
    alpha = wp.float32(frame_idx - first) / wp.float32(second - first)
    noise[row_idx, frame_idx * num_dofs + joint_idx] = (1.0 - alpha) * noise[
        row_idx, first * num_dofs + joint_idx
    ] + alpha * noise[row_idx, second * num_dofs + joint_idx]


@wp.kernel
def _set_shifted_actions_kernel(
    src: wp.array2d(dtype=wp.float32),
    warm_start: wp.array1d(dtype=wp.int32),
    num_frames: int,
    num_dofs: int,
    dst: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()
    frame_idx = dof_idx / num_dofs
    joint_idx = dof_idx % num_dofs
    if warm_start[0] == 0:
        dst[batch_idx, dof_idx] = 0.0
    elif frame_idx < num_frames - 1:
        dst[batch_idx, dof_idx] = src[batch_idx, (frame_idx + 1) * num_dofs + joint_idx]
    else:
        dst[batch_idx, dof_idx] = src[batch_idx, (num_frames - 1) * num_dofs + joint_idx]


# --- acceleration solver ---------------------------------------------------
class AccelerationParticleSolver(ParticleSolver):
    """Particle solver that samples acceleration actions and rolls out trajectories."""

    def __init__(
        self,
        terms: list,
        config: ParticleSolverConfig,
        trajectory_config: "MppiTrajectoryOptimizerConfig",
        state: RobotState,
        current_q: wp.array,
        num_frames: int,
        dt: float,
        device: str,
        score_task: Optional[TrajectorySeedScoreTask] = None,
    ):
        score_terms = [score_task] if score_task is not None else None
        super().__init__(terms=terms, config=config, device=device, score_terms=score_terms)
        self.trajectory_config = trajectory_config
        self.num_frames = num_frames
        self.num_dofs = state.robot.spec.num_actuated_joints
        self.dt = dt
        self.current_q = current_q
        self.warm_start = wp.zeros(1, dtype=wp.int32, device=self.device)
        dt_schedule = np.full(num_frames, dt, dtype=np.float32)
        if trajectory_config.dt_max > dt:
            base_frames = int(trajectory_config.dt_base_ratio * num_frames)
            dt_schedule[base_frames:] = np.linspace(
                dt, trajectory_config.dt_max, num_frames - base_frames, dtype=np.float32
            )
        self.dt_schedule = wp.array(dt_schedule, dtype=wp.float32, device=self.device)
        batch_size = state.q.shape[0]
        tangent_dim = num_frames * self.num_dofs
        self.q0 = wp.zeros((batch_size, self.num_dofs), dtype=wp.float32, device=self.device)
        self.prev_q = wp.zeros_like(self.q0)
        self.v0 = wp.zeros_like(self.q0)
        self.mean_action = wp.zeros((batch_size, tangent_dim), dtype=wp.float32, device=self.device)
        self.particle_action = wp.zeros(
            (batch_size * config.num_particles, tangent_dim), dtype=wp.float32, device=self.device
        )
        self.action_scratch = wp.zeros_like(self.mean_action)
        spec = state.robot.spec
        velocity_limit = spec.actuated_joint_velocity_limits.astype(np.float32)
        self.velocity_limit = wp.array(velocity_limit, dtype=wp.float32, device=self.device)
        limits = spec.actuated_joint_limits.astype(np.float32)
        self.q_lower = wp.array(limits[:, 0] - trajectory_config.q_limit_margin, dtype=wp.float32, device=self.device)
        self.q_upper = wp.array(limits[:, 1] + trajectory_config.q_limit_margin, dtype=wp.float32, device=self.device)
        self.knot_count = round(trajectory_config.knot_ratio * config.num_particles)
        if self.knot_count > 0:
            assert trajectory_config.num_knots >= 2, "knot sampling needs at least two knots"
        knots = np.round(np.linspace(0, num_frames - 1, trajectory_config.num_knots)).astype(np.int32)
        self.knot_frames = wp.array(knots, dtype=wp.int32, device=self.device)

    def _solve_stage(self, stage_idx: int, warmup: bool = False) -> wp.array:
        stage = cast(_MppiStage, self._updaters[stage_idx])
        mean_var = self._stage_vars[stage_idx]
        mean_state = cast(RobotState, mean_var.get("robot"))
        particle_state = cast(RobotState, stage.particle_var.get("robot"))
        terms = self.terms[stage_idx]
        config = cast(ParticleSolverConfig, self.config)
        trajectory_config = self.trajectory_config
        wp.copy(self.action_scratch, self.mean_action)
        wp.launch(
            _set_acceleration_start_state_kernel,
            dim=(stage.instances, self.num_dofs),
            inputs=[mean_state.q, self.current_q, self.warm_start, self.dt],
            outputs=[self.q0, self.prev_q, self.v0],
            device=self.device,
        )
        wp.launch(
            _set_shifted_actions_kernel,
            dim=(stage.instances, stage.dim),
            inputs=[self.action_scratch, self.warm_start, self.num_frames, self.num_dofs],
            outputs=[self.mean_action],
            device=self.device,
        )
        self._reset_std(stage, config.std_scale if config.std_scale is not None else 1.0)
        b0, b1, b2 = trajectory_config.noise_filter
        for iteration in range(list(config.stages)[stage_idx].iters):
            self._sample_noise(stage, iteration)
            if self.knot_count > 0:
                wp.launch(
                    _compute_knot_smoothed_noise_kernel,
                    dim=(stage.instances * stage.num_particles, self.num_frames, self.num_dofs),
                    inputs=[
                        self.knot_frames,
                        trajectory_config.num_knots,
                        stage.num_particles,
                        self.knot_count,
                        self.num_dofs,
                        stage.noise,
                    ],
                    device=self.device,
                )
            wp.launch(
                _compute_smoothed_noise_kernel,
                dim=(stage.instances * stage.num_particles, self.num_dofs),
                inputs=[stage.noise, b0, b1, b2, self.num_frames, self.num_dofs],
                device=self.device,
            )
            wp.launch(
                _mppi_broadcast_add_kernel,
                dim=(stage.instances * stage.num_particles, stage.dim),
                inputs=[self.mean_action, stage.noise, stage.num_particles],
                outputs=[self.particle_action],
                device=self.device,
            )
            wp.launch(
                _compute_acceleration_rollout_kernel,
                dim=(self.particle_action.shape[0], self.num_dofs),
                inputs=[
                    self.q0,
                    self.v0,
                    self.particle_action,
                    self.dt_schedule,
                    trajectory_config.accel_limit,
                    self.velocity_limit,
                    self.q_lower,
                    self.q_upper,
                    stage.num_particles,
                    self.num_frames,
                    self.num_dofs,
                ],
                outputs=[particle_state.q],
                device=self.device,
            )
            particle_state.invalidate()
            self._accumulate_cost(stage.particle_var, terms, stage.particle_residual, stage.particle_costs)
            self._update_distribution(stage)
            wp.launch(
                _mppi_add_inplace_kernel,
                dim=(stage.instances, stage.dim),
                inputs=[stage.delta, self.mean_action],
                device=self.device,
            )
        wp.launch(
            _compute_acceleration_rollout_kernel,
            dim=(self.mean_action.shape[0], self.num_dofs),
            inputs=[
                self.q0,
                self.v0,
                self.mean_action,
                self.dt_schedule,
                trajectory_config.accel_limit,
                self.velocity_limit,
                self.q_lower,
                self.q_upper,
                1,
                self.num_frames,
                self.num_dofs,
            ],
            outputs=[mean_state.q],
            device=self.device,
        )
        mean_state.invalidate()
        self._accumulate_cost(mean_var, terms, stage.mean_residual, stage.costs)
        return stage.costs


# --- configuration ---------------------------------------------------------
@dataclasses.dataclass
class MppiTrajectoryOptimizerConfig:
    """Configuration for position- or acceleration-controlled trajectory MPPI."""

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
    max_iter: int = 50
    num_seeds: int = 1
    use_cuda_graph: bool = True
    accel_smoothness_weight: float = 0.0
    accel2_smoothness_weight: float = 0.0
    smooth_boundary: bool = False
    lock_endpoints: bool = False
    free_goal_frame: bool = True
    collision_sweep_steps: int = 0
    collision_penalty: Literal["smooth", "plain", "surface_distance"] = "plain"
    num_particles: int = 64
    beta: float = 0.1
    init_std: float = 0.3
    control_space: Literal["position", "acceleration"] = "position"
    knot_ratio: float = 0.0
    min_std: float = 0.01
    accel_limit: float = 10.0
    dt_max: float = 0.0
    q_limit_margin: float = 0.0
    lock_dim: int = 0
    base_seed: int = 0
    gamma: float = 0.9
    dt_base_ratio: float = 0.5
    noise_filter: Tuple[float, float, float] = (0.3, 0.3, 0.4)
    num_knots: int = 5
    settle_radius: float = 0.0
    goal_rest_weight: float = 0.0
    link_position_weight: float = 0.0


# --- optimizer -------------------------------------------------------------
class MppiTrajectoryOptimizer:
    """Trajectory MPPI with optional caller-owned state for receding-horizon updates.

    Lifecycle:
        1. `__init__` stores configuration and creates the active-DOF mask.
        2. `warmup(batch_size)` builds batch-shaped tasks, buffers, execution graphs, and selects:
           a. `ParticleSolver` updates trajectory positions directly.
           b. `AccelerationParticleSolver` updates acceleration actions and rolls out trajectories.
        3. `solve(...)` uses the same path for offline, cold-online, and warm-online optimization:
           a. `_set_inputs(...)` updates tasks and restores or initializes the optimization state.
           b. `_solver.solve(...)` runs all stages through the selected solver's `_solve_stage(...)`.
           c. `wp.copy(...)` saves the trajectory, previous configuration, and mean action to
              `online_state` when provided.
    """

    def __init__(
        self,
        config: MppiTrajectoryOptimizerConfig,
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
        if config.lock_endpoints and num_frames >= 2:
            lock_mask[: self.num_dofs] = 0.0
            lock_mask[-self.num_dofs :] = 0.0
        self._lock_endpoints_mask_np = lock_mask
        self._active_dof_mask_wp = wp.from_numpy(lock_mask, dtype=wp.float32, device=device)

    def set_active_joint_mask(self, joint_mask: Sequence[float]):
        """Lock or unlock actuated joints without rebuilding the optimizer."""
        joint_mask_np = np.asarray(joint_mask, dtype=np.float32)
        self._active_dof_mask_wp.assign(np.tile(joint_mask_np, self.num_frames) * self._lock_endpoints_mask_np)

    def _set_inputs(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        traj_q_init: Optional[wp.array],
        target_q_per_seed: wp.array,
        scene_indices: wp.array,
        online_state: Optional["OnlineState"],
        explore_scale: Optional[Union[float, wp.array]],
    ) -> bool:
        """Set task inputs and return whether this solve starts from `traj_q_init`."""
        target = self._goal_vec7_buf
        goal_q = target_q_per_seed
        sq = warp_utils.repeat(start_q, self._num_seeds, out=self._sq_buf)
        if T_world_target is not None:
            warp_utils.repeat(T_world_target, self._num_seeds, out=target)
        if target_q is not None:
            goal_q = warp_utils.repeat(target_q, self._num_seeds, out=self._target_q_per_seed_buf)
        si = warp_utils.repeat(scene_indices, self._num_seeds, out=self._scene_indices_buf)
        for task in self._collision_tasks:
            task.set_scene_indices(si)
        if self._start_task is not None:
            self._start_task.dense_task.set_rest_state(sq)
        if self._goal_task is not None:
            self._goal_task.dense_task.set_rest_state(goal_q)
        if self._frame_task is not None:
            self._frame_task.dense_task.set_target(target)
        goal_q = target_q if target_q is not None else target_q_per_seed
        if self._rest_goal_task is not None:
            wp.copy(cast(wp.array, self._rest_goal_task.dense_task.rest_q), goal_q.reshape((1, self.num_dofs)))
        if self._link_task is not None:
            # the task stores targets link-major; this one has a single link
            targets = cast(wp.array, self._link_task.target_positions)
            targets = targets.reshape((targets.shape[0], targets.shape[1], 3))
            wp.launch(
                _set_link_position_targets_kernel,
                dim=(targets.shape[0], self.num_frames),
                inputs=[target],
                outputs=[targets],
                device=self.device,
            )

        acceleration_solver = self._solver if isinstance(self._solver, AccelerationParticleSolver) else None
        cold_start = traj_q_init is not None or online_state is None or online_state.prev_q_traj is None
        if cold_start:
            if traj_q_init is None:
                warp_utils.repeat(
                    start_q,
                    self.num_frames,
                    out=self._state.q.reshape((start_q.shape[0] * self.num_frames, self.num_dofs)),
                )
            else:
                wp.copy(self._state.q, traj_q_init)
            self._var.invalidate()
            self._settle_scale.fill_(1.0)
            if acceleration_solver is not None:
                acceleration_solver.warm_start.zero_()
            return True
        assert online_state is not None
        if self._num_seeds != 1:
            raise ValueError("Online MPPI requires num_seeds=1.")
        if acceleration_solver is not None:
            if online_state.prev_start_q is None or online_state.mean_action is None:
                raise ValueError("Acceleration MPPI state requires prev_start_q and mean_action.")
            wp.copy(acceleration_solver.prev_q, online_state.prev_start_q)
            wp.copy(acceleration_solver.mean_action, online_state.mean_action)
            acceleration_solver.warm_start.fill_(1)
        else:
            assert self._config.lock_dim >= self.num_dofs, "position update must lock frame zero"
            wp.copy(self._shift_scratch, online_state.prev_q_traj)
            wp.launch(
                _set_shifted_pinned_trajectory_kernel,
                dim=self._state.q.shape,
                inputs=[self._shift_scratch, sq, self.num_frames],
                outputs=[self._state.q],
                device=self.device,
            )
            self._var.invalidate()
        if isinstance(explore_scale, wp.array):
            wp.copy(self._settle_scale, explore_scale)
        elif explore_scale is not None:
            self._settle_scale.fill_(explore_scale)
        elif T_world_target is None:
            self._settle_scale.fill_(1.0)
        else:
            start_state = self.robot.forward_kinematics(self.robot.state(q=sq))
            wp.launch(
                _compute_exploration_scale_kernel,
                dim=start_q.shape[0],
                inputs=[start_state.T_world_link, target, self.ee_link_index, self._config.settle_radius],
                outputs=[self._settle_scale],
                device=self.device,
            )
        return False

    def warmup(self, batch_size: int):
        """Build the batch-shaped tasks, buffers, and MPPI solver."""
        if self._initialized:
            return
        config = self._config
        actual_batch = batch_size * config.num_seeds
        self.batch_size = batch_size
        scene_indices = wp.zeros(actual_batch, dtype=wp.int32, device=self.device)
        (
            self._tasks,
            self._start_task,
            self._collision_tasks,
            self._goal_task,
            self._frame_task,
        ) = _build_trajectory_tasks(
            config,
            self.robot,
            self.ee_link_index,
            self.num_frames,
            actual_batch,
            self._dt,
            self._scene,
            scene_indices,
            self.device,
        )
        self._rest_goal_task: Optional[TrajectoryTask] = None
        self._link_task: Optional[TrajectoryPositionTask] = None
        if config.goal_rest_weight > 0:
            self._rest_goal_task = TrajectoryTask(
                RestTask(
                    robot=self.robot,
                    rest_q=np.zeros(self.num_dofs, dtype=np.float32),
                    weight=config.goal_rest_weight,
                ),
                num_frames=self.num_frames,
            )
            self._tasks.append(self._rest_goal_task)
        if config.link_position_weight > 0:
            self._link_task = TrajectoryPositionTask(
                robot=self.robot,
                frame_index=self.ee_link_index,
                target_positions=np.zeros((actual_batch * config.num_particles, self.num_frames, 3), dtype=np.float32),
                weight=config.link_position_weight,
            )
            self._tasks.append(self._link_task)
        q = wp.zeros((actual_batch, self.num_frames, self.num_dofs), dtype=wp.float32, device=self.device)
        self._state = self.robot.state(q=q)
        self._var = VarValues(robot=self._state)
        self._goal_vec7_buf = wp.zeros(actual_batch, dtype=wp_vec7, device=self.device)
        self._sq_buf = wp.zeros((actual_batch, self.num_dofs), dtype=wp.float32, device=self.device)
        self._target_q_per_seed_buf = wp.zeros((actual_batch, self.num_dofs), dtype=wp.float32, device=self.device)
        self._scene_indices_buf = wp.zeros(actual_batch, dtype=wp.int32, device=self.device)
        self._settle_scale = wp.ones(actual_batch, dtype=wp.float32, device=self.device)
        score_task = (
            TrajectorySeedScoreTask(
                self.robot,
                self.ee_link_index,
                self.num_frames - 1,
                self._goal_vec7_buf,
                lambda: self._solver.candidate_scores,
                self._collision_tasks,
                self._num_seeds,
                False,
            )
            if self._num_seeds > 1
            else None
        )
        particle_config = ParticleSolverConfig(
            stages=[StageConfig(config.num_seeds, config.max_iter, 0.0)],
            cuda_graph_mode="full" if config.use_cuda_graph else "none",
            num_particles=config.num_particles,
            beta=config.beta,
            init_std=config.init_std,
            min_std=config.min_std,
            lock_dim=config.lock_dim,
            base_seed=config.base_seed,
            gamma=config.gamma,
            active_dof_mask=self._active_dof_mask_wp,
            std_scale=self._settle_scale,
        )
        if self._rest_goal_task is not None:
            self._rest_goal_task.init_buffers(self.device)
        if self._link_task is not None:
            self._link_task.init_buffers(self.device)
        if config.control_space == "acceleration":
            self._solver = AccelerationParticleSolver(
                terms=[self._tasks],
                config=particle_config,
                trajectory_config=config,
                state=self._state,
                current_q=self._sq_buf,
                num_frames=self.num_frames,
                dt=self._dt,
                device=self.device,
                score_task=score_task,
            )
        else:
            self._shift_scratch = wp.zeros_like(self._state.q)
            score_terms = [score_task] if score_task is not None else None
            self._solver = ParticleSolver([self._tasks], particle_config, self.device, score_terms)
        self._solver.setup(self._var)
        self._initialized = True
        if self._num_seeds == 1:
            zero_q = wp.zeros((batch_size, self.num_dofs), dtype=wp.float32, device=self.device)
            zero_traj = wp.zeros_like(self._state.q)
            zero_scene = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
            self.solve(zero_q, None, None, zero_traj, zero_q, zero_scene)
        wp.synchronize_device(self.device)

    def solve(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        traj_q_init: Optional[wp.array],
        target_q_per_seed: wp.array,
        scene_indices: wp.array,
        online_state: Optional["OnlineState"] = None,
        explore_scale: Optional[Union[float, wp.array]] = None,
    ) -> Tuple[RobotState, wp.array]:
        """Solve from an initial trajectory or update caller-owned MPPI state.

        Args:
            start_q: Batched current configurations.
            T_world_target: Optional batched pose goals.
            target_q: Optional batched joint-space goals.
            traj_q_init: Initial trajectories for a cold solve.
            target_q_per_seed: Goal configurations for every seed.
            scene_indices: Scene index for each query.
            online_state: Optional caller-owned state for online updates.
            explore_scale: Optional scalar or per-query exploration scale.

        Returns:
            The selected robot state and final ranking costs.

        """
        batch_size = start_q.shape[0]
        if not self._initialized:
            self.warmup(batch_size)
        elif batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: expected {self.batch_size}, got {batch_size}")
        cold_start = self._set_inputs(
            start_q,
            T_world_target,
            target_q,
            traj_q_init,
            target_q_per_seed,
            scene_indices,
            online_state,
            explore_scale,
        )
        best_var, costs = self._solver.solve(self._var)
        robot_state = cast(RobotState, best_var.get("robot"))
        if cold_start:
            self.robot.forward_kinematics(robot_state)
        if online_state is not None:
            if online_state.prev_q_traj is None:
                online_state.prev_q_traj = wp.empty_like(robot_state.q)
            wp.copy(online_state.prev_q_traj, robot_state.q)
            if isinstance(self._solver, AccelerationParticleSolver):
                if cold_start:
                    online_state.prev_start_q = wp.zeros_like(self._solver.prev_q)
                    online_state.mean_action = wp.zeros_like(self._solver.mean_action)
                wp.copy(cast(wp.array, online_state.prev_start_q), self._solver.prev_q)
                wp.copy(cast(wp.array, online_state.mean_action), self._solver.mean_action)
        return robot_state, costs
