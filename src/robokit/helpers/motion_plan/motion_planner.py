# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import numpy as np
import warp as wp

from robokit.geom import WarpScene
from robokit.helpers.ik import IK, IKConfig
from robokit.helpers.motion_plan.gradient_trajectory_optimizer import (
    GradientTrajectoryOptimizer,
    GradientTrajectoryOptimizerConfig,
    build_default_gradient_trajectory_optimizer_config,
)
from robokit.helpers.motion_plan.mppi_trajectory_optimizer import (
    MppiTrajectoryOptimizer,
    MppiTrajectoryOptimizerConfig,
)
from robokit.helpers.motion_plan.trajectory_postprocessor import (
    EndpointSnap,
    LaplacianShortcut,
    TrajectoryPostprocessor,
)
from robokit.helpers.motion_plan.trajectory_retimer import TrajectoryRetimer, TrajectoryRetimerConfig
from robokit.lie.se3 import se3_from_matrix
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.utils import warp_utils
from robokit.utils.warp_utils import wp_vec7


if TYPE_CHECKING:
    import torch


# --- kernels ---------------------------------------------------------------
@wp.kernel
def _build_linear_trajectory_kernel(
    start_q: wp.array2d(dtype=wp.float32),
    target_q_per_seed: wp.array2d(dtype=wp.float32),
    num_seeds: int,
    num_frames: int,
    num_dofs: int,
    out: wp.array3d(dtype=wp.float32),
):
    tid = wp.tid()
    b = tid // (num_frames * num_dofs)
    f = (tid // num_dofs) % num_frames
    d = tid % num_dofs
    alpha = wp.float32(f) / wp.float32(num_frames - 1)
    start = start_q[b // num_seeds, d]
    out[b, f, d] = start * (1.0 - alpha) + target_q_per_seed[b, d] * alpha


@wp.kernel
def _append_q_history_kernel(
    q_history: wp.array3d(dtype=wp.float32),
    start_q: wp.array2d(dtype=wp.float32),
):
    batch, dof = wp.tid()
    q_history[batch, 0, dof] = q_history[batch, 1, dof]
    q_history[batch, 1, dof] = q_history[batch, 2, dof]
    q_history[batch, 2, dof] = start_q[batch, dof]


@wp.kernel
def _build_seed_trajectory_kernel(
    start_q: wp.array2d(dtype=wp.float32),
    target_q_per_seed: wp.array2d(dtype=wp.float32),
    rng_seed: int,
    active_joint_mask: wp.array1d(dtype=wp.float32),
    joint_limits: wp.array2d(dtype=wp.float32),
    sample_range: float,
    num_seeds: int,
    num_frames: int,
    num_dofs: int,
    num_direct_seeds: int,
    out: wp.array3d(dtype=wp.float32),
):
    tid = wp.tid()
    seed_row = tid // (num_frames * num_dofs)
    frame = (tid // num_dofs) % num_frames
    dof = tid % num_dofs
    batch = seed_row // num_seeds
    seed = seed_row % num_seeds
    start = start_q[batch, dof]
    if active_joint_mask[dof] == 0.0:
        out[seed_row, frame, dof] = start
        return
    target = target_q_per_seed[seed_row, dof]
    alpha = wp.float32(frame) / wp.float32(num_frames - 1)
    direct = start + alpha * (target - start)
    if seed < num_direct_seeds:
        out[seed_row, frame, dof] = direct
        return
    random_state = wp.rand_init(wp.int32(rng_seed), wp.int32(seed_row * num_dofs + dof))
    joint_range = joint_limits[dof, 1] - joint_limits[dof, 0]
    offset = (wp.randf(random_state) - 0.5) * sample_range * joint_range
    blend = 16.0 * alpha * alpha * (1.0 - alpha) * (1.0 - alpha)
    out[seed_row, frame, dof] = wp.clamp(direct + blend * offset, joint_limits[dof, 0], joint_limits[dof, 1])


@wp.kernel
def _pin_inactive_joints_kernel(
    start_q: wp.array2d(dtype=wp.float32),
    active_joint_mask: wp.array1d(dtype=wp.float32),
    num_seeds: int,
    target_q: wp.array2d(dtype=wp.float32),
):
    row, dof = wp.tid()
    if active_joint_mask[dof] == 0.0:
        target_q[row, dof] = start_q[row // num_seeds, dof]


@wp.kernel
def _expand_ik_goal_seeds_kernel(
    ik_goal_q: wp.array2d(dtype=wp.float32),
    goal_offset: int,
    num_goal_seeds: int,
    total_goal_seeds: int,
    num_trajectory_seeds: int,
    target_q_per_seed: wp.array2d(dtype=wp.float32),
):
    row, dof = wp.tid()
    batch = row // num_trajectory_seeds
    trajectory_seed = row % num_trajectory_seeds
    repeats = num_trajectory_seeds // num_goal_seeds
    goal_seed = trajectory_seed // repeats
    target_q_per_seed[row, dof] = ik_goal_q[batch * total_goal_seeds + goal_offset + goal_seed, dof]


@wp.kernel
def _set_best_retry_result_kernel(
    scores: wp.array1d(dtype=wp.float32),
    penetrations: wp.array1d(dtype=wp.float32),
    q: wp.array3d(dtype=wp.float32),
    always_retry: int,
    best_score: wp.array1d(dtype=wp.float32),
    best_pen: wp.array1d(dtype=wp.float32),
    condition: wp.array1d(dtype=wp.int32),
    final_q: wp.array3d(dtype=wp.float32),
):
    batch = wp.tid()
    if scores[batch] < best_score[batch]:
        best_score[batch] = scores[batch]
        best_pen[batch] = penetrations[batch]
        for frame in range(q.shape[1]):
            for dof in range(q.shape[2]):
                final_q[batch, frame, dof] = q[batch, frame, dof]
    pen = best_pen[batch]
    if always_retry != 0 or best_score[batch] >= 1.0e6 or (pen > 1.0e-5 and pen < 0.005):
        wp.atomic_max(condition, 0, 1)


# --- configuration ---------------------------------------------------------
def build_default_motion_plan_ik_config() -> IKConfig:
    """Build the default goal-IK configuration for motion planning."""
    return (
        IKConfig(
            solver=MultiSeedSolverConfig(
                stages=[
                    StageConfig(num_seeds=64, iters=6, lm_lambda=10.0),
                    StageConfig(num_seeds=4, iters=10, lm_lambda=1.0),
                    StageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
                ],
                cuda_graph_mode="full",
            )
        )
        .add(PositionTask(weight=20.0))
        .add(RotationTask(weight=10.0))
        .add(PositionLimit(weight=50.0))
    )


@dataclass
class MotionPlannerConfig:
    """Configuration for offline or online motion planning."""

    num_timesteps: int = 32
    dt: float = 0.25
    use_cuda_graph: bool = True
    q_goal: bool = False  # plan to joint-space goals (target_q) instead of pose goals (T_world_target)
    ik_config: IKConfig = field(default_factory=build_default_motion_plan_ik_config)
    expand_ik_goal_seeds: bool = False  # Repeat fewer IK goals across trajectory seeds.
    ik_collision_weight: float = 20.0
    ik_collision_margin: Optional[float] = None
    trajectory_optimizer: Union[GradientTrajectoryOptimizerConfig, MppiTrajectoryOptimizerConfig] = field(
        default_factory=build_default_gradient_trajectory_optimizer_config
    )
    finetune: Optional[TrajectoryRetimerConfig] = None
    always_finetune_retry: bool = False
    enable_retry: bool = True
    max_retries: int = 1
    postprocessors: Tuple[Type[TrajectoryPostprocessor], ...] = (LaplacianShortcut, EndpointSnap)


# --- results ---------------------------------------------------------------
ArrayT = TypeVar("ArrayT")


@dataclass
class OnlineState:
    """Previous IK goal plus executed history or MPPI state retained between updates."""

    prev_q_traj: Optional[wp.array] = None
    prev_goal_q: Optional[wp.array] = None
    prev_start_q: Optional[wp.array] = None
    mean_action: Optional[wp.array] = None
    q_history: Optional[wp.array] = None
    history_count: int = 0


@dataclass
class MotionPlanResult(Generic[ArrayT]):
    """Planner output with offline `motion_time` or online `online_state`."""

    q_traj: ArrayT
    motion_time: Optional[ArrayT] = None
    online_state: Optional[OnlineState] = None


# --- planner ---------------------------------------------------------------
class MotionPlanner:
    """Batched motion planning with goal IK, trajectory optimization, retry, postprocessing.

    Lifecycle:
        1. `__init__` copies configuration and constructs goal IK, trajectory optimization,
           postprocessing, and retiming components.
        2. `warmup(batch_size)` fixes the batch size and allocates all batch-shaped state.
        3. `solve_offline(...)` builds seeds, solves and retries trajectories, runs ordered
           postprocessors, then computes `motion_time`.
        4. `solve_online(...)` runs a fresh gradient solve with executed-history smoothing, or
           updates the retained MPPI action distribution.

    Optimizer dispatch:
        `solve_offline()`
        ├── `GradientTrajectoryOptimizer.solve()`
        └── `MppiTrajectoryOptimizer.solve()`

        `solve_online()`
        ├── `GradientTrajectoryOptimizer.solve()`
        └── `MppiTrajectoryOptimizer.solve(online_state=...)`

    Args:
        config: Planner configuration.
        robot: Robot model.
        ee_link_name_or_index: End-effector link name or index.
        scene: Optional collision scene.
        device: Warp device.
    """

    # --- setup ---
    def __init__(
        self,
        config: MotionPlannerConfig,
        robot: Robot,
        ee_link_name_or_index: Union[str, int],
        scene: Optional[WarpScene] = None,
        device: str = "cuda:0",
    ):
        self.robot = robot
        self.config = copy.deepcopy(config)
        self.device = wp.get_device(device)
        if isinstance(ee_link_name_or_index, str):
            self.ee_link_index = robot.link_names.index(ee_link_name_or_index)
            self.ee_link_name = ee_link_name_or_index
        else:
            self.ee_link_index = int(ee_link_name_or_index)
            self.ee_link_name = robot.link_names[self.ee_link_index]

        self._num_frames = self.config.num_timesteps + 1
        self._batch_size: Optional[int] = None
        # geometries must not change after construction; update them in place with `geom.update(...)`
        self._scene = scene if scene is not None else WarpScene(1, self.device)
        self._active_joint_mask_np = np.ones(robot.num_actuated_joints, dtype=np.float32)
        self._active_joint_mask_wp = wp.ones(robot.num_actuated_joints, dtype=wp.float32, device=self.device)
        self._retimer = TrajectoryRetimer(robot, self._num_frames, self.config.dt, self.config.finetune, device)
        self._joint_limits_wp = wp.from_numpy(
            robot.spec.actuated_joint_limits.astype(np.float32), dtype=wp.float32, device=self.device
        )
        self._postprocessors = tuple(
            processor(
                robot,
                self.ee_link_index,
                self._num_frames,
                self._scene,
                self._retimer,
                self.config.use_cuda_graph,
                device,
            )
            for processor in self.config.postprocessors
        )

        if self.config.max_retries < 1:
            raise ValueError("max_retries must be >= 1; use enable_retry=False to disable retries.")

        # Retry requires multi-seed pose L-BFGS because it consumes penetration scores and beam indices.
        cfg = self.config.trajectory_optimizer
        cfg.use_cuda_graph = self.config.use_cuda_graph
        if isinstance(cfg, GradientTrajectoryOptimizerConfig) and cfg.init_sample_range < 0.0:
            raise ValueError("init_sample_range must be >= 0.")
        self._retry_enabled = (
            self.config.enable_retry
            and not self.config.q_goal
            and isinstance(cfg, GradientTrajectoryOptimizerConfig)
            and cfg.num_seeds > 1
            and cfg.solver == "lbfgs"
        )

        # The IK final stage provides goal seeds; retries reserve additional groups of those goals.
        stages = self.config.ik_config.solver.stages
        num_trajectory_seeds = cfg.num_seeds
        self._num_ik_goal_seeds = stages[-1].num_seeds
        if not self.config.q_goal and self._num_ik_goal_seeds != num_trajectory_seeds:
            if not self.config.expand_ik_goal_seeds:
                raise ValueError(
                    f"IK returns {self._num_ik_goal_seeds} goal seeds but trajectory optimization uses "
                    f"{num_trajectory_seeds}; set expand_ik_goal_seeds=True to expand them explicitly."
                )
            if num_trajectory_seeds % self._num_ik_goal_seeds != 0:
                raise ValueError("Trajectory seed count must be divisible by IK goal seed count.")
        ik_config = copy.deepcopy(self.config.ik_config)
        self._num_ik_goal_seeds_total = self._num_ik_goal_seeds
        if self._retry_enabled:
            self._num_ik_goal_seeds_total *= 1 + self.config.max_retries
            ik_config.solver.stages[-1].num_seeds = self._num_ik_goal_seeds_total
            for stage_idx in range(len(ik_config.solver.stages) - 2, -1, -1):
                ik_config.solver.stages[stage_idx].num_seeds = max(
                    ik_config.solver.stages[stage_idx].num_seeds,
                    ik_config.solver.stages[stage_idx + 1].num_seeds,
                )
        for geometry in self._scene.geoms:
            ik_config.add(
                SceneCollisionTask(
                    robot=robot,
                    scene=self._scene,
                    geometry=geometry,
                    scene_indices=None,
                    weight=self.config.ik_collision_weight,
                    margin=(
                        cfg.collision_margin
                        if self.config.ik_collision_margin is None
                        else self.config.ik_collision_margin
                    ),
                )
            )
        self._ik_helper = IK(ik_config, robot=robot, link=self.ee_link_index, device=self.device)
        self._ik_goal_q: Optional[wp.array] = None

        optimizer_type = (
            MppiTrajectoryOptimizer if isinstance(cfg, MppiTrajectoryOptimizerConfig) else GradientTrajectoryOptimizer
        )
        self._trajectory_optimizer = optimizer_type(
            config=cfg,
            robot=robot,
            ee_link_name_or_index=self.ee_link_index,
            scene=self._scene,
            num_frames=self._num_frames,
            dt=self.config.dt,
            device=self.device,
        )

    def warmup(self, batch_size: int):
        """Fix the planner batch size and allocate its batch-shaped buffers.

        Args:
            batch_size: Number of queries solved together.

        """
        if self._batch_size is not None:
            if batch_size != self._batch_size:
                raise ValueError(f"Batch size mismatch: expected {self._batch_size}, got {batch_size}")
            return

        num_dofs = self.robot.num_actuated_joints
        n_total = batch_size * self.config.trajectory_optimizer.num_seeds
        self._default_scene_indices = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
        self._target_q_per_seed_buf = wp.empty((n_total, num_dofs), dtype=wp.float32, device=self.device)
        self._traj_q_init_buf = wp.empty((n_total, self._num_frames, num_dofs), dtype=wp.float32, device=self.device)
        self._retimer.warmup(batch_size)
        self._final_q_buf = wp.empty((batch_size, self._num_frames, num_dofs), dtype=wp.float32, device=self.device)
        if batch_size > 1 and self._ik_helper.config.solver.cuda_graph_mode == "full":
            self._ik_helper.config.solver.cuda_graph_mode = "iter"
        # Joint-space planning skips goal-IK warmup; direct `solve_goal_ik` calls still initialize it lazily.
        if not self.config.q_goal:
            self._ik_helper.warmup(batch_size)
        self._trajectory_optimizer.warmup(batch_size)
        if self._retry_enabled:
            self._retry_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._retry_best_score = wp.empty(batch_size, dtype=wp.float32, device=self.device)
            self._retry_best_pen = wp.empty(batch_size, dtype=wp.float32, device=self.device)
            self._retry_penetrations = wp.zeros(batch_size, dtype=wp.float32, device=self.device)
            self._retry_collision_tasks = []
            self._retry_collision_jacobians = []
            if self.config.trajectory_optimizer.collision_weight > 0:
                self._final_q_buf.zero_()
                retry_var = VarValues(robot=self.robot.state(q=self._final_q_buf))
                for geometry in self._scene.geoms or (None,):
                    task = TrajectoryCollisionTask(
                        robot=self.robot,
                        scene=self._scene,
                        geometry=geometry,
                        scene_indices=self._default_scene_indices,
                        num_frames=self._num_frames,
                        weight=self.config.trajectory_optimizer.collision_weight,
                        margin=self.config.trajectory_optimizer.collision_margin,
                        free_goal_frame=True,
                    )
                    jacobian = wp.empty(
                        (batch_size, task.residual_dim * num_dofs), dtype=wp.float32, device=self.device
                    )
                    task.compute_weighted_sparse_jacobian_values(retry_var, out_jacobian_values=jacobian)
                    self._retry_collision_tasks.append(task)
                    self._retry_collision_jacobians.append(jacobian)
        for processor in self._postprocessors:
            processor.warmup(batch_size, self._active_joint_mask_np)
        self._batch_size = batch_size

    def set_active_joint_mask(self, joint_mask: Sequence[float]):
        """Set which actuated joints IK and trajectory optimization may change.

        Args:
            joint_mask: One value per actuated joint; zero pins the joint.
        """
        joint_mask_np = np.asarray(joint_mask, dtype=np.float32)
        self._active_joint_mask_np = joint_mask_np.copy()
        self._active_joint_mask_wp.assign(joint_mask_np)
        self._ik_helper.set_active_joint_mask(joint_mask_np.tolist())
        self._trajectory_optimizer.set_active_joint_mask(joint_mask_np)
        for processor in self._postprocessors:
            processor.set_active_joint_mask(joint_mask_np)

    # --- goal inverse kinematics ---
    def solve_goal_ik(
        self,
        T_world_target: wp.array,
        scene_indices: Optional[wp.array] = None,
        init_q: Optional[wp.array] = None,
        rest_q: Optional[wp.array] = None,
    ) -> wp.array:
        """Solve batched Warp pose targets.

        Args:
            T_world_target: Batched pose goals.
            scene_indices: Optional scene index for each query.
            init_q: Optional IK sampling centers with shape `(batch, dofs)`.
            rest_q: Optional IK rest configurations with shape `(batch, dofs)`.

        Returns:
            Best goal configurations with shape `(batch, dofs)`.
        """
        batch_size = T_world_target.shape[0]
        self.warmup(batch_size)
        scene_indices = scene_indices if scene_indices is not None else self._default_scene_indices
        q = self._ik_helper.solve(
            T_world_target=T_world_target.reshape((batch_size, 1)),
            init_state=self.robot.state(q=init_q) if init_q is not None else None,
            rest_state=self.robot.state(q=rest_q) if rest_q is not None else None,
            scene_indices=scene_indices,
        ).q
        return cast(
            wp.array,
            q.reshape((batch_size, self._ik_helper.num_solutions, self.robot.num_actuated_joints))[:, 0],
        )

    def solve_goal_ik_numpy(
        self,
        T_world_target: np.ndarray,
        scene_indices: Optional[np.ndarray] = None,
        init_q: Optional[np.ndarray] = None,
        rest_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """NumPy wrapper for `solve_goal_ik`."""
        T_world_target_np = np.asarray(T_world_target, dtype=np.float32)
        single = T_world_target_np.ndim == 1
        if single:
            T_world_target_np = T_world_target_np[None]
        target = wp.from_numpy(T_world_target_np, dtype=wp_vec7, device=self.device)
        scene_indices_wp = None
        if scene_indices is not None:
            scene_indices = np.asarray(scene_indices, dtype=np.int32)
            if np.any(scene_indices < 0) or np.any(scene_indices >= self._scene.num_scenes):
                raise ValueError("scene_indices contains an out-of-range scene index.")
            scene_indices_wp = wp.from_numpy(scene_indices, dtype=wp.int32, device=self.device)
        init_q_wp = (
            wp.from_numpy(np.atleast_2d(np.asarray(init_q, np.float32)), dtype=wp.float32, device=self.device)
            if init_q is not None
            else None
        )
        rest_q_wp = (
            wp.from_numpy(np.atleast_2d(np.asarray(rest_q, np.float32)), dtype=wp.float32, device=self.device)
            if rest_q is not None
            else None
        )
        result_np = self.solve_goal_ik(target, scene_indices_wp, init_q_wp, rest_q_wp).numpy().copy()
        return result_np[0] if single else result_np

    # --- shared internals ---
    def _build_target_q_per_seed(
        self,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
        target_q_init: Optional[wp.array] = None,
        rest_q: Optional[wp.array] = None,
    ) -> wp.array:
        """Build trajectory goal seeds from a joint goal or pose-goal IK."""
        if self.config.q_goal and (T_world_target is not None or target_q is None):
            raise ValueError("MotionPlannerConfig.q_goal=True requires target_q without T_world_target.")
        if not self.config.q_goal and T_world_target is None:
            raise ValueError(
                "MotionPlannerConfig.q_goal=False requires T_world_target; target_q may provide its solved IK goal."
            )
        num_seeds = self.config.trajectory_optimizer.num_seeds
        num_dofs = self.robot.num_actuated_joints
        batch_size = target_q.shape[0] if target_q is not None else cast(wp.array, T_world_target).shape[0]
        n_total = batch_size * num_seeds
        if target_q is not None and target_q_init is not None:
            raise ValueError("target_q_init only initializes pose-goal IK.")
        if target_q is not None:  # repeat joint-space goals
            warp_utils.repeat(target_q, num_seeds, out=self._target_q_per_seed_buf)
        else:  # solve pose-goal IK
            init_state = self.robot.state(q=target_q_init) if target_q_init is not None else None
            rest_state = init_state if rest_q is target_q_init else None
            if rest_state is None and rest_q is not None:
                rest_state = self.robot.state(q=rest_q)
            result = self._ik_helper.solve(
                T_world_target=cast(wp.array, T_world_target).reshape((batch_size, 1)),
                prev_state=init_state if rest_state is init_state else None,
                init_state=init_state,
                rest_state=rest_state,
                scene_indices=scene_indices,
            )
            self._ik_goal_q = result.q
            if self._num_ik_goal_seeds_total == 1:
                warp_utils.repeat(self._ik_goal_q, num_seeds, out=self._target_q_per_seed_buf)
            else:
                wp.launch(
                    _expand_ik_goal_seeds_kernel,
                    dim=(n_total, num_dofs),
                    inputs=[
                        self._ik_goal_q,
                        0,
                        self._num_ik_goal_seeds,
                        self._num_ik_goal_seeds_total,
                        num_seeds,
                        self._target_q_per_seed_buf,
                    ],
                    device=self.device,
                )

        return self._target_q_per_seed_buf

    def _build_initial_trajectories(
        self,
        start_q: wp.array,
        target_q_per_seed: wp.array,
    ) -> wp.array:
        """Build fresh trajectory seeds from each start to its joint goals."""
        num_seeds = self.config.trajectory_optimizer.num_seeds
        num_dofs = self.robot.num_actuated_joints
        n_total = start_q.shape[0] * num_seeds
        traj_q_init_buf = self._traj_q_init_buf
        wp.launch(
            _pin_inactive_joints_kernel,
            dim=target_q_per_seed.shape,
            inputs=[start_q, self._active_joint_mask_wp, num_seeds, target_q_per_seed],
            device=self.device,
        )
        if num_seeds > 1:
            wp.launch(
                _build_seed_trajectory_kernel,
                dim=n_total * self._num_frames * num_dofs,
                inputs=[
                    start_q,
                    target_q_per_seed,
                    0,
                    self._active_joint_mask_wp,
                    self._joint_limits_wp,
                    cast(GradientTrajectoryOptimizerConfig, self.config.trajectory_optimizer).init_sample_range,
                    num_seeds,
                    self._num_frames,
                    num_dofs,
                    1,
                    traj_q_init_buf,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _build_linear_trajectory_kernel,
                dim=[n_total * self._num_frames * num_dofs],
                inputs=[start_q, target_q_per_seed, num_seeds, self._num_frames, num_dofs, traj_q_init_buf],
                device=self.device,
            )
        return traj_q_init_buf

    # --- offline planning ---
    def _solve_offline_attempt(
        self,
        round_index: int,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
        target_q_per_seed: wp.array,
        traj_q_init: wp.array,
        retries: int,
    ):
        """One optimization round: round 0 uses the prebuilt seeds, retries reseed from the IK funnel."""
        optimizer = self._trajectory_optimizer
        batch_size = start_q.shape[0]
        num_seeds = self.config.trajectory_optimizer.num_seeds
        num_dofs = self.robot.num_actuated_joints
        if round_index > 0:  # reseed: this round's slice of the refined IK candidate pool
            wp.launch(
                _expand_ik_goal_seeds_kernel,
                dim=(batch_size * num_seeds, num_dofs),
                inputs=[
                    self._ik_goal_q,
                    round_index * self._num_ik_goal_seeds,
                    self._num_ik_goal_seeds,
                    self._num_ik_goal_seeds_total,
                    num_seeds,
                    target_q_per_seed,
                ],
                device=self.device,
            )
            wp.launch(
                _pin_inactive_joints_kernel,
                dim=(batch_size * num_seeds, num_dofs),
                inputs=[start_q, self._active_joint_mask_wp, num_seeds, target_q_per_seed],
                device=self.device,
            )
            wp.launch(
                _build_seed_trajectory_kernel,
                dim=batch_size * num_seeds * self._num_frames * num_dofs,
                inputs=[
                    start_q,
                    target_q_per_seed,
                    round_index,
                    self._active_joint_mask_wp,
                    self._joint_limits_wp,
                    cast(GradientTrajectoryOptimizerConfig, self.config.trajectory_optimizer).init_sample_range,
                    num_seeds,
                    self._num_frames,
                    num_dofs,
                    num_seeds // 2,
                    traj_q_init,
                ],
                device=self.device,
            )
        state, scores = optimizer.solve(
            start_q=start_q,
            T_world_target=T_world_target,
            target_q=target_q if round_index == 0 else None,
            traj_q_init=traj_q_init,
            target_q_per_seed=target_q_per_seed,
            scene_indices=scene_indices,
        )
        if retries == 0:
            wp.copy(self._final_q_buf, state.q)
            return
        self._retry_penetrations.zero_()
        retry_var = VarValues(robot=state)
        for task, jacobian in zip(self._retry_collision_tasks, self._retry_collision_jacobians):
            task.set_scene_indices(scene_indices)
            task.compute_weighted_sparse_jacobian_values(retry_var, out_jacobian_values=jacobian)
            task.accumulate_max_penetration(self._retry_penetrations)
        self._retry_condition.fill_(0)
        wp.launch(
            _set_best_retry_result_kernel,
            dim=batch_size,
            inputs=[
                scores,
                self._retry_penetrations,
                state.q,
                int(self.config.always_finetune_retry) if round_index == 0 else 0,
                self._retry_best_score,
                self._retry_best_pen,
                self._retry_condition,
                self._final_q_buf,
            ],
            device=self.device,
        )

    def solve_offline(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array] = None,
        target_q: Optional[wp.array] = None,
        scene_indices: Optional[wp.array] = None,
        target_q_init: Optional[wp.array] = None,
    ) -> MotionPlanResult[wp.array]:
        """Plan batched Warp trajectories.

        Lifecycle:
            1. `warmup(...)` allocates batch buffers.
            2. `_build_target_q_per_seed(...)` builds joint goals, using IK when needed.
            3. `_build_initial_trajectories(...)` samples trajectory seeds.
            4. `_solve_offline_attempt(...)` optimizes and optionally retries with new seeds.
            5. `TrajectoryPostprocessor.solve(...)` applies configured postprocessing.
            6. `TrajectoryRetimer.compute_motion_time(...)` computes shared segment timing.

        Args:
            start_q: Starting configurations with shape `(batch, dofs)`.
            T_world_target: Optional `wp_vec7` pose goals with shape `(batch,)`.
            target_q: Optional joint-space goals with shape `(batch, dofs)`.
            scene_indices: Optional scene indices with shape `(batch,)`.
            target_q_init: Optional IK initialization with shape `(batch, dofs)`.

        Returns:
            Trajectories with shape `(batch, num_timesteps + 1, dofs)` and motion times with shape `(batch,)`.
        """
        batch_size = start_q.shape[0]
        self.warmup(batch_size)
        scene_indices = scene_indices if scene_indices is not None else self._default_scene_indices
        if (T_world_target is None) == (target_q is None):
            raise ValueError("Provide exactly one of T_world_target or target_q for offline planning.")

        # --- build seeds ---
        target_q_per_seed = self._build_target_q_per_seed(
            T_world_target,
            target_q,
            scene_indices,
            target_q_init=target_q_init,
            rest_q=start_q,
        )
        traj_q_init = self._build_initial_trajectories(start_q, target_q_per_seed)

        # --- solve trajectories ---
        # retries solve the full batch because optimization terms do not accept per-row masks
        retries = self.config.max_retries if self._retry_enabled else 0
        if retries:
            self._retry_best_score.fill_(1.0e30)
        args = dict(
            start_q=start_q,
            T_world_target=T_world_target,
            target_q=target_q,
            scene_indices=scene_indices,
            target_q_per_seed=target_q_per_seed,
            traj_q_init=traj_q_init,
            retries=retries,
        )
        self._solve_offline_attempt(round_index=0, **args)
        for round_index in range(1, 1 + retries):
            wp.capture_if(self._retry_condition, self._solve_offline_attempt, round_index=round_index, **args)
        for processor in self._postprocessors:
            processor.solve(self._final_q_buf, T_world_target, target_q, scene_indices)
        return MotionPlanResult(
            q_traj=self._final_q_buf,
            motion_time=self._retimer.compute_motion_time(self._final_q_buf),
        )

    def solve_offline_numpy(
        self,
        start_q: np.ndarray,
        T_world_target: Optional[np.ndarray] = None,
        target_q: Optional[np.ndarray] = None,
        scene_indices: Optional[np.ndarray] = None,
        target_q_init: Optional[np.ndarray] = None,
    ) -> MotionPlanResult[np.ndarray]:
        """NumPy wrapper for `solve_offline`."""
        start_q = np.asarray(start_q, dtype=np.float32)
        if start_q.ndim == 1:
            start_q = start_q[None]
        start_q_wp = wp.from_numpy(start_q, dtype=wp.float32, device=self.device)

        scene_indices_wp = None
        if scene_indices is not None:
            scene_indices = np.asarray(scene_indices, dtype=np.int32)
            if np.any(scene_indices < 0) or np.any(scene_indices >= self._scene.num_scenes):
                raise ValueError("scene_indices contains an out-of-range scene index.")
            scene_indices_wp = wp.from_numpy(scene_indices, dtype=wp.int32, device=self.device)

        T_world_target_wp = None
        if T_world_target is not None:
            T_world_target = np.asarray(T_world_target, dtype=np.float32)
            if T_world_target.ndim == 1:
                T_world_target = T_world_target[None]
            T_world_target_wp = wp.from_numpy(T_world_target, dtype=wp_vec7, device=self.device)

        target_q_wp = None
        if target_q is not None:
            target_q = np.asarray(target_q, dtype=np.float32)
            if target_q.ndim == 1:
                target_q = target_q[None]
            target_q_wp = wp.from_numpy(target_q, dtype=wp.float32, device=self.device)

        target_q_init_wp = None
        if target_q_init is not None:
            target_q_init = np.asarray(target_q_init, dtype=np.float32)
            if target_q_init.ndim == 1:
                target_q_init = target_q_init[None]
            target_q_init_wp = wp.from_numpy(target_q_init, dtype=wp.float32, device=self.device)

        result = self.solve_offline(start_q_wp, T_world_target_wp, target_q_wp, scene_indices_wp, target_q_init_wp)
        return MotionPlanResult(q_traj=result.q_traj.numpy().copy(), motion_time=result.motion_time.numpy().copy())

    def solve_offline_torch(
        self,
        start_q: "torch.Tensor",
        T_world_target: Optional["torch.Tensor"] = None,
        target_q: Optional["torch.Tensor"] = None,
        scene_indices: Optional["torch.Tensor"] = None,
        target_q_init: Optional["torch.Tensor"] = None,
    ) -> MotionPlanResult["torch.Tensor"]:
        """Torch wrapper for `solve_offline`."""
        if start_q.ndim == 1:
            start_q = start_q[None]
        start_q_wp = wp.from_torch(start_q.contiguous(), dtype=wp.float32)

        T_world_target_wp = None
        if T_world_target is not None:
            if T_world_target.ndim == 2:
                T_world_target = T_world_target[None]
            T_world_target_wp = se3_from_matrix(wp.from_torch(T_world_target.contiguous(), dtype=wp.mat44))

        target_q_wp = None
        if target_q is not None:
            if target_q.ndim == 1:
                target_q = target_q[None]
            target_q_wp = wp.from_torch(target_q.contiguous(), dtype=wp.float32)

        scene_indices_wp = None
        if scene_indices is not None:
            scene_indices_wp = wp.from_torch(scene_indices.contiguous(), dtype=wp.int32)

        target_q_init_wp = None
        if target_q_init is not None:
            if target_q_init.ndim == 1:
                target_q_init = target_q_init[None]
            target_q_init_wp = wp.from_torch(target_q_init.contiguous(), dtype=wp.float32)

        result = self.solve_offline(start_q_wp, T_world_target_wp, target_q_wp, scene_indices_wp, target_q_init_wp)
        return MotionPlanResult(
            q_traj=wp.to_torch(result.q_traj).clone(), motion_time=wp.to_torch(result.motion_time).clone()
        )

    # --- online planning ---
    def _solve_online_mppi(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
        online_state: Optional[OnlineState],
    ) -> Tuple[wp.array, OnlineState]:
        """Solve one MPPI update with its retained trajectory or action distribution."""
        online_state = online_state or OnlineState()
        target_q_per_seed = self._build_target_q_per_seed(
            T_world_target,
            target_q,
            scene_indices,
            rest_q=start_q,
        )
        wp.launch(
            _pin_inactive_joints_kernel,
            dim=target_q_per_seed.shape,
            inputs=[
                start_q,
                self._active_joint_mask_wp,
                self.config.trajectory_optimizer.num_seeds,
                target_q_per_seed,
            ],
            device=self.device,
        )
        state, _ = cast(MppiTrajectoryOptimizer, self._trajectory_optimizer).solve(
            start_q=start_q,
            T_world_target=T_world_target,
            target_q=target_q,
            traj_q_init=None,
            target_q_per_seed=target_q_per_seed,
            scene_indices=scene_indices,
            online_state=online_state,
        )
        return state.q, online_state

    def _solve_online_gradient(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
        online_state: Optional[OnlineState],
    ) -> Tuple[wp.array, OnlineState]:
        """Solve one fresh gradient trajectory.

        Lifecycle:
            1. Build joint goals, stabilizing pose-goal IK with the previous joint goal.
            2. Sample fresh trajectories from the current configurations to those goals.
            3. Append the current configurations to the executed history.
            4. Optimize the trajectories and save the joint goal as the next IK initialization.
        """
        # --- build goals and seeds ---
        target_q_init = (
            online_state.prev_goal_q
            if target_q is None and T_world_target is not None and online_state is not None
            else None
        )
        rest_q = target_q_init if target_q_init is not None and self._ik_helper.num_solutions == 1 else start_q
        target_q_per_seed = self._build_target_q_per_seed(
            T_world_target,
            target_q,
            scene_indices,
            target_q_init=target_q_init,
            rest_q=rest_q,
        )
        traj_q_init = self._build_initial_trajectories(start_q, target_q_per_seed)

        # --- update online state ---
        if online_state is None:
            online_state = OnlineState()
        if online_state.prev_goal_q is None:
            online_state.prev_goal_q = wp.empty(
                (start_q.shape[0], self.robot.num_actuated_joints), dtype=wp.float32, device=self.device
            )
        wp.copy(
            online_state.prev_goal_q,
            target_q_per_seed.reshape(
                (start_q.shape[0], self.config.trajectory_optimizer.num_seeds, self.robot.num_actuated_joints)
            )[:, 0],
        )
        if online_state.q_history is None:
            online_state.q_history = wp.zeros(
                (start_q.shape[0], 3, self.robot.num_actuated_joints), dtype=wp.float32, device=self.device
            )
            online_state.history_count = 0
        elif online_state.q_history.shape != (start_q.shape[0], 3, self.robot.num_actuated_joints):
            raise ValueError("online_state.q_history shape does not match start_q.")
        wp.launch(
            _append_q_history_kernel,
            dim=start_q.shape,
            inputs=[online_state.q_history, start_q],
            device=self.device,
        )
        online_state.history_count = min(online_state.history_count + 1, 3)

        # --- solve trajectory ---
        state, _ = cast(GradientTrajectoryOptimizer, self._trajectory_optimizer).solve(
            start_q=start_q,
            T_world_target=T_world_target,
            target_q=target_q,
            traj_q_init=traj_q_init,
            target_q_per_seed=target_q_per_seed,
            scene_indices=scene_indices,
            q_history=online_state.q_history,
            history_count=online_state.history_count,
        )
        return state.q, online_state

    def solve_online(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array] = None,
        target_q: Optional[wp.array] = None,
        scene_indices: Optional[wp.array] = None,
        online_state: Optional[OnlineState] = None,
    ) -> MotionPlanResult[wp.array]:
        """Solve one online planning update.

        Args:
            start_q: Current configurations with shape `(batch, dofs)`.
            T_world_target: Optional `wp_vec7` pose goals with shape `(batch,)`.
            target_q: Optional solved IK goals or joint-space goals with shape `(batch, dofs)`.
            scene_indices: Optional scene indices with shape `(batch,)`.
            online_state: Optional state returned by the previous update.

        Returns:
            Optimized trajectories and caller-owned online state.
        """
        self.warmup(start_q.shape[0])
        scene_indices = scene_indices if scene_indices is not None else self._default_scene_indices
        if isinstance(self._trajectory_optimizer, MppiTrajectoryOptimizer):
            q_traj, online_state = self._solve_online_mppi(
                start_q, T_world_target, target_q, scene_indices, online_state
            )
        else:
            q_traj, online_state = self._solve_online_gradient(
                start_q, T_world_target, target_q, scene_indices, online_state
            )
        wp.copy(self._final_q_buf, q_traj)
        return MotionPlanResult(q_traj=self._final_q_buf, online_state=online_state)

    def solve_online_numpy(
        self,
        start_q: np.ndarray,
        T_world_target: Optional[np.ndarray] = None,
        target_q: Optional[np.ndarray] = None,
        scene_indices: Optional[np.ndarray] = None,
        online_state: Optional[OnlineState] = None,
    ) -> MotionPlanResult[np.ndarray]:
        """NumPy wrapper for `solve_online`."""
        start_q = np.asarray(start_q, dtype=np.float32)
        if start_q.ndim == 1:
            start_q = start_q[None]
        start_q_wp = wp.from_numpy(start_q, dtype=wp.float32, device=self.device)

        T_world_target_wp = None
        if T_world_target is not None:
            T_world_target = np.asarray(T_world_target, dtype=np.float32)
            if T_world_target.ndim == 1:
                T_world_target = T_world_target[None]
            T_world_target_wp = wp.from_numpy(T_world_target, dtype=wp_vec7, device=self.device)

        target_q_wp = None
        if target_q is not None:
            target_q = np.asarray(target_q, dtype=np.float32)
            if target_q.ndim == 1:
                target_q = target_q[None]
            target_q_wp = wp.from_numpy(target_q, dtype=wp.float32, device=self.device)

        scene_indices_wp = None
        if scene_indices is not None:
            scene_indices = np.atleast_1d(np.asarray(scene_indices, dtype=np.int32))
            scene_indices_wp = wp.from_numpy(scene_indices, dtype=wp.int32, device=self.device)

        result = self.solve_online(
            start_q_wp,
            T_world_target_wp,
            target_q_wp,
            scene_indices_wp,
            online_state,
        )
        return MotionPlanResult(q_traj=result.q_traj.numpy().copy(), online_state=result.online_state)
