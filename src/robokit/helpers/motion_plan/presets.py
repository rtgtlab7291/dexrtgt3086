"""Motion planner preset configurations.

Use `dataclasses.replace` for per-run changes; nested `trajectory_optimizer` objects are shared.

Presets:
    - `offline` and `offline_lbfgs` run goal IK, retry, shortcut, snap, and retiming.
    - `online_lbfgs` and `online_lm` solve fresh trajectories with executed-history smoothing.
    - `online_mppi` uses one planner with caller-owned MPPI state.
"""

import dataclasses
from typing import Literal, Optional

from robokit.helpers.ik import IKConfig
from robokit.helpers.motion_plan.gradient_trajectory_optimizer import GradientTrajectoryOptimizerConfig
from robokit.helpers.motion_plan.motion_planner import MotionPlannerConfig
from robokit.helpers.motion_plan.mppi_trajectory_optimizer import MppiTrajectoryOptimizerConfig
from robokit.helpers.motion_plan.trajectory_retimer import TrajectoryRetimerConfig
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.rotation_task import RotationTask


# --- builders --------------------------------------------------------------
def build_offline_ik_config(
    num_seeds: int = 8,
    cuda_graph_mode: Literal["none", "full", "iter"] = "full",
    num_goal_seeds: Optional[int] = None,
) -> IKConfig:
    """Build the offline goal-IK multi-seed funnel.

    Args:
        num_seeds: Number of IK candidates refined after global sampling.
        cuda_graph_mode: CUDA graph capture mode.
        num_goal_seeds: Number of final joint goals returned per query. Defaults to `num_seeds`.

    Returns:
        A `64 -> search -> goal` IK configuration.
    """
    num_goal_seeds = num_seeds if num_goal_seeds is None else num_goal_seeds
    if num_goal_seeds > num_seeds:
        raise ValueError("num_goal_seeds must not exceed num_seeds.")
    return (
        IKConfig(
            solver=MultiSeedSolverConfig(
                stages=[
                    StageConfig(num_seeds=max(64, num_seeds * 2), iters=8, lm_lambda=10.0),
                    StageConfig(
                        num_seeds=num_seeds if num_seeds > 1 else 4,
                        iters=16,
                        lm_lambda=1.0,
                    ),
                    StageConfig(num_seeds=num_goal_seeds, iters=0, lm_lambda=1.0),
                ],
                cuda_graph_mode=cuda_graph_mode,
            )
        )
        .add(PositionTask(weight=20.0))
        .add(RotationTask(weight=40.0))
        .add(PositionLimit(weight=50.0))
        .add(RestTask(weight=0.05))
    )


# --- offline planning ------------------------------------------------------
_offline_traj = GradientTrajectoryOptimizerConfig(
    position_weight=5000.0,
    orientation_weight=5000.0,
    smoothness_weight=200.0,
    collision_weight=1000.0,
    lm_lambda=0.01,
    solver="lm",
    goal_config_weight=50.0,
    velocity_limit_weight=5.0,
    self_collision_weight=0.0,
    collision_margin=0.025,
    accel_smoothness_weight=0.0,
    accel2_smoothness_weight=30.0,
    num_seeds=8,
    max_iter=50,
)

offline = MotionPlannerConfig(
    ik_config=build_offline_ik_config(),
    finetune=TrajectoryRetimerConfig(dt_scale=0.9),
    trajectory_optimizer=_offline_traj,
)
offline_lbfgs = dataclasses.replace(
    offline,
    # Jerk-limited retiming keeps jerk near 100 at about 20% extra motion time.
    finetune=TrajectoryRetimerConfig(dt_scale=1.0, jerk_limit=100.0, max_dt=1.0),
    trajectory_optimizer=dataclasses.replace(_offline_traj, solver="lbfgs", start_config_weight=2000.0),
)


# --- joint-space planning --------------------------------------------------
# Joint-space planning pins both endpoints and optimizes only the interior trajectory.
_q_goal_traj = GradientTrajectoryOptimizerConfig(
    position_weight=0.0,
    orientation_weight=0.0,
    smoothness_weight=200.0,
    collision_weight=1000.0,
    lm_lambda=0.01,
    solver="lbfgs",
    start_config_weight=0.0,
    goal_config_weight=0.0,
    velocity_limit_weight=5.0,
    self_collision_weight=0.0,
    collision_margin=0.025,
    accel_smoothness_weight=0.0,
    accel2_smoothness_weight=30.0,
    lock_endpoints=True,
    num_seeds=8,
    max_iter=100,
)
q_goal = dataclasses.replace(
    offline,
    q_goal=True,
    finetune=TrajectoryRetimerConfig(
        dt_scale=1.0, jerk_limit=100.0, max_dt=1.0
    ),  # jerk-limited retiming (as offline_lbfgs)
    trajectory_optimizer=_q_goal_traj,
)


# --- online gradient planning ----------------------------------------------
# Online L-BFGS replans fresh collision-aware trajectories from the current state.
online_lbfgs = dataclasses.replace(
    offline_lbfgs,
    num_timesteps=15,
    dt=0.1,
    ik_collision_weight=500.0,
    ik_collision_margin=0.001,
    finetune=None,
    enable_retry=False,
    postprocessors=(),
    ik_config=build_offline_ik_config(num_goal_seeds=1),
    expand_ik_goal_seeds=True,
    trajectory_optimizer=dataclasses.replace(
        offline_lbfgs.trajectory_optimizer,
        num_seeds=16,  # seeds batch near-free on GPU; iters past ~10 buy little success per tick-ms
        max_iter=10,
        position_weight=0.0,
        orientation_weight=0.0,
        goal_config_weight=5000.0,
        accel_smoothness_weight=0.25,
        accel2_smoothness_weight=3.0,
        collision_goal_radius=3.0,
        keep_init_seed=False,
        lock_start=True,
        free_goal_frame=False,
    ),
)

# Online LM solves the same fresh planning problem with sparse Gauss-Newton steps.
online_lm = dataclasses.replace(
    online_lbfgs,
    trajectory_optimizer=dataclasses.replace(
        online_lbfgs.trajectory_optimizer,
        solver="lm",
        num_seeds=8,
        max_iter=5,
        cg_max_iter=4,
        accel2_smoothness_weight=3.0,
    ),
)


# --- online MPPI planning --------------------------------------------------
# Reactive acceleration MPPI uses caller-owned state and live IK goals without a separate discovery planner.
online_mppi = MotionPlannerConfig(
    num_timesteps=31,  # -> num_frames = 32 (the MPPI horizon)
    dt=0.03,  # control step + rollout base dt
    use_cuda_graph=False,  # cuda_graph_mode="none": eager solve()
    enable_retry=False,
    postprocessors=(),
    ik_config=build_offline_ik_config(num_goal_seeds=1),  # tuned goal IK (rot 40 + rest)
    trajectory_optimizer=MppiTrajectoryOptimizerConfig(
        position_weight=600.0,
        orientation_weight=150.0,
        smoothness_weight=0.0,
        collision_weight=1500.0,
        start_config_weight=0.0,
        goal_config_weight=0.0,
        velocity_limit_weight=0.0,
        position_limit_weight=300.0,
        rest_weight=0.0,
        self_collision_weight=0.0,
        collision_margin=0.005,
        accel_smoothness_weight=1.0,
        accel2_smoothness_weight=8.0,
        num_seeds=1,
        max_iter=5,  # MPPI updates per tick
        collision_sweep_steps=4,
        collision_penalty="smooth",
        num_particles=384,
        beta=0.2,
        init_std=1.5,
        min_std=0.005,
        settle_radius=0.05,
        control_space="acceleration",
        knot_ratio=0.7,
        accel_limit=15.0,
        dt_max=0.05,
        q_limit_margin=0.10,
        goal_rest_weight=40.0,
        link_position_weight=300.0,
    ),
)


__all__ = [
    "build_offline_ik_config",
    "offline",
    "offline_lbfgs",
    "q_goal",
    "online_lbfgs",
    "online_lm",
    "online_mppi",
]
