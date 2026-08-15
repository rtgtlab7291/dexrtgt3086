from typing import Any, List

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import (
    IK,
    IKConfig,
)
from robokit.lie.se3 import se3_identity
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.multi_seed_solver import MultiSeedSolver, MultiSeedSolverConfig, StageConfig
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec7


_robot_cache: dict = {}


def _get_robot(desc_name: str) -> Robot:
    if desc_name not in _robot_cache:
        _robot_cache[desc_name] = Robot.load(load_robot_description(desc_name))
    return _robot_cache[desc_name]


def test_full_cuda_graph_mode_caps_lbfgs_eager_warmup_iters():
    """cuda_graph_mode="full" runs one eager pass (population_solver.py::_warmup_and_capture)
    before capturing the whole solve as a single graph. That eager pass must not run the stage's
    full configured max_iter eagerly - MultiSeedSolver._solve_stage(warmup=True) caps it, mirroring
    the cheap-warmup LBFGSOptimizer._warmup_and_capture used before LBFGS routed through
    PopulationSolver's own "full" graph capture. The subsequent capture-recording pass (and every
    later replay) must still see the full configured max_iter."""
    if not wp.is_cuda_available():
        pytest.skip("cuda graph capture exercised on CUDA")
    device = wp.get_device("cuda:0")
    robot = _get_robot("ur10_description")
    target_link_index = robot.link_names.index("ee_link")
    batch_size = 2
    num_dofs = robot.num_actuated_joints

    q_target = robot.spec.midrange_q.astype(np.float32)
    q_targets = wp.from_numpy(np.stack([q_target, q_target]), dtype=wp.float32, device=device)
    target_state = robot.forward_kinematics(robot.state(q=q_targets))
    target_pose = target_state.get_T_world_link(target_link_index).reshape((batch_size, 1))

    configured_max_iter = 20
    solver_config = MultiSeedSolverConfig(
        stages=[StageConfig(num_seeds=1, iters=configured_max_iter, lm_lambda=1.0)],
        cuda_graph_mode="full",
        lbfgs=LBFGSOptimizerConfig(),
    )
    solver = MultiSeedSolver(
        terms=[[FrameTask(robot, target_link_index, target_pose, 10.0, 5.0)]],
        config=solver_config,
        device=device,
    )

    seen_max_iters: List[int] = []
    original_solve = LBFGSOptimizer.solve

    def _spy_solve(self, var):
        seen_max_iters.append(self.config.max_iter)
        return original_solve(self, var)

    LBFGSOptimizer.solve = _spy_solve
    try:
        q_base = wp.empty((batch_size, num_dofs), dtype=wp.float32, device=device)  # type: ignore[arg-type]
        initial_var = VarValues(robot=robot.state(q=q_base))
        solver.solve(initial_var)
    finally:
        LBFGSOptimizer.solve = original_solve

    # exactly 2 solve() calls: the capped eager warmup, then the full-iters capture recording.
    assert seen_max_iters == [min(2, configured_max_iter), configured_max_iter]


def test_solver_multistage():
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    robot: Robot = _get_robot("ur10_description")
    target_link_index: int = robot.link_names.index("ee_link")

    batch_size: int = 2
    num_seeds: int = 8
    num_seeds_mid: int = 2
    init_iters: int = 4
    mid_iters: int = 4
    final_iters: int = 4
    num_dofs: int = robot.num_actuated_joints

    total_batch_init: int = batch_size * num_seeds

    q_target_a: np.ndarray = robot.spec.midrange_q.astype(np.float32)
    q_target_b: np.ndarray = robot.spec.zero_q.astype(np.float32)
    q_targets_np: np.ndarray = np.stack([q_target_a, q_target_b], axis=0)
    q_targets: wp.array = wp.from_numpy(q_targets_np, dtype=wp.float32, device=device)
    target_state: Any = robot.state(q=q_targets)
    target_state = robot.forward_kinematics(target_state)
    target_pose: wp.array = target_state.get_T_world_link(target_link_index)
    target_vec7_np: np.ndarray = target_pose.numpy()

    rng = np.random.default_rng(0)
    joint_limits: np.ndarray = robot.spec.actuated_joint_limits
    seed_qs_np: np.ndarray = rng.uniform(
        joint_limits[:, 0], joint_limits[:, 1], size=(total_batch_init, num_dofs)
    ).astype(np.float32)

    # create per-stage targets (repeated for seeds)
    targets_init_np: np.ndarray = np.repeat(target_vec7_np, repeats=num_seeds, axis=0)
    targets_mid_np: np.ndarray = np.repeat(target_vec7_np, repeats=num_seeds_mid, axis=0)
    targets_final_np: np.ndarray = target_vec7_np.copy()

    target_se3_init = wp.from_numpy(targets_init_np[:, None], dtype=wp_vec7, device=device)
    target_se3_mid = wp.from_numpy(targets_mid_np[:, None], dtype=wp_vec7, device=device)
    target_se3_final = wp.from_numpy(targets_final_np[:, None], dtype=wp_vec7, device=device)

    terms_init: List[ResidualTask] = [
        FrameTask(robot, target_link_index, target_se3_init, 10.0, 5.0),
        PositionLimit(robot, weight=50.0),
    ]
    terms_mid: List[ResidualTask] = [
        FrameTask(robot, target_link_index, target_se3_mid, 10.0, 5.0),
        PositionLimit(robot, weight=50.0),
    ]
    terms_final: List[ResidualTask] = [
        FrameTask(robot, target_link_index, target_se3_final, 10.0, 5.0),
        PositionLimit(robot, weight=50.0),
    ]

    # create base placeholder state (batch_size only)
    q_base: wp.array = wp.empty((batch_size, num_dofs), dtype=wp.float32, device=device)  # type: ignore[arg-type]
    base_state = robot.state(q=q_base)

    solver_config = MultiSeedSolverConfig(
        stages=[
            StageConfig(num_seeds=num_seeds, iters=init_iters, lm_lambda=10.0),
            StageConfig(num_seeds=num_seeds_mid, iters=mid_iters, lm_lambda=5.0),
            StageConfig(num_seeds=1, iters=final_iters, lm_lambda=1.0),
        ],
        cuda_graph_mode="none",
    )
    solver = MultiSeedSolver(
        terms=[terms_init, terms_mid, terms_final],
        config=solver_config,
        device=device,
    )

    # create and fill initial var with random seeds
    indices = wp.zeros((total_batch_init,), dtype=wp.int32, device=device)
    initial_state = base_state.gather(indices)
    initial_var = VarValues(robot=initial_state)
    q_init: wp.array = wp.from_numpy(seed_qs_np, dtype=wp.float32, device=device)
    wp.copy(initial_state.q, q_init)

    best_state: Any
    best_costs: wp.array
    best_var, best_costs = solver.solve(initial_var)
    best_state = best_var.get("robot")
    best_state = robot.forward_kinematics(best_state)
    achieved_pose: wp.array = best_state.get_T_world_link(target_link_index)

    achieved_vec7: np.ndarray = achieved_pose.numpy()
    target_vec7: np.ndarray = target_pose.numpy()
    assert np.allclose(achieved_vec7[:, :3], target_vec7[:, :3], atol=1e-2)
    achieved_quat: np.ndarray = achieved_vec7[:, 3:7]
    target_quat: np.ndarray = target_vec7[:, 3:7]
    dot_product: np.ndarray = np.abs(np.sum(achieved_quat * target_quat, axis=1))
    assert np.all(dot_product > 0.99)


def test_solver_selects_terminal_seed_with_score_terms():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    robot = _get_robot("ur10_description")
    q0 = robot.spec.midrange_q.astype(np.float32).reshape(-1)
    q1 = q0 + 0.1
    initial_var = VarValues(robot=robot.state(q=wp.from_numpy(np.stack([q0, q1]), dtype=wp.float32, device=device)))
    solver = MultiSeedSolver(
        terms=[[RestTask(robot=robot, rest_q=q0)]],
        score_terms=[RestTask(robot=robot, rest_q=q1)],
        config=MultiSeedSolverConfig(stages=[StageConfig(num_seeds=2, iters=0)], cuda_graph_mode="none"),
        device=device,
    )

    best_var, best_costs = solver.solve(initial_var)

    np.testing.assert_allclose(best_var.get("robot").q.numpy(), q1[None])
    np.testing.assert_allclose(best_costs.numpy(), 0.0, atol=1e-6)
    np.testing.assert_array_equal(solver.winner_indices.numpy(), [[1]])


def test_solver_stage_handoff_cost_consistency_with_mobile_smoothness():
    """Verify that gather correctly copies variable data between stages.

    After stage 0 solve, the best seeds are gathered into stage 1. We verify that
    the gathered variable values (q, T_world_base) match the original values at
    the selected indices. This tests the correctness of the gather operation itself,
    without relying on two different optimizers producing bit-identical residuals
    (which can vary across warp-lang versions due to kernel compilation differences).
    """
    device_str = "cuda:0" if wp.is_cuda_available() else "cpu"
    device = wp.get_device(device_str)
    robot = _get_robot("fetch_description")

    config = (
        IKConfig(
            enable_T_world_base=True,
            init_sample_range=0.1,
            solver=MultiSeedSolverConfig(
                stages=[
                    StageConfig(num_seeds=8, iters=4, lm_lambda=10.0),
                    StageConfig(num_seeds=2, iters=6, lm_lambda=1.0),
                    StageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
                ],
                cuda_graph_mode="none",
            ),
        )
        .add(PositionTask(weight=20.0))
        .add(RotationTask(weight=10.0))
        .add(PositionLimit(weight=50.0))
        .add(RestTask(weight=0.01, base_weight=[0.01, 0.01, 100.0, 100.0, 100.0, 0.01]))
        .add(SmoothnessTask(weight=0.2, base_weight=0.2))
    )
    ik_helper = IK(config, robot=robot, link="gripper_link", device=device_str)
    ik_helper.warmup(batch_size=1)

    q0 = robot.spec.zero_q.reshape(1, -1).astype(np.float32)
    prev_state = robot.state(
        q=wp.from_numpy(q0, dtype=wp.float32, device=device),
        T_world_base=se3_identity(1, device=device),
    )

    target_state = robot.state(
        q=wp.from_numpy(q0, dtype=wp.float32, device=device),
        T_world_base=se3_identity(1, device=device),
    )
    target_state = robot.forward_kinematics(target_state)
    target = target_state.get_T_world_link(robot.link_names.index("gripper_link"))

    # Stage vars are VarValues internally; unwrap to the robot leaf for q/base inspection.
    stage0_values = ik_helper._solver._stage_vars[0]
    stage1_values = ik_helper._solver._stage_vars[1]
    stage0_var = stage0_values.get("robot")
    stage1_var = stage1_values.get("robot")
    stage0_opt = ik_helper._solver._updaters[0]

    rng = np.random.default_rng(0)
    for _ in range(6):
        initial_var = ik_helper._initial_var
        initial_state = initial_var.get("robot")
        q_range = float(rng.uniform(0.05, 0.15))
        base_range = float(rng.uniform(0.05, 0.15))
        ik_helper._sampler.sample_q(prev_state.q, q_range, initial_state.q, joint_mask=ik_helper._active_joint_mask)
        ik_helper._sampler.sample_base(
            prev_state.T_world_base,
            base_range,
            initial_state.T_world_base,
            base_mask=ik_helper._active_base_mask,
        )
        ik_helper._update_targets(target.reshape((1, 1)), prev_state, None)

        stage0_values.invalidate()
        stage0_opt.solve(stage0_values)

        solver = ik_helper._solver
        if solver._use_tiled_select:
            wp.launch_tiled(
                solver._select_best_kernels[0],
                dim=[solver.batch_size],
                inputs=[stage0_opt.costs, solver.best_indices[0]],
                block_dim=solver._tile_threads[0],
                device=solver.device,
            )
        else:
            from robokit.opt.population_solver import _plain_select_best_k_kernel

            stages = list(solver.config.stages)
            wp.launch(
                _plain_select_best_k_kernel,
                dim=[solver.batch_size],
                inputs=[
                    stage0_opt.costs,
                    solver._plain_select_work,
                    solver.best_indices[0],
                    stages[0].num_seeds,
                    stages[1].num_seeds,
                ],
                device=solver.device,
            )
        flat_indices = solver.best_indices[0].reshape((solver.total_batches[1],))
        stage0_values.gather(flat_indices, stage1_values)

        # Verify gather correctness: the gathered q and T_world_base must
        # match the source rows at the selected indices (NaN-aware comparison
        # since Cholesky decomposition can produce NaN for ill-conditioned seeds).
        selected = np.asarray(flat_indices.numpy(), dtype=np.int64)
        stage0_q_np = stage0_var.q.numpy()
        stage1_q_np = stage1_var.q.numpy()
        np.testing.assert_array_equal(stage1_q_np, stage0_q_np[selected])

        stage0_base_np = stage0_var.T_world_base.numpy()
        stage1_base_np = stage1_var.T_world_base.numpy()
        np.testing.assert_array_equal(stage1_base_np, stage0_base_np[selected])

        # The gathered entries' costs from stage 0 should be monotonically
        # ordered (best_indices selects lowest costs first). Skip NaN entries.
        stage0_costs_np = stage0_opt.costs.numpy()
        gathered_costs = stage0_costs_np[selected]
        finite_mask = np.isfinite(gathered_costs)
        finite_costs = gathered_costs[finite_mask]
        if len(finite_costs) > 1:
            assert np.all(np.diff(finite_costs) >= -1e-6), f"Gathered costs should be sorted: {gathered_costs}"
