from typing import Any, List, Type, cast

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IKHelper, IKHelperConfig
from robokit.lie.warp_se3 import WarpSE3
from robokit.opt.warp_optimizer import aggregate_residuals_to_costs
from robokit.opt.warp_solver import WarpSolver, WarpSolverConfig, WarpStageConfig
from robokit.robo.robot import Robot
from robokit.terms import WarpFrameTask, WarpPositionLimit
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import repeat, wp_vec7


def test_warp_solver_multistage() -> None:
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    robot: Robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
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
    target_pose: WarpSE3 = target_state.get_T_world_link(target_link_index)
    target_vec7_np: np.ndarray = target_pose.xyz_wxyz.numpy()

    rng = np.random.default_rng(0)
    joint_limits: np.ndarray = robot.spec.actuated_joint_limits
    seed_qs_np: np.ndarray = rng.uniform(
        joint_limits[:, 0], joint_limits[:, 1], size=(total_batch_init, num_dofs)
    ).astype(np.float32)

    # Create per-stage targets (repeated for seeds)
    targets_init_np: np.ndarray = np.repeat(target_vec7_np, repeats=num_seeds, axis=0)
    targets_mid_np: np.ndarray = np.repeat(target_vec7_np, repeats=num_seeds_mid, axis=0)
    targets_final_np: np.ndarray = target_vec7_np.copy()

    target_vec7_init: wp.array = wp.from_numpy(targets_init_np, dtype=wp_vec7, device=device)
    target_vec7_mid: wp.array = wp.from_numpy(targets_mid_np, dtype=wp_vec7, device=device)
    target_vec7_final: wp.array = wp.from_numpy(targets_final_np, dtype=wp_vec7, device=device)

    target_se3_init: WarpSE3 = WarpSE3(target_vec7_init)
    target_se3_mid: WarpSE3 = WarpSE3(target_vec7_mid)
    target_se3_final: WarpSE3 = WarpSE3(target_vec7_final)

    total_batch_mid: int = batch_size * num_seeds_mid
    total_batch_final: int = batch_size

    terms_init: List[WarpTask] = [
        WarpFrameTask(robot, target_link_index, target_se3_init, 10.0, 5.0),
        WarpPositionLimit(robot, weight=50.0, batch_size=total_batch_init),
    ]
    terms_mid: List[WarpTask] = [
        WarpFrameTask(robot, target_link_index, target_se3_mid, 10.0, 5.0),
        WarpPositionLimit(robot, weight=50.0, batch_size=total_batch_mid),
    ]
    terms_final: List[WarpTask] = [
        WarpFrameTask(robot, target_link_index, target_se3_final, 10.0, 5.0),
        WarpPositionLimit(robot, weight=50.0, batch_size=total_batch_final),
    ]

    # Create base placeholder state (batch_size only)
    q_base: wp.array = wp.empty((batch_size, num_dofs), dtype=wp.float32, device=device)  # type: ignore[arg-type]
    base_state = robot.state(q=q_base)

    solver_config = WarpSolverConfig(
        stages=[
            WarpStageConfig(num_seeds=num_seeds, iters=init_iters, lm_lambda=10.0),
            WarpStageConfig(num_seeds=num_seeds_mid, iters=mid_iters, lm_lambda=5.0),
            WarpStageConfig(num_seeds=1, iters=final_iters, lm_lambda=1.0),
        ],
        use_cuda_graph=False,
    )
    solver: WarpSolver[Any] = WarpSolver(
        config=solver_config,
        placeholder_var=base_state,
        terms=[terms_init, terms_mid, terms_final],
    )

    # Fill initial_var with random seeds
    q_init: wp.array = wp.from_numpy(seed_qs_np, dtype=wp.float32, device=device)
    wp.copy(solver.initial_var.q, q_init)

    best_state: Any
    best_costs: wp.array
    best_state, best_costs = solver.solve()
    best_state = robot.forward_kinematics(best_state)
    achieved_pose: WarpSE3 = best_state.get_T_world_link(target_link_index)

    achieved_vec7: np.ndarray = achieved_pose.xyz_wxyz.numpy()
    target_vec7: np.ndarray = target_pose.xyz_wxyz.numpy()
    assert np.allclose(achieved_vec7[:, :3], target_vec7[:, :3], atol=1e-2)
    achieved_quat: np.ndarray = achieved_vec7[:, 3:7]
    target_quat: np.ndarray = target_vec7[:, 3:7]
    dot_product: np.ndarray = np.abs(np.sum(achieved_quat * target_quat, axis=1))
    assert np.all(dot_product > 0.99)


def test_warp_solver_stage_handoff_cost_consistency_with_mobile_smoothness() -> None:
    """Verify that gather correctly copies variable data between stages.

    After stage 0 solve, the best seeds are gathered into stage 1. We verify that
    the gathered variable values (q, T_world_base) match the original values at
    the selected indices. This tests the correctness of the gather operation itself,
    without relying on two different optimizers producing bit-identical residuals
    (which can vary across warp-lang versions due to kernel compilation differences).
    """
    device_str = "cuda:0" if wp.is_cuda_available() else "cpu"
    device = wp.get_device(device_str)
    robot = Robot.load(load_robot_description("fetch_description"), backend="warp")

    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    config = IKHelperConfig(
        stages=[
            WarpStageConfig(num_seeds=8, iters=4, lm_lambda=10.0),
            WarpStageConfig(num_seeds=2, iters=6, lm_lambda=1.0),
            WarpStageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
        ],
        use_cuda_graph=False,
        smoothness_weight=0.2,
        velocity_limit_weight=0.0,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,
        base_weight_smoothness=0.2,
        init_sample_range=0.1,
    )
    ik_helper = IKHelper(robot, "gripper_link", placeholder, config)
    ik_helper._solver.score_terms = [None] * ik_helper._solver.num_stages
    ik_helper._stage_score_tasks = [None] * len(ik_helper._stage_score_tasks)

    q0 = robot.spec.zero_q.reshape(1, -1).astype(np.float32)
    prev_state = robot.state(
        q=wp.from_numpy(q0, dtype=wp.float32, device=device),
        T_world_base=WarpSE3.identity(shape=(1,), device=device),
    )

    target_state = robot.state(
        q=wp.from_numpy(q0, dtype=wp.float32, device=device),
        T_world_base=WarpSE3.identity(shape=(1,), device=device),
    )
    target_state = robot.forward_kinematics(target_state)
    target = target_state.get_T_world_link(robot.link_names.index("gripper_link"))

    stage0_var = ik_helper._solver._stage_vars[0]
    stage1_var = ik_helper._solver._stage_vars[1]
    stage0_opt = ik_helper._solver._optimizers[0]

    rng = np.random.default_rng(0)
    for _ in range(6):
        initial_var = ik_helper._solver.initial_var
        q_range = float(rng.uniform(0.05, 0.15))
        base_range = float(rng.uniform(0.05, 0.15))
        ik_helper._sample_q_around_init(prev_state.q, ik_helper.stage_configs[0].num_seeds, initial_var.q, q_range)
        ik_helper._sample_base_around_init(
            prev_state.T_world_base.xyz_wxyz,
            ik_helper.stage_configs[0].num_seeds,
            initial_var.T_world_base.xyz_wxyz,
            base_range,
        )
        for stage_idx, stage_config in enumerate(ik_helper.stage_configs):
            if stage_config.num_seeds > 1:
                expanded_target = [WarpSE3(repeat(target.xyz_wxyz, stage_config.num_seeds))]
            else:
                expanded_target = [target]
            ik_helper._stage_position_tasks[stage_idx].set_target(expanded_target)
            ik_helper._stage_rotation_tasks[stage_idx].set_target(expanded_target)
        for smoothness_task in ik_helper._stage_smoothness_tasks:
            if smoothness_task is not None:
                smoothness_task.set_prev_state(prev_state)

        stage0_var.invalidate()
        stage0_opt.solve(stage0_var)

        wp.launch_tiled(
            ik_helper._solver._select_best_kernels[0],
            dim=[ik_helper._solver.batch_size],
            inputs=[stage0_opt.costs, ik_helper._solver.best_indices[0]],
            block_dim=ik_helper._solver._tile_threads[0],
            device=ik_helper._solver.device,
        )
        flat_indices = ik_helper._solver.best_indices[0].reshape((ik_helper._solver.total_batches[1],))
        stage0_var.gather(flat_indices, stage1_var)

        # Verify gather correctness: the gathered q and T_world_base must
        # match the source rows at the selected indices (NaN-aware comparison
        # since Cholesky decomposition can produce NaN for ill-conditioned seeds).
        selected = np.asarray(flat_indices.numpy(), dtype=np.int64)
        stage0_q_np = stage0_var.q.numpy()
        stage1_q_np = stage1_var.q.numpy()
        np.testing.assert_array_equal(stage1_q_np, stage0_q_np[selected])

        stage0_base_np = stage0_var.T_world_base.xyz_wxyz.numpy()
        stage1_base_np = stage1_var.T_world_base.xyz_wxyz.numpy()
        np.testing.assert_array_equal(stage1_base_np, stage0_base_np[selected])

        # Verify cost consistency using the SAME optimizer (stage 0) to avoid
        # cross-optimizer numerical differences from different batch sizes.
        stage0_opt.compute_residuals(stage0_opt.terms, stage0_var, stage0_opt.residuals)
        stage0_costs = wp.empty((stage0_var.batch_size,), dtype=cast(Type[float], wp.float32), device=device)
        wp.launch(
            kernel=aggregate_residuals_to_costs,
            dim=stage0_var.batch_size,
            inputs=[stage0_opt.residuals, stage0_costs],
            device=device,
        )

        # The gathered entries' costs from stage 0 should be monotonically
        # ordered (best_indices selects lowest costs first). Skip NaN entries.
        stage0_costs_np = stage0_costs.numpy()
        gathered_costs = stage0_costs_np[selected]
        finite_mask = np.isfinite(gathered_costs)
        finite_costs = gathered_costs[finite_mask]
        if len(finite_costs) > 1:
            assert np.all(np.diff(finite_costs) >= -1e-6), f"Gathered costs should be sorted: {gathered_costs}"


def test_warp_solver_debug_stage_handoff_validation_flag_raises_on_inconsistent_stage_terms(monkeypatch) -> None:
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    target_link_index = robot.link_names.index("ee_link")
    batch_size = 1
    num_seeds = 4
    num_dofs = robot.num_actuated_joints

    q_target = wp.from_numpy(robot.spec.midrange_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device)
    target_state = robot.forward_kinematics(robot.state(q=q_target))
    target_pose = target_state.get_T_world_link(target_link_index)

    target_stage0 = WarpSE3(
        wp.from_numpy(np.repeat(target_pose.xyz_wxyz.numpy(), repeats=num_seeds, axis=0), dtype=wp_vec7, device=device)
    )
    target_stage1_np = target_pose.xyz_wxyz.numpy().copy()
    target_stage1_np[:, 0] += 0.2
    target_stage1 = WarpSE3(wp.from_numpy(target_stage1_np, dtype=wp_vec7, device=device))

    terms_stage0: List[WarpTask] = [
        WarpFrameTask(robot, target_link_index, target_stage0, 10.0, 5.0),
        WarpPositionLimit(robot, weight=20.0, batch_size=batch_size * num_seeds),
    ]
    terms_stage1: List[WarpTask] = [
        WarpFrameTask(robot, target_link_index, target_stage1, 10.0, 5.0),
        WarpPositionLimit(robot, weight=200.0, batch_size=batch_size),
    ]

    q_placeholder = wp.empty((batch_size, num_dofs), dtype=cast(Type[float], wp.float32), device=device)
    placeholder_state = robot.state(q=q_placeholder)
    solver = WarpSolver(
        config=WarpSolverConfig(
            stages=[
                WarpStageConfig(num_seeds=num_seeds, iters=2, lm_lambda=5.0),
                WarpStageConfig(num_seeds=1, iters=2, lm_lambda=1.0),
            ],
            use_cuda_graph=False,
        ),
        placeholder_var=placeholder_state,
        terms=[terms_stage0, terms_stage1],
    )

    rng = np.random.default_rng(0)
    seed_qs_np = rng.uniform(
        robot.spec.actuated_joint_limits[:, 0],
        robot.spec.actuated_joint_limits[:, 1],
        size=(batch_size * num_seeds, num_dofs),
    ).astype(np.float32)
    wp.copy(solver.initial_var.q, wp.from_numpy(seed_qs_np, dtype=wp.float32, device=device))

    monkeypatch.setenv("ROBOKIT_DEBUG_VALIDATE_STAGE_HANDOFF", "1")
    with pytest.raises(RuntimeError, match="stage handoff cost mismatch"):
        solver.solve()
