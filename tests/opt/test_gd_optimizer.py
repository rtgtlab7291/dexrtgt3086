from typing import Optional

import numpy as np
import pytest
import warp as wp

from robokit.opt.gd_optimizer import GDOptimizer, GDOptimizerConfig
from robokit.opt.var_values import VarValues
from robokit.terms import FrameTask, PositionLimit
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.utils.warp_utils import wp_device_type, wp_vec7


def _target_pose(batch_size: int = 1, device: Optional[wp_device_type] = None) -> wp.array:
    target = np.tile([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0], (batch_size, 1)).astype(np.float32)
    return wp.from_numpy(target[:, None], dtype=wp_vec7, device=device)


class TestGDOptimizer:
    @pytest.mark.xfail(
        reason="Warp tiled matmul sporadically produces NaN with 6x6 dimensions (CPU CI, multiple python versions)",
        strict=False,
    )
    def test_cost_decreases(self, ur10_robot):
        robot = ur10_robot
        target_pose = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=50, learning_rate=1e-4, use_early_stopping=False)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        # record initial cost
        state = robot.state(q=q_init)
        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_cost = optimizer.costs.numpy()[0]

        # run GD
        state = robot.state(q=q_init)
        values, costs = optimizer.solve(VarValues(robot=state))
        state = values.get("robot")
        final_cost = costs.numpy()[0]

        assert costs is optimizer.costs
        assert final_cost < initial_cost, f"Cost should decrease: {initial_cost} -> {final_cost}"

    @pytest.mark.xfail(
        reason="Warp tiled matmul sporadically produces NaN with 6x6 dimensions (CPU CI, multiple python versions)",
        strict=False,
    )
    def test_multi_task(self, ur10_robot):
        robot = ur10_robot
        target_pose = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        terms = [
            FrameTask(
                robot=robot,
                frame_index=frame_index,
                T_world_target=target_pose,
                position_weight=10.0,
                orientation_weight=2.0,
            ),
            PositionLimit(robot=robot, weight=2.0),
        ]

        config = GDOptimizerConfig(max_iter=50, learning_rate=1e-4, use_early_stopping=False)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=terms, config=config)

        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"Cost should decrease: {initial_cost} -> {final_cost}"

    def test_early_stopping(self, ur10_robot):
        robot = ur10_robot
        target_pose = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        # very loose tolerance so it stops early
        config = GDOptimizerConfig(max_iter=1000, learning_rate=1e-4, cost_tol=100.0, use_early_stopping=True)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        assert optimizer.costs.numpy()[0] <= 100.0

    def test_batch(self, ur10_robot):
        batch_size = 4
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        robot = ur10_robot

        target_pose = _target_pose(batch_size, device)
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=20, learning_rate=1e-4, use_early_stopping=False)
        q_init = wp.from_numpy(
            np.tile(robot.spec.zero_q, (batch_size, 1)).astype(np.float32), dtype=wp.float32, device=device
        )
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config, device=device)

        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_costs = optimizer.costs.numpy().copy()

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        final_costs = optimizer.costs.numpy()

        assert optimizer.costs.shape == (batch_size,)
        assert np.all(final_costs < initial_costs), "All batch costs should decrease"

    def test_adam_cost_decreases(self, ur10_robot):
        robot = ur10_robot
        target_pose = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=50, learning_rate=0.01, optimizer_type="adam", use_early_stopping=False)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        state = robot.state(q=q_init)
        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"Adam cost should decrease: {initial_cost} -> {final_cost}"

    def test_adam_batch(self, ur10_robot):
        batch_size = 4
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        robot = ur10_robot

        target_pose = _target_pose(batch_size, device)
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=20, learning_rate=0.01, optimizer_type="adam", use_early_stopping=False)
        q_init = wp.from_numpy(
            np.tile(robot.spec.zero_q, (batch_size, 1)).astype(np.float32), dtype=wp.float32, device=device
        )
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config, device=device)

        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_costs = optimizer.costs.numpy().copy()

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        final_costs = optimizer.costs.numpy()

        assert optimizer.costs.shape == (batch_size,)
        assert np.all(final_costs < initial_costs), "All Adam batch costs should decrease"

    def test_adam_autodiff(self, ur10_robot):
        robot = ur10_robot
        target_pose = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(
            max_iter=50, learning_rate=0.01, optimizer_type="adam", gradient_mode="autodiff", use_early_stopping=False
        )
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"Adam autodiff cost should decrease: {initial_cost} -> {final_cost}"

    def test_adamw_cost_decreases(self, ur10_robot):
        robot = ur10_robot
        target_pose = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(
            max_iter=50, learning_rate=0.01, optimizer_type="adamw", weight_decay=0.01, use_early_stopping=False
        )
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        state = robot.state(q=q_init)
        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"AdamW cost should decrease: {initial_cost} -> {final_cost}"

    def test_autodiff_gradient(self, ur10_robot):
        robot = ur10_robot
        target_pose = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=50, learning_rate=0.001, gradient_mode="autodiff", use_early_stopping=False)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"Cost should decrease: {initial_cost} -> {final_cost}"


class TestLRSchedule:
    """Per-iter learning-rate schedule via `GDOptimizerConfig.lr_schedule`."""

    def _make_optimizer(self, panda_robot, *, lr_schedule, learning_rate, max_iter, optimizer_type):
        robot = panda_robot
        B, T, D = 1, 6, robot.num_actuated_joints
        q_np = np.broadcast_to(robot.spec.zero_q.astype(np.float32), (B, T, D)).copy()
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        smooth_task = TrajectorySmoothnessTask(robot=robot, num_frames=T, weight=1.0)
        config = GDOptimizerConfig(
            max_iter=max_iter,
            learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            optimizer_type=optimizer_type,
            weight_decay=0.0,
            gradient_mode="analytic_gradient",
            use_early_stopping=False,
        )
        return GDOptimizer(terms=[smooth_task], config=config), state

    def test_constant_when_unset(self, panda_robot):
        # Default behavior preserved: lr_schedule=None means cfg.learning_rate is used as-is each iter.
        optimizer, state = self._make_optimizer(
            panda_robot, lr_schedule=None, learning_rate=0.05, max_iter=5, optimizer_type="adamw"
        )
        optimizer.solve(VarValues(robot=state))[0].get("robot")
        assert optimizer._lr_iter == 5

    def test_lr_schedule_called_per_iter_with_zero_indexed_counter(self, panda_robot):
        # Schedule receives (iter_idx_zero_indexed, base_lr); iter_idx must be 0..max_iter-1.
        seen = []

        def record(it, base):
            seen.append((it, base))
            return base

        optimizer, state = self._make_optimizer(
            panda_robot, lr_schedule=record, learning_rate=0.07, max_iter=4, optimizer_type="adamw"
        )
        optimizer.solve(VarValues(robot=state))[0].get("robot")
        assert seen == [(0, 0.07), (1, 0.07), (2, 0.07), (3, 0.07)]

    def test_cosine_schedule_matches_torch(self, panda_robot):
        # Per-iter LR values produced by our schedule must match torch.optim.lr_scheduler.CosineAnnealingLR.
        torch = pytest.importorskip("torch")

        T_max, eta_min, base_lr = 20, 5e-5, 0.02

        def cosine_lr(it, base):
            import math

            return eta_min + 0.5 * (base - eta_min) * (1.0 + math.cos(math.pi * it / T_max))

        param = torch.zeros(1, requires_grad=True)
        torch_opt = torch.optim.AdamW([param], lr=base_lr)
        torch_sched = torch.optim.lr_scheduler.CosineAnnealingLR(torch_opt, T_max=T_max, eta_min=eta_min)
        torch_lrs = []
        for _ in range(T_max):
            torch_lrs.append(torch_opt.param_groups[0]["lr"])
            torch_sched.step()

        ours = [cosine_lr(it, base_lr) for it in range(T_max)]
        np.testing.assert_allclose(ours, torch_lrs, rtol=0, atol=1e-12)

    def test_solve_with_cosine_schedule_decreases_cost(self, panda_robot):
        # End-to-end smoke: cosine schedule applied through GDOptimizer.solve still drives cost down.
        import math

        T_max, eta_min, base_lr = 40, 5e-5, 0.05

        def cosine_lr(it, base):
            return eta_min + 0.5 * (base - eta_min) * (1.0 + math.cos(math.pi * it / T_max))

        np.random.seed(11)
        robot = panda_robot
        B, T, D = 1, 8, robot.num_actuated_joints
        q_np = np.broadcast_to(robot.spec.midrange_q.astype(np.float32), (B, T, D)).copy() + (
            np.random.randn(B, T, D) * 0.3
        ).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        smooth_task = TrajectorySmoothnessTask(robot=robot, num_frames=T, weight=1.0)
        config = GDOptimizerConfig(
            max_iter=T_max,
            learning_rate=base_lr,
            lr_schedule=cosine_lr,
            optimizer_type="adamw",
            weight_decay=0.0,
            gradient_mode="analytic_gradient",
            use_early_stopping=False,
        )
        optimizer = GDOptimizer(terms=[smooth_task], config=config)
        optimizer._build(VarValues(robot=state))
        optimizer._compute_cost_and_gradient(VarValues(robot=state))
        initial_cost = float(optimizer.costs.numpy()[0])
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        final_cost = float(optimizer.costs.numpy()[0])
        assert final_cost < initial_cost, f"cosine-schedule cost should decrease: {initial_cost} -> {final_cost}"
