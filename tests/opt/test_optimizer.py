from typing import Optional, Sequence

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizer, SparseLMOptimizerConfig
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms import FrameTask, PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.task import SparseTask, SparsityPattern
from robokit.utils.warp_utils import wp_device_type, wp_vec7


_robot_cache: dict = {}


def _target_pose(
    values: Sequence[float] = (0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0),
    batch_size: int = 1,
    device: Optional[wp_device_type] = None,
) -> wp.array:
    target = np.tile(np.asarray(values, dtype=np.float32), (batch_size, 1))
    return wp.from_numpy(target[:, None], dtype=wp_vec7, device=device)


def _get_robot(desc_name: str) -> Robot:
    if desc_name not in _robot_cache:
        _robot_cache[desc_name] = Robot.load(load_robot_description(desc_name))
    return _robot_cache[desc_name]


class _MutableResidualSparseTask(SparseTask):
    def __init__(self, residual_dim: int):
        self._residual_dim = residual_dim
        self.gain = 1.0
        self.residual_weight = None

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_residual(self, var, *args, out_residual=None, row_offset=0, **kwargs):
        del args, kwargs
        del row_offset
        if out_residual is not None:
            return out_residual
        return wp.zeros((var.batch_size, self.residual_dim), dtype=wp.float32, device=var.device)

    def compute_weighted_jacobian_analytic(self, var, *args, out_jacobian=None, row_offset=0, col_offset=0, **kwargs):
        del args, kwargs
        del row_offset, col_offset
        if out_jacobian is not None:
            return out_jacobian
        return wp.zeros((var.batch_size, self.residual_dim, var.tangent_dim), dtype=wp.float32, device=var.device)

    def compute_sparse_jacobian_pattern(self, var, offset=0, **kwargs):
        del kwargs
        row_indices = np.arange(self.residual_dim, dtype=np.int32) + int(offset)
        col_indices = np.zeros((self.residual_dim,), dtype=np.int32)
        pattern = SparsityPattern()
        pattern.row_indices = wp.from_numpy(row_indices, dtype=wp.int32, device=var.device)
        pattern.col_indices = wp.from_numpy(col_indices, dtype=wp.int32, device=var.device)
        return pattern

    def compute_weighted_sparse_jacobian_values(self, var, jacobian_values_buffer=None, offset=0, **kwargs):
        del offset, kwargs
        if jacobian_values_buffer is not None:
            return jacobian_values_buffer
        return wp.zeros((var.batch_size, self.residual_dim), dtype=wp.float32, device=var.device)


def test_ik():
    from robokit.terms.dense.frame_task import FrameTask
    from robokit.terms.dense.position_limit import PositionLimit

    robot = _get_robot("ur10_description")
    target_pose = _target_pose()
    frame_index = robot.link_names.index("ee_link")
    terms = [
        FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=1.0,
            orientation_weight=0.2,
        ),
        PositionLimit(robot=robot, weight=2.0),
    ]
    q_placeholder = wp.empty((1, robot.spec.num_actuated_joints), dtype=wp.float32)
    placeholder_state = robot.state(q=q_placeholder)
    config = LMOptimizerConfig(num_dofs=placeholder_state.tangent_dim, lm_lambda=1.0)
    ik_optimizer = LMOptimizer(terms=terms, config=config)
    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
    state = robot.state(q=q_init)
    state = ik_optimizer.solve(VarValues(robot=state))[0].get("robot")
    state = robot.forward_kinematics(state)
    achieved_pose = state.get_T_world_link(frame_index)
    assert np.allclose(achieved_pose.numpy(), target_pose.numpy()[:, 0], atol=1e-3)


def test_sparse_lm_optimizer_rejects_active_dof_mask_shape():
    robot = _get_robot("panda_description")
    q = wp.zeros((1, robot.num_actuated_joints), dtype=wp.float32)
    state = robot.state(q=q)
    active_dof_mask = wp.ones((state.tangent_dim + 1,), dtype=wp.float32, device=state.device)

    with pytest.raises(ValueError, match="active_dof_mask"):
        SparseLMOptimizer(
            term=_MutableResidualSparseTask(1),
            batch_size=1,
            total_tangent_dim=state.tangent_dim,
            device=state.device,
            active_dof_mask=active_dof_mask,
        )


def test_mobile_ik_with_base_pose():
    """Test mobile IK optimizing both robot q and T_world_base pose variables."""
    target_link_name = "panda_hand"
    robot = _get_robot("panda_description")

    batch_size = 1
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

    T_world_base_np = np.array([0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device)

    q_np = robot.spec.midrange_q
    q = wp.from_numpy(q_np.reshape(1, -1), dtype=wp.float32, device=device)
    state = robot.state(q=q, T_world_base=T_world_base)

    target_pose_np = np.array([0.6, 0.0, 0.55, 0.0, 0.707, 0.0, -0.707], dtype=np.float32).reshape(1, 7)
    target_pose = wp.from_numpy(target_pose_np[:, None], dtype=wp_vec7, device=device)
    frame_index = robot.link_names.index(target_link_name)

    terms = []

    frame_task = FrameTask(
        robot=robot,
        frame_index=frame_index,
        T_world_target=target_pose,
        position_weight=10.0,
        orientation_weight=2.0,
    )
    terms.append(frame_task)

    position_limit = PositionLimit(
        robot=robot,
        weight=2.0,
    )
    terms.append(position_limit)

    optimizer_config = LMOptimizerConfig(
        num_dofs=state.tangent_dim,
        max_iter=50,
        lm_lambda=1.0,
        lambda_factor=2.0,
        lambda_min=1e-6,
        lambda_max=1e6,
    )

    ik_optimizer = LMOptimizer(
        terms=terms,
        config=optimizer_config,
        device=device,
    )

    state = robot.state(q=q, T_world_base=T_world_base)
    state = ik_optimizer.solve(VarValues(robot=state))[0].get("robot")
    state = robot.forward_kinematics(state)

    achieved_pose = state.get_T_world_link(frame_index)
    assert np.allclose(achieved_pose.numpy()[..., :3], target_pose.numpy()[:, 0, :3], atol=1e-3)

    achieved_quat = achieved_pose.numpy()[..., 3:]
    target_quat = target_pose.numpy()[:, 0, 3:]
    dot_product = np.abs(np.sum(achieved_quat * target_quat))
    assert dot_product > 0.999, f"Quaternion mismatch: dot product {dot_product}"

    assert state.T_world_base is not None
    assert state.q.shape == (batch_size, robot.num_actuated_joints)


def test_ik_status():
    from robokit.terms.dense.frame_task import FrameTask
    from robokit.terms.dense.position_limit import PositionLimit

    robot = _get_robot("ur10_description")
    target_pose = _target_pose()
    frame_index = robot.link_names.index("ee_link")
    terms = [
        FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=1.0,
            orientation_weight=0.2,
        ),
        PositionLimit(robot=robot, weight=2.0),
    ]
    q_placeholder = wp.empty((1, robot.spec.num_actuated_joints), dtype=wp.float32)
    placeholder_state = robot.state(q=q_placeholder)
    config = LMOptimizerConfig(num_dofs=placeholder_state.tangent_dim, cost_tol=1.0)

    ik_optimizer = LMOptimizer(terms=terms, config=config)
    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
    state = robot.state(q=q_init)

    state = ik_optimizer.solve(VarValues(robot=state))[0].get("robot")
    costs_np = ik_optimizer.costs.numpy()
    assert costs_np[0] <= 1.0, "Optimization should succeed with loose tolerance"


def test_sparse_lm_optimizer_fails_fast_on_residual_layout_drift():
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    robot = _get_robot("ur10_description")
    q_init = wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device)
    state = robot.state(q=q_init)

    task = _MutableResidualSparseTask(residual_dim=1)
    optimizer = SparseLMOptimizer(
        term=[task],
        batch_size=state.batch_size,
        total_tangent_dim=state.tangent_dim,
        device=device,
        config=SparseLMOptimizerConfig(max_iter=1, use_early_stopping=False),
    )

    task._residual_dim = 2
    with pytest.raises(ValueError, match="residual layout mismatch"):
        optimizer.solve(VarValues(robot=state))[0].get("robot")


class TestLMOptimizerVariableBatchSize:
    @staticmethod
    def _build(batch_size: int, device):
        robot = _get_robot("ur10_description")
        target_pose = _target_pose(batch_size=batch_size, device=device)
        frame_index = robot.link_names.index("ee_link")
        task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )
        q = wp.from_numpy(
            np.tile(robot.spec.zero_q, (batch_size, 1)).astype(np.float32), dtype=wp.float32, device=device
        )
        state = robot.state(q=q)
        return robot, task, state

    def test_variable_batch_size(self):
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        robot, task1, state1 = self._build(1, device)

        config = LMOptimizerConfig(num_dofs=state1.tangent_dim, max_iter=5, use_early_stopping=False)
        optimizer = LMOptimizer(terms=[task1], config=config, device=device)
        optimizer.solve(VarValues(robot=state1))[0].get("robot")
        assert optimizer.costs.shape == (1,)

        _, task2, state2 = self._build(2, device)
        optimizer.solve(VarValues(robot=state2))[0].get("robot")
        assert optimizer.costs.shape == (2,)

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_cuda_graph_rejects_batch_change(self):
        device = wp.get_device("cuda:0")
        robot, task1, state1 = self._build(1, device)

        config = LMOptimizerConfig(
            num_dofs=state1.tangent_dim, max_iter=5, use_early_stopping=False, use_cuda_graph=True
        )
        optimizer = LMOptimizer(terms=[task1], config=config, device=device)
        optimizer.solve(VarValues(robot=state1))[0].get("robot")

        _, task2, state2 = self._build(2, device)
        with pytest.raises(ValueError, match="batch size changed"):
            optimizer.solve(VarValues(robot=state2))[0].get("robot")


# ==============================================================================
# Regression tests for Codex review (2026-04-23)
# ==============================================================================


def test_max_iter_zero_populates_current_cost():
    """With max_iter=0 the optimizer must populate self.costs with 0.5*||r||^2
    at the current state (not leave stale zeros). MultiSeedSolver reads this
    buffer as the stage score when the final stage has iters=0.
    """
    robot = _get_robot("ur10_description")
    # Target deliberately far from q=zero_q so cost is clearly nonzero.
    target_pose = _target_pose()
    frame_index = robot.link_names.index("ee_link")
    frame_task = FrameTask(
        robot=robot,
        frame_index=frame_index,
        T_world_target=target_pose,
        position_weight=1.0,
        orientation_weight=0.5,
    )
    q_placeholder = wp.empty((1, robot.spec.num_actuated_joints), dtype=wp.float32)
    placeholder_state = robot.state(q=q_placeholder)

    # Reference cost: run one iter of a fresh solver, read the current-state cost
    # it computes in _propose_body (before any accept).
    ref_opt = LMOptimizer(
        terms=[frame_task],
        config=LMOptimizerConfig(num_dofs=placeholder_state.tangent_dim, max_iter=1, use_early_stopping=False),
    )
    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
    state = robot.state(q=q_init)
    ref_opt.solve(VarValues(robot=state))[0].get("robot")
    # Re-run to compute the cost at q_init cleanly (solve mutates state).
    state = robot.state(q=q_init)
    ref_opt.solve(VarValues(robot=state))[0].get("robot")
    # Now build a baseline by computing the residual directly.
    residual_buf = wp.zeros((1, frame_task.residual_dim), dtype=wp.float32)
    state = robot.state(q=q_init)
    frame_task.compute_weighted_residual(VarValues(robot=state), out_residual=residual_buf, row_offset=0)
    expected_cost = 0.5 * float(np.sum(residual_buf.numpy() ** 2))
    assert expected_cost > 1e-3, "Test setup bug: initial cost should be clearly nonzero"

    # max_iter=0 optimizer: must report the same cost, not stale zeros.
    zero_opt = LMOptimizer(
        terms=[frame_task],
        config=LMOptimizerConfig(num_dofs=placeholder_state.tangent_dim, max_iter=0, use_early_stopping=False),
    )
    state = robot.state(q=q_init)
    _, costs = zero_opt.solve(VarValues(robot=state))
    got_cost = float(costs.numpy()[0])
    assert costs is zero_opt.costs
    assert abs(got_cost - expected_cost) / expected_cost < 1e-3, (
        f"max_iter=0 left stale costs: got {got_cost}, expected {expected_cost}"
    )


def test_active_dof_mask_matches_reduced_problem():
    """active_dof_mask must apply to the normal equations, not just zero δ.

    The correct behavior zeros frozen Jacobian columns before forming
    ``(JᵀJ + λI) δ = -Jᵀr``. The active-DOF step then matches the reduced
    problem's solution (active rows/cols of ``JᵀJ``, active entries of
    ``Jᵀr``). Post-solve δ-masking instead solves the full system and then
    zeros the frozen δ entries, which biases the active-DOF step by cross
    terms through the frozen columns.

    We verify this numerically: run one tiny-λ LM iteration with a mask,
    then compare ``optimizer.delta`` against the analytic reduced-problem
    solution built from the same ``J``, ``r``, ``λ``.
    """
    robot = _get_robot("ur10_description")
    target = _target_pose()
    frame_index = robot.link_names.index("ee_link")
    frame_task = FrameTask(
        robot=robot,
        frame_index=frame_index,
        T_world_target=target,
        position_weight=10.0,
        orientation_weight=2.0,
    )

    q_init = wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)
    placeholder_state = robot.state(q=q_init)
    tangent_dim = placeholder_state.tangent_dim
    device = placeholder_state.device

    # Freeze a joint that genuinely couples with the task (joint 1 = shoulder).
    frozen_idx = 1
    mask_np = np.ones((tangent_dim,), dtype=np.float32)
    mask_np[frozen_idx] = 0.0
    active_mask = wp.from_numpy(mask_np, dtype=wp.float32, device=device)

    lm_lambda = 1e-2
    cfg = LMOptimizerConfig(
        num_dofs=tangent_dim,
        max_iter=1,
        lm_lambda=lm_lambda,
        use_early_stopping=False,
        active_dof_mask=active_mask,
    )
    opt = LMOptimizer(terms=[frame_task], config=cfg, device=device)
    state = robot.state(q=q_init)
    opt.solve(VarValues(robot=state))[0].get("robot")

    # Expected: build J, r at q_init, apply mask, solve reduced system.
    R = frame_task.residual_dim
    state_ref = robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32, device=device))
    r_buf = wp.zeros((1, R), dtype=wp.float32, device=device)
    j_buf = wp.zeros((1, R, tangent_dim), dtype=wp.float32, device=device)
    vals_ref = VarValues(robot=state_ref)
    frame_task.compute_weighted_residual(vals_ref, out_residual=r_buf, row_offset=0)
    frame_task.compute_weighted_jacobian(vals_ref, out_jacobian=j_buf, row_offset=0)
    J = j_buf.numpy()[0]  # (R, D)
    r = r_buf.numpy()[0]  # (R,)
    # Mask Jacobian columns for frozen DOFs (the correct pre-mask behavior).
    J_masked = J * mask_np.reshape(1, -1)
    JtJ = J_masked.T @ J_masked
    Jtr = J_masked.T @ r
    A = JtJ + lm_lambda * np.eye(tangent_dim, dtype=np.float32)
    # The frozen rows/cols of A are [λ on diag, 0 elsewhere]; solve restricts
    # δ[frozen] = 0 automatically.
    expected_delta = np.linalg.solve(A, -Jtr).astype(np.float32)

    got_delta = opt.delta.numpy()[0]
    # Frozen entry must be zero in both.
    assert abs(got_delta[frozen_idx]) < 1e-6, f"δ[frozen] = {got_delta[frozen_idx]} (should be 0)"
    # Active entries must match the reduced-problem solution.
    active_mask_np = mask_np.astype(bool)
    diff = got_delta[active_mask_np] - expected_delta[active_mask_np]
    err = float(np.max(np.abs(diff)))
    assert err < 1e-4, (
        f"active-DOF δ doesn't match reduced problem; max-abs diff = {err:.3e}\n"
        f"got:      {got_delta[active_mask_np]}\n"
        f"expected: {expected_delta[active_mask_np]}"
    )


def test_active_dof_mask_freezes_world_base_translation():
    """LM ``active_dof_mask`` must hard-lock the **world** base translation for
    masked translation axes — not just zero the twist delta.

    The LM-delta mask alone leaves world xy free when the base rotates
    (``V(w)·v`` coupling in ``T_new = T_old @ exp_se3(xi)``). The
    ``integrate_floating_base_kernel`` projection ensures masked world
    translation axes snap back to their previous values regardless of
    rotation. We verify by placing a wrist target far in +x (unreachable
    from the locked base), masking world xy, and asserting the solved
    base xy stays at the reference.
    """
    robot = _get_robot("panda_description")
    target = _target_pose((1.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0))
    frame_index = robot.link_names.index("panda_hand")
    frame_task = PositionTask(robot=robot, frame_index=frame_index, T_world_target=target, weight=1.5)

    ref_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    q_init = wp.from_numpy(robot.spec.midrange_q.reshape(1, -1).astype(np.float32), dtype=wp.float32)
    base_init = wp.from_numpy(ref_base_np, dtype=wp_vec7)
    placeholder_state = robot.state(q=q_init, T_world_base=base_init)
    placeholder_state.has_floating_base = True
    tangent_dim = placeholder_state.tangent_dim
    device = placeholder_state.device

    # Freeze world-frame translation x and y (tangent indices 0, 1).
    mask_np = np.ones((tangent_dim,), dtype=np.float32)
    mask_np[0] = 0.0
    mask_np[1] = 0.0
    active_mask = wp.from_numpy(mask_np, dtype=wp.float32, device=device)

    cfg = LMOptimizerConfig(
        num_dofs=tangent_dim,
        max_iter=20,
        lm_lambda=1.0,
        use_early_stopping=False,
        active_dof_mask=active_mask,
    )
    opt = LMOptimizer(terms=[frame_task], config=cfg, device=device)
    state = robot.state(q=q_init, T_world_base=wp.from_numpy(ref_base_np.copy(), dtype=wp_vec7))
    state.has_floating_base = True
    opt.solve(VarValues(robot=state))[0].get("robot")

    solved_xy = state.T_world_base.numpy()[0, :2]
    assert np.linalg.norm(solved_xy) < 1e-4, f"base xy should be locked at (0, 0), got {solved_xy}"


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_chunked_accum_solve_matches_dense_normal_equations():
    """For R > _LM_FUSED_MAX_ROWS the wide Jacobian overflows one block's shared memory, so the solve
    streams JᵀJ in row-blocks (_accum_kernel_chunked) then solves the small system (_cholesky_kernel).
    The damped step must match the dense numpy normal-equations solve."""
    from robokit.opt.lm_optimizer import (
        _LM_FUSED_MAX_ROWS,
        _LM_SOLVE_CHUNK,
        TILE_THREADS,
        _accum_kernel_chunked,
        _cholesky_kernel,
    )

    dev = wp.get_device("cuda:0")
    R, D, lam = _LM_FUSED_MAX_ROWS + 400, 35, 0.7  # > threshold forces the chunked path
    C = _LM_SOLVE_CHUNK
    RP = -(-R // C) * C  # pad rows to a multiple of the chunk size
    NC = RP // C
    rng = np.random.default_rng(0)
    Jr = rng.standard_normal((R, D)).astype(np.float32)
    rr = rng.standard_normal(R).astype(np.float32)
    Jp = np.zeros((RP, D), np.float32)
    Jp[:R] = Jr
    rp = np.zeros((RP, 1), np.float32)
    rp[:R, 0] = rr
    J = wp.from_numpy(Jp.reshape(1, NC, C, D), dtype=wp.float32, device=dev)
    rc = wp.from_numpy(rp.reshape(1, NC, C, 1), dtype=wp.float32, device=dev)
    H = wp.zeros((1, D, D), dtype=wp.float32, device=dev)
    g = wp.zeros((1, D, 1), dtype=wp.float32, device=dev)
    costs = wp.zeros((1,), dtype=wp.float32, device=dev)
    lm = wp.from_numpy(np.array([lam], np.float32), dtype=wp.float32, device=dev)
    delta = wp.zeros((1, D), dtype=wp.float32, device=dev)
    pred = wp.zeros((1,), dtype=wp.float32, device=dev)
    wp.launch_tiled(
        _accum_kernel_chunked(D, C),
        dim=[1],
        inputs=[J, rc, NC],
        outputs=[H, g, costs],
        block_dim=TILE_THREADS,
        device=dev,
    )
    wp.launch_tiled(
        _cholesky_kernel(D), dim=[1], inputs=[H, g, lm], outputs=[delta, pred], block_dim=TILE_THREADS, device=dev
    )
    delta_ref = np.linalg.solve(Jr.T @ Jr + lam * np.eye(D, dtype=np.float32), -(Jr.T @ rr))
    np.testing.assert_allclose(delta.numpy()[0], delta_ref, rtol=0, atol=2e-3)
    assert abs(float(costs.numpy()[0]) - 0.5 * float(rr @ rr)) < 1e-1
