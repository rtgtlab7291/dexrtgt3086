"""Unit tests for Warp backend terms."""

from typing import Literal

import numpy as np
import pytest
import trimesh
import warp as wp
from scipy.spatial.transform import Rotation

from robokit.geom import BoxGeom, MeshGeom, SphereGeom, VolumeGeom, WarpScene
from robokit.geom.sdf_volume import mesh_to_sdf_volume
from robokit.lie.se3 import SE3Var, se3_identity
from robokit.opt.var_values import VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.manipulability_task import ManipulabilityTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.rotation_task import AxisLimitTask, RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.utils.warp_utils import stack, wp_device_type, wp_vec7


def _make_box_mesh(device: wp_device_type, half_extent: float) -> wp.Mesh:
    he = float(half_extent)
    v = np.array(
        [
            [-he, -he, -he],
            [he, -he, -he],
            [he, he, -he],
            [-he, he, -he],
            [-he, -he, he],
            [he, -he, he],
            [he, he, he],
            [-he, he, he],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 5, 1],
            [0, 4, 5],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int32,
    )
    return wp.Mesh(
        points=wp.array(v, dtype=wp.vec3, device=device),
        indices=wp.array(np.ravel(f), dtype=int, device=device),
    )


def _make_plane_mesh_x(device: wp_device_type, x: float, half_extent: float) -> wp.Mesh:
    he = float(half_extent)
    x0 = float(x)
    v = np.array(
        [
            [x0, -he, -he],
            [x0, he, -he],
            [x0, he, he],
            [x0, -he, he],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )
    return wp.Mesh(
        points=wp.array(v, dtype=wp.vec3, device=device),
        indices=wp.array(np.ravel(f), dtype=int, device=device),
    )


class TestWarpPositionLimit:
    """Test PositionLimit with and without floating base."""

    def test_position_limit_fixed_base(self, panda_robot):
        robot = panda_robot
        q_np = robot.spec.zero_q
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)

        position_limit = PositionLimit(robot=robot, weight=1.0)
        residual = position_limit.compute_weighted_residual(VarValues(robot=state))
        jacobian = position_limit.compute_weighted_jacobian(VarValues(robot=state))

        assert residual.shape == (1, robot.num_actuated_joints)
        assert jacobian.shape == (1, robot.num_actuated_joints, robot.num_actuated_joints)

        # at zero_q, residual should be small (depends on limits)
        residual_np = residual.numpy()
        assert np.all(residual_np >= 0)

    def test_position_limit_floating_base(self, panda_robot):
        robot = panda_robot
        q_np = robot.spec.zero_q
        q = wp.from_numpy(q_np, dtype=wp.float32)
        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np.reshape(1, 7), dtype=wp_vec7)
        state = robot.state(q=q, T_world_base=T_base)

        position_limit = PositionLimit(robot=robot, weight=1.0)
        residual = position_limit.compute_weighted_residual(VarValues(robot=state))
        jacobian = position_limit.compute_weighted_jacobian(VarValues(robot=state))

        assert residual.shape == (1, robot.num_actuated_joints)
        # jacobian should have base columns (6) + joint columns
        assert jacobian.shape == (1, robot.num_actuated_joints, 6 + robot.num_actuated_joints)


class TestWarpPositionLimitCostAndGradient:
    """Validate compute_weighted_cost_and_gradient (Gauss-Newton form: cost = 0.5*||r||^2, grad = J^T r)."""

    def _setup_violating(self, robot, weight=1.0, residual_mode: Literal["abs", "sqrt_abs", "exp_barrier"] = "abs"):
        device = wp.get_device("cpu")
        D = robot.num_actuated_joints
        limits = robot.spec.actuated_joint_limits.astype(np.float32)
        # B=2: row 0 violates upper by 0.1 on every dof, row 1 violates lower by 0.05.
        q_np = np.zeros((2, D), dtype=np.float32)
        q_np[0] = limits[:, 1] + 0.1
        q_np[1] = limits[:, 0] - 0.05
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device))
        task = PositionLimit(robot=robot, weight=weight, residual_mode=residual_mode)
        cost_buf = wp.zeros((2,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((2, state.tangent_dim), dtype=wp.float32, device=device)
        return state, task, cost_buf, grad_buf, device

    def test_cost_matches_residual_form(self, panda_robot):
        state, task, cost_buf, grad_buf, _ = self._setup_violating(panda_robot, weight=2.0)
        residual = task.compute_weighted_residual(VarValues(robot=state))
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        expected = 0.5 * (residual.numpy() ** 2).sum(axis=-1)
        assert np.allclose(cost_buf.numpy(), expected, rtol=1e-5, atol=1e-6), (
            f"cost mismatch: got {cost_buf.numpy()} expected {expected}"
        )

    def test_gradient_matches_jacobian_times_residual(self, panda_robot):
        state, task, cost_buf, grad_buf, _ = self._setup_violating(panda_robot, weight=2.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        expected = np.einsum("bri,br->bi", jacobian, residual)
        assert np.allclose(grad_buf.numpy(), expected, rtol=1e-5, atol=1e-6), (
            f"max abs err {np.max(np.abs(grad_buf.numpy() - expected)):.3e}"
        )

    @pytest.mark.parametrize("residual_mode", ["abs", "exp_barrier"])
    def test_gradient_matches_autodiff(self, panda_robot, residual_mode):
        state, task, cost_buf, grad_buf, device = self._setup_violating(
            panda_robot, weight=2.0, residual_mode=residual_mode
        )
        velocity = wp.zeros_like(state.q, requires_grad=True)
        proposed = state.integrate(velocity)
        residual_buf = wp.zeros(
            (state.batch_size, task.residual_dim), dtype=wp.float32, device=device, requires_grad=True
        )
        tape = wp.Tape()
        with tape:
            proposed_var = state.integrate(velocity, out=proposed)
            r = task.compute_weighted_residual(VarValues(robot=proposed_var), out_residual=residual_buf)
        tape.backward(grads={r: r})
        autodiff_grad = velocity.grad.numpy()
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        assert np.allclose(grad_buf.numpy(), autodiff_grad, rtol=1e-4, atol=1e-5), (
            f"max abs err {np.max(np.abs(grad_buf.numpy() - autodiff_grad)):.3e}"
        )

    def test_gradient_matches_finite_differences(self, panda_robot):
        state, task, cost_buf, grad_buf, device = self._setup_violating(panda_robot, weight=1.5)
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        analytic = grad_buf.numpy()

        q_np = state.q.numpy().copy()
        eps = 1e-4
        np.random.seed(0)
        D = q_np.shape[-1]
        sample_pairs = [(b, d) for b in range(q_np.shape[0]) for d in np.random.choice(D, size=4, replace=False)]
        max_err = 0.0
        for b, d in sample_pairs:
            qp = q_np.copy()
            qp[b, d] += eps
            sp = panda_robot.state(q=wp.from_numpy(qp, dtype=wp.float32, device=device))
            cb_p = wp.zeros_like(cost_buf)
            gb_p = wp.zeros_like(grad_buf)
            task.compute_weighted_cost_and_gradient(VarValues(robot=sp), out_cost=cb_p, out_gradient=gb_p)
            cp = cb_p.numpy()[b]

            qm = q_np.copy()
            qm[b, d] -= eps
            sm = panda_robot.state(q=wp.from_numpy(qm, dtype=wp.float32, device=device))
            cb_m = wp.zeros_like(cost_buf)
            gb_m = wp.zeros_like(grad_buf)
            task.compute_weighted_cost_and_gradient(VarValues(robot=sm), out_cost=cb_m, out_gradient=gb_m)
            cm = cb_m.numpy()[b]

            fd = (cp - cm) / (2.0 * eps)
            err = abs(fd - float(analytic[b, d]))
            max_err = max(max_err, err)
        assert max_err < 5e-3, f"finite-diff mismatch: max abs err {max_err:.3e}"

    def test_zero_when_inactive(self, panda_robot):
        device = wp.get_device("cpu")
        q = wp.from_numpy(panda_robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        state = panda_robot.state(q=q)
        task = PositionLimit(robot=panda_robot, weight=1.0)
        cost_buf = wp.zeros((1,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        assert np.max(np.abs(cost_buf.numpy())) < 1e-6
        assert np.max(np.abs(grad_buf.numpy())) < 1e-6

    def test_accumulation_into_shared_buffers(self, panda_robot):
        state, task, cost_buf, grad_buf, _ = self._setup_violating(panda_robot, weight=1.0)
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        single_cost = cost_buf.numpy().copy()
        single_grad = grad_buf.numpy().copy()
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        assert np.allclose(cost_buf.numpy(), 2.0 * single_cost, rtol=1e-5, atol=1e-6)
        assert np.allclose(grad_buf.numpy(), 2.0 * single_grad, rtol=1e-5, atol=1e-6)

    def test_floating_base_column_offset(self, panda_robot):
        device = wp.get_device("cpu")
        D = panda_robot.num_actuated_joints
        limits = panda_robot.spec.actuated_joint_limits.astype(np.float32)
        q_np = (limits[:, 1] + 0.1).reshape(1, D)
        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7, device=device)
        state = panda_robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device), T_world_base=T_base)
        task = PositionLimit(robot=panda_robot, weight=1.0)
        cost_buf = wp.zeros((1,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        grad = grad_buf.numpy()
        # First 6 columns are the floating base; position-limit must not write there.
        assert np.max(np.abs(grad[:, :6])) < 1e-6, f"floating-base columns should be zero, got {grad[:, :6]}"
        assert np.max(np.abs(grad[:, 6:])) > 1e-3, "actuated-joint columns must have non-zero gradient"


class TestWarpRestTask:
    """Test RestTask."""

    def test_rest_task_fixed_base(self, panda_robot):
        robot = panda_robot
        rest_q = robot.spec.midrange_q
        q = wp.from_numpy(rest_q + 0.1, dtype=wp.float32)
        state = robot.state(q=q)

        rest_task = RestTask(robot=robot, rest_q=rest_q, weight=0.1)
        residual = rest_task.compute_weighted_residual(VarValues(robot=state))

        # residual should be weighted 0.1
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        expected = 0.1 * 0.1  # diff * weight
        assert np.allclose(residual_np, expected, atol=1e-5)

    def test_rest_task_fixed_base_batched(self, panda_robot):
        robot = panda_robot
        batch_size = 4
        num_seeds = 2
        total_batch = batch_size * num_seeds
        rest_q = robot.spec.midrange_q
        q_np = np.tile(rest_q + 0.1, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))

        rest_task = RestTask(robot=robot, rest_q=rest_q, weight=0.1)
        residual = rest_task.compute_weighted_residual(VarValues(robot=state))

        assert residual.shape == (total_batch, robot.num_actuated_joints)
        residual_np = residual.numpy()
        expected = 0.1 * 0.1
        assert np.allclose(residual_np, expected, atol=1e-5)

        # verify set_rest_state with correct batch size
        new_rest_q = wp.from_numpy(np.tile(rest_q + 0.05, (batch_size, 1)).astype(np.float32), dtype=wp.float32)
        rest_task.set_rest_state(new_rest_q)
        residual2 = rest_task.compute_weighted_residual(VarValues(robot=state))
        expected2 = 0.05 * 0.1
        assert np.allclose(residual2.numpy(), expected2, atol=1e-5)

    def test_rest_task_base_only(self, panda_robot):
        robot = panda_robot
        T_world_base = se3_identity(shape=(1,))
        state = robot.state(q=robot.zero_q, T_world_base=T_world_base)
        task = RestTask(
            robot=robot,
            T_world_base_rest=T_world_base,
            weight=0.0,
            base_weight=1.0,
            include_joints=False,
        )

        residual = task.compute_weighted_residual(VarValues(robot=state))
        jacobian = task.compute_weighted_jacobian_analytic(VarValues(robot=state))

        assert residual.shape == (1, 6)
        assert jacobian.shape == (1, 6, state.tangent_dim)
        assert np.allclose(residual.numpy(), 0.0)


class TestWarpSmoothnessTask:
    """Test SmoothnessTask."""

    def test_smoothness_task_fixed_base(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        state_prev = robot.state(q=q_prev)
        state_curr = robot.state(q=q_curr)

        smooth_task = SmoothnessTask(robot=robot, weight=0.5)
        residual = smooth_task.compute_weighted_residual(VarValues(robot=state_curr), state_prev)

        # residual should be weighted difference
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        expected = 0.1 * 0.5  # diff * weight
        assert np.allclose(residual_np, expected, atol=1e-5)

    def test_smoothness_task_no_prev_state(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q)

        smooth_task = SmoothnessTask(robot=robot, weight=0.5)
        residual = smooth_task.compute_weighted_residual(VarValues(robot=state))

        # should return buffer (already zeroed by caller or empty)
        assert residual is not None

    def test_smoothness_task_floating_base_residual_dim_is_static(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)
        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)
        state_prev = robot.state(q=q_prev, T_world_base=T_base)
        state_curr = robot.state(q=q_curr, T_world_base=T_base)

        smooth_task = SmoothnessTask(
            robot=robot,
            prev_var=state_prev,
            weight=0.5,
            base_weight=0.2,
        )
        residual_dim_before = smooth_task.residual_dim

        residual = smooth_task.compute_weighted_residual(VarValues(robot=state_curr))
        jacobian = smooth_task.compute_weighted_jacobian(VarValues(robot=state_curr))

        assert residual_dim_before == robot.num_actuated_joints + 6
        assert smooth_task.residual_dim == residual_dim_before
        assert residual.shape == (1, residual_dim_before)
        assert jacobian.shape == (1, residual_dim_before, 6 + robot.num_actuated_joints)

    def test_smoothness_task_residual_dim_static_after_set_prev_state_with_floating_base(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)
        state_prev = robot.state(q=q_prev, T_world_base=T_base)
        state_curr = robot.state(q=q_curr, T_world_base=T_base)

        smooth_task = SmoothnessTask(robot=robot, prev_var=None, weight=0.5, base_weight=None)
        residual_dim_before = smooth_task.residual_dim
        smooth_task.set_prev_state(state_prev)
        residual = smooth_task.compute_weighted_residual(VarValues(robot=state_curr))
        jacobian = smooth_task.compute_weighted_jacobian(VarValues(robot=state_curr))

        assert residual_dim_before == robot.num_actuated_joints
        assert smooth_task.residual_dim == residual_dim_before
        assert residual.shape == (1, residual_dim_before)
        assert jacobian.shape == (1, residual_dim_before, 6 + robot.num_actuated_joints)

    def test_smoothness_task_base_enabled_rejects_nonfloating_var(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)
        state_prev = robot.state(q=q_prev, T_world_base=T_base)
        state_curr = robot.state(q=q_curr)

        smooth_task = SmoothnessTask(
            robot=robot,
            prev_var=state_prev,
            weight=0.5,
            base_weight=0.2,
        )
        with pytest.raises(ValueError, match="configured with base residuals"):
            smooth_task.compute_weighted_residual(VarValues(robot=state_curr))


class TestWarpFrameVectorTask:
    """Tests for the unified pair-vector alignment task.

    Inherits the historical FrameVectorDistanceTask cases (Huber + gating)
    and adds an L2-mode case to cover the FrameRetargetingTask path.
    """

    def _make_state(self, robot):
        q_values = wp.from_numpy(robot.spec.zero_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q_values)
        robot.forward_kinematics(state)
        return state

    def _make_robot_vector(self, state, origin_link_index, task_link_index):
        all_link_positions = state.T_world_link.numpy()[0, :, :3]
        origin_position = all_link_positions[origin_link_index]
        task_position = all_link_positions[task_link_index]
        return task_position - origin_position

    def test_soft_gate_validation(self, panda_robot):
        dummy_targets = np.zeros((1, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="must both be set"):
            FrameVectorTask(
                robot=panda_robot,
                origin_link_indices=[0],
                task_link_indices=[0],
                targets=dummy_targets,
                soft_gate_start_distance=0.02,
            )
        with pytest.raises(ValueError, match="must be greater"):
            FrameVectorTask(
                robot=panda_robot,
                origin_link_indices=[0],
                task_link_indices=[0],
                targets=dummy_targets,
                soft_gate_start_distance=0.02,
                soft_gate_full_distance=0.02,
            )
        with pytest.raises(ValueError, match="non-negative"):
            FrameVectorTask(
                robot=panda_robot,
                origin_link_indices=[0],
                task_link_indices=[0],
                targets=dummy_targets,
                soft_gate_start_distance=0.02,
                soft_gate_full_distance=-0.01,
            )

    def test_soft_gate_residual_transitions(self, panda_robot):
        state = self._make_state(panda_robot)
        origin_link_index = panda_robot.link_names.index("panda_link0")
        task_link_index = panda_robot.link_names.index("panda_hand")
        robot_vector = self._make_robot_vector(state, origin_link_index, task_link_index)
        robot_vector_norm = float(np.linalg.norm(robot_vector))
        robot_direction = robot_vector / robot_vector_norm
        reference_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(robot_direction, reference_direction))) > 0.95:
            reference_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        orthogonal_direction = np.cross(robot_direction, reference_direction)
        orthogonal_direction = orthogonal_direction / np.linalg.norm(orthogonal_direction)

        perturbation = 0.02
        target_far = robot_vector + perturbation * robot_direction
        target_mid = robot_vector + perturbation * orthogonal_direction
        target_near = robot_vector - perturbation * robot_direction

        soft_gate_start_distance = robot_vector_norm + 0.5 * perturbation
        soft_gate_full_distance = robot_vector_norm - 0.5 * perturbation

        task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=np.array([target_far], dtype=np.float32),
            huber_delta=0.05,
            huber_on_norm=True,
            soft_gate_start_distance=soft_gate_start_distance,
            soft_gate_full_distance=soft_gate_full_distance,
        )
        task.init_buffers(state.q.device)

        residual_far = float(task.compute_weighted_residual(VarValues(robot=state)).numpy()[0, 0])
        task.set_targets(np.array([target_mid], dtype=np.float32))
        residual_mid = float(task.compute_weighted_residual(VarValues(robot=state)).numpy()[0, 0])
        task.set_targets(np.array([target_near], dtype=np.float32))
        residual_near = float(task.compute_weighted_residual(VarValues(robot=state)).numpy()[0, 0])

        assert residual_far > 0.0
        assert residual_near > residual_mid
        assert residual_mid > residual_far

    def test_soft_gate_disabled_keeps_legacy_behavior(self, panda_robot):
        state = self._make_state(panda_robot)
        origin_link_index = panda_robot.link_names.index("panda_link0")
        task_link_index = panda_robot.link_names.index("panda_hand")
        robot_vector = self._make_robot_vector(state, origin_link_index, task_link_index)
        robot_direction = robot_vector / np.linalg.norm(robot_vector)
        target_vector = robot_vector + 0.01 * robot_direction
        target_vectors = np.array([target_vector], dtype=np.float32)

        legacy_task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=target_vectors,
            huber_delta=0.05,
            huber_on_norm=True,
        )
        gated_task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=target_vectors,
            huber_delta=0.05,
            huber_on_norm=True,
            soft_gate_start_distance=30.0,
            soft_gate_full_distance=29.0,
        )

        legacy_residual = legacy_task.compute_weighted_residual(VarValues(robot=state)).numpy()
        gated_residual = gated_task.compute_weighted_residual(VarValues(robot=state)).numpy()
        legacy_jacobian = legacy_task.compute_weighted_jacobian(VarValues(robot=state)).numpy()
        gated_jacobian = gated_task.compute_weighted_jacobian(VarValues(robot=state)).numpy()

        assert np.allclose(legacy_residual, gated_residual, atol=1e-6, rtol=1e-6)
        assert np.allclose(legacy_jacobian, gated_jacobian, atol=1e-6, rtol=1e-6)

    def test_l2_mode_three_channel_residual(self, panda_robot):
        """`huber_delta=None` emits a 3-channel L2 residual per pair."""
        state = self._make_state(panda_robot)
        origin_link_index = panda_robot.link_names.index("panda_link0")
        task_link_index = panda_robot.link_names.index("panda_hand")
        robot_vector = self._make_robot_vector(state, origin_link_index, task_link_index)
        target = robot_vector + np.array([0.02, -0.01, 0.005], dtype=np.float32)

        task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=np.array([target], dtype=np.float32),
            huber_delta=None,
        )
        task.init_buffers(state.q.device)

        assert task.residual_dim == 3
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]
        # per-pair diff = robot - target, weight_scale = sqrt(1.0) = 1.0
        expected = robot_vector - target
        np.testing.assert_allclose(residual, expected, atol=1e-5)

    def test_direction_task(self, panda_robot):
        state = self._make_state(panda_robot)
        origin_link_index = panda_robot.link_names.index("panda_link0")
        task_link_index = panda_robot.link_names.index("panda_hand")
        robot_vector = self._make_robot_vector(state, origin_link_index, task_link_index)
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(robot_vector / np.linalg.norm(robot_vector), reference))) > 0.95:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        target = np.cross(robot_vector, reference)

        task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=np.array([target], dtype=np.float32),
            direction_only=True,
        )
        expected = robot_vector - target / np.linalg.norm(target) * np.linalg.norm(robot_vector)
        np.testing.assert_allclose(
            task.compute_weighted_residual(VarValues(robot=state)).numpy(), [expected], atol=1e-5
        )
        analytic = task.compute_weighted_jacobian(VarValues(robot=state)).numpy()
        autodiff = autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy()
        np.testing.assert_allclose(analytic, autodiff, atol=1e-5)


class TestWarpVelocityLimitTask:
    """Test VelocityLimitTask."""

    def test_velocity_limit_task(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.5, dtype=wp.float32)  # large change

        state_prev = robot.state(q=q_prev)
        state_curr = robot.state(q=q_curr)

        dt = 0.1
        vel_task = VelocityLimitTask(robot=robot, dt=dt, weight=1.0)
        residual = vel_task.compute_weighted_residual(VarValues(robot=state_curr), state_prev)
        jacobian = vel_task.compute_weighted_jacobian(VarValues(robot=state_curr), state_prev)

        # residual should be positive where velocity exceeds limits
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        assert np.all(residual_np >= 0)

        # jacobian shape should be correct
        assert jacobian.shape == (1, robot.num_actuated_joints, robot.num_actuated_joints)


class TestWarpIntegration:
    """Test RobotState.integrate()."""

    def test_integrate_fixed_base_preserves_world_base(self, panda_robot):
        q = wp.from_numpy(panda_robot.spec.zero_q, dtype=wp.float32)
        state = panda_robot.state(q=q)
        expected = np.array([[0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        wp.copy(state.T_world_base, wp.from_numpy(expected, dtype=wp_vec7))
        out = panda_robot.state(q=wp.clone(state.q))

        state.integrate(wp.zeros((1, panda_robot.num_actuated_joints), dtype=wp.float32), out=out)

        np.testing.assert_array_equal(out.T_world_base.numpy(), expected)

    def test_integrate_floating_base(self, panda_robot):
        robot = panda_robot
        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np.reshape(1, 7), dtype=wp_vec7)
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q, T_world_base=T_base)

        # create velocity with base motion
        velocity_np = np.zeros(6 + robot.num_actuated_joints, dtype=np.float32)
        velocity_np[0] = 0.1  # move base in x
        velocity = wp.from_numpy(velocity_np.reshape(1, -1), dtype=wp.float32)

        new_state = state.integrate(velocity)

        # check base has moved
        new_T_base_np = new_state.T_world_base.numpy()[0]
        assert new_T_base_np[0] > 0  # x should have increased


class TestWarpAutodiffJacobian:
    """Test consistency between analytic and autodiff Jacobians for Warp terms."""

    def test_frame_task_jacobian(self, panda_robot):
        robot = panda_robot
        q_np = np.random.randn(robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        q = wp.from_numpy(q_np, dtype=wp.float32)

        T_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_base)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")

        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        frame_task = FrameTask(
            robot=robot,
            frame_index=ee_idx,
            T_world_target=T_world_target.reshape((1, 1)),
            position_weight=1.0,
            orientation_weight=1.0,
        )

        analytic_jacobian = frame_task.compute_weighted_jacobian_analytic(VarValues(robot=state))
        autodiff_jacobian = autodiff_weighted_jacobian(frame_task, VarValues(robot=state))

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4, rtol=1e-4)

    def test_collision_task_jacobian(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        assert robot.spec.has_collision_spheres

        mesh = _make_box_mesh(device=device, half_extent=2.0)

        np.random.seed(0)
        q_np = np.random.randn(robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        q = wp.from_numpy(q_np, dtype=wp.float32, device=device)
        state = robot.state(q=q)
        state = robot.forward_kinematics(state)
        state = robot.transform_collision_spheres(state)
        state = robot.compute_motion_subspace(state)

        sphere_indices = list(range(min(6, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        wm = WarpScene(1, mesh.device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=0.01,
            sphere_indices=sphere_indices,
        )

        task.compute_weighted_residual(VarValues(robot=state))
        analytic_jacobian = task.compute_weighted_jacobian_analytic(VarValues(robot=state))
        autodiff_jacobian = autodiff_weighted_jacobian(task, VarValues(robot=state))

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert analytic_np.shape == autodiff_np.shape
        assert np.allclose(analytic_np, autodiff_np, atol=3e-4, rtol=3e-4)


class TestWarpCollisionTaskResidual:
    def test_residual_near_zero_when_far_from_mesh(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        assert robot.spec.has_collision_spheres

        mesh = _make_box_mesh(device=device, half_extent=2.0)

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_base_np = np.array([[100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7, device=device)
        state = robot.state(q=q, T_world_base=T_base)
        state = robot.transform_collision_spheres(state)

        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        wm = WarpScene(1, mesh.device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=0.0,
            sphere_indices=sphere_indices,
        )

        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]
        assert residual.shape == (len(sphere_indices),)
        assert float(np.max(residual)) < 1e-6

    def test_residual_positive_when_surface_within_radius(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        assert robot.spec.has_collision_spheres

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7, device=device)
        state = robot.forward_kinematics(robot.state(q=q, T_world_base=T_base))
        state = robot.transform_collision_spheres(state)

        centers_np = state.collision_sphere_centers_world.numpy()[0]
        c = centers_np[0]
        r = float(robot.spec.local_collision_sphere_radii[0])

        # Place a plane at distance r/2 from the sphere center along +x.
        mesh = _make_plane_mesh_x(device=device, x=float(c[0]) + 0.5 * r, half_extent=10.0)

        sphere_indices = [0]
        wm = WarpScene(1, mesh.device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=0.0,
            sphere_indices=sphere_indices,
        )

        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]
        assert residual.shape == (1,)
        assert float(residual[0]) > 1e-4

    def test_residual_near_zero_when_surface_farther_than_radius(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        assert robot.spec.has_collision_spheres

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7, device=device)
        state = robot.forward_kinematics(robot.state(q=q, T_world_base=T_base))
        state = robot.transform_collision_spheres(state)

        centers_np = state.collision_sphere_centers_world.numpy()[0]
        c = centers_np[0]
        r = float(robot.spec.local_collision_sphere_radii[0])

        mesh = _make_plane_mesh_x(device=device, x=float(c[0]) + 10.0 * r, half_extent=10.0)

        wm = WarpScene(1, mesh.device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=0.0,
            sphere_indices=[0],
        )

        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]
        assert residual.shape == (1,)
        assert float(residual[0]) < 1e-6

    def test_position_limit_jacobian(self, panda_robot):
        robot = panda_robot
        q_np = (np.zeros(robot.num_actuated_joints) + 10.0).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        pos_limit = PositionLimit(robot=robot, weight=1.0)

        analytic_jacobian = pos_limit.compute_weighted_jacobian_analytic(VarValues(robot=state))
        autodiff_jacobian = autodiff_weighted_jacobian(pos_limit, VarValues(robot=state))

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4)

    def test_rest_task_jacobian_fixed_base(self, panda_robot):
        robot = panda_robot
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)

        rest_task = RestTask(
            robot=robot,
            rest_q=robot.spec.midrange_q,
            weight=1.0,
        )

        analytic_jacobian = rest_task.compute_weighted_jacobian_analytic(VarValues(robot=state))
        autodiff_jacobian = autodiff_weighted_jacobian(rest_task, VarValues(robot=state))

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4, rtol=1e-4)


class TestWarpSceneDistanceTask:
    def test_residual_jacobian_and_direct_gradient(self, panda_robot):
        device = wp.get_device("cpu")
        robot = panda_robot
        q_np = (robot.spec.midrange_q + 0.1).astype(np.float32)
        state = robot.forward_kinematics(robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device)))
        hand_idx = robot.link_names.index("panda_hand")
        hand_x = float(state.T_world_link.numpy()[0, hand_idx, 0])
        scene = WarpScene(1, device).add(MeshGeom([_make_plane_mesh_x(device, hand_x + 0.2, 10.0)], [0, 1]))
        local_points = wp.zeros((1, 1), dtype=wp.vec3, device=device)
        link_indices = wp.from_numpy(np.array([[hand_idx]], dtype=np.int32), dtype=wp.int32, device=device)
        task = SceneDistanceTask(
            robot=robot,
            warp_meshes=scene,
            num_contact_points=1,
            weight=2.0,
            local_contact_points=local_points,
            contact_points_link_indices=link_indices,
        )
        values = VarValues(robot=state)
        residual = task.compute_weighted_residual(values).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(values).numpy()
        cost = wp.zeros(1, dtype=wp.float32, device=device)
        gradient = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(values, out_cost=cost, out_gradient=gradient)

        np.testing.assert_allclose(residual, [[0.4]], atol=1e-5)
        np.testing.assert_allclose(cost.numpy(), 0.5 * (residual**2).sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(gradient.numpy(), np.einsum("brd,br->bd", jacobian, residual), atol=1e-6)


class TestWarpCollisionTaskTrajectoryCost:
    """Validate the analytic-gradient trajectory branch of SceneCollisionTask."""

    def _setup(self, robot, B: int = 2, T: int = 4, padding: float = 0.05, box_pos=(0.0, 0.0, 0.5), box_half=0.5):
        device = wp.get_device("cpu")
        # Build a translated box mesh by directly shifting vertices.
        he = float(box_half)
        v = (
            np.array(
                [
                    [-he, -he, -he],
                    [he, -he, -he],
                    [he, he, -he],
                    [-he, he, -he],
                    [-he, -he, he],
                    [he, -he, he],
                    [he, he, he],
                    [-he, he, he],
                ],
                dtype=np.float32,
            )
            + np.asarray(box_pos, dtype=np.float32)[None, :]
        )
        f = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 5, 1],
                [0, 4, 5],
                [3, 2, 6],
                [3, 6, 7],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2],
            ],
            dtype=np.int32,
        )
        mesh_local = wp.Mesh(
            points=wp.array(v, dtype=wp.vec3, device=device),
            indices=wp.array(np.ravel(f), dtype=int, device=device),
        )
        wm = WarpScene(1, mesh_local.device).add(MeshGeom([mesh_local], np.array([0, 1], dtype=np.int32)))

        np.random.seed(0)
        D = robot.num_actuated_joints
        q_np = (np.random.randn(B, T, D).astype(np.float32) * 0.05) + robot.spec.midrange_q[None, None, :].astype(
            np.float32
        )
        q_wp = wp.from_numpy(q_np, dtype=wp.float32, device=device)
        # Panda is fixed-base; do NOT pass T_world_base (would flip has_floating_base=True).
        state = robot.state(q=q_wp)

        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=padding,
            sphere_indices=sphere_indices,
        )
        cost_buf = wp.zeros((B,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((B, state.tangent_dim), dtype=wp.float32, device=device)
        return state, task, cost_buf, grad_buf, B, T, D, padding

    def test_trajectory_cost_zero_for_far_box(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf, B, T, D, _ = self._setup(
            panda_robot_with_collision, padding=0.0, box_pos=(100.0, 0.0, 0.0), box_half=0.1
        )
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        cost = cost_buf.numpy()
        grad = grad_buf.numpy()
        assert np.max(cost) < 1e-6
        assert np.max(np.abs(grad)) < 1e-6

    def test_trajectory_cost_positive_for_intersecting_box(self, panda_robot_with_collision):
        # Place a large padding so any sphere within ~0.5m of the box surface is "active".
        state, task, cost_buf, grad_buf, B, T, D, padding = self._setup(
            panda_robot_with_collision, padding=0.5, box_pos=(0.4, 0.0, 0.5), box_half=0.05
        )
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        cost = cost_buf.numpy()
        grad = grad_buf.numpy()
        assert np.min(cost) > 1e-4, f"expected positive cost, got {cost}"
        assert np.max(np.abs(grad)) > 1e-4, f"expected non-zero gradient, got max-abs {np.max(np.abs(grad))}"

        signed_dists = task._direct_signed_dists.numpy().reshape(B, T, -1)[:, :, task._sphere_indices_np]
        gaps = signed_dists - task.robot.spec.local_collision_sphere_radii[task._sphere_indices_np]
        residuals = np.where(
            gaps < 0.0,
            0.5 * padding - gaps,
            np.where(gaps <= padding, 0.5 * (gaps - padding) ** 2 / (padding + 1e-6), 0.0),
        )
        np.testing.assert_allclose(cost, residuals.sum(axis=(1, 2)) ** 2 / B, rtol=1e-5, atol=1e-6)

    def test_trajectory_endpoint_gradients_are_zero(self, panda_robot_with_collision):
        # With skip_trajectory_endpoints=True (default), gradient at frames 0 and T-1 must be zero.
        state, task, cost_buf, grad_buf, B, T, D, _ = self._setup(
            panda_robot_with_collision, padding=0.5, box_pos=(0.4, 0.0, 0.5), box_half=0.05
        )
        assert task.skip_trajectory_endpoints is True
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        single_tangent_dim = state.tangent_dim // T
        grad = grad_buf.numpy().reshape(B, T, single_tangent_dim)
        assert np.max(np.abs(grad[:, 0, :])) < 1e-6, "gradient at frame 0 must be zero (pinned start)"
        assert np.max(np.abs(grad[:, -1, :])) < 1e-6, "gradient at frame T-1 must be zero (pinned end)"
        # And interior must have at least some non-zero entries.
        assert np.max(np.abs(grad[:, 1:-1, :])) > 1e-4, "interior frames must have non-zero gradient"

    def test_trajectory_gradient_matches_finite_differences(self, panda_robot_with_collision):
        # Verify analytic gradient against central finite differences on q.
        state, task, cost_buf_a, grad_buf_a, B, T, D, padding = self._setup(
            panda_robot_with_collision, B=1, T=4, padding=0.05, box_pos=(0.0, 0.0, 0.5), box_half=0.5
        )
        task.skip_trajectory_endpoints = False  # FD needs gradient defined on all frames
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf_a, out_gradient=grad_buf_a)
        analytic_grad = grad_buf_a.numpy().reshape(B, T, -1)

        device = wp.get_device("cpu")
        eps = 1e-4
        q_np = state.q.numpy().copy()

        def _eval_cost(q_perturbed: np.ndarray) -> np.ndarray:
            q_p = wp.from_numpy(q_perturbed.astype(np.float32), dtype=wp.float32, device=device)
            sp = panda_robot_with_collision.state(q=q_p)
            cb = wp.zeros((B,), dtype=wp.float32, device=device)
            task.compute_weighted_cost_and_gradient(VarValues(robot=sp), out_cost=cb)
            return cb.numpy()

        # Sample only a few (b, t, d) for speed.
        np.random.seed(1)
        D_q = q_np.shape[-1]  # number of actuated DOFs in q (axis we perturb)
        sample_pairs = [(0, t, d) for t in range(T) for d in np.random.choice(D_q, size=min(3, D_q), replace=False)]
        max_err = 0.0
        for b, t, d in sample_pairs:
            qp = q_np.copy()
            qp[b, t, d] += eps
            cp = _eval_cost(qp)
            qm = q_np.copy()
            qm[b, t, d] -= eps
            cm = _eval_cost(qm)
            fd = (cp[b] - cm[b]) / (2.0 * eps)
            ana = float(analytic_grad[b, t, d])
            err = abs(fd - ana)
            max_err = max(max_err, float(err))
        assert max_err < 5e-3, f"finite-diff gradient mismatch: max abs err {max_err:.3e}"


class TestWarpCollisionTaskCostAndGradientSumSquared:
    """Validate `compute_weighted_cost_and_gradient(..., cost_form='sum_squared')` for per-pose grasp-optim use."""

    def _setup(self, robot, B: int = 2, padding: float = 0.05, box_pos=(0.4, 0.0, 0.5), box_half: float = 0.1):
        device = wp.get_device("cpu")
        v = np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=np.float32,
        )
        v = v * float(box_half) + np.array(box_pos, dtype=np.float32)
        f = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 5, 1],
                [0, 4, 5],
                [3, 2, 6],
                [3, 6, 7],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2],
            ],
            dtype=np.int32,
        )
        mesh_local = wp.Mesh(
            points=wp.array(v, dtype=wp.vec3, device=device),
            indices=wp.array(np.ravel(f), dtype=int, device=device),
        )
        wm = WarpScene(1, mesh_local.device).add(MeshGeom([mesh_local], np.array([0, 1], dtype=np.int32)))

        np.random.seed(0)
        D = robot.num_actuated_joints
        # per-pose state (B, D) - not a trajectory
        q_np = (np.random.randn(B, D).astype(np.float32) * 0.05) + robot.spec.midrange_q[None, :].astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device))
        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=2.0,
            margin=padding,
            sphere_indices=sphere_indices,
        )
        cost_buf = wp.zeros((B,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((B, state.tangent_dim), dtype=wp.float32, device=device)
        return state, task, cost_buf, grad_buf, B, D

    def test_cost_matches_residual_form(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf, B, _ = self._setup(panda_robot_with_collision, padding=0.5, box_half=0.05)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        expected = 0.5 * (residual**2).sum(axis=-1)
        assert expected.max() > 1e-3, "expected positive cost (spheres should be inside padding region)"
        np.testing.assert_allclose(cost_buf.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_gradient_matches_jacobian_times_residual(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf, B, _ = self._setup(panda_robot_with_collision, padding=0.5, box_half=0.05)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        # need motion subspace before analytic jacobian
        task.robot.forward_kinematics(state)
        task.robot.compute_motion_subspace(state)
        jacobian = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        expected = np.einsum("bri,br->bi", jacobian, residual)
        assert np.max(np.abs(expected)) > 1e-5, "expected non-zero gradient"
        np.testing.assert_allclose(grad_buf.numpy(), expected, rtol=1e-4, atol=1e-5)

    def test_zero_when_inactive(self, panda_robot_with_collision):
        # box far away → no sphere within padding → zero cost and gradient
        state, task, cost_buf, grad_buf, _, _ = self._setup(
            panda_robot_with_collision, padding=0.0, box_pos=(100.0, 0.0, 0.0), box_half=0.1
        )
        task.compute_weighted_cost_and_gradient(
            VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        assert np.max(np.abs(cost_buf.numpy())) < 1e-6
        assert np.max(np.abs(grad_buf.numpy())) < 1e-6

    def test_default_cost_form_unchanged(self, panda_robot_with_collision):
        # Sanity: omitting cost_form falls back to the trajectory `squared_sum` path,
        # which requires a trajectory state. Since our per-pose state isn't trajectory,
        # the default path runs the trajectory direct-gradient kernel with T=1.
        state, task, cost_buf, grad_buf, _, _ = self._setup(panda_robot_with_collision, padding=0.5, box_half=0.05)
        # Default form should not raise.
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf)
        assert np.all(np.isfinite(cost_buf.numpy()))
        assert np.all(np.isfinite(grad_buf.numpy()))

    def test_accumulation_into_shared_buffers(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf, _, _ = self._setup(panda_robot_with_collision, padding=0.5, box_half=0.05)
        task.compute_weighted_cost_and_gradient(
            VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        single_cost = cost_buf.numpy().copy()
        single_grad = grad_buf.numpy().copy()
        task.compute_weighted_cost_and_gradient(
            VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        np.testing.assert_allclose(cost_buf.numpy(), 2.0 * single_cost, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_buf.numpy(), 2.0 * single_grad, rtol=1e-5, atol=1e-6)

    def _setup_sqrt(self, robot, B: int = 2, padding: float = 0.05, box_pos=(0.4, 0.0, 0.5), box_half: float = 0.1):
        """Setup variant using residual_mode='sqrt_abs' - matches what GDGraspOptHelper uses in production."""
        device = wp.get_device("cpu")
        v = np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=np.float32,
        ) * float(box_half) + np.array(box_pos, dtype=np.float32)
        f = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 5, 1],
                [0, 4, 5],
                [3, 2, 6],
                [3, 6, 7],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2],
            ],
            dtype=np.int32,
        )
        mesh_local = wp.Mesh(
            points=wp.array(v, dtype=wp.vec3, device=device),
            indices=wp.array(np.ravel(f), dtype=int, device=device),
        )
        wm = WarpScene(1, mesh_local.device).add(MeshGeom([mesh_local], np.array([0, 1], dtype=np.int32)))
        np.random.seed(0)
        D = robot.num_actuated_joints
        q_np = (np.random.randn(B, D).astype(np.float32) * 0.05) + robot.spec.midrange_q[None, :].astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device))
        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=2.0,
            margin=padding,
            sphere_indices=sphere_indices,
            residual_mode="sqrt_abs",
            residual_eps=1e-6,
        )
        cost_buf = wp.zeros((B,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((B, state.tangent_dim), dtype=wp.float32, device=device)
        return state, task, cost_buf, grad_buf

    def test_sqrt_mode_cost_matches_residual_form(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf = self._setup_sqrt(panda_robot_with_collision, padding=0.5, box_half=0.05)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        expected = 0.5 * (residual**2).sum(axis=-1)
        np.testing.assert_allclose(cost_buf.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_sqrt_mode_gradient_matches_finite_differences(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf = self._setup_sqrt(panda_robot_with_collision, B=1, padding=0.5, box_half=0.05)
        task.compute_weighted_cost_and_gradient(
            VarValues(robot=state), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        analytic = grad_buf.numpy()[0]
        device = wp.get_device("cpu")

        q_np = state.q.numpy().copy()
        eps_q = 1e-4
        np.random.seed(0)
        D = q_np.shape[-1]
        sample_dofs = np.random.choice(D, size=4, replace=False)
        max_err = 0.0
        for d in sample_dofs:
            qp = q_np.copy()
            qp[0, d] += eps_q
            sp = panda_robot_with_collision.state(q=wp.from_numpy(qp, dtype=wp.float32, device=device))
            r_p = task.compute_weighted_residual(VarValues(robot=sp)).numpy()
            cp = 0.5 * (r_p**2).sum(axis=-1)[0]
            qm = q_np.copy()
            qm[0, d] -= eps_q
            sm = panda_robot_with_collision.state(q=wp.from_numpy(qm, dtype=wp.float32, device=device))
            r_m = task.compute_weighted_residual(VarValues(robot=sm)).numpy()
            cm = 0.5 * (r_m**2).sum(axis=-1)[0]
            fd = (cp - cm) / (2.0 * eps_q)
            err = abs(fd - float(analytic[d]))
            max_err = max(max_err, err)
        assert max_err < 5e-3, f"sqrt-mode FD mismatch: max abs err {max_err:.3e}"


class TestWarpPositionTask:
    """Test PositionTask residual and Jacobian correctness."""

    def test_residual_is_world_frame_position_error(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)

        ee_idx = robot.link_names.index("panda_hand")
        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        weight = 2.0
        pos_task = PositionTask(robot, ee_idx, T_world_target.reshape((1, 1)), weight=weight)
        pos_res = pos_task.compute_weighted_residual(VarValues(robot=state)).numpy()

        ee_pose = state.T_world_link.numpy()[0, ee_idx]
        ee_pos = ee_pose[:3]
        target_pos = target_np[0, :3]
        expected = weight * (target_pos - ee_pos)

        assert np.allclose(pos_res[0], expected, atol=1e-5)

    def test_jacobian_analytic_vs_autodiff(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")
        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        pos_task = PositionTask(robot, ee_idx, T_world_target.reshape((1, 1)), weight=2.0)

        analytic_jac = pos_task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        autodiff_jac = autodiff_weighted_jacobian(pos_task, VarValues(robot=state)).numpy()

        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)


class TestWarpManipulabilityTask:
    """Test ManipulabilityTask residual and Jacobian correctness."""

    @staticmethod
    def _ee_position(robot, ee_idx, q_np):
        state = robot.state(q=wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32))
        robot.forward_kinematics(state)
        return state.get_T_world_link(ee_idx).numpy()[0, :3]

    def test_manipulability_value_matches_fk_jacobian(self, panda_robot):
        # Independent reference: J = d(ee position)/dq via finite differences of FK, then the
        # Yoshikawa measure sqrt(det(J Jᵀ)) - exactly pyroki's manipulability definition. The task
        # residual is weight / (manip + eps), so back out manip and compare.
        robot = panda_robot
        np.random.seed(0)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.3 + robot.spec.midrange_q).astype(np.float32)
        ee_idx = robot.link_names.index("panda_hand")
        weight, eps = 0.01, 1e-6

        h = 1e-4
        J = np.zeros((3, robot.num_actuated_joints))
        for d in range(robot.num_actuated_joints):
            qp, qm = q_np.copy(), q_np.copy()
            qp[d] += h
            qm[d] -= h
            J[:, d] = (self._ee_position(robot, ee_idx, qp) - self._ee_position(robot, ee_idx, qm)) / (2 * h)
        manip_ref = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        res = float(
            ManipulabilityTask(robot, ee_idx, weight=weight, epsilon=eps)
            .compute_weighted_residual(VarValues(robot=state))
            .numpy()[0, 0]
        )
        manip_task = weight / res - eps
        assert np.isclose(manip_task, manip_ref, rtol=5e-3, atol=1e-4)

    def test_jacobian_analytic_vs_finite_difference(self, panda_robot):
        # The residual depends on the kinematic Jacobian (S_world), so Warp autodiff would need a
        # second kinematic derivative it does not support; central finite differences are the ground
        # truth.
        robot = panda_robot
        np.random.seed(0)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.3 + robot.spec.midrange_q).astype(np.float32)
        ee_idx = robot.link_names.index("panda_hand")
        task = ManipulabilityTask(robot, ee_idx, weight=0.01)

        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        analytic_jac = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()[0, 0]

        def residual(q: np.ndarray) -> float:
            st = robot.state(q=wp.from_numpy(q.astype(np.float32), dtype=wp.float32))
            return float(task.compute_weighted_residual(VarValues(robot=st)).numpy()[0, 0])

        eps = 1e-3
        fd_jac = np.zeros_like(analytic_jac)
        for d in range(robot.num_actuated_joints):
            qp, qm = q_np.copy(), q_np.copy()
            qp[d] += eps
            qm[d] -= eps
            fd_jac[d] = (residual(qp) - residual(qm)) / (2 * eps)

        assert np.allclose(analytic_jac, fd_jac, atol=2e-3, rtol=1e-2)


class TestWarpRotationTask:
    """Test RotationTask residual and Jacobian correctness."""

    def test_residual_is_world_frame_rotation_error(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)

        ee_idx = robot.link_names.index("panda_hand")
        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        weight = 3.0
        rot_task = RotationTask(robot, ee_idx, T_world_target.reshape((1, 1)), weight=weight)
        rot_res = rot_task.compute_weighted_residual(VarValues(robot=state)).numpy()

        ee_pose = state.T_world_link.numpy()[0, ee_idx]
        q_actual_wxyz = ee_pose[3:7]
        q_target_wxyz = target_np[0, 3:7]
        r_actual = Rotation.from_quat([q_actual_wxyz[1], q_actual_wxyz[2], q_actual_wxyz[3], q_actual_wxyz[0]])
        r_target = Rotation.from_quat([q_target_wxyz[1], q_target_wxyz[2], q_target_wxyz[3], q_target_wxyz[0]])
        r_err = r_actual * r_target.inv()
        expected = weight * r_err.as_rotvec().astype(np.float32)
        assert np.allclose(rot_res[0], expected, atol=1e-5)

    def test_jacobian_analytic_vs_autodiff_near_identity(self, panda_robot):
        """Test rotation Jacobian near identity where geometric omega matches exact Jacobian."""
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")
        # use the actual EE pose as target so rotation error is near zero
        ee_pose = state.T_world_link.numpy()[0, ee_idx]
        target_np = ee_pose.reshape(1, 7).astype(np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        rot_task = RotationTask(robot, ee_idx, T_world_target.reshape((1, 1)), weight=3.0)

        analytic_jac = rot_task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        autodiff_jac = autodiff_weighted_jacobian(rot_task, VarValues(robot=state)).numpy()

        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)


class TestWarpPositionTaskPerLinkWeights:
    """Test PositionTask per-link weight support."""

    def test_per_link_weights_residual(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = PositionTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=[2.0, 0.0])
        res = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        ee0 = state.T_world_link.numpy()[0, idx0, :3]
        expected0 = 2.0 * (t0_np[0, :3] - ee0)
        assert np.allclose(res[:3], expected0, atol=1e-5)
        assert np.allclose(res[3:6], 0.0, atol=1e-7)

    def test_set_weight_dynamic(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = PositionTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=1.0)
        task.set_weight([0.0, 3.0])
        res = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        ee1 = state.T_world_link.numpy()[0, idx1, :3]
        expected1 = 3.0 * (t1_np[0, :3] - ee1)
        assert np.allclose(res[:3], 0.0, atol=1e-7)
        assert np.allclose(res[3:6], expected1, atol=1e-5)

    def test_per_link_weights_jacobian(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = PositionTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=[2.0, 0.5])
        analytic = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        autodiff = autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy()
        assert np.allclose(analytic, autodiff, atol=1e-4, rtol=1e-4)


class TestWarpRotationTaskPerLinkWeights:
    """Test RotationTask per-link weight support."""

    def test_per_link_weights_residual(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = RotationTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=[3.0, 0.0])
        res = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        # frame 0 should have non-zero residual, frame 1 should be zero
        assert np.linalg.norm(res[:3]) > 0.01
        assert np.allclose(res[3:6], 0.0, atol=1e-7)

    def test_set_weight_dynamic(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = RotationTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=1.0)
        task.set_weight([0.0, 2.0])
        res = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        assert np.allclose(res[:3], 0.0, atol=1e-7)
        assert np.linalg.norm(res[3:6]) > 0.01

    def test_per_link_weights_jacobian(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        # use actual poses as targets so rotation error is near zero (for Jacobian accuracy)
        t0_pose = state.T_world_link.numpy()[0, idx0].reshape(1, 7).astype(np.float32)
        t1_pose = state.T_world_link.numpy()[0, idx1].reshape(1, 7).astype(np.float32)
        t0 = wp.from_numpy(t0_pose, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_pose, dtype=wp_vec7)

        task = RotationTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=[2.0, 0.5])
        analytic = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        autodiff = autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy()
        assert np.allclose(analytic, autodiff, atol=1e-4, rtol=1e-4)


class TestWarpAxisLimitTask:
    def test_residual_and_jacobian(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy((robot.spec.midrange_q + 0.1).astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        frame_index = robot.link_names.index("panda_hand")
        quat_wxyz = state.T_world_link.numpy()[0, frame_index, 3:]
        axis_world = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).apply([0.0, 0.0, 1.0])
        perpendicular = np.cross(axis_world, [1.0, 0.0, 0.0])
        if np.linalg.norm(perpendicular) < 1e-6:
            perpendicular = np.cross(axis_world, [0.0, 1.0, 0.0])
        world_axis = axis_world + perpendicular / np.linalg.norm(perpendicular)
        world_axis /= np.linalg.norm(world_axis)

        task = AxisLimitTask(
            robot,
            frame_index,
            local_axis=(0.0, 0.0, 1.0),
            world_axis=tuple(world_axis),
            min_angle=np.pi / 2,
            weight=3.0,
        )
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        np.testing.assert_allclose(residual[0, 0], 3.0 * np.dot(axis_world, world_axis), atol=1e-5)

        analytic = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        autodiff = autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy()
        np.testing.assert_allclose(analytic, autodiff, atol=1e-4, rtol=1e-4)

    def test_max_angle(self, panda_robot):
        robot = panda_robot
        state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        frame_index = robot.link_names.index("panda_hand")
        quat_wxyz = state.T_world_link.numpy()[0, frame_index, 3:]
        axis_world = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).apply([0.0, 0.0, 1.0])
        task = AxisLimitTask(
            robot,
            frame_index,
            world_axis=tuple(-axis_world),
            max_angle=0.5,
            weight=2.0,
        )
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        np.testing.assert_allclose(residual[0, 0], 2.0 * (np.cos(0.5) + 1.0), atol=1e-5)


class TestWarpFixedFrameTaskJacobian:
    def test_jacobian_fixed_base(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        foot_indices = [robot.link_names.index("panda_hand")]
        targets = [state.get_T_world_link(foot_indices[0])]

        target = stack(targets, axis=1)
        tasks = [
            PositionTask(robot, foot_indices, target, weight=5.0, fixed_target=True),
            RotationTask(robot, foot_indices, target, weight=2.0, fixed_target=True),
        ]
        analytic_jac = np.concatenate(
            [task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy() for task in tasks], axis=1
        )
        autodiff_jac = np.concatenate(
            [autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy() for task in tasks], axis=1
        )

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)

    def test_jacobian_floating_base(self, g1_robot):
        robot = g1_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        T_base_np = np.array([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_base)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        foot_indices = [
            robot.link_names.index("left_ankle_roll_link"),
            robot.link_names.index("right_ankle_roll_link"),
        ]
        targets = [state.get_T_world_link(i) for i in foot_indices]

        target = stack(targets, axis=1)
        tasks = [
            PositionTask(robot, foot_indices, target, weight=5.0, fixed_target=True),
            RotationTask(robot, foot_indices, target, weight=2.0, fixed_target=True),
        ]
        analytic_jac = np.concatenate(
            [task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy() for task in tasks], axis=1
        )
        autodiff_jac = np.concatenate(
            [autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy() for task in tasks], axis=1
        )

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)


class TestWarpComPositionTaskJacobian:
    def test_jacobian_floating_base(self, g1_robot):
        robot = g1_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        T_base_np = np.array([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_base)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        target_com = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        task = ComPositionTask(robot=robot, target_com_position=target_com, weight=1.0)

        analytic_jac = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        autodiff_jac = autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy()

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)

    def test_jacobian_fixed_base(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        target_com = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        task = ComPositionTask(robot=robot, target_com_position=target_com, weight=2.0)

        analytic_jac = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        autodiff_jac = autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy()

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)

    def test_residual_values(self, g1_robot):
        robot = g1_robot
        q = wp.from_numpy(robot.spec.zero_q.astype(np.float32), dtype=wp.float32)

        T_base_np = np.array([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_base)
        robot.forward_kinematics(state)

        # compute expected CoM manually
        link_poses = state.T_world_link.numpy()[0]  # [num_links, 7]
        masses = robot.spec.link_masses
        local_coms = robot.spec.link_local_com_positions

        com_expected = np.zeros(3)
        for i in range(robot.spec.num_links):
            m = masses[i]
            if m > 0:
                pos = link_poses[i, :3]
                q_wxyz = link_poses[i, 3:]
                rot = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                com_world = pos + rot.apply(local_coms[i])
                com_expected += m * com_world
        com_expected /= robot.spec.total_mass

        target_com = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        weight = 2.0
        task = ComPositionTask(robot=robot, target_com_position=target_com, weight=weight)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        expected_residual = weight * (target_com - com_expected.astype(np.float32))
        assert np.allclose(residual, expected_residual, atol=1e-4)


class TestMultiSeedConsistency:
    """Test that multi-seed (instance-major) layout produces consistent results.

    Key invariant: with num_seeds=K and all seeds having the same q per instance,
    every seed must produce identical residual and Jacobian values.
    """

    NUM_SEEDS = 4
    ACTUAL_BATCH = 2

    def _make_expanded_state(self, robot, rng):
        """Create un-expanded q and instance-major expanded state."""
        num_joints = robot.num_actuated_joints
        lo = robot.spec.actuated_joint_limits[:, 0]
        hi = robot.spec.actuated_joint_limits[:, 1]
        q_np = rng.uniform(lo, hi, size=(self.ACTUAL_BATCH, num_joints)).astype(np.float32)
        q_expanded = np.repeat(q_np, self.NUM_SEEDS, axis=0)
        state = robot.state(q=wp.from_numpy(q_expanded, dtype=wp.float32))
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        return q_np, state

    def _assert_seeds_consistent(self, array_np, name: str):
        """Assert all seeds for each instance produce the same value."""
        for i in range(self.ACTUAL_BATCH):
            base = i * self.NUM_SEEDS
            ref = array_np[base]
            for s in range(1, self.NUM_SEEDS):
                np.testing.assert_allclose(
                    array_np[base + s],
                    ref,
                    atol=1e-5,
                    rtol=1e-5,
                    err_msg=f"{name}: instance {i}, seed {s} differs from seed 0",
                )

    def test_position_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(42)
        q_np, state = self._make_expanded_state(robot, rng)

        target_link_idx = robot.link_names.index("panda_link7")
        single_state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(single_state)
        target_pose = single_state.get_T_world_link(target_link_idx)

        task = PositionTask(robot, target_link_idx, target_pose.reshape((target_pose.shape[0], 1)), weight=1.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues(robot=state)).numpy()

        total_batch = self.ACTUAL_BATCH * self.NUM_SEEDS
        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_seeds_consistent(residual, "PositionTask residual")
        self._assert_seeds_consistent(jacobian, "PositionTask jacobian")

    def test_rotation_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(43)
        q_np, state = self._make_expanded_state(robot, rng)

        target_link_idx = robot.link_names.index("panda_link7")
        single_state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(single_state)
        target_pose = single_state.get_T_world_link(target_link_idx)

        task = RotationTask(robot, target_link_idx, target_pose.reshape((target_pose.shape[0], 1)), weight=1.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues(robot=state)).numpy()

        total_batch = self.ACTUAL_BATCH * self.NUM_SEEDS
        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_seeds_consistent(residual, "RotationTask residual")
        self._assert_seeds_consistent(jacobian, "RotationTask jacobian")

    def test_frame_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(44)
        q_np, state = self._make_expanded_state(robot, rng)

        target_link_idx = robot.link_names.index("panda_link7")
        single_state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(single_state)
        target_pose = single_state.get_T_world_link(target_link_idx)

        task = FrameTask(
            robot,
            target_link_idx,
            target_pose.reshape((target_pose.shape[0], 1)),
            position_weight=1.0,
            orientation_weight=1.0,
        )
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues(robot=state)).numpy()

        total_batch = self.ACTUAL_BATCH * self.NUM_SEEDS
        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_seeds_consistent(residual, "FrameTask residual")
        self._assert_seeds_consistent(jacobian, "FrameTask jacobian")

    def test_rest_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(45)
        q_np, state = self._make_expanded_state(robot, rng)

        total_batch = self.ACTUAL_BATCH * self.NUM_SEEDS
        rest_q = robot.spec.midrange_q
        task = RestTask(robot=robot, rest_q=rest_q, weight=0.1)
        task.set_rest_state(wp.from_numpy(q_np[:, : robot.num_actuated_joints].copy(), dtype=wp.float32))

        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues(robot=state)).numpy()

        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_seeds_consistent(residual, "RestTask residual")
        self._assert_seeds_consistent(jacobian, "RestTask jacobian")

    def test_velocity_limit_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(46)
        q_np, state = self._make_expanded_state(robot, rng)

        total_batch = self.ACTUAL_BATCH * self.NUM_SEEDS
        prev_q = rng.uniform(
            robot.spec.actuated_joint_limits[:, 0],
            robot.spec.actuated_joint_limits[:, 1],
            size=(self.ACTUAL_BATCH, robot.num_actuated_joints),
        ).astype(np.float32)
        prev_state = robot.state(q=wp.from_numpy(prev_q, dtype=wp.float32))

        task = VelocityLimitTask(robot=robot, dt=0.1, weight=1.0)
        task.set_prev_state(prev_state)

        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues(robot=state)).numpy()

        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_seeds_consistent(residual, "VelocityLimitTask residual")
        self._assert_seeds_consistent(jacobian, "VelocityLimitTask jacobian")

    def test_smoothness_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(47)
        q_np, state = self._make_expanded_state(robot, rng)

        total_batch = self.ACTUAL_BATCH * self.NUM_SEEDS
        prev_q = rng.uniform(
            robot.spec.actuated_joint_limits[:, 0],
            robot.spec.actuated_joint_limits[:, 1],
            size=(self.ACTUAL_BATCH, robot.num_actuated_joints),
        ).astype(np.float32)
        prev_state = robot.state(q=wp.from_numpy(prev_q, dtype=wp.float32))

        task = SmoothnessTask(robot=robot, weight=1.0)

        residual = task.compute_weighted_residual(VarValues(robot=state), prev_var=prev_state).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues(robot=state), prev_var=prev_state).numpy()

        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_seeds_consistent(residual, "SmoothnessTask residual")
        self._assert_seeds_consistent(jacobian, "SmoothnessTask jacobian")


class TestWarpSelfCollisionTask:
    def test_residual_shape_and_nonneg(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)

        task = SelfCollisionTask(robot=robot, weight=1.0, margin=0.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        assert residual.shape == (task.residual_dim,)
        assert float(np.min(residual)) >= 0.0

    def test_residual_positive_when_colliding(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        # use large margin to force collisions even at midrange
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)

        task = SelfCollisionTask(robot=robot, weight=1.0, margin=1.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        assert float(np.max(residual)) > 0.0

    def test_jacobian_analytic_vs_autodiff(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        np.random.seed(42)
        q_np = np.random.randn(robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        q = wp.from_numpy(q_np, dtype=wp.float32)

        T_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_base)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        task = SelfCollisionTask(robot=robot, weight=1.0, margin=0.5)

        analytic_jac = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        autodiff_jac = autodiff_weighted_jacobian(task, VarValues(robot=state)).numpy()

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=3e-4, rtol=3e-4)

    def test_pair_filtering_reduces_pairs(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        task_filtered = SelfCollisionTask(robot=robot, filter_adjacent_links=True)
        task_all = SelfCollisionTask(robot=robot, filter_adjacent_links=False)

        num_spheres = len(robot.spec.local_collision_sphere_centers)
        assert task_all._num_pairs == num_spheres * (num_spheres - 1) // 2
        assert task_filtered._num_pairs < task_all._num_pairs

    def test_batch_consistency(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        np.random.seed(0)
        batch_size = 4
        q_np = np.random.randn(batch_size, robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q

        # batch evaluation
        q_batch = wp.from_numpy(q_np, dtype=wp.float32)
        state_batch = robot.state(q=q_batch)
        task_batch = SelfCollisionTask(robot=robot, weight=1.0, margin=0.5)
        residual_batch = task_batch.compute_weighted_residual(VarValues(robot=state_batch)).numpy()

        # individual evaluations
        for i in range(batch_size):
            q_single = wp.from_numpy(q_np[i], dtype=wp.float32)
            state_single = robot.state(q=q_single)
            task_single = SelfCollisionTask(robot=robot, weight=1.0, margin=0.5)
            residual_single = task_single.compute_weighted_residual(VarValues(robot=state_single)).numpy()[0]
            # The parallel active-pair selection fills rows in an arbitrary order, so compare the
            # active SET (cost/normal-equations are permutation-invariant) by sorting the rows.
            assert np.allclose(np.sort(residual_batch[i]), np.sort(residual_single), atol=1e-6)


class TestWarpCapsuleSelfCollisionTask:
    def test_residual_shape_and_nonneg(self, panda_robot_with_capsules):
        robot = panda_robot_with_capsules
        assert robot.spec.has_link_capsules
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)

        task = SelfCollisionTask(robot=robot, representation="capsule", weight=1.0, margin=0.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        assert residual.shape == (task.residual_dim,)
        assert float(np.min(residual)) >= 0.0

    def test_residual_positive_when_colliding(self, panda_robot_with_capsules):
        robot = panda_robot_with_capsules
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)

        task = SelfCollisionTask(robot=robot, representation="capsule", weight=1.0, margin=1.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]

        assert float(np.max(residual)) > 0.0

    @pytest.mark.parametrize("residual_mode", ["abs", "sqrt_abs"])
    def test_jacobian_matches_finite_differences(self, panda_robot_with_capsules, residual_mode):
        robot = panda_robot_with_capsules
        task = SelfCollisionTask(
            robot=robot, representation="capsule", weight=1.0, margin=0.5, residual_mode=residual_mode
        )
        n = robot.num_actuated_joints
        T_base = wp.from_numpy(np.array([[0, 0, 0, 1, 0, 0, 0]], np.float32), dtype=wp_vec7)

        def state_at(q_np):
            st = robot.state(q=wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32), T_world_base=T_base)
            st = robot.forward_kinematics(st)
            st = robot.compute_motion_subspace(st)
            return st

        def sum_residual(q_np):
            return float(task.compute_weighted_residual(VarValues(robot=state_at(q_np))).numpy()[0].sum())

        np.random.seed(7)
        q0 = np.random.randn(n).astype(np.float32) * 0.15 + robot.spec.midrange_q

        # All link pairs fit within max_active_pairs, so the active set is order-invariant: compare the
        # gradient of the summed residual (column sums of J) against finite differences. This sidesteps
        # the discrete top-K row ordering, which a per-row comparison would trip on.
        jac = task.compute_weighted_jacobian_analytic(VarValues(robot=state_at(q0))).numpy()[0]
        grad_analytic = jac.sum(axis=0)[-n:]  # joint columns

        eps = 1e-4
        grad_fd = np.zeros(n, dtype=np.float64)
        for d in range(n):
            qp, qm = q0.copy(), q0.copy()
            qp[d] += eps
            qm[d] -= eps
            grad_fd[d] = (sum_residual(qp) - sum_residual(qm)) / (2.0 * eps)

        assert np.allclose(grad_analytic, grad_fd, atol=2e-2)

    def test_batch_consistency(self, panda_robot_with_capsules):
        robot = panda_robot_with_capsules
        np.random.seed(0)
        batch_size = 4
        q_np = np.random.randn(batch_size, robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q

        q_batch = wp.from_numpy(q_np, dtype=wp.float32)
        task_batch = SelfCollisionTask(robot=robot, representation="capsule", weight=1.0, margin=0.5)
        residual_batch = task_batch.compute_weighted_residual(VarValues(robot=robot.state(q=q_batch))).numpy()

        for i in range(batch_size):
            q_single = wp.from_numpy(q_np[i], dtype=wp.float32)
            task_single = SelfCollisionTask(robot=robot, representation="capsule", weight=1.0, margin=0.5)
            residual_single = task_single.compute_weighted_residual(VarValues(robot=robot.state(q=q_single))).numpy()[0]
            # The parallel active-pair selection fills rows in an arbitrary order, so compare the
            # active SET (cost/normal-equations are permutation-invariant) by sorting the rows.
            assert np.allclose(np.sort(residual_batch[i]), np.sort(residual_single), atol=1e-6)


class TestWarpLinkSphereSelfCollisionTask:
    """One-sphere-per-link mode (subsumes the old SelfPenetrationTask)."""

    def test_residual_shape_and_nonneg(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        state = robot.state(q=wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32))
        task = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=0.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]
        assert residual.shape == (task.residual_dim,)
        assert float(np.min(residual)) >= 0.0

    def test_residual_positive_when_colliding(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        state = robot.state(q=wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32))
        task = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=1.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]
        assert float(np.max(residual)) > 0.0

    def test_jacobian_matches_finite_differences(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        task = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=0.4)
        n = robot.num_actuated_joints
        T_base = wp.from_numpy(np.array([[0, 0, 0, 1, 0, 0, 0]], np.float32), dtype=wp_vec7)

        def state_at(q_np):
            st = robot.state(q=wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32), T_world_base=T_base)
            st = robot.forward_kinematics(st)
            return robot.compute_motion_subspace(st)

        def sum_residual(q_np):
            return float(task.compute_weighted_residual(VarValues(robot=state_at(q_np))).numpy()[0].sum())

        np.random.seed(7)
        q0 = np.random.randn(n).astype(np.float32) * 0.15 + robot.spec.midrange_q
        # Column sums of J are order-invariant, so they sidestep the discrete top-K row ordering.
        grad_analytic = (
            task.compute_weighted_jacobian_analytic(VarValues(robot=state_at(q0))).numpy()[0].sum(axis=0)[-n:]
        )
        eps = 1e-4
        grad_fd = np.zeros(n, dtype=np.float64)
        for d in range(n):
            qp, qm = q0.copy(), q0.copy()
            qp[d] += eps
            qm[d] -= eps
            grad_fd[d] = (sum_residual(qp) - sum_residual(qm)) / (2.0 * eps)
        assert np.allclose(grad_analytic, grad_fd, atol=2e-2)

    def test_equivalence_to_point_pair(self, panda_robot_with_collision):
        """radius=0, center=origin reproduces the old point-pair E_spen: max(0, margin - ||o_i - o_j||)."""
        robot = panda_robot_with_collision
        state = robot.forward_kinematics(
            robot.state(q=wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32))
        )
        margin = 0.15
        task = SelfCollisionTask(
            robot=robot,
            representation="link_sphere",
            link_sphere_radius=0.0,
            link_sphere_center="origin",
            margin=margin,
        )
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]
        origins = state.T_world_link.numpy().reshape(robot.spec.num_links, 7)[:, :3]
        pairs = task._pair_indices_np
        assert pairs is not None
        d = np.linalg.norm(origins[pairs[:, 0]] - origins[pairs[:, 1]], axis=1)
        expected = np.maximum(0.0, margin - d)
        assert int((expected > 0).sum()) <= task.max_active_pairs  # no top-K capping for this margin
        assert np.allclose(np.sort(residual[residual > 0]), np.sort(expected[expected > 0]), atol=1e-5)

    def test_batch_consistency(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        np.random.seed(0)
        q_np = np.random.randn(4, robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        task_b = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=0.4)
        residual_batch = task_b.compute_weighted_residual(
            VarValues(robot=robot.state(q=wp.from_numpy(q_np, dtype=wp.float32)))
        ).numpy()
        for i in range(4):
            task_s = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=0.4)
            residual_single = task_s.compute_weighted_residual(
                VarValues(robot=robot.state(q=wp.from_numpy(q_np[i], dtype=wp.float32)))
            ).numpy()[0]
            assert np.allclose(np.sort(residual_batch[i]), np.sort(residual_single), atol=1e-6)

    def test_cost_and_gradient_finite_differences(self, panda_robot_with_collision):
        """GD/L-BFGS path: analytic dC/dq vs finite differences of the cost-only call."""
        robot = panda_robot_with_collision
        n = robot.num_actuated_joints
        np.random.seed(3)
        q0 = (np.random.randn(2, n).astype(np.float32) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        task = SelfCollisionTask(robot=robot, representation="link_sphere", weight=2.0, margin=0.4)

        def cost_at(q_np):
            st = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
            cb = wp.zeros((q_np.shape[0],), dtype=wp.float32)
            task.compute_weighted_cost_and_gradient(VarValues(robot=st), out_cost=cb, out_gradient=None)
            return cb.numpy()

        st0 = robot.state(q=wp.from_numpy(q0, dtype=wp.float32))
        cb = wp.zeros((2,), dtype=wp.float32)
        gb = wp.zeros((2, st0.tangent_dim), dtype=wp.float32)
        task.compute_weighted_cost_and_gradient(VarValues(robot=st0), out_cost=cb, out_gradient=gb)
        analytic = gb.numpy()

        eps = 1e-4
        np.random.seed(0)
        max_err = 0.0
        for b in range(2):
            for d in np.random.choice(n, size=4, replace=False):
                qp, qm = q0.copy(), q0.copy()
                qp[b, d] += eps
                qm[b, d] -= eps
                fd = (cost_at(qp)[b] - cost_at(qm)[b]) / (2.0 * eps)
                max_err = max(max_err, abs(analytic[b, d] - fd))
        assert max_err < 2e-2, f"max grad err {max_err:.3e}"


def _scene_0_single(device: wp_device_type) -> WarpScene:
    """Reference scene 0: 2 meshes + 1 sphere primitive."""
    plane = _make_plane_mesh_x(device=device, x=0.3, half_extent=10.0)
    box = _make_box_mesh(device=device, half_extent=0.2)
    return (
        WarpScene(num_scenes=1, device=device)
        .add(MeshGeom(meshes=[plane, box], scene_offsets=np.array([0, 2], dtype=np.int32)))
        .add(SphereGeom(radii=np.array([0.1], dtype=np.float32), scene_offsets=np.array([0, 1], dtype=np.int32)))
    )


def _scene_1_single(device: wp_device_type) -> WarpScene:
    """Reference scene 1: 1 mesh + 1 box primitive."""
    plane = _make_plane_mesh_x(device=device, x=0.5, half_extent=10.0)
    return (
        WarpScene(num_scenes=1, device=device)
        .add(MeshGeom(meshes=[plane], scene_offsets=np.array([0, 1], dtype=np.int32)))
        .add(
            BoxGeom(
                half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
                scene_offsets=np.array([0, 1], dtype=np.int32),
            )
        )
    )


def _scene_2_single(device: wp_device_type) -> WarpScene:
    """Reference scene 2 (for uneven-partition test): 1 mesh only."""
    box = _make_box_mesh(device=device, half_extent=0.15)
    return WarpScene(num_scenes=1, device=device).add(
        MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32))
    )


def _multi_scene_2(device: wp_device_type) -> WarpScene:
    """Two-scene scene matching `_scene_0_single` ⊕ `_scene_1_single`:
    3 meshes split 2/1, sphere primitive in scene 0, box primitive in scene 1."""
    plane03 = _make_plane_mesh_x(device=device, x=0.3, half_extent=10.0)
    box02 = _make_box_mesh(device=device, half_extent=0.2)
    plane05 = _make_plane_mesh_x(device=device, x=0.5, half_extent=10.0)
    return (
        WarpScene(num_scenes=2, device=device)
        .add(MeshGeom(meshes=[plane03, box02, plane05], scene_offsets=np.array([0, 2, 3], dtype=np.int32)))
        .add(SphereGeom(radii=np.array([0.1], dtype=np.float32), scene_offsets=np.array([0, 1, 1], dtype=np.int32)))
        .add(
            BoxGeom(
                half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
                scene_offsets=np.array([0, 0, 1], dtype=np.int32),
            )
        )
    )


def _multi_scene_3(device: wp_device_type) -> WarpScene:
    """Three-scene scene matching `_scene_0_single` ⊕ `_scene_1_single` ⊕ `_scene_2_single`:
    4 meshes split 2/1/1, sphere primitive in scene 0, box primitive in scene 1, none in scene 2."""
    plane03 = _make_plane_mesh_x(device=device, x=0.3, half_extent=10.0)
    box02 = _make_box_mesh(device=device, half_extent=0.2)
    plane05 = _make_plane_mesh_x(device=device, x=0.5, half_extent=10.0)
    box015 = _make_box_mesh(device=device, half_extent=0.15)
    return (
        WarpScene(num_scenes=3, device=device)
        .add(
            MeshGeom(
                meshes=[plane03, box02, plane05, box015],
                scene_offsets=np.array([0, 2, 3, 4], dtype=np.int32),
            )
        )
        .add(SphereGeom(radii=np.array([0.1], dtype=np.float32), scene_offsets=np.array([0, 1, 1, 1], dtype=np.int32)))
        .add(
            BoxGeom(
                half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
                scene_offsets=np.array([0, 0, 1, 1], dtype=np.int32),
            )
        )
    )


def _run_scene_task(robot, q_np, scene, device, scene_indices=None, broad_phase=False):
    """Build state + task from `(robot, q_np, scene)` and return
    `(residual_np, jacobian_np)`. Used by both multi-scene and per-scene reference runs."""
    state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device))
    state = robot.forward_kinematics(state)
    state = robot.transform_collision_spheres(state)
    state = robot.compute_motion_subspace(state)
    task = SceneCollisionTask(
        robot=robot,
        scene=scene,
        scene_indices=scene_indices,
        use_link_bounding_sphere_filter=broad_phase,
        weight=1.0,
        margin=0.02,
        residual_mode="abs",
    )
    residual = task.compute_weighted_residual(VarValues(robot=state)).numpy().copy()
    jacobian = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy().copy()
    return residual, jacobian


class TestWarpSceneCollisionTaskMultiScene:
    """Multi-scene `SceneCollisionTask`: one task driving a multi-scene `WarpScene`
    with a batch-partition must match the concat of per-scene single-scene reference tasks
    run on the corresponding batch slices. Exercised with heterogeneous geometry counts and
    mixed primitive types across scenes."""

    def _assert_multi_scene_matches_refs(self, robot, q_np, partition, scene_builders, device, broad_phase):
        multi_scene = _multi_scene_3(device) if len(partition) == 3 else _multi_scene_2(device)
        bounds = np.concatenate([partition, [q_np.shape[0]]])
        scene_indices = wp.from_numpy(
            np.repeat(np.arange(len(partition), dtype=np.int32), np.diff(bounds)), dtype=wp.int32, device=device
        )
        r_multi, j_multi = _run_scene_task(
            robot,
            q_np,
            multi_scene,
            device,
            scene_indices=scene_indices,
            broad_phase=broad_phase,
        )
        ref_r_parts, ref_j_parts = [], []
        for s, build_ref_scene in enumerate(scene_builders):
            lo, hi = int(bounds[s]), int(bounds[s + 1])
            if hi == lo:
                continue
            r_s, j_s = _run_scene_task(
                robot,
                q_np[lo:hi],
                build_ref_scene(device),
                device,
                broad_phase=broad_phase,
            )
            ref_r_parts.append(r_s)
            ref_j_parts.append(j_s)
        r_ref = np.concatenate(ref_r_parts, axis=0)
        j_ref = np.concatenate(ref_j_parts, axis=0)
        assert r_multi.shape == r_ref.shape
        assert j_multi.shape == j_ref.shape
        assert np.allclose(r_multi, r_ref, atol=1e-5, rtol=1e-5)
        assert np.allclose(j_multi, j_ref, atol=1e-5, rtol=1e-5)

    def test_matches_per_scene_single_scene_reference(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        np.random.seed(0)
        B = 4
        q_np = (np.random.randn(B, robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        self._assert_multi_scene_matches_refs(
            robot,
            q_np,
            partition=np.array([0, B // 2], dtype=np.int64),
            scene_builders=[_scene_0_single, _scene_1_single],
            device=device,
            broad_phase=False,
        )

    def test_broad_phase_filter_multi_scene(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        np.random.seed(1)
        B = 4
        q_np = (np.random.randn(B, robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        self._assert_multi_scene_matches_refs(
            robot,
            q_np,
            partition=np.array([0, B // 2], dtype=np.int64),
            scene_builders=[_scene_0_single, _scene_1_single],
            device=device,
            broad_phase=True,
        )

    def test_uneven_partition(self, panda_robot_with_collision):
        """3 scenes with partition [0, 1, B-1] → scene 0 gets 1 batch entry, scene 1 gets B-2,
        scene 2 gets 1. Each scene has different geometry counts (2+1 / 1+1 / 1+0)."""
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        np.random.seed(2)
        B = 6
        q_np = (np.random.randn(B, robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        self._assert_multi_scene_matches_refs(
            robot,
            q_np,
            partition=np.array([0, 1, B - 1], dtype=np.int64),
            scene_builders=[_scene_0_single, _scene_1_single, _scene_2_single],
            device=device,
            broad_phase=False,
        )

    def test_rejects_bad_scene_indices(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        scene = _multi_scene_2(device)
        q_np = np.zeros((4, robot.num_actuated_joints), dtype=np.float32)
        with pytest.raises(TypeError, match="wp.int32"):
            _run_scene_task(robot, q_np, scene, device, scene_indices=wp.zeros(4, dtype=wp.float32))
        with pytest.raises(TypeError, match="1D"):
            _run_scene_task(robot, q_np, scene, device, scene_indices=wp.zeros((4, 1), dtype=wp.int32))
        with pytest.raises(ValueError, match="batch_size"):
            _run_scene_task(robot, q_np, scene, device, scene_indices=wp.zeros(3, dtype=wp.int32))


class TestWarpCollisionTaskBroadPhaseFilter:
    """`use_link_bounding_sphere_filter=True` must produce the same cost+gradient
    as the no-filter path on a `WarpScene` SDF volume - for both per-pose and
    trajectory states."""

    def _build_scene_and_state(self, robot, B: int, T: int, padding: float, box_pos, box_half: float, device_str: str):
        if not wp.is_cuda_available():
            pytest.skip("CUDA required for SDF volumes")

        device = wp.get_device(device_str)
        full_extents = (2 * box_half, 2 * box_half, 2 * box_half)
        box_mesh = trimesh.creation.box(full_extents)
        box_mesh.apply_translation(np.asarray(box_pos, dtype=np.float64))
        vol = mesh_to_sdf_volume(box_mesh, voxel_size=box_half / 8.0, padding=box_half * 0.5, device=device_str)
        scene = WarpScene(num_scenes=1, device=device_str).add(
            VolumeGeom(sdf_volumes=[vol], scene_offsets=np.array([0, 1], dtype=np.int32))
        )

        np.random.seed(0)
        D = robot.num_actuated_joints
        if T == 1:
            q_np = (np.random.randn(B, D).astype(np.float32) * 0.05) + robot.spec.midrange_q[None, :].astype(np.float32)
        else:
            q_np = (np.random.randn(B, T, D).astype(np.float32) * 0.05) + robot.spec.midrange_q[None, None, :].astype(
                np.float32
            )
        q_wp = wp.from_numpy(q_np, dtype=wp.float32, device=device)
        state = robot.state(q=q_wp)

        sphere_indices = list(range(int(robot.spec.local_collision_sphere_centers.shape[0])))
        cost_buf = wp.zeros((B,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((B, state.tangent_dim), dtype=wp.float32, device=device)
        return state, scene, sphere_indices, cost_buf, grad_buf, padding, device

    def _run_two_tasks(self, robot, state, scene, sphere_indices, padding):
        device = state.q.device
        cost_a = wp.zeros((state.batch_size,), dtype=wp.float32, device=device)
        grad_a = wp.zeros((state.batch_size, state.tangent_dim), dtype=wp.float32, device=device)
        cost_b = wp.zeros((state.batch_size,), dtype=wp.float32, device=device)
        grad_b = wp.zeros((state.batch_size, state.tangent_dim), dtype=wp.float32, device=device)
        task_no_filter = SceneCollisionTask(
            robot=robot,
            scene=scene,
            weight=2.0,
            margin=padding,
            sphere_indices=sphere_indices,
            use_link_bounding_sphere_filter=False,
        )
        task_with_filter = SceneCollisionTask(
            robot=robot,
            scene=scene,
            weight=2.0,
            margin=padding,
            sphere_indices=sphere_indices,
            use_link_bounding_sphere_filter=True,
        )
        task_no_filter.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost_a, out_gradient=grad_a)
        task_with_filter.compute_weighted_cost_and_gradient(
            VarValues(robot=state), out_cost=cost_b, out_gradient=grad_b
        )
        return cost_a.numpy(), grad_a.numpy(), cost_b.numpy(), grad_b.numpy()

    def test_filter_matches_no_filter_per_pose(self, panda_robot_with_collision):
        state, scene, sphere_indices, _, _, padding, _ = self._build_scene_and_state(
            panda_robot_with_collision,
            B=4,
            T=1,
            padding=0.05,
            box_pos=(0.4, 0.0, 0.5),
            box_half=0.1,
            device_str="cuda:0",
        )
        cost_a, grad_a, cost_b, grad_b = self._run_two_tasks(
            panda_robot_with_collision, state, scene, sphere_indices, padding
        )
        np.testing.assert_allclose(cost_a, cost_b, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(grad_a, grad_b, atol=1e-4, rtol=1e-4)
        # And the filter must actually skip something here (some link bounding spheres are far from the box).
        assert np.max(np.abs(grad_a)) > 1e-4, "expected non-trivial gradient signal"

    def test_filter_matches_no_filter_trajectory(self, panda_robot_with_collision):
        state, scene, sphere_indices, _, _, padding, _ = self._build_scene_and_state(
            panda_robot_with_collision,
            B=2,
            T=4,
            padding=0.05,
            box_pos=(0.4, 0.0, 0.5),
            box_half=0.1,
            device_str="cuda:0",
        )
        cost_a, grad_a, cost_b, grad_b = self._run_two_tasks(
            panda_robot_with_collision, state, scene, sphere_indices, padding
        )
        np.testing.assert_allclose(cost_a, cost_b, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(grad_a, grad_b, atol=1e-4, rtol=1e-4)
        assert np.max(np.abs(grad_a)) > 1e-4, "expected non-trivial gradient signal"


class TestWarpBasePositionLimit:
    """Test PositionLimit base bounds: residual, sign, frame, fixed-base guard, and offset."""

    def _state(self, robot, z: float, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
        T_base_np = np.array([0.0, 0.0, z, *quat_wxyz], dtype=np.float32).reshape(1, 7)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        return robot.state(q=q, T_world_base=T_base)

    def test_residual_below_lo(self, panda_robot):
        state = self._state(panda_robot, z=0.1)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=5.0)
        r = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        assert r.shape == (1, 1)
        np.testing.assert_allclose(r[0, 0], 5.0 * (0.3 - 0.1), atol=1e-5)

    def test_residual_above_hi(self, panda_robot):
        state = self._state(panda_robot, z=1.0)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=5.0)
        r = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        np.testing.assert_allclose(r[0, 0], 5.0 * (1.0 - 0.76), atol=1e-5)

    def test_residual_inside_range_is_zero(self, panda_robot):
        state = self._state(panda_robot, z=0.5)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=5.0)
        r = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        np.testing.assert_allclose(r[0, 0], 0.0, atol=1e-7)

    def test_jacobian_sign_identity_rotation(self, panda_robot):
        # below lo: sign = -1, identity rotation → jacobian translation row = (0,0,-w)
        state = self._state(panda_robot, z=0.1)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=7.0)
        J = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        assert J.shape == (1, 1, 6 + panda_robot.num_actuated_joints)
        np.testing.assert_allclose(J[0, 0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(J[0, 0, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(J[0, 0, 2], -7.0, atol=1e-6)
        np.testing.assert_allclose(J[0, 0, 3:], 0.0, atol=1e-6)

        # above hi: sign = +1
        state2 = self._state(panda_robot, z=1.0)
        J2 = task.compute_weighted_jacobian_analytic(VarValues(robot=state2)).numpy()
        np.testing.assert_allclose(J2[0, 0, 2], 7.0, atol=1e-6)

    def test_jacobian_frame_aware(self, panda_robot):
        # 90° yaw: body x-axis points along world y. d(world_z)/d(body_twist) row should be (0,0,1)*sign*weight still,
        # because rotation about z preserves world-z. Use 90° pitch (about world y) instead, which maps body-z → world-x.
        # quat for 90° pitch about world y: wxyz = (cos(45°), 0, sin(45°), 0)
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        state = self._state(panda_robot, z=0.1, quat_wxyz=(c, 0.0, s, 0.0))
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=1.0)
        J = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        # R = quat_to_R for this quaternion (pitch by 90°):
        #   R = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]] = [[0,0,1],[0,1,0],[-1,0,0]]
        # row 2 (world z) of R = (-1, 0, 0). sign = -1 (below lo). So J[0..2] = (+1, 0, 0) * weight.
        np.testing.assert_allclose(J[0, 0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(J[0, 0, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(J[0, 0, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(J[0, 0, 3:], 0.0, atol=1e-6)

    def test_no_floating_base_returns_zero(self, panda_robot):
        # Fixed-base RobotState: residual and jacobian buffers stay at their zero initialization.
        q = wp.from_numpy(panda_robot.spec.zero_q, dtype=wp.float32)
        state = panda_robot.state(q=q)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=10.0)
        r = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        J = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()
        np.testing.assert_allclose(r, 0.0, atol=1e-7)
        np.testing.assert_allclose(J, 0.0, atol=1e-7)

    def test_col_offset_writes_correct_columns(self, panda_robot):
        state = self._state(panda_robot, z=0.1)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=3.0)
        # a leading pad var pushes the "robot" leaf's tangent block off zero
        pad = SE3Var(se3_identity(shape=(1,), device=state.q.device))
        vals = VarValues(pad=pad, robot=state)
        col_offset = vals.tangent_offset("robot")
        J = wp.zeros((1, 1, vals.tangent_dim), dtype=wp.float32, device=state.q.device)
        task.compute_weighted_jacobian_analytic(vals, out_jacobian=J)
        J_np = J.numpy()
        # below lo, identity rotation → row entry at col_offset+2 is -weight
        np.testing.assert_allclose(J_np[0, 0, col_offset + 2], -3.0, atol=1e-6)
        # Columns outside [col_offset, col_offset+6) remain zero.
        np.testing.assert_allclose(J_np[0, 0, :col_offset], 0.0, atol=1e-7)
        np.testing.assert_allclose(J_np[0, 0, col_offset + 6 :], 0.0, atol=1e-7)

    def test_cost_gradient_matches_residual_jacobian(self, panda_robot):
        state = self._state(panda_robot, z=0.1)
        values = VarValues(robot=state)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=3.0)
        residual = task.compute_weighted_residual(values).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(values).numpy()
        cost = wp.zeros(1, dtype=wp.float32, device=state.q.device)
        gradient = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=state.q.device)
        task.compute_weighted_cost_and_gradient(values, out_cost=cost, out_gradient=gradient)
        np.testing.assert_allclose(cost.numpy(), 0.5 * (residual**2).sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(gradient.numpy(), np.einsum("brd,br->bd", jacobian, residual), atol=1e-6)

    def test_combined_joint_and_base_rows(self, panda_robot):
        q = panda_robot.spec.actuated_joint_limits[:, 0] - 0.1
        T_base = wp.from_numpy(np.array([[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp_vec7)
        state = panda_robot.state(q=wp.from_numpy(q.astype(np.float32)), T_world_base=T_base)
        task = PositionLimit(panda_robot, weight=2.0, base_axis=2, base_bounds=(0.3, 0.76), base_weight=5.0)
        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        assert residual.shape == (1, panda_robot.num_actuated_joints + 1)
        np.testing.assert_allclose(residual[0, :-1], 0.2, atol=1e-5)
        np.testing.assert_allclose(residual[0, -1], 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
