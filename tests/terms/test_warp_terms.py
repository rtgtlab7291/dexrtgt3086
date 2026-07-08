"""Unit tests for Warp backend terms."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.lie.warp_se3 import WarpSE3
from robokit.robo.robot import Robot
from robokit.terms.warp.base_damping_task import WarpBaseDampingTask
from robokit.terms.warp.base_step_limit import WarpBaseStepLimit
from robokit.terms.warp.collision_task import WarpCollisionTask
from robokit.terms.warp.frame_task import WarpFrameTask
from robokit.terms.warp.position_limit import WarpPositionLimit
from robokit.terms.warp.position_task import WarpPositionTask
from robokit.terms.warp.rest_task import WarpRestTask
from robokit.terms.warp.rotation_task import WarpRotationTask
from robokit.terms.warp.smoothness_task import WarpSmoothnessTask
from robokit.terms.warp.velocity_limit_task import WarpVelocityLimitTask
from robokit.utils.warp_utils import wp_device_type, wp_vec7


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


@pytest.fixture
def panda_robot():
    """Load Panda robot with Warp backend."""
    urdf = load_robot_description("panda_description")
    return Robot.load(urdf, backend="warp")


class TestWarpPositionLimit:
    """Test WarpPositionLimit with and without floating base."""

    def test_position_limit_fixed_base(self, panda_robot):
        robot = panda_robot
        q_np = robot.spec.zero_q
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)

        position_limit = WarpPositionLimit(robot=robot, weight=1.0, batch_size=1)
        residual = position_limit.compute_weighted_residual(state)
        jacobian = position_limit.compute_weighted_jacobian(state)

        assert residual.shape == (1, robot.num_actuated_joints)
        assert jacobian.shape == (1, robot.num_actuated_joints, robot.num_actuated_joints)

        # At zero_q, residual should be small (depends on limits)
        residual_np = residual.numpy()
        assert np.all(residual_np >= 0)

    def test_position_limit_floating_base(self, panda_robot):
        robot = panda_robot
        q_np = robot.spec.zero_q
        q = wp.from_numpy(q_np, dtype=wp.float32)
        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        T_base = WarpSE3(wp.from_numpy(T_base_np.reshape(1, 7), dtype=wp_vec7))
        state = robot.state(q=q, T_world_base=T_base)

        position_limit = WarpPositionLimit(robot=robot, weight=1.0, batch_size=1)
        residual = position_limit.compute_weighted_residual(state)
        jacobian = position_limit.compute_weighted_jacobian(state)

        assert residual.shape == (1, robot.num_actuated_joints)
        # Jacobian should have base columns (6) + joint columns
        assert jacobian.shape == (1, robot.num_actuated_joints, 6 + robot.num_actuated_joints)


class TestWarpRestTask:
    """Test WarpRestTask."""

    def test_rest_task_fixed_base(self, panda_robot):
        robot = panda_robot
        rest_q = robot.spec.midrange_q
        q = wp.from_numpy(rest_q + 0.1, dtype=wp.float32)
        state = robot.state(q=q)

        rest_task = WarpRestTask(robot=robot, rest_q=rest_q, weight=0.1, batch_size=1)
        residual = rest_task.compute_weighted_residual(state)

        # Residual should be weighted 0.1
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        expected = 0.1 * 0.1  # diff * weight
        assert np.allclose(residual_np, expected, atol=1e-5)


class TestWarpSmoothnessTask:
    """Test WarpSmoothnessTask."""

    def test_smoothness_task_fixed_base(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        state_prev = robot.state(q=q_prev)
        state_curr = robot.state(q=q_curr)

        smooth_task = WarpSmoothnessTask(robot=robot, weight=0.5, batch_size=1)
        residual = smooth_task.compute_weighted_residual(state_curr, state_prev)

        # Residual should be weighted difference
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        expected = 0.1 * 0.5  # diff * weight
        assert np.allclose(residual_np, expected, atol=1e-5)

    def test_smoothness_task_no_prev_state(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q)

        smooth_task = WarpSmoothnessTask(robot=robot, weight=0.5, batch_size=1)
        residual = smooth_task.compute_weighted_residual(state)

        # Should return buffer (already zeroed by caller or empty)
        assert residual is not None

    def test_smoothness_task_floating_base_residual_dim_is_static(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)
        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_base = WarpSE3(wp.from_numpy(T_base_np, dtype=wp_vec7))
        state_prev = robot.state(q=q_prev, T_world_base=T_base)
        state_curr = robot.state(q=q_curr, T_world_base=T_base)

        smooth_task = WarpSmoothnessTask(
            robot=robot,
            prev_var=state_prev,
            weight=0.5,
            base_weight=0.2,
            batch_size=1,
        )
        residual_dim_before = smooth_task.residual_dim

        residual = smooth_task.compute_weighted_residual(state_curr)
        jacobian = smooth_task.compute_weighted_jacobian(state_curr)

        assert residual_dim_before == robot.num_actuated_joints + 6
        assert smooth_task.residual_dim == residual_dim_before
        assert residual.shape == (1, residual_dim_before)
        assert jacobian.shape == (1, residual_dim_before, 6 + robot.num_actuated_joints)

    def test_smoothness_task_residual_dim_static_after_set_prev_state_with_floating_base(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_base = WarpSE3(wp.from_numpy(T_base_np, dtype=wp_vec7))
        state_prev = robot.state(q=q_prev, T_world_base=T_base)
        state_curr = robot.state(q=q_curr, T_world_base=T_base)

        smooth_task = WarpSmoothnessTask(robot=robot, prev_var=None, weight=0.5, base_weight=None, batch_size=1)
        residual_dim_before = smooth_task.residual_dim
        smooth_task.set_prev_state(state_prev)
        residual = smooth_task.compute_weighted_residual(state_curr)
        jacobian = smooth_task.compute_weighted_jacobian(state_curr)

        assert residual_dim_before == robot.num_actuated_joints
        assert smooth_task.residual_dim == residual_dim_before
        assert residual.shape == (1, residual_dim_before)
        assert jacobian.shape == (1, residual_dim_before, 6 + robot.num_actuated_joints)

    def test_smoothness_task_base_enabled_rejects_nonfloating_var(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_base = WarpSE3(wp.from_numpy(T_base_np, dtype=wp_vec7))
        state_prev = robot.state(q=q_prev, T_world_base=T_base)
        state_curr = robot.state(q=q_curr)

        smooth_task = WarpSmoothnessTask(
            robot=robot,
            prev_var=state_prev,
            weight=0.5,
            base_weight=0.2,
            batch_size=1,
        )
        with pytest.raises(ValueError, match="configured with base residuals"):
            smooth_task.compute_weighted_residual(state_curr)


class TestWarpBaseDampingTask:
    """Test WarpBaseDampingTask."""

    def test_base_damping_floating_base(self, panda_robot):
        robot = panda_robot
        T_base_np = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        T_base = WarpSE3(wp.from_numpy(T_base_np.reshape(1, 7), dtype=wp_vec7))
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q, T_world_base=T_base)

        damping_task = WarpBaseDampingTask(robot=robot, weight=0.1, batch_size=1)
        residual = damping_task.compute_weighted_residual(state)
        jacobian = damping_task.compute_weighted_jacobian(state)

        # Residual should be 6D
        assert residual.shape == (1, 6)

        # Jacobian should be [1, 6, 6 + num_joints]
        assert jacobian.shape == (1, 6, 6 + robot.num_actuated_joints)


class TestWarpVelocityLimitTask:
    """Test WarpVelocityLimitTask."""

    def test_velocity_limit_task(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.5, dtype=wp.float32)  # Large change

        state_prev = robot.state(q=q_prev)
        state_curr = robot.state(q=q_curr)

        dt = 0.1
        vel_task = WarpVelocityLimitTask(robot=robot, dt=dt, weight=1.0, batch_size=1)
        residual = vel_task.compute_weighted_residual(state_curr, state_prev)
        jacobian = vel_task.compute_weighted_jacobian(state_curr, state_prev)

        # Residual should be positive where velocity exceeds limits
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        assert np.all(residual_np >= 0)

        # Jacobian shape should be correct
        assert jacobian.shape == (1, robot.num_actuated_joints, robot.num_actuated_joints)


class TestWarpIntegration:
    """Test WarpRobotState.integrate() with floating base."""

    def test_integrate_floating_base(self, panda_robot):
        robot = panda_robot
        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        T_base = WarpSE3(wp.from_numpy(T_base_np.reshape(1, 7), dtype=wp_vec7))
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q, T_world_base=T_base)

        # Create velocity with base motion
        velocity_np = np.zeros(6 + robot.num_actuated_joints, dtype=np.float32)
        velocity_np[0] = 0.1  # Move base in x
        velocity = wp.from_numpy(velocity_np.reshape(1, -1), dtype=wp.float32)

        new_state = state.integrate(velocity)

        # Check base has moved
        new_T_base_np = new_state.T_world_base.xyz_wxyz.numpy()[0]
        assert new_T_base_np[0] > 0  # x should have increased


class TestWarpBaseStepLimit:
    """Test WarpBaseStepLimit (soft constraint)."""

    def test_base_step_limit_floating_base(self, panda_robot):
        robot = panda_robot
        T_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        T_base = WarpSE3(wp.from_numpy(T_base_np.reshape(1, 7), dtype=wp_vec7))
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q, T_world_base=T_base)

        # Lock z, roll, pitch for planar base (indices 2, 3, 4)
        limit = WarpBaseStepLimit(lock_indices=[2, 3, 4], weight=10.0, batch_size=1)
        residual = limit.compute_weighted_residual(state)
        jacobian = limit.compute_weighted_jacobian(state)

        # Should have 3 residuals (one per locked DOF)
        assert residual.shape == (1, 3)

        # Jacobian should be [1, 3, 14] for batch=1, 3 locked, 6 base + 8 joints
        assert jacobian.shape == (1, 3, 6 + robot.num_actuated_joints)

        # Check Jacobian structure: weighted identity for locked columns
        jac_np = jacobian.numpy()[0]
        assert np.allclose(jac_np[:, 2], np.array([10.0, 0.0, 0.0]))  # col 2 (z) with weight
        assert np.allclose(jac_np[:, 3], np.array([0.0, 10.0, 0.0]))  # col 3 (roll)
        assert np.allclose(jac_np[:, 4], np.array([0.0, 0.0, 10.0]))  # col 4 (pitch)

        # Joint columns should be zero
        assert np.allclose(jac_np[:, 6:], 0.0)

    def test_base_step_limit_no_floating_base(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q)

        limit = WarpBaseStepLimit(lock_indices=[2, 3, 4], weight=10.0, batch_size=1)
        residual = limit.compute_weighted_residual(state)

        # Should return zeros when no floating base
        assert residual.shape == (1, 3)
        assert np.allclose(residual.numpy(), 0.0)


class TestWarpAutodiffJacobian:
    """Test consistency between analytic and autodiff Jacobians for Warp terms."""

    def test_frame_task_jacobian(self, panda_robot):
        robot = panda_robot
        q_np = np.random.randn(robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        q = wp.from_numpy(q_np, dtype=wp.float32)

        T_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = WarpSE3(wp.from_numpy(T_base_np, dtype=wp_vec7))

        state = robot.state(q=q, T_world_base=T_base)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")

        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = WarpSE3(wp.from_numpy(target_np, dtype=wp_vec7))

        frame_task = WarpFrameTask(
            robot=robot,
            frame_index=ee_idx,
            T_world_target=T_world_target,
            position_weight=1.0,
            orientation_weight=1.0,
        )

        analytic_jacobian = frame_task.compute_weighted_jacobian_analytic(state)
        autodiff_jacobian = frame_task.compute_weighted_jacobian_autodiff(state)

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4, rtol=1e-4)

    def test_collision_task_jacobian(self):
        device = wp.get_device("cpu")
        robot = Robot.load(
            load_robot_description("panda_description"),
            backend="warp",
            load_collision_spheres=True,
            collision_spheres_path=str(
                Path(__file__).parent.parent.parent / "assets" / "collision_spheres" / "franka.yaml"
            ),
        )
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
        task = WarpCollisionTask(
            robot=robot,
            scene_meshes=[mesh],
            weight=1.0,
            margin=0.01,
            beta=25.0,
            sphere_indices=sphere_indices,
        )

        task.compute_weighted_residual(state)
        analytic_jacobian = task.compute_weighted_jacobian_analytic(state)
        autodiff_jacobian = task.compute_weighted_jacobian_autodiff(state)

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert analytic_np.shape == autodiff_np.shape
        assert np.allclose(analytic_np, autodiff_np, atol=3e-4, rtol=3e-4)


class TestWarpCollisionTaskResidual:
    def test_residual_near_zero_when_far_from_mesh(self):
        device = wp.get_device("cpu")
        robot = Robot.load(
            load_robot_description("panda_description"),
            backend="warp",
            load_collision_spheres=True,
            collision_spheres_path=str(
                Path(__file__).parent.parent.parent / "assets" / "collision_spheres" / "franka.yaml"
            ),
        )
        assert robot.spec.has_collision_spheres

        mesh = _make_box_mesh(device=device, half_extent=2.0)

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_base_np = np.array([[100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = WarpSE3(wp.from_numpy(T_base_np, dtype=wp_vec7, device=device))
        state = robot.state(q=q, T_world_base=T_base)
        state = robot.transform_collision_spheres(state)

        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        task = WarpCollisionTask(
            robot=robot,
            scene_meshes=[mesh],
            weight=1.0,
            margin=0.0,
            beta=50.0,
            sphere_indices=sphere_indices,
        )

        residual = task.compute_weighted_residual(state).numpy()[0]
        assert residual.shape == (len(sphere_indices),)
        assert float(np.max(residual)) < 1e-6

    def test_residual_positive_when_surface_within_radius(self):
        device = wp.get_device("cpu")
        robot = Robot.load(
            load_robot_description("panda_description"),
            backend="warp",
            load_collision_spheres=True,
            collision_spheres_path=str(
                Path(__file__).parent.parent.parent / "assets" / "collision_spheres" / "franka.yaml"
            ),
        )
        assert robot.spec.has_collision_spheres

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = WarpSE3(wp.from_numpy(T_base_np, dtype=wp_vec7, device=device))
        state = robot.forward_kinematics(robot.state(q=q, T_world_base=T_base))
        state = robot.transform_collision_spheres(state)

        centers_np = state.world_collision_sphere_centers.numpy()[0]
        c = centers_np[0]
        r = float(robot.spec.local_collision_sphere_radii[0])

        # Place a plane at distance r/2 from the sphere center along +x.
        mesh = _make_plane_mesh_x(device=device, x=float(c[0]) + 0.5 * r, half_extent=10.0)

        sphere_indices = [0]
        task = WarpCollisionTask(
            robot=robot,
            scene_meshes=[mesh],
            weight=1.0,
            margin=0.0,
            beta=25.0,
            sphere_indices=sphere_indices,
        )

        residual = task.compute_weighted_residual(state).numpy()[0]
        assert residual.shape == (1,)
        assert float(residual[0]) > 1e-4

    def test_residual_near_zero_when_surface_farther_than_radius(self):
        device = wp.get_device("cpu")
        robot = Robot.load(
            load_robot_description("panda_description"),
            backend="warp",
            load_collision_spheres=True,
            collision_spheres_path=str(
                Path(__file__).parent.parent.parent / "assets" / "collision_spheres" / "franka.yaml"
            ),
        )
        assert robot.spec.has_collision_spheres

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = WarpSE3(wp.from_numpy(T_base_np, dtype=wp_vec7, device=device))
        state = robot.forward_kinematics(robot.state(q=q, T_world_base=T_base))
        state = robot.transform_collision_spheres(state)

        centers_np = state.world_collision_sphere_centers.numpy()[0]
        c = centers_np[0]
        r = float(robot.spec.local_collision_sphere_radii[0])

        mesh = _make_plane_mesh_x(device=device, x=float(c[0]) + 10.0 * r, half_extent=10.0)

        task = WarpCollisionTask(
            robot=robot,
            scene_meshes=[mesh],
            weight=1.0,
            margin=0.0,
            beta=50.0,
            sphere_indices=[0],
        )

        residual = task.compute_weighted_residual(state).numpy()[0]
        assert residual.shape == (1,)
        assert float(residual[0]) < 1e-6

    def test_position_limit_jacobian(self, panda_robot):
        robot = panda_robot
        q_np = (np.zeros(robot.num_actuated_joints) + 10.0).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        pos_limit = WarpPositionLimit(robot=robot, weight=1.0, batch_size=1)

        analytic_jacobian = pos_limit.compute_weighted_jacobian_analytic(state)
        autodiff_jacobian = pos_limit.compute_weighted_jacobian_autodiff(state)

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4)

    def test_rest_task_jacobian_fixed_base(self, panda_robot):
        robot = panda_robot
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)

        rest_task = WarpRestTask(
            robot=robot,
            rest_q=robot.spec.midrange_q,
            weight=1.0,
            batch_size=1,
        )

        analytic_jacobian = rest_task.compute_weighted_jacobian_analytic(state)
        autodiff_jacobian = rest_task.compute_weighted_jacobian_autodiff(state)

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4, rtol=1e-4)


class TestWarpPositionTask:
    """Test WarpPositionTask residual and Jacobian correctness."""

    def test_residual_is_world_frame_position_error(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)

        ee_idx = robot.link_names.index("panda_hand")
        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = WarpSE3(wp.from_numpy(target_np, dtype=wp_vec7))

        weight = 2.0
        pos_task = WarpPositionTask(robot, ee_idx, T_world_target, weight=weight)
        pos_res = pos_task.compute_weighted_residual(state).numpy()

        ee_pose = state.T_world_link.xyz_wxyz.numpy()[0, ee_idx]
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
        T_world_target = WarpSE3(wp.from_numpy(target_np, dtype=wp_vec7))

        pos_task = WarpPositionTask(robot, ee_idx, T_world_target, weight=2.0)

        analytic_jac = pos_task.compute_weighted_jacobian_analytic(state).numpy()
        autodiff_jac = pos_task.compute_weighted_jacobian_autodiff(state).numpy()

        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)


class TestWarpRotationTask:
    """Test WarpRotationTask residual and Jacobian correctness."""

    def test_residual_is_world_frame_rotation_error(self, panda_robot):
        from scipy.spatial.transform import Rotation

        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)

        ee_idx = robot.link_names.index("panda_hand")
        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = WarpSE3(wp.from_numpy(target_np, dtype=wp_vec7))

        weight = 3.0
        rot_task = WarpRotationTask(robot, ee_idx, T_world_target, weight=weight)
        rot_res = rot_task.compute_weighted_residual(state).numpy()

        ee_pose = state.T_world_link.xyz_wxyz.numpy()[0, ee_idx]
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
        # Use the actual EE pose as target so rotation error is near zero
        ee_pose = state.T_world_link.xyz_wxyz.numpy()[0, ee_idx]
        target_np = ee_pose.reshape(1, 7).astype(np.float32)
        T_world_target = WarpSE3(wp.from_numpy(target_np, dtype=wp_vec7))

        rot_task = WarpRotationTask(robot, ee_idx, T_world_target, weight=3.0)

        analytic_jac = rot_task.compute_weighted_jacobian_analytic(state).numpy()
        autodiff_jac = rot_task.compute_weighted_jacobian_autodiff(state).numpy()

        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
