"""Unit tests for Torch backend terms."""

import pytest
import torch
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.lie.torch_se3 import TorchSE3
from robokit.robo.robot import Robot
from robokit.terms.torch.base_damping_task import TorchBaseDampingTask
from robokit.terms.torch.base_step_limit import TorchBaseStepLimit
from robokit.terms.torch.frame_task import TorchFrameTask
from robokit.terms.torch.position_limit import TorchPositionLimit
from robokit.terms.torch.rest_task import TorchRestTask
from robokit.terms.torch.smoothness_task import TorchSmoothnessTask
from robokit.terms.torch.velocity_limit_task import TorchVelocityLimitTask


@pytest.fixture
def panda_robot():
    """Load Panda robot with Torch backend."""
    urdf = load_robot_description("panda_description")
    return Robot.load(urdf, backend="torch")


class TestTorchPositionLimit:
    """Test TorchPositionLimit with and without floating base."""

    def test_position_limit_fixed_base(self, panda_robot):
        robot = panda_robot
        state = robot.state(q=robot.zero_q)

        position_limit = TorchPositionLimit(robot=robot, weight=1.0)
        residual = position_limit.compute_residual(state)
        jacobian = position_limit.compute_jacobian(state)

        assert residual.shape == (robot.num_actuated_joints,)
        assert jacobian.shape == (robot.num_actuated_joints, robot.num_actuated_joints)

        # At zero_q, residual should be small (depends on limits)
        assert torch.all(residual >= 0)

    def test_position_limit_floating_base(self, panda_robot):
        robot = panda_robot
        T_base = TorchSE3(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        state = robot.state(q=robot.zero_q, T_world_base=T_base)

        position_limit = TorchPositionLimit(robot=robot, weight=1.0)
        residual = position_limit.compute_residual(state)
        jacobian = position_limit.compute_jacobian(state)

        assert residual.shape == (robot.num_actuated_joints,)
        # Jacobian should have base columns (6) + joint columns
        assert jacobian.shape == (robot.num_actuated_joints, 6 + robot.num_actuated_joints)

        # Base columns should be zero
        assert torch.allclose(jacobian[:, :6], torch.zeros_like(jacobian[:, :6]))


class TestTorchRestTask:
    """Test TorchRestTask."""

    def test_rest_task_fixed_base(self, panda_robot):
        robot = panda_robot
        rest_q = robot.midrange_q
        state = robot.state(q=rest_q + 0.1)

        rest_task = TorchRestTask(robot=robot, rest_q=rest_q, weight=0.1)
        residual = rest_task.compute_residual(state)
        jacobian = rest_task.compute_jacobian(state)

        # Residual should be 0.1 for all joints
        assert torch.allclose(residual, torch.tensor(0.1), atol=1e-6)

        # Jacobian should be identity matrix
        assert jacobian.shape == (robot.num_actuated_joints, robot.num_actuated_joints)
        assert torch.allclose(jacobian, torch.eye(robot.num_actuated_joints))

    def test_rest_task_floating_base(self, panda_robot):
        robot = panda_robot
        rest_q = robot.midrange_q
        T_base = TorchSE3(torch.tensor([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        # Use unbatched identity to match T_base
        T_base_rest = TorchSE3(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

        state = robot.state(q=rest_q + 0.1, T_world_base=T_base)

        rest_task = TorchRestTask(
            robot=robot, rest_q=rest_q, T_world_base_rest=T_base_rest, weight=0.1, base_weight=0.05
        )
        residual = rest_task.compute_residual(state)
        jacobian = rest_task.compute_jacobian(state)

        # Residual should have joint part (8 dims) + base part (6 dims)
        assert residual.shape == (robot.num_actuated_joints + 6,)

        # Jacobian should be [num_joints + 6, 6 + num_joints]
        assert jacobian.shape == (robot.num_actuated_joints + 6, 6 + robot.num_actuated_joints)


class TestTorchSmoothnessTask:
    """Test TorchSmoothnessTask."""

    def test_smoothness_task_fixed_base(self, panda_robot):
        robot = panda_robot
        state_prev = robot.state(q=robot.zero_q)
        state_curr = robot.state(q=robot.zero_q + 0.1)

        smooth_task = TorchSmoothnessTask(robot=robot, weight=0.5)
        residual = smooth_task.compute_residual(state_curr, state_prev)
        jacobian = smooth_task.compute_jacobian(state_curr, state_prev)

        # Residual should be 0.1 for all joints
        assert torch.allclose(residual, torch.tensor(0.1), atol=1e-6)

        # Jacobian should be identity
        assert jacobian.shape == (robot.num_actuated_joints, robot.num_actuated_joints)
        assert torch.allclose(jacobian, torch.eye(robot.num_actuated_joints))

    def test_smoothness_task_no_prev_state(self, panda_robot):
        robot = panda_robot
        state = robot.state(q=robot.zero_q)

        smooth_task = TorchSmoothnessTask(robot=robot, weight=0.5)
        residual = smooth_task.compute_residual(state)

        # Should return zeros when no prev_state
        assert torch.allclose(residual, torch.zeros_like(residual))


class TestTorchBaseDampingTask:
    """Test TorchBaseDampingTask."""

    def test_base_damping_floating_base(self, panda_robot):
        robot = panda_robot
        T_base = TorchSE3(torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]))
        state = robot.state(q=robot.zero_q, T_world_base=T_base)

        damping_task = TorchBaseDampingTask(robot=robot, weight=0.1)
        residual = damping_task.compute_residual(state)
        jacobian = damping_task.compute_jacobian(state)

        # Residual should be 6D (base twist)
        assert residual.shape == (6,)

        # Jacobian should be [6, 6 + num_joints]
        assert jacobian.shape == (6, 6 + robot.num_actuated_joints)

        # Joint columns should be zero
        assert torch.allclose(jacobian[:, 6:], torch.zeros_like(jacobian[:, 6:]))

    def test_base_damping_no_floating_base(self, panda_robot):
        robot = panda_robot
        state = robot.state(q=robot.zero_q)

        damping_task = TorchBaseDampingTask(robot=robot, weight=0.1)
        residual = damping_task.compute_residual(state)

        # Should return zeros when no floating base
        assert residual.shape == (6,)
        assert torch.allclose(residual, torch.zeros_like(residual))


class TestTorchVelocityLimitTask:
    """Test TorchVelocityLimitTask."""

    def test_velocity_limit_task(self, panda_robot):
        robot = panda_robot
        state_prev = robot.state(q=robot.zero_q)
        state_curr = robot.state(q=robot.zero_q + 0.5)  # Large change

        dt = 0.1
        vel_task = TorchVelocityLimitTask(robot=robot, dt=dt, prev_state_var=state_prev, weight=1.0)
        residual = vel_task.compute_residual(state_curr)
        jacobian = vel_task.compute_jacobian(state_curr)

        # Residual should be positive where velocity exceeds limits
        assert residual.shape == (robot.num_actuated_joints,)
        assert torch.all(residual >= 0)

        # Jacobian shape should be correct
        assert jacobian.shape == (robot.num_actuated_joints, robot.num_actuated_joints)


class TestTorchFrameTask:
    """Test TorchFrameTask with floating base."""

    def test_frame_task_floating_base(self, panda_robot):
        robot = panda_robot
        T_base = TorchSE3(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        state = robot.state(q=robot.zero_q, T_world_base=T_base)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")
        T_ee = state.get_T_world_link(ee_idx)

        # Create frame task
        frame_task = TorchFrameTask(
            robot=robot,
            frame_index=ee_idx,
            T_world_target=T_ee,  # Same as current
            position_weight=1.0,
            orientation_weight=0.2,
        )

        residual = frame_task.compute_residual(state)
        jacobian = frame_task.compute_jacobian(state)

        # Residual should be near zero (tracking itself)
        assert residual.shape == (6,)
        assert torch.allclose(residual, torch.zeros(6), atol=1e-4)

        # Jacobian should include base columns
        assert jacobian.shape == (6, 6 + robot.num_actuated_joints)


class TestTorchBaseStepLimit:
    """Test TorchBaseStepLimit (soft constraint)."""

    def test_base_step_limit_floating_base(self, panda_robot):
        robot = panda_robot
        T_base = TorchSE3(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        state = robot.state(q=robot.zero_q, T_world_base=T_base)

        # Lock z, roll, pitch for planar base (indices 2, 3, 4)
        limit = TorchBaseStepLimit(lock_indices=[2, 3, 4], weight=10.0)
        residual = limit.compute_residual(state)
        jacobian = limit.compute_jacobian(state)

        # Should have 3 residuals (one per locked DOF)
        assert residual.shape == (3,)

        # Jacobian should be [3, 14] for 3 locked + 6 base + 8 joints
        assert jacobian.shape == (3, 6 + robot.num_actuated_joints)

        # Check Jacobian structure: identity for locked columns, zero elsewhere
        assert torch.allclose(jacobian[:, 2], torch.tensor([1.0, 0.0, 0.0]))  # col 2 (z)
        assert torch.allclose(jacobian[:, 3], torch.tensor([0.0, 1.0, 0.0]))  # col 3 (roll)
        assert torch.allclose(jacobian[:, 4], torch.tensor([0.0, 0.0, 1.0]))  # col 4 (pitch)

        # Joint columns should be zero
        assert torch.allclose(jacobian[:, 6:], torch.zeros_like(jacobian[:, 6:]))

    def test_base_step_limit_no_floating_base(self, panda_robot):
        robot = panda_robot
        state = robot.state(q=robot.zero_q)

        limit = TorchBaseStepLimit(lock_indices=[2, 3, 4], weight=10.0)
        residual = limit.compute_residual(state)

        # Should return zeros when no floating base
        assert residual.shape == (3,)
        assert torch.allclose(residual, torch.zeros(3))


class TestTorchAutodiffJacobian:
    """Test consistency between analytic and autodiff Jacobians."""

    @pytest.mark.parametrize("batch_shape", [(), (1,), (1, 1)])
    def test_frame_task_jacobian(self, panda_robot, batch_shape):
        robot = panda_robot
        q_shape = batch_shape + (robot.num_actuated_joints,)
        q = torch.randn(q_shape) * 0.1 + robot.midrange_q

        if batch_shape:
            T_base_data = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).expand(batch_shape + (7,))
            T_base = TorchSE3(T_base_data)
        else:
            T_base = TorchSE3(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

        state = robot.state(q=q, T_world_base=T_base)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")

        target_data = torch.tensor([0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]).to(q.device)
        if batch_shape:
            target_data = target_data.expand(batch_shape + (7,))
        T_world_target = TorchSE3(target_data)

        frame_task = TorchFrameTask(
            robot=robot,
            frame_index=ee_idx,
            T_world_target=T_world_target,
            position_weight=1.0,
            orientation_weight=1.0,
        )

        analytic_jacobian = frame_task.compute_jacobian_analytic(state)
        autodiff_jacobian = frame_task.compute_jacobian_autodiff(state)

        assert torch.allclose(analytic_jacobian, autodiff_jacobian, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(), (1,), (1, 1)])
    def test_position_limit_jacobian(self, panda_robot, batch_shape):
        robot = panda_robot
        q_shape = batch_shape + (robot.num_actuated_joints,)
        q = torch.zeros(q_shape) + 10.0

        state = robot.state(q=q)

        pos_limit = TorchPositionLimit(robot=robot, weight=1.0)

        analytic_jacobian = pos_limit.compute_jacobian_analytic(state)
        autodiff_jacobian = pos_limit.compute_jacobian_autodiff(state)

        assert torch.allclose(analytic_jacobian, autodiff_jacobian, atol=1e-4)

    @pytest.mark.parametrize("batch_shape", [(), (1,), (1, 1)])
    def test_rest_task_jacobian(self, panda_robot, batch_shape):
        robot = panda_robot
        q_shape = batch_shape + (robot.num_actuated_joints,)
        q = torch.randn(q_shape) * 0.1 + robot.midrange_q

        if batch_shape:
            T_base_data = torch.randn(batch_shape + (7,))
            T_base_data[..., 3:] = torch.nn.functional.normalize(T_base_data[..., 3:], dim=-1)
            T_base = TorchSE3(T_base_data)
        else:
            T_base = TorchSE3(torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]))

        T_base_rest = TorchSE3(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

        state = robot.state(q=q, T_world_base=T_base)

        rest_task = TorchRestTask(
            robot=robot, rest_q=robot.midrange_q, T_world_base_rest=T_base_rest, weight=1.0, base_weight=1.0
        )

        analytic_jacobian = rest_task.compute_jacobian_analytic(state)
        autodiff_jacobian = rest_task.compute_jacobian_autodiff(state)

        assert torch.allclose(analytic_jacobian, autodiff_jacobian, atol=1e-4, rtol=1e-4)
