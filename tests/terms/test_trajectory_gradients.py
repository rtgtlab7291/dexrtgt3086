"""Trajectory-term gradients (sparse J^T r path): validate against finite differences and residuals."""

import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.sparse.trajectory_velocity_limit_task import TrajectoryVelocityLimitTask
from robokit.terms.task import sparse_cost_and_gradient
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.warp_utils import wp_vec7


BATCH = 2
FRAMES = 8


def _q_traj(robot, seed=0):
    rng = np.random.default_rng(seed)
    limits = robot.spec.actuated_joint_limits
    mid, span = (limits[:, 0] + limits[:, 1]) / 2, limits[:, 1] - limits[:, 0]
    q = mid[None, None] + 0.45 * span[None, None] * rng.uniform(-1, 1, size=(BATCH, FRAMES, limits.shape[0]))
    return q.astype(np.float32)


def _state(robot, q_np):
    return robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))


def _direct(task, state):
    cost = wp.zeros((BATCH,), dtype=wp.float32, device=state.q.device)
    grad = wp.zeros((BATCH, state.tangent_dim), dtype=wp.float32, device=state.q.device)
    task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost, out_gradient=grad)
    return cost.numpy(), grad.numpy()


def _sparse_ref(task, state):
    cost = wp.zeros((BATCH,), dtype=wp.float32, device=state.q.device)
    grad = wp.zeros((BATCH, state.tangent_dim), dtype=wp.float32, device=state.q.device)
    sparse_cost_and_gradient(task, VarValues(robot=state), out_cost=cost, out_gradient=grad)
    return cost.numpy(), grad.numpy()


def _fd_gradient(task, robot, q_np, sample_dims, eps=1e-3):
    def cost_at(q):
        r = task.compute_weighted_residual(VarValues(robot=_state(robot, q))).numpy()
        return 0.5 * (r.astype(np.float64) ** 2).sum(axis=-1)

    grads = {}
    for b, f, d in sample_dims:
        qp, qm = q_np.copy(), q_np.copy()
        qp[b, f, d] += eps
        qm[b, f, d] -= eps
        grads[(b, f, d)] = (cost_at(qp)[b] - cost_at(qm)[b]) / (2 * eps)
    return grads


class TestTrajectoryTaskCostAndGradient:
    def _check(self, task, robot, q_np, fd=True, rtol=1e-4, atol=1e-5):
        state = _state(robot, q_np)
        cost, grad = _direct(task, state)
        ref_cost, ref_grad = _sparse_ref(task, _state(robot, q_np))
        assert np.allclose(cost, ref_cost, rtol=1e-5, atol=1e-6), f"cost {cost} vs sparse {ref_cost}"
        assert np.allclose(grad, ref_grad, rtol=rtol, atol=atol), (
            f"max abs err vs sparse: {np.max(np.abs(grad - ref_grad)):.3e}"
        )
        # cost-only call (line-search path)
        cost_only = wp.zeros((BATCH,), dtype=wp.float32, device=state.q.device)
        task.compute_weighted_cost_and_gradient(VarValues(robot=_state(robot, q_np)), out_cost=cost_only)
        assert np.allclose(cost_only.numpy(), cost, rtol=1e-5, atol=1e-6)
        if not fd:
            return
        num_dofs = q_np.shape[-1]
        rng = np.random.default_rng(1)
        samples = [(b, int(rng.integers(FRAMES)), int(rng.integers(num_dofs))) for b in range(BATCH) for _ in range(4)]
        fd_grads = _fd_gradient(task, robot, q_np, samples)
        for (b, f, d), g_fd in fd_grads.items():
            g = grad[b, f * num_dofs + d]
            assert np.isclose(g, g_fd, rtol=5e-2, atol=1e-3), f"({b},{f},{d}): direct {g} vs fd {g_fd}"

    def test_start_config(self, panda_robot):
        dense = RestTask(robot=panda_robot, rest_q=panda_robot.spec.zero_q, weight=37.0)
        task = TrajectoryTask(dense, FRAMES, frame_indices=[0])
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_rest_config(self, panda_robot):
        dense = RestTask(robot=panda_robot, rest_q=panda_robot.spec.midrange_q, weight=0.5)
        task = TrajectoryTask(dense, FRAMES)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_smoothness(self, panda_robot):
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=2.0)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_jerk_smoothness(self, panda_robot):
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=1.5, dt=0.25, order=3)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_jerk_smoothness_with_executed_history(self, panda_robot):
        q_traj = _q_traj(panda_robot)
        history_np = _q_traj(panda_robot, seed=1)[:, :3]
        history_np[:, 2] = q_traj[:, 0]
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=1.5, dt=0.25, order=3)
        task.set_history(wp.from_numpy(history_np, dtype=wp.float32), num_seeds=1, history_valid=True)
        residual = task.compute_weighted_residual(VarValues(robot=_state(panda_robot, q_traj))).numpy()
        expected = 1.5 * np.diff(np.concatenate([history_np[:, :2], q_traj], axis=1), n=3, axis=1) / 0.25**3
        np.testing.assert_allclose(residual, expected.reshape(BATCH, -1), rtol=1e-5, atol=1e-4)
        self._check(task, panda_robot, q_traj)

    def test_accel2_smoothness(self, panda_robot):
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=1.5, dt=0.25, order=2)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_accel2_smoothness_with_executed_history(self, panda_robot):
        q_traj = _q_traj(panda_robot)
        history_np = _q_traj(panda_robot, seed=1)[:, :3]
        history_np[:, 2] = q_traj[:, 0]
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=1.5, dt=0.25, order=2)
        task.set_history(wp.from_numpy(history_np, dtype=wp.float32), num_seeds=1, history_valid=True)
        residual = task.compute_weighted_residual(VarValues(robot=_state(panda_robot, q_traj))).numpy()
        expected = 1.5 * np.diff(np.concatenate([history_np[:, :2], q_traj], axis=1), n=2, axis=1) / 0.25**2
        np.testing.assert_allclose(residual, expected[:, 1:].reshape(BATCH, -1), rtol=1e-5, atol=1e-4)
        self._check(task, panda_robot, q_traj)

    def test_velocity_limit(self, panda_robot):
        # dt small enough that random frame-to-frame jumps exceed the velocity limits
        task = TrajectoryVelocityLimitTask(panda_robot, FRAMES, dt=0.05, weight=3.0)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_joint_position_limit(self, panda_robot):
        task = TrajectoryTask(PositionLimit(robot=panda_robot, weight=10.0), FRAMES)
        q = _q_traj(panda_robot)
        limits = panda_robot.spec.actuated_joint_limits
        q[:, ::2] = limits[None, :, 1] + 0.1  # violate upper limits on alternating frames
        self._check(task, panda_robot, q)

    def test_end_frame_pose(self, panda_robot):
        robot = panda_robot
        ee = len(robot.link_names) - 1
        goal_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        target_np = np.tile(goal_state.get_T_world_link(ee).numpy(), (BATCH, 1))
        target = wp.from_numpy(target_np, dtype=wp_vec7)
        dense = FrameTask(
            robot=robot, frame_index=ee, T_world_target=target, position_weight=50.0, orientation_weight=20.0
        )
        task = TrajectoryTask(dense, FRAMES, frame_indices=[-1])
        self._check(task, robot, _q_traj(robot), fd=False, rtol=1e-3, atol=1e-4)

    def test_self_collision(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        # large margin so several sphere pairs are active
        task = TrajectorySelfCollisionTask(robot, FRAMES, weight=5.0, margin=0.1)
        if task.num_pairs == 0:
            pytest.skip("no active collision pairs")
        self._check(task, robot, _q_traj(robot), fd=False, rtol=1e-3, atol=1e-4)
