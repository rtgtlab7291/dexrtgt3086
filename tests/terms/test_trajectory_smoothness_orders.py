"""Order-1 closed form and finite-difference validation for every smoothness order."""

import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask


_FRAMES = 6
_BATCH = 2


@pytest.fixture
def device():
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _dense_from_sparse(task, var):
    pattern = task.compute_sparse_jacobian_pattern(var)
    values = task.compute_weighted_sparse_jacobian_values(var).numpy()
    rows = pattern.row_indices.numpy()
    cols = pattern.col_indices.numpy()
    out = np.zeros((var.batch_size, task.residual_dim, var.tangent_dim), dtype=np.float64)
    for b in range(var.batch_size):
        for i in range(len(rows)):
            out[b, rows[i], cols[i]] += values[b, i]
    return out


def _traj(robot, device, seed=4):
    rng = np.random.default_rng(seed)
    limits = robot.spec.actuated_joint_limits
    q = rng.uniform(limits[:, 0], limits[:, 1], (_BATCH, _FRAMES, robot.spec.num_actuated_joints)).astype(np.float32)
    return wp.from_numpy(q, dtype=wp.float32, device=device)


def _var(robot, q_wp):
    return VarValues(robot=robot.state(q=q_wp))


def _history(robot, device, seed=9):
    rng = np.random.default_rng(seed)
    h = rng.uniform(-0.2, 0.2, (_BATCH, 3, robot.spec.num_actuated_joints)).astype(np.float32)
    return wp.from_numpy(h, dtype=wp.float32, device=device)


def _compare(legacy, merged, var):
    np.testing.assert_allclose(
        merged.compute_weighted_residual(var).numpy(),
        legacy.compute_weighted_residual(var).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(_dense_from_sparse(merged, var), _dense_from_sparse(legacy, var), atol=1e-5, rtol=1e-5)


class TestOrderEquivalence:
    def test_order1_matches_legacy_default(self, panda_robot, device):
        """order=1 with dt=1 must be bit-identical to the historical velocity task."""
        robot = panda_robot
        q = _traj(robot, device)
        var = _var(robot, q)
        rng = np.random.default_rng(2)
        reference = rng.uniform(-0.1, 0.1, (_BATCH, _FRAMES, robot.spec.num_actuated_joints)).astype(np.float32)

        merged = TrajectorySmoothnessTask(robot=robot, num_frames=_FRAMES, weight=1.7, order=1, reference_q=reference)
        merged.init_buffers(device)
        residual = merged.compute_weighted_residual(var).numpy()

        q_np = q.numpy()
        expected = 1.7 * ((q_np[:, 1:] - q_np[:, :-1]) - (reference[:, 1:] - reference[:, :-1]))
        np.testing.assert_allclose(residual, expected.reshape(_BATCH, -1), atol=1e-5, rtol=1e-5)


class TestOrderJacobian:
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_jacobian_matches_finite_difference(self, panda_robot, device, order):
        robot = panda_robot
        q = _traj(robot, device, seed=13)
        history = _history(robot, device)

        task = TrajectorySmoothnessTask(robot=robot, num_frames=_FRAMES, weight=1.3, order=order, dt=0.05)
        task.init_buffers(device)
        task.set_history(history, 1, True)

        analytic = _dense_from_sparse(task, _var(robot, q))

        q0 = q.numpy()
        num_dofs = q0.shape[2]
        eps = 1e-3
        fd = np.zeros_like(analytic)
        for t in range(_FRAMES):
            for d in range(num_dofs):
                sides = []
                for sign in (1.0, -1.0):
                    qp = q0.copy()
                    qp[:, t, d] += sign * eps
                    probe = _var(robot, wp.from_numpy(qp, dtype=wp.float32, device=device))
                    sides.append(task.compute_weighted_residual(probe).numpy().astype(np.float64))
                fd[:, :, t * num_dofs + d] = (sides[0] - sides[1]) / (2.0 * eps)
        np.testing.assert_allclose(analytic, fd, atol=1e-3, rtol=1e-3)


class TestExpandedBatch:
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_reference_and_history_follow_the_source_row(self, panda_robot, device, order):
        """The L-BFGS line search evaluates a batch-expanded var; per-batch buffers must broadcast."""
        robot = panda_robot
        q = _traj(robot, device, seed=21)
        rng = np.random.default_rng(3)
        reference = rng.uniform(-0.3, 0.3, (_BATCH, _FRAMES, robot.spec.num_actuated_joints)).astype(np.float32)

        task = TrajectorySmoothnessTask(
            robot=robot, num_frames=_FRAMES, weight=0.7, order=order, dt=0.05, reference_q=reference
        )
        task.init_buffers(device)
        task.set_history(_history(robot, device), 1, True)

        base = _var(robot, q)
        expected = np.repeat(task.compute_weighted_residual(base).numpy(), 4, axis=0)
        expected_jac = np.repeat(_dense_from_sparse(task, base), 4, axis=0)

        indices = wp.from_numpy(np.repeat(np.arange(_BATCH), 4).astype(np.int32), dtype=wp.int32, device=device)
        expanded = VarValues(robot=base.get("robot").gather(indices))
        np.testing.assert_allclose(task.compute_weighted_residual(expanded).numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(_dense_from_sparse(task, expanded), expected_jac, atol=1e-6)
