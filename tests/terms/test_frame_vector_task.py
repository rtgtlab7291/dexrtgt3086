"""Finite-difference validation of every FrameVectorTask residual mode."""

import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.utils.warp_utils import wp_vec7


_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_ORIGINS = [1, 3, 5]
_TASKS = [4, 6, 7]
_BATCH = 3


def _state(robot, floating_base, seed=0):
    rng = np.random.default_rng(seed)
    limits = robot.spec.actuated_joint_limits
    q = rng.uniform(limits[:, 0], limits[:, 1], (_BATCH, robot.spec.num_actuated_joints)).astype(np.float32)
    T_world_base = None
    if floating_base:
        base = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (_BATCH, 1))
        base[:, :3] = rng.uniform(-0.2, 0.2, (_BATCH, 3))
        quat = rng.normal(size=(_BATCH, 4)).astype(np.float32)
        base[:, 3:] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        T_world_base = wp.from_numpy(base, dtype=wp_vec7, device=_DEVICE)
    return robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=_DEVICE), T_world_base=T_world_base)


def _evaluate(robot, task, state):
    task.init_buffers(_DEVICE)
    var_values = VarValues(robot=state)
    robot.forward_kinematics(state)
    robot.compute_motion_subspace(state)
    residual = task.compute_weighted_residual(var_values).numpy()
    jacobian = task.compute_weighted_jacobian_analytic(var_values).numpy()
    return residual, jacobian


def _fd_jacobian(robot, task, state, eps=1e-3):
    q0 = state.q.numpy().copy()
    num_dofs = q0.shape[1]
    rows = task.residual_dim
    out = np.zeros((_BATCH, rows, num_dofs), dtype=np.float64)
    for j in range(num_dofs):
        perturbed = []
        for sign in (1.0, -1.0):
            q = q0.copy()
            q[:, j] += sign * eps
            probe = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=_DEVICE))
            robot.forward_kinematics(probe)
            perturbed.append(task.compute_weighted_residual(VarValues(robot=probe)).numpy().astype(np.float64))
        out[:, :, j] = (perturbed[0] - perturbed[1]) / (2.0 * eps)
    return out


class TestFrameVectorTaskJacobian:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(),
            dict(huber_delta=0.02),
            dict(huber_delta=0.02, huber_on_norm=True),
            dict(direction_only=True),
            dict(origin_link_indices=[-1, -1, -1]),
            dict(soft_gate_start_distance=0.12, soft_gate_full_distance=0.02),
        ],
    )
    def test_jacobian_matches_finite_difference(self, panda_robot, kwargs):
        rng = np.random.default_rng(7)
        targets = rng.normal(scale=0.1, size=(_BATCH, 3, 3)).astype(np.float32)
        task = FrameVectorTask(
            robot=panda_robot,
            task_link_indices=_TASKS,
            targets=targets,
            weight=np.array([1.0, 2.0, 0.5], dtype=np.float32),
            scale=1.3,
            **{"origin_link_indices": _ORIGINS, **kwargs},
        )
        state = _state(panda_robot, floating_base=False, seed=11)
        analytic = _evaluate(panda_robot, task, state)[1]
        fd = _fd_jacobian(panda_robot, task, state)
        np.testing.assert_allclose(analytic.astype(np.float64), fd, atol=2e-3, rtol=2e-3)
