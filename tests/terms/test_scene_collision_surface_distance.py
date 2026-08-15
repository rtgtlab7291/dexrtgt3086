"""Finite-difference validation of SceneCollisionTask(penalty="surface_distance")."""

import numpy as np
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.opt.var_values import VarValues
from robokit.terms.dense.scene_collision_task import SceneCollisionTask


_DEVICE = wp.get_device("cpu")


def _box_scene(half_extent=0.35):
    from test_terms import _make_box_mesh

    mesh = _make_box_mesh(device=_DEVICE, half_extent=half_extent)
    return WarpScene(1, mesh.device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))


def _state(robot, seed=0, batch=2):
    rng = np.random.default_rng(seed)
    q = (robot.spec.midrange_q + rng.normal(scale=0.1, size=robot.num_actuated_joints)).astype(np.float32)
    state = robot.state(q=wp.from_numpy(np.tile(q, (batch, 1)), dtype=wp.float32, device=_DEVICE))
    state = robot.forward_kinematics(state)
    state = robot.transform_collision_spheres(state)
    return robot.compute_motion_subspace(state)


def _evaluate(task, state):
    var_values = VarValues(robot=state)
    task.compute_weighted_residual(var_values)
    return task.compute_weighted_jacobian_analytic(var_values).numpy()


class TestSurfaceDistanceJacobian:
    def test_jacobian_matches_finite_difference(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        scene = _box_scene()
        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        task = SceneCollisionTask(
            robot=robot, scene=scene, sphere_indices=sphere_indices, weight=1.5, penalty="surface_distance"
        )

        state = _state(robot, seed=3)
        analytic = _evaluate(task, state)

        q0 = state.q.numpy().copy()
        eps = 1e-4
        fd = np.zeros_like(analytic)
        for j in range(q0.shape[1]):
            sides = []
            for sign in (1.0, -1.0):
                q = q0.copy()
                q[:, j] += sign * eps
                probe = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=_DEVICE))
                probe = robot.forward_kinematics(probe)
                probe = robot.transform_collision_spheres(probe)
                sides.append(task.compute_weighted_residual(VarValues(robot=probe)).numpy())
            fd[:, :, j] = (sides[0] - sides[1]) / (2.0 * eps)
        np.testing.assert_allclose(analytic, fd, atol=2e-3, rtol=2e-3)
