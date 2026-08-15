import numpy as np
import pytest
import warp as wp

from robokit.geom import SphereGeom, WarpScene
from robokit.opt.var_values import VarValues
from robokit.terms.dense.force_closure_task import ForceClosureTask
from robokit.utils.warp_utils import wp_vec7


class TestForceClosureTask:
    def test_surface_normal_and_force_jacobian(self, panda_robot) -> None:
        device = wp.get_device("cpu")
        radius = 0.1
        scene = WarpScene(1, device).add(
            SphereGeom(
                radii=np.array([radius], dtype=np.float32),
                scene_offsets=np.array([0, 1], dtype=np.int32),
            )
        )
        q = wp.from_numpy(panda_robot.spec.zero_q, dtype=wp.float32, device=device)
        T_world_base = wp.from_numpy(
            np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            dtype=wp_vec7,
            device=device,
        )
        state = panda_robot.state(q=q, T_world_base=T_world_base)
        local_points = wp.from_numpy(
            np.array([[[radius, 0.0, 0.0], [0.0, radius, 0.0]]], dtype=np.float32),
            dtype=wp.vec3,
            device=device,
        )
        link_indices = wp.from_numpy(np.zeros((1, 2), dtype=np.int32), dtype=wp.int32, device=device)
        task = ForceClosureTask(
            robot=panda_robot,
            num_contact_points=2,
            warp_meshes=scene,
            contact_force_weights=(2.0, 0.5),
            local_contact_points=local_points,
            contact_points_link_indices=link_indices,
        )

        residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()[0]
        np.testing.assert_allclose(residual[:3], [2.0, 0.5, 0.0], atol=1e-6)

        analytic = task.compute_weighted_jacobian_analytic(VarValues(robot=state)).numpy()[0, :3, :3]
        finite_difference = np.zeros((3, 3), dtype=np.float32)
        eps = 1.0e-4
        for col_idx in range(3):
            velocity = np.zeros((1, state.tangent_dim), dtype=np.float32)
            velocity[0, col_idx] = eps
            plus = state.integrate(wp.from_numpy(velocity, dtype=wp.float32, device=device))
            velocity[0, col_idx] = -eps
            minus = state.integrate(wp.from_numpy(velocity, dtype=wp.float32, device=device))
            finite_difference[:, col_idx] = (
                task.compute_weighted_residual(VarValues(robot=plus)).numpy()[0, :3]
                - task.compute_weighted_residual(VarValues(robot=minus)).numpy()[0, :3]
            ) / (2.0 * eps)

        assert np.max(np.abs(analytic)) > 1.0
        np.testing.assert_allclose(analytic, finite_difference, atol=2e-2, rtol=2e-3)

        cost = wp.zeros((1,), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(VarValues(robot=state), out_cost=cost)
        np.testing.assert_allclose(cost.numpy(), [2.125], atol=1e-6)

        normals = wp.zeros((1, 2), dtype=wp.vec3, device=device)
        with pytest.raises(ValueError, match="must be provided together"):
            task.compute_weighted_residual(VarValues(robot=state), precomputed_contact_points_world=local_points)
        with pytest.raises(ValueError, match="must be provided together"):
            task.compute_weighted_residual(VarValues(robot=state), precomputed_normals=normals)
