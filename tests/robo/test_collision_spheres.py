from pathlib import Path

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def _reference_world_collision_sphere_centers(
    T_world_link: np.ndarray,
    local_collision_sphere_centers: np.ndarray,
    collision_spheres_link_indices: np.ndarray,
) -> np.ndarray:
    num_elements = T_world_link.shape[0]
    num_spheres = local_collision_sphere_centers.shape[0]

    T_world_sphere_np = T_world_link[:, collision_spheres_link_indices]
    T_world_sphere = WarpSE3(wp.from_numpy(T_world_sphere_np.reshape(-1, 7), dtype=wp_vec7))

    local_points_tiled = np.tile(local_collision_sphere_centers, (num_elements, 1))
    world_points = T_world_sphere.apply(wp.from_numpy(local_points_tiled, dtype=wp.vec3))
    world_points_np = world_points.numpy().reshape(num_elements, num_spheres, 3)

    return world_points_np


@pytest.fixture
def robot():
    return Robot.load(
        load_robot_description("panda_description"),
        backend="warp",
        load_collision_spheres=True,
        collision_spheres_path=str(
            Path(__file__).parent.parent.parent / "assets" / "collision_spheres" / "franka.yaml"
        ),
    )


def test_world_collision_spheres_correctness_single(robot) -> None:
    assert robot.spec.has_collision_spheres

    state = robot.forward_kinematics(robot.state())
    state = robot.transform_collision_spheres(state)
    world_centers_np = state.world_collision_sphere_centers.numpy()

    T_links_np = state.T_world_link.xyz_wxyz.numpy().reshape(state.num_elements, robot.spec.num_links, 7)
    ref = _reference_world_collision_sphere_centers(
        T_links_np,
        robot.spec.local_collision_sphere_centers.astype(np.float32),
        robot.spec.collision_spheres_link_indices.astype(np.int32),
    )

    assert world_centers_np.shape == ref.shape
    assert np.allclose(world_centers_np, ref, atol=1e-5)


def test_world_collision_spheres_batch_shape_and_values(robot) -> None:
    state = robot.forward_kinematics(robot.state(q=robot.sample_q(num_samples=4)))
    state = robot.transform_collision_spheres(state)
    world_centers_np = state.world_collision_sphere_centers.numpy()

    assert world_centers_np.shape == (4, robot.spec.local_collision_sphere_centers.shape[0], 3)

    T_links_np = state.T_world_link.xyz_wxyz.numpy().reshape(state.num_elements, robot.spec.num_links, 7)
    ref = _reference_world_collision_sphere_centers(
        T_links_np,
        robot.spec.local_collision_sphere_centers.astype(np.float32),
        robot.spec.collision_spheres_link_indices.astype(np.int32),
    )

    assert np.allclose(world_centers_np, ref, atol=1e-5)


def test_world_collision_spheres_changes_with_base_transform(robot) -> None:
    q = wp.from_numpy(robot.spec.zero_q.astype(np.float32), dtype=wp.float32)

    state_identity = robot.forward_kinematics(robot.state(q=q))
    state_identity = robot.transform_collision_spheres(state_identity)
    centers_identity = state_identity.world_collision_sphere_centers.numpy()

    T_world_base = WarpSE3(
        wp.from_numpy(
            np.array([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0], dtype=np.float32).reshape(1, 7), dtype=wp_vec7
        )
    )
    state_shifted = robot.forward_kinematics(robot.state(q=q, T_world_base=T_world_base))
    state_shifted = robot.transform_collision_spheres(state_shifted)
    centers_shifted = state_shifted.world_collision_sphere_centers.numpy()

    assert centers_identity.shape == centers_shifted.shape
    assert not np.allclose(centers_identity, centers_shifted)


def test_world_collision_spheres_invalidation_on_integrate(robot) -> None:
    state = robot.forward_kinematics(robot.state())
    state = robot.transform_collision_spheres(state)
    _ = state.world_collision_sphere_centers
    assert state.is_collision_spheres_computed

    velocity = wp.zeros((state.batch_size, state.tangent_dim), dtype=wp.float32, device=state.q.device)
    next_state = state.integrate(velocity)

    assert not next_state.is_collision_spheres_computed

    next_state = robot.forward_kinematics(next_state)
    next_state = robot.transform_collision_spheres(next_state)
    assert next_state.is_collision_spheres_computed
    assert next_state.world_collision_sphere_centers.numpy().shape[0] == next_state.num_elements


def test_world_collision_spheres_zero_length_when_not_loaded() -> None:
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp", load_collision_spheres=False)
    assert not robot.spec.has_collision_spheres

    state = robot.forward_kinematics(robot.state())

    assert state.world_collision_sphere_centers.shape[-1] == 0
