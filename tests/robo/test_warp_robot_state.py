import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.robo import Robot


@pytest.fixture
def panda_robot():
    urdf_path = load_robot_description("panda_description")
    return Robot.load(urdf_path, backend="warp")


def test_gather_basic(panda_robot):
    robot = panda_robot
    q = wp.from_numpy(np.array([[0.1] * 8, [0.2] * 8, [0.3] * 8], dtype=np.float32), dtype=wp.float32)
    state = robot.state(q=q)
    state = robot.forward_kinematics(state)

    indices = wp.from_numpy(np.array([2, 0], dtype=np.int32))
    gathered_state = state.gather(indices)

    assert gathered_state.q.shape == (2, 8)
    assert np.allclose(gathered_state.q.numpy()[0], [0.3] * 8, atol=1e-6)
    assert np.allclose(gathered_state.q.numpy()[1], [0.1] * 8, atol=1e-6)


def test_gather_preserves_fk(panda_robot):
    robot = panda_robot
    q = wp.from_numpy(
        np.array(
            [[0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.2] * 8],
            dtype=np.float32,
        ),
        dtype=wp.float32,
    )
    state = robot.state(q=q)
    state = robot.forward_kinematics(state)

    indices = wp.from_numpy(np.array([1, 2, 0], dtype=np.int32))
    gathered_state = state.gather(indices)

    assert gathered_state.is_fk_computed
    assert gathered_state.T_world_link.xyz_wxyz.shape == (3, robot.spec.num_links)
    for link_idx in range(robot.spec.num_links):
        assert np.allclose(
            gathered_state.T_world_link.xyz_wxyz.numpy()[0, link_idx],
            state.T_world_link.xyz_wxyz.numpy()[1, link_idx],
            atol=1e-6,
        )
        assert np.allclose(
            gathered_state.T_world_link.xyz_wxyz.numpy()[1, link_idx],
            state.T_world_link.xyz_wxyz.numpy()[2, link_idx],
            atol=1e-6,
        )
        assert np.allclose(
            gathered_state.T_world_link.xyz_wxyz.numpy()[2, link_idx],
            state.T_world_link.xyz_wxyz.numpy()[0, link_idx],
            atol=1e-6,
        )


def test_gather_with_dest(panda_robot):
    robot = panda_robot
    q = wp.from_numpy(np.array([[0.1] * 8, [0.2] * 8, [0.3] * 8, [0.4] * 8], dtype=np.float32), dtype=wp.float32)
    state = robot.state(q=q)
    state = robot.forward_kinematics(state)

    indices = wp.from_numpy(np.array([3, 1], dtype=np.int32))

    dest_q = wp.empty((2, 8), dtype=wp.float32, device=q.device)  # type: ignore
    dest_state = robot.state(q=dest_q)

    gathered_state = state.gather(indices, dest=dest_state)

    assert gathered_state is dest_state
    assert np.allclose(gathered_state.q.numpy()[0], [0.4] * 8, atol=1e-6)
    assert np.allclose(gathered_state.q.numpy()[1], [0.2] * 8, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
