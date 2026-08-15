import numpy as np
import warp as wp

from robokit.utils.warp_utils import wp_vec7


def test_typed_transform_buffers(panda_robot):
    state = panda_robot.state()

    assert state.T_world_base.dtype == wp_vec7
    assert state.T_world_joint.dtype == wp_vec7
    assert state.T_world_link.dtype == wp_vec7
    assert state.T_world_base.shape == (1,)
    assert state.T_world_joint.shape == (1, panda_robot.spec.num_joints)
    assert state.T_world_link.shape == (1, panda_robot.spec.num_links)
    assert state.T_world_base.is_contiguous
    assert state.T_world_joint.is_contiguous
    assert state.T_world_link.is_contiguous
    assert np.allclose(state.T_world_base.numpy(), [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])


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


def test_gather_marks_fk_stale(panda_robot):
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

    assert not gathered_state.is_fk_computed
    gathered_state = robot.forward_kinematics(gathered_state)
    assert gathered_state.T_world_link.shape == (3, robot.spec.num_links)
    for link_idx in range(robot.spec.num_links):
        assert np.allclose(
            gathered_state.T_world_link.numpy()[0, link_idx],
            state.T_world_link.numpy()[1, link_idx],
            atol=1e-6,
        )
        assert np.allclose(
            gathered_state.T_world_link.numpy()[1, link_idx],
            state.T_world_link.numpy()[2, link_idx],
            atol=1e-6,
        )
        assert np.allclose(
            gathered_state.T_world_link.numpy()[2, link_idx],
            state.T_world_link.numpy()[0, link_idx],
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

    gathered_state = state.gather(indices, out=dest_state)

    assert gathered_state is dest_state
    assert np.allclose(gathered_state.q.numpy()[0], [0.4] * 8, atol=1e-6)
    assert np.allclose(gathered_state.q.numpy()[1], [0.2] * 8, atol=1e-6)


def test_accept_basic(panda_robot):
    robot = panda_robot
    q_src = wp.from_numpy(np.array([[0.1] * 8, [0.2] * 8, [0.3] * 8], dtype=np.float32), dtype=wp.float32)
    q_dst = wp.from_numpy(np.array([[0.9] * 8, [0.8] * 8, [0.7] * 8], dtype=np.float32), dtype=wp.float32)
    src = robot.state(q=q_src)
    dst = robot.state(q=q_dst)
    src = robot.forward_kinematics(src)
    dst = robot.forward_kinematics(dst)

    mask = wp.from_numpy(np.array([1, 0, 1], dtype=np.int32))
    dst.accept(mask, src)

    assert np.allclose(dst.q.numpy()[0], [0.1] * 8, atol=1e-6)
    assert np.allclose(dst.q.numpy()[1], [0.8] * 8, atol=1e-6)
    assert np.allclose(dst.q.numpy()[2], [0.3] * 8, atol=1e-6)

    for link_idx in range(robot.spec.num_links):
        assert np.allclose(
            dst.T_world_link.numpy()[0, link_idx],
            src.T_world_link.numpy()[0, link_idx],
            atol=1e-6,
        )
        assert np.allclose(
            dst.T_world_link.numpy()[2, link_idx],
            src.T_world_link.numpy()[2, link_idx],
            atol=1e-6,
        )

    assert dst.is_fk_computed
    assert not dst.is_motion_subspace_computed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
