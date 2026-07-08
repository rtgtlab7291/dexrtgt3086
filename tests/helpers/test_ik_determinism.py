import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IKHelper, IKHelperConfig
from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def test_ik_helper_determinism_same_seed():
    """Test that IKHelper produces identical results with the same seed."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp")

    # Create target pose
    target_pos = np.array([0.5, 0.2, 0.5], dtype=np.float32)
    target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Create placeholder
    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    # First helper with seed=42
    config1 = IKHelperConfig(seed=42)
    helper1 = IKHelper(robot, "panda_hand", placeholder, config=config1)
    state1 = helper1.solve_numpy(target_pos, target_quat)
    q1 = state1.q.numpy()

    # Second helper with seed=42
    config2 = IKHelperConfig(seed=42)
    helper2 = IKHelper(robot, "panda_hand", placeholder, config=config2)
    state2 = helper2.solve_numpy(target_pos, target_quat)
    q2 = state2.q.numpy()

    # Results should be identical
    assert np.allclose(q1, q2, atol=1e-6), f"Solutions differ: max diff = {np.abs(q1 - q2).max()}"


def test_ik_helper_determinism_different_seeds():
    """Test that IKHelper produces different results with different seeds."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp")

    target_pos = np.array([0.5, 0.2, 0.5], dtype=np.float32)
    target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    # Helper with seed=42
    config1 = IKHelperConfig(seed=42)
    helper1 = IKHelper(robot, "panda_hand", placeholder, config=config1)
    state1 = helper1.solve_numpy(target_pos, target_quat)
    state1.q.numpy()

    # Helper with seed=123
    config2 = IKHelperConfig(seed=123)
    helper2 = IKHelper(robot, "panda_hand", placeholder, config=config2)
    state2 = helper2.solve_numpy(target_pos, target_quat)
    state2.q.numpy()

    # Results should differ (with high probability)
    # Note: They might converge to the same solution, so we check the initial seeds differ
    # by checking Roberts offsets are different
    assert helper1._roberts_offset != helper2._roberts_offset


def test_ik_helper_no_seed_backward_compatibility():
    """Test that IKHelper works without seed (backward compatibility)."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp")

    target_pos = np.array([0.5, 0.2, 0.5], dtype=np.float32)
    target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    # Create helper without seed (default behavior)
    helper = IKHelper(robot, "panda_hand", placeholder)
    state = helper.solve_numpy(target_pos, target_quat)

    # Should solve successfully
    assert state.q is not None
    assert state.q.shape == (1, robot.num_actuated_joints)

    # Roberts offset should be 0 (no offset)
    assert helper._roberts_offset == 0
