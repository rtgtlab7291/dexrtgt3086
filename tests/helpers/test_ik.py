"""Tests for IK examples - ensures all IK examples can initialize and solve successfully."""

from typing import List, Tuple, Type, cast

import numpy as np
import pytest
import torch
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IKHelper, IKHelperConfig
from robokit.helpers.ik_cpu import CPUIKHelper
from robokit.lie.warp_se3 import WarpSE3
from robokit.opt.warp_optimizer import aggregate_residuals_to_costs
from robokit.opt.warp_solver import WarpStageConfig
from robokit.robo import Robot
from robokit.terms.warp.frame_task import WarpFrameTask
from robokit.terms.warp.position_score import WarpCompositeScoreTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.torch import quaternion_to_matrix, rot_tl_to_tf_mat


# ==============================================================================
# Example Configurations
# ==============================================================================

# Format: (robot_desc, target_link_names, batch_size, config_kwargs, check_orientation)
BASIC_EXAMPLE = pytest.param(
    "panda_description",
    ["panda_hand"],
    1,
    {},
    True,
    id="basic",
)

ADVANCED_EXAMPLE = pytest.param(
    "panda_description",
    ["panda_hand"],
    1,
    {"rest_weight": 0.1, "smoothness_weight": 0.3, "velocity_limit_weight": 1.0, "dt": 0.1},
    True,
    id="advanced",
)

BATCH_EXAMPLE = pytest.param(
    "panda_description",
    ["panda_hand"],
    10,
    {},
    True,
    id="batch",
)

BIMANUAL_EXAMPLE = pytest.param(
    "yumi_description",
    ["yumi_link_7_r", "yumi_link_7_l"],
    1,
    {
        "position_weight": 1.0,
        "orientation_weight": 1.0,
        "rest_weight": 0.01,
        "smoothness_weight": 1.0,
        "velocity_limit_weight": 1.0,
        "dt": 0.1,
    },
    True,
    id="bimanual",
)

MIMIC_JOINTS_EXAMPLE = pytest.param(
    "ability_hand_description",
    ["thumb_anchor", "index_anchor", "middle_anchor", "ring_anchor", "pinky_anchor"],
    1,
    {
        "position_weight": 1.0,
        "orientation_weight": 0.0,
        "rest_weight": 0.0,
        "smoothness_weight": 1.0,
        "velocity_limit_weight": 1.0,
        "dt": 0.1,
    },
    False,  # Position-only IK
    id="mimic_joints",
)

HUMANOID_EXAMPLE = pytest.param(
    "g1_description",
    [
        "pelvis_contour_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ],
    1,
    {
        "position_weight": 1.5,
        "orientation_weight": 0.3,
        "rest_weight": 0.01,
        "smoothness_weight": 1.0,
        "velocity_limit_weight": 1.0,
        "dt": 0.1,
        "enable_T_world_base": True,
        "base_damping_weight": 0.5,
        "base_step_limit_indices": [2, 3, 4],
        "base_step_limit_weight": 100.0,
        "base_weight_rest": 0.01,
        "base_weight_smoothness": 0.5,
    },
    True,
    id="humanoid",
)

MOBILE_EXAMPLE = pytest.param(
    "fetch_description",
    ["gripper_link"],
    1,
    {
        "position_weight": 10.0,
        "orientation_weight": 5.0,
        "rest_weight": 0.01,
        "dt": 0.1,
        "enable_T_world_base": True,
        "base_damping_weight": 0.5,
        "base_step_limit_indices": [2, 3, 4],
        "base_step_limit_weight": 100.0,
        "base_weight_rest": 0.01,
        "base_weight_smoothness": 0.0,
        "init_sample_range": 0.1,
        "seed": 42,
    },
    True,
    id="mobile",
)

GPU_IK_EXAMPLES = [
    BASIC_EXAMPLE,
    ADVANCED_EXAMPLE,
    BATCH_EXAMPLE,
    BIMANUAL_EXAMPLE,
    MIMIC_JOINTS_EXAMPLE,
    HUMANOID_EXAMPLE,
    MOBILE_EXAMPLE,
]

CPU_ONLY_EXAMPLE = pytest.param(
    "panda_description",
    ["panda_hand"],
    id="cpu_only",
)

BIMANUAL_CPU_ONLY_EXAMPLE = pytest.param(
    "yumi_description",
    ["yumi_link_7_r", "yumi_link_7_l"],
    id="bimanual_cpu_only",
)

CPU_IK_EXAMPLES = [
    CPU_ONLY_EXAMPLE,
    BIMANUAL_CPU_ONLY_EXAMPLE,
]


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_initial_target_poses(
    robot_desc: str, target_link_names: List[str], batch_size: int, device: str
) -> List[WarpSE3]:
    """Get initial target poses from forward kinematics at zero configuration."""
    urdf = load_robot_description(robot_desc)
    numpy_robot = Robot.load(urdf, backend="numpy")
    numpy_state = numpy_robot.state(q=numpy_robot.zero_q)
    numpy_state = numpy_robot.forward_kinematics(numpy_state)

    targets = []
    for link_name in target_link_names:
        link_idx = numpy_robot.link_names.index(link_name)
        link_pose = numpy_state.get_T_world_link(link_idx)
        pose_np = np.concatenate([link_pose.xyz, link_pose.quat_wxyz]).astype(np.float32)
        pose_np = np.tile(pose_np.reshape(1, 7), (batch_size, 1))
        targets.append(WarpSE3(wp.from_numpy(pose_np, dtype=wp_vec7, device=device)))
    return targets


def get_initial_target_poses_numpy(
    robot: Robot, target_link_names: List[str]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Get initial target poses as numpy arrays for CPU IK helper."""
    state = robot.state(q=robot.spec.zero_q)
    state = robot.forward_kinematics(state)

    positions = []
    quats = []
    for link_name in target_link_names:
        link_idx = robot.link_names.index(link_name)
        link_pose = state.get_T_world_link(link_idx)
        positions.append(np.asarray(link_pose.xyz, dtype=np.float32))
        quats.append(np.asarray(link_pose.quat_wxyz, dtype=np.float32))
    return positions, quats


def validate_joint_limits(q: np.ndarray, joint_limits: np.ndarray, tolerance: float = 1e-3):
    """Validate that joint positions are within limits."""
    lower = joint_limits[:, 0] - tolerance
    upper = joint_limits[:, 1] + tolerance

    if q.ndim == 1:
        q = q.reshape(1, -1)

    for batch_idx in range(q.shape[0]):
        q_batch = q[batch_idx]
        assert np.all(q_batch >= lower), f"Batch {batch_idx}: joints below lower limit"
        assert np.all(q_batch <= upper), f"Batch {batch_idx}: joints above upper limit"


def quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angular distance between two quaternions (wxyz format) in radians."""
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def validate_ik_solution_warp(
    robot: Robot,
    solved_state,
    target_link_names: List[str],
    target_positions: List[np.ndarray],
    target_quats: List[np.ndarray],
    position_tolerance: float = 0.01,
    orientation_tolerance: float = 0.1,
    check_orientation: bool = True,
    batch_idx: int = 0,
):
    """Validate IK solution by running FK on the solved state and comparing to targets.

    Args:
        robot: Robot instance (warp backend)
        solved_state: Solved state from IK (will run FK on it)
        target_link_names: List of target link names
        target_positions: List of target positions (xyz)
        target_quats: List of target quaternions (wxyz)
        position_tolerance: Maximum position error in meters
        orientation_tolerance: Maximum orientation error in radians
        check_orientation: Whether to check orientation (some examples use position-only)
        batch_idx: Batch index to validate
    """
    fk_state = robot.forward_kinematics(solved_state)

    for idx, link_name in enumerate(target_link_names):
        link_idx = robot.link_names.index(link_name)
        link_pose = fk_state.get_T_world_link(link_idx)
        link_xyz = link_pose.xyz.numpy()[batch_idx]
        link_quat = link_pose.quat_wxyz.numpy()[batch_idx]

        pos_error = np.linalg.norm(link_xyz - target_positions[idx])
        assert pos_error < position_tolerance, (
            f"Link {link_name}: position error {pos_error:.4f}m > {position_tolerance}m"
        )

        if check_orientation:
            orient_error = quaternion_angular_distance(link_quat, target_quats[idx])
            assert orient_error < orientation_tolerance, (
                f"Link {link_name}: orientation error {orient_error:.4f}rad > {orientation_tolerance}rad"
            )


def validate_ik_solution_numpy(
    robot: Robot,
    solved_q: np.ndarray,
    target_link_names: List[str],
    target_positions: List[np.ndarray],
    target_quats: List[np.ndarray],
    position_tolerance: float = 0.01,
    orientation_tolerance: float = 0.1,
    check_orientation: bool = True,
):
    """Validate IK solution for CPU backend by running FK and comparing to targets."""
    state = robot.state(q=solved_q)
    state = robot.forward_kinematics(state)

    for idx, link_name in enumerate(target_link_names):
        link_idx = robot.link_names.index(link_name)
        link_pose = state.get_T_world_link(link_idx)

        pos_error = np.linalg.norm(link_pose.xyz - target_positions[idx])
        assert pos_error < position_tolerance, (
            f"Link {link_name}: position error {pos_error:.4f}m > {position_tolerance}m"
        )

        if check_orientation:
            orient_error = quaternion_angular_distance(np.asarray(link_pose.quat_wxyz), target_quats[idx])
            assert orient_error < orientation_tolerance, (
                f"Link {link_name}: orientation error {orient_error:.4f}rad > {orientation_tolerance}rad"
            )


def get_target_arrays_from_warp_se3(
    targets: List[WarpSE3], batch_idx: int = 0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract position and quaternion arrays from WarpSE3 targets."""
    positions = []
    quats = []
    for target in targets:
        pose_np = target.xyz_wxyz.numpy()[batch_idx]
        positions.append(pose_np[:3])
        quats.append(pose_np[3:])
    return positions, quats


# ==============================================================================
# GPU-based IK Tests (IKHelper)
# ==============================================================================


@pytest.mark.parametrize(
    ("robot_desc_name", "target_link_names", "batch_size", "config_kwargs", "check_orientation"),
    GPU_IK_EXAMPLES,
)
def test_ik_example_gpu_initialization(
    robot_desc_name: str,
    target_link_names: List[str],
    batch_size: int,
    config_kwargs: dict,
    check_orientation: bool,
):
    """Test that GPU-based IK examples can initialize successfully."""
    del check_orientation  # Used only in solve tests
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description(robot_desc_name)
    robot = Robot.load(urdf, backend="warp")

    placeholder_targets = []
    for _ in target_link_names:
        placeholder_np = np.zeros((batch_size, 7), dtype=np.float32)
        placeholder_np[:, 3] = 1.0
        placeholder_targets.append(WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device)))

    config = IKHelperConfig(**config_kwargs) if config_kwargs else None

    if len(target_link_names) == 1:
        ik_helper = IKHelper(robot, target_link_names[0], placeholder_targets[0], config)
    else:
        ik_helper = IKHelper(robot, target_link_names, placeholder_targets, config)

    assert ik_helper is not None
    assert ik_helper.num_joints == robot.num_actuated_joints
    assert ik_helper.num_frames == len(target_link_names)


@pytest.mark.parametrize(
    ("robot_desc_name", "target_link_names", "batch_size", "config_kwargs", "check_orientation"),
    GPU_IK_EXAMPLES,
)
def test_ik_example_gpu_solve(
    robot_desc_name: str,
    target_link_names: List[str],
    batch_size: int,
    config_kwargs: dict,
    check_orientation: bool,
):
    """Test that GPU-based IK examples can solve and reach targets correctly."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description(robot_desc_name)
    robot = Robot.load(urdf, backend="warp")

    placeholder_targets = []
    for _ in target_link_names:
        placeholder_np = np.zeros((batch_size, 7), dtype=np.float32)
        placeholder_np[:, 3] = 1.0
        placeholder_targets.append(WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device)))

    config = IKHelperConfig(**config_kwargs) if config_kwargs else None

    if len(target_link_names) == 1:
        ik_helper = IKHelper(robot, target_link_names[0], placeholder_targets[0], config)
    else:
        ik_helper = IKHelper(robot, target_link_names, placeholder_targets, config)

    targets = get_initial_target_poses(robot_desc_name, target_link_names, batch_size, device)

    if len(target_link_names) == 1:
        state = ik_helper.solve(targets[0])
    else:
        state = ik_helper.solve(targets)

    q_np = state.q.numpy()
    assert q_np.shape == (batch_size, robot.num_actuated_joints)
    validate_joint_limits(q_np, robot.spec.actuated_joint_limits)

    target_positions, target_quats = get_target_arrays_from_warp_se3(targets, batch_idx=0)
    validate_ik_solution_warp(
        robot,
        state,
        target_link_names,
        target_positions,
        target_quats,
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        check_orientation=check_orientation,
        batch_idx=0,
    )


# ==============================================================================
# CPU-based IK Tests (CPUIKHelper)
# ==============================================================================


@pytest.mark.parametrize(
    ("robot_desc_name", "target_link_names"),
    CPU_IK_EXAMPLES,
)
def test_ik_example_cpu_initialization(
    robot_desc_name: str,
    target_link_names: List[str],
):
    """Test that CPU-based IK examples can initialize successfully."""
    urdf = load_robot_description(robot_desc_name)
    robot = Robot.load(urdf, backend="numpy")

    if len(target_link_names) == 1:
        ik_helper = CPUIKHelper(robot, target_link_names[0])
    else:
        ik_helper = CPUIKHelper(robot, target_link_names)

    assert ik_helper is not None
    assert ik_helper.num_frames == len(target_link_names)


@pytest.mark.parametrize(
    ("robot_desc_name", "target_link_names"),
    CPU_IK_EXAMPLES,
)
def test_ik_example_cpu_solve(
    robot_desc_name: str,
    target_link_names: List[str],
):
    """Test that CPU-based IK examples can solve and reach targets correctly."""
    urdf = load_robot_description(robot_desc_name)
    robot = Robot.load(urdf, backend="numpy")

    if len(target_link_names) == 1:
        ik_helper = CPUIKHelper(robot, target_link_names[0])
    else:
        ik_helper = CPUIKHelper(robot, target_link_names)

    positions, quats = get_initial_target_poses_numpy(robot, target_link_names)

    if len(target_link_names) == 1:
        solved_q = ik_helper.solve_numpy(positions[0], quats[0])
    else:
        solved_q = ik_helper.solve_numpy(positions, quats)

    assert solved_q.shape == (robot.num_actuated_joints,)
    validate_joint_limits(solved_q, robot.spec.actuated_joint_limits)

    validate_ik_solution_numpy(
        robot,
        solved_q,
        target_link_names,
        positions,
        quats,
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        check_orientation=True,
    )


# ==============================================================================
# Batch Torch Interface Test
# ==============================================================================


def test_batch_torch_matrix_interface():
    """Test the batch_torch example's solve_torch interface and correctness."""
    batch_size = 10
    device_str = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp")

    placeholder_np = np.zeros((batch_size, 7), dtype=np.float32)
    placeholder_np[:, 3] = 1.0
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device_str))

    ik_helper = IKHelper(robot, "panda_hand", placeholder)

    target_pos = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32, device=device_str)
    target_wxyz = torch.tensor([0.0, 0.707, 0.0, 0.707], dtype=torch.float32, device=device_str)
    target_wxyz = target_wxyz / target_wxyz.norm()
    rot_mat = quaternion_to_matrix(target_wxyz)
    target_matrix = rot_tl_to_tf_mat(rot_mat=rot_mat, tl=target_pos)
    target_matrices = target_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

    result = ik_helper.solve_torch(target_matrices)

    q_np = result.q.cpu().numpy()
    assert q_np.shape == (batch_size, robot.num_actuated_joints)
    validate_joint_limits(q_np, robot.spec.actuated_joint_limits)

    target_pos_np = target_pos.cpu().numpy()
    target_quat_np = target_wxyz.cpu().numpy()
    solved_state = robot.state(q=wp.from_torch(result.q))
    validate_ik_solution_warp(
        robot,
        solved_state,
        ["panda_hand"],
        [target_pos_np],
        [target_quat_np],
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        check_orientation=True,
        batch_idx=0,
    )


# ==============================================================================
# solve_numpy Interface Test
# ==============================================================================


def test_basic_solve_numpy_interface():
    """Test the basic example's solve_numpy convenience interface and correctness."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp")

    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    ik_helper = IKHelper(robot, "panda_hand", placeholder)

    target_pos = np.array([0.5, 0.2, 0.5], dtype=np.float32)
    target_quat_wxyz = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    state = ik_helper.solve_numpy(target_pos, target_quat_wxyz)

    q_np = state.q.numpy()
    assert q_np.shape == (1, robot.num_actuated_joints)
    validate_joint_limits(q_np, robot.spec.actuated_joint_limits)

    validate_ik_solution_warp(
        robot,
        state,
        ["panda_hand"],
        [target_pos],
        [target_quat_wxyz],
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        check_orientation=True,
        batch_idx=0,
    )


# ==============================================================================
# Mobile Base Tests
# ==============================================================================


def test_mobile_base_output():
    """Test that mobile base examples output valid T_world_base and reach targets."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("fetch_description")
    robot = Robot.load(urdf, backend="warp")

    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    config = IKHelperConfig(
        position_weight=10.0,
        orientation_weight=5.0,
        rest_weight=0.01,
        dt=0.1,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,
        seed=42,
    )

    ik_helper = IKHelper(robot, "gripper_link", placeholder, config)

    target_pos = np.array([0.6, 0.0, 0.55], dtype=np.float32)
    target_quat = np.array([0.0, 0.707, 0.0, -0.707], dtype=np.float32)
    target_quat = target_quat / np.linalg.norm(target_quat)
    target_np = np.concatenate([target_pos, target_quat]).reshape(1, 7)
    target = WarpSE3(wp.from_numpy(target_np, dtype=wp_vec7, device=device))

    state = ik_helper.solve(target)

    assert state.T_world_base is not None
    base_np = state.T_world_base.xyz_wxyz.numpy()[0]
    assert base_np.shape == (7,)

    quat = base_np[3:]
    quat_norm = np.linalg.norm(quat)
    assert np.isclose(quat_norm, 1.0, atol=1e-4), f"Quaternion not normalized: {quat_norm}"

    validate_ik_solution_warp(
        robot,
        state,
        ["gripper_link"],
        [target_pos],
        [target_quat],
        position_tolerance=0.02,
        orientation_tolerance=0.2,
        check_orientation=True,
        batch_idx=0,
    )


def test_humanoid_mobile_base_output():
    """Test that humanoid example outputs valid T_world_base and reaches targets."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("g1_description")
    robot = Robot.load(urdf, backend="warp")
    target_link_names = [
        "pelvis_contour_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ]

    placeholder_targets = []
    for _ in target_link_names:
        placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        placeholder_targets.append(WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device)))

    config = IKHelperConfig(
        position_weight=1.5,
        orientation_weight=0.3,
        rest_weight=0.01,
        smoothness_weight=1.0,
        velocity_limit_weight=1.0,
        dt=0.1,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,
        base_weight_smoothness=0.5,
    )

    ik_helper = IKHelper(robot, target_link_names, placeholder_targets, config)

    targets = get_initial_target_poses("g1_description", target_link_names, 1, device)
    state = ik_helper.solve(targets)

    assert state.T_world_base is not None
    base_np = state.T_world_base.xyz_wxyz.numpy()[0]
    assert base_np.shape == (7,)

    quat = base_np[3:]
    quat_norm = np.linalg.norm(quat)
    assert np.isclose(quat_norm, 1.0, atol=1e-4), f"Quaternion not normalized: {quat_norm}"

    target_positions, target_quats = get_target_arrays_from_warp_se3(targets, batch_idx=0)
    validate_ik_solution_warp(
        robot,
        state,
        target_link_names,
        target_positions,
        target_quats,
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        check_orientation=True,
        batch_idx=0,
    )


def test_humanoid_multi_step_solve():
    """Test that humanoid IK tracks targets across multiple solve calls with prev_state.

    This mirrors the loop in examples/ik/humanoid.py where solved state is fed back
    as prev_state. Regressions here (e.g. CUDA graph buffer pointer invalidation)
    cause the solver to reject all steps and make zero progress.
    """
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("g1_description")
    robot = Robot.load(urdf, backend="warp")
    target_link_names = [
        "pelvis_contour_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ]

    placeholder_targets = []
    for _ in target_link_names:
        placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        placeholder_targets.append(WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device)))

    config = IKHelperConfig(
        position_weight=1.5,
        orientation_weight=0.3,
        rest_weight=0.01,
        smoothness_weight=1.0,
        velocity_limit_weight=1.0,
        dt=0.1,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,
        base_weight_smoothness=0.5,
    )

    ik_helper = IKHelper(robot, target_link_names, placeholder_targets, config)

    # Initial prev_state with floating base (matches humanoid.py)
    prev_q = robot.spec.zero_q
    prev_state = robot.state(
        q=wp.from_numpy(prev_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device),
        T_world_base=WarpSE3.identity(shape=(1,), device=device),
    )

    targets = get_initial_target_poses("g1_description", target_link_names, 1, device)

    # Step 1: Solve with prev_state
    state = ik_helper.solve(targets, prev_state=prev_state)

    target_positions, target_quats = get_target_arrays_from_warp_se3(targets, batch_idx=0)
    validate_ik_solution_warp(
        robot,
        state,
        target_link_names,
        target_positions,
        target_quats,
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        check_orientation=True,
    )

    # Step 2: Perturb targets slightly and solve again with previous solution as prev_state
    perturbed_targets = []
    for target in targets:
        pose_np = target.xyz_wxyz.numpy().copy()
        pose_np[0, :3] += np.array([0.01, 0.0, 0.0], dtype=np.float32)
        perturbed_targets.append(WarpSE3(wp.from_numpy(pose_np, dtype=wp_vec7, device=device)))

    state2 = ik_helper.solve(perturbed_targets, prev_state=state)

    perturbed_positions, perturbed_quats = get_target_arrays_from_warp_se3(perturbed_targets, batch_idx=0)
    validate_ik_solution_warp(
        robot,
        state2,
        target_link_names,
        perturbed_positions,
        perturbed_quats,
        position_tolerance=0.02,
        orientation_tolerance=0.2,
        check_orientation=True,
    )


def test_ik_optimizer_residual_layout_consistent_with_mobile_smoothness():
    """Ensure stage optimizer residual layouts stay consistent before and after solve."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("fetch_description")
    robot = Robot.load(urdf, backend="warp")

    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    config = IKHelperConfig(
        stages=[
            WarpStageConfig(num_seeds=8, iters=4, lm_lambda=10.0),
            WarpStageConfig(num_seeds=2, iters=6, lm_lambda=1.0),
            WarpStageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
        ],
        use_cuda_graph=False,
        smoothness_weight=0.2,
        velocity_limit_weight=0.0,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,
        base_weight_smoothness=0.2,
        init_sample_range=0.1,
    )
    ik_helper = IKHelper(robot, "gripper_link", placeholder, config)

    for optimizer in ik_helper._solver._optimizers:
        expected_total = sum(term.residual_dim for term in optimizer.terms)
        assert optimizer.total_residual_dim == expected_total

    target = get_initial_target_poses("fetch_description", ["gripper_link"], 1, device)[0]
    prev_state = robot.state(
        q=wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device),
        T_world_base=WarpSE3.identity(shape=(1,), device=device),
    )
    ik_helper.solve(target, prev_state=prev_state, init_state=prev_state)

    for optimizer in ik_helper._solver._optimizers:
        expected_total = sum(term.residual_dim for term in optimizer.terms)
        assert optimizer.total_residual_dim == expected_total


def _assert_stage_optimizer_layout(ik_helper: IKHelper) -> None:
    for optimizer in ik_helper._solver._optimizers:
        expected_total = sum(term.residual_dim for term in optimizer.terms)
        assert optimizer.total_residual_dim == expected_total


@pytest.mark.parametrize(
    ("use_smoothness",),
    [(True,), (False,)],
)
def test_ik_score_task_layout_consistent(use_smoothness: bool):
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    robot = Robot.load(load_robot_description("fetch_description"), backend="warp")

    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    config = IKHelperConfig(
        stages=[
            WarpStageConfig(num_seeds=8, iters=4, lm_lambda=10.0),
            WarpStageConfig(num_seeds=2, iters=6, lm_lambda=1.0),
            WarpStageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
        ],
        use_cuda_graph=False,
        smoothness_weight=0.2,
        velocity_limit_weight=0.0,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,
        base_weight_smoothness=0.2,
        score_position_weight=1.0,
        score_orientation_weight=0.5,
        score_smoothness_weight=0.5 if use_smoothness else 0.0,
        init_sample_range=0.1,
        seed=0,
    )
    ik_helper = IKHelper(robot, "gripper_link", placeholder, config)
    stage0_score_task = ik_helper._stage_score_tasks[0]
    if use_smoothness:
        assert isinstance(stage0_score_task, WarpCompositeScoreTask)
    else:
        assert isinstance(stage0_score_task, WarpFrameTask)

    _assert_stage_optimizer_layout(ik_helper)

    target = get_initial_target_poses("fetch_description", ["gripper_link"], 1, device)[0]
    prev_state = robot.state(
        q=wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device),
        T_world_base=WarpSE3.identity(shape=(1,), device=device),
    )
    ik_helper.solve(target, prev_state=prev_state, init_state=prev_state)
    _assert_stage_optimizer_layout(ik_helper)


def test_ik_frozen_target_continuity_regression():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    robot = Robot.load(load_robot_description("fetch_description"), backend="warp")

    placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    config = IKHelperConfig(
        stages=[
            WarpStageConfig(num_seeds=8, iters=4, lm_lambda=10.0),
            WarpStageConfig(num_seeds=2, iters=6, lm_lambda=1.0),
            WarpStageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
        ],
        use_cuda_graph=False,
        smoothness_weight=0.2,
        velocity_limit_weight=0.0,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,
        base_weight_smoothness=0.2,
        init_sample_range=0.1,
        seed=0,
    )
    ik_helper = IKHelper(robot, "gripper_link", placeholder, config)
    ik_helper._solver.score_terms = [None] * ik_helper._solver.num_stages
    ik_helper._stage_score_tasks = [None] * len(ik_helper._stage_score_tasks)

    prev_state = robot.state(
        q=wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device),
        T_world_base=WarpSE3.identity(shape=(1,), device=device),
    )
    target = get_initial_target_poses("fetch_description", ["gripper_link"], 1, device)[0]

    final_optimizer = ik_helper._solver._optimizers[-1]
    final_costs = wp.empty((1,), dtype=cast(Type[float], wp.float32), device=device)

    cost_history = []
    q_deltas = []
    prev_q_np = None
    for _ in range(20):
        state = ik_helper.solve(target, prev_state=prev_state, init_state=prev_state)

        final_optimizer.compute_residuals(final_optimizer.terms, state, final_optimizer.residuals)
        wp.launch(
            kernel=aggregate_residuals_to_costs,
            dim=1,
            inputs=[final_optimizer.residuals, final_costs],
            device=device,
        )
        cost_history.append(float(final_costs.numpy()[0]))

        q_np = state.q.numpy().copy()
        if prev_q_np is not None:
            q_deltas.append(float(np.linalg.norm(q_np - prev_q_np)))
        prev_q_np = q_np

        base_np = state.T_world_base.xyz_wxyz.numpy().copy()
        prev_state = robot.state(
            q=wp.from_numpy(q_np, dtype=wp.float32, device=device),
            T_world_base=WarpSE3(wp.from_numpy(base_np, dtype=wp_vec7, device=device)),
        )

    assert len(cost_history) == 20
    assert np.all(np.isfinite(np.array(cost_history)))
    assert max(q_deltas) < 0.12
    assert float(np.max(np.diff(np.array(cost_history)))) <= 1e-4


# ==============================================================================
# Dynamic Sample Range Override Tests
# ==============================================================================


def test_init_sample_range_override_has_effect():
    """Verify that init_sample_range actually changes sampling variance."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp")

    placeholder_np = np.zeros((1, 7), dtype=np.float32)
    placeholder_np[:, 3] = 1.0
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    config = IKHelperConfig(seed=42)
    ik_helper = IKHelper(robot, "panda_hand", placeholder, config)

    init_q_np = robot.spec.midrange_q.reshape(1, -1).astype(np.float32)
    init_q_wp = wp.from_numpy(init_q_np, dtype=wp.float32, device=device)

    num_seeds = ik_helper.stage_configs[0].num_seeds
    out_q = wp.zeros((num_seeds, robot.num_actuated_joints), dtype=wp.float32, device=device)

    ik_helper._sample_q_around_init(init_q_wp, num_seeds, out_q, sample_range=0.0)
    q_zero_range = out_q.numpy().copy()
    var_zero = np.var(q_zero_range, axis=0).sum()

    ik_helper._sample_q_around_init(init_q_wp, num_seeds, out_q, sample_range=0.5)
    q_large_range = out_q.numpy().copy()
    var_large = np.var(q_large_range, axis=0).sum()

    assert var_zero < 1e-10, f"Expected zero variance with range=0.0, got {var_zero}"
    assert var_large > 0.01, f"Expected significant variance with range=0.5, got {var_large}"


def test_base_init_sample_range_override_has_effect():
    """Verify that base_init_sample_range actually changes base sampling variance."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("fetch_description")
    robot = Robot.load(urdf, backend="warp")

    placeholder_np = np.zeros((1, 7), dtype=np.float32)
    placeholder_np[:, 3] = 1.0
    placeholder = WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device))

    config = IKHelperConfig(
        seed=42,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
    )
    ik_helper = IKHelper(robot, "gripper_link", placeholder, config)

    init_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    init_base_wp = wp.from_numpy(init_base_np, dtype=wp_vec7, device=device)

    num_seeds = ik_helper.stage_configs[0].num_seeds
    out_base = wp.zeros(num_seeds, dtype=wp_vec7, device=device)

    ik_helper._sample_base_around_init(init_base_wp, num_seeds, out_base, sample_range=0.0)
    base_zero_range = out_base.numpy().copy()
    var_zero = np.var(base_zero_range[:, :2], axis=0).sum()

    ik_helper._sample_base_around_init(init_base_wp, num_seeds, out_base, sample_range=0.5)
    base_large_range = out_base.numpy().copy()
    var_large = np.var(base_large_range[:, :2], axis=0).sum()

    assert var_zero < 1e-10, f"Expected zero variance with range=0.0, got {var_zero}"
    assert var_large > 0.01, f"Expected significant variance with range=0.5, got {var_large}"
