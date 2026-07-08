import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.lie.warp_se3 import WarpSE3
from robokit.opt.sparse_warp_optimizer import SparseWarpOptimizer, SparseWarpOptimizerConfig
from robokit.opt.warp_optimizer import WarpLMOptimizer, WarpLMOptimizerConfig
from robokit.robo.robot import Robot
from robokit.terms import WarpFrameTask, WarpPositionLimit
from robokit.terms.terms import SparseWarpTask, WarpSparsityPattern, WarpTask
from robokit.utils.warp_utils import wp_vec7


class _MutableResidualDenseTask(WarpTask):
    def __init__(self, residual_dim: int):
        self._residual_dim = residual_dim
        self.gain = 1.0
        self.residual_weight = None

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_residual(self, var, *args, residual_buffer=None, row_offset=0, **kwargs):
        del args, kwargs
        del row_offset
        if residual_buffer is not None:
            return residual_buffer
        return wp.zeros((var.batch_size, self.residual_dim), dtype=wp.float32, device=var.device)

    def compute_weighted_jacobian_analytic(
        self, var, *args, jacobian_buffer=None, row_offset=0, col_offset=0, **kwargs
    ):
        del args, kwargs
        del row_offset, col_offset
        if jacobian_buffer is not None:
            return jacobian_buffer
        return wp.zeros((var.batch_size, self.residual_dim, var.tangent_dim), dtype=wp.float32, device=var.device)


class _MutableResidualSparseTask(SparseWarpTask):
    def __init__(self, residual_dim: int):
        self._residual_dim = residual_dim
        self.gain = 1.0
        self.residual_weight = None

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_residual(self, var, *args, residual_buffer=None, row_offset=0, **kwargs):
        del args, kwargs
        del row_offset
        if residual_buffer is not None:
            return residual_buffer
        return wp.zeros((var.batch_size, self.residual_dim), dtype=wp.float32, device=var.device)

    def compute_weighted_jacobian_analytic(
        self, var, *args, jacobian_buffer=None, row_offset=0, col_offset=0, **kwargs
    ):
        del args, kwargs
        del row_offset, col_offset
        if jacobian_buffer is not None:
            return jacobian_buffer
        return wp.zeros((var.batch_size, self.residual_dim, var.tangent_dim), dtype=wp.float32, device=var.device)

    def compute_sparse_jacobian_pattern(self, var, offset=0, **kwargs):
        del kwargs
        row_indices = np.arange(self.residual_dim, dtype=np.int32) + int(offset)
        col_indices = np.zeros((self.residual_dim,), dtype=np.int32)
        pattern = WarpSparsityPattern()
        pattern.row_indices = wp.from_numpy(row_indices, dtype=wp.int32, device=var.device)
        pattern.col_indices = wp.from_numpy(col_indices, dtype=wp.int32, device=var.device)
        return pattern

    def compute_weighted_sparse_jacobian_values(self, var, jacobian_values_buffer=None, offset=0, **kwargs):
        del offset, kwargs
        if jacobian_values_buffer is not None:
            return jacobian_values_buffer
        return wp.zeros((var.batch_size, self.residual_dim), dtype=wp.float32, device=var.device)


def test_ik():
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    target_pose = WarpSE3(wp.from_numpy(np.array([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]), dtype=wp_vec7))
    ik_optimizer = robot.build_inverse_kinematics_optimizer("ee_link", target_pose)
    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
    state = robot.state(q=q_init)
    state = ik_optimizer.solve(state)
    state = robot.forward_kinematics(state)
    achieved_pose = state.get_T_world_link(robot.link_names.index("ee_link"))
    assert np.allclose(achieved_pose.xyz.numpy(), target_pose.xyz.numpy(), atol=1e-3)
    assert np.allclose(achieved_pose.quat_wxyz.numpy(), target_pose.quat_wxyz.numpy(), atol=1e-3)


def test_mobile_ik_with_base_pose():
    """Test mobile IK optimizing both robot q and T_world_base pose variables."""
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"
    robot = Robot.load(urdf, backend="warp")

    batch_size = 1
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

    T_world_base_np = np.array([0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    T_world_base = WarpSE3(wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device))

    q_np = robot.spec.midrange_q
    q = wp.from_numpy(q_np.reshape(1, -1), dtype=wp.float32, device=device)
    state = robot.state(q=q, T_world_base=T_world_base)

    target_pose_np = np.array([0.6, 0.0, 0.55, 0.0, 0.707, 0.0, -0.707], dtype=np.float32).reshape(1, 7)
    target_pose = WarpSE3(wp.from_numpy(target_pose_np, dtype=wp_vec7, device=device))
    frame_index = robot.link_names.index(target_link_name)

    terms = []

    frame_task = WarpFrameTask(
        robot=robot,
        frame_index=frame_index,
        T_world_target=target_pose,
        position_weight=10.0,
        orientation_weight=2.0,
    )
    terms.append(frame_task)

    position_limit = WarpPositionLimit(
        robot=robot,
        weight=2.0,
        batch_size=batch_size,
    )
    terms.append(position_limit)

    optimizer_config = WarpLMOptimizerConfig(
        max_iter=50,
        lm_lambda=1.0,
        lambda_factor=2.0,
        lambda_min=1e-6,
        lambda_max=1e6,
        verbose=False,
    )

    ik_optimizer = WarpLMOptimizer(
        terms=terms,
        device=device,
        config=optimizer_config,
        placeholder_var=state,
    )

    state = robot.state(q=q, T_world_base=T_world_base)
    state = ik_optimizer.solve(state)
    state = robot.forward_kinematics(state)

    achieved_pose = state.get_T_world_link(frame_index)
    assert np.allclose(achieved_pose.xyz.numpy(), target_pose.xyz.numpy(), atol=1e-3)

    achieved_quat = achieved_pose.quat_wxyz.numpy()
    target_quat = target_pose.quat_wxyz.numpy()
    dot_product = np.abs(np.sum(achieved_quat * target_quat))
    assert dot_product > 0.999, f"Quaternion mismatch: dot product {dot_product}"

    assert state.T_world_base is not None
    assert state.q.shape == (batch_size, robot.num_actuated_joints)


def test_ik_status():
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    target_pose = WarpSE3(wp.from_numpy(np.array([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]), dtype=wp_vec7))

    # Create config with known tolerance
    config = WarpLMOptimizerConfig(cost_tol=1.0)  # Very loose tolerance to ensure success

    ik_optimizer = robot.build_inverse_kinematics_optimizer("ee_link", target_pose, optimizer_config=config)
    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
    state = robot.state(q=q_init)

    # Solve with return_status=True
    state, status = ik_optimizer.solve(state, return_status=True)

    assert isinstance(status, wp.array)
    assert status.shape == (1,)

    status_np = status.numpy()
    assert status_np[0] == 1, "Optimization should succeed with loose tolerance"


def test_dense_optimizer_fails_fast_on_residual_layout_drift() -> None:
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    q_init = wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device)
    state = robot.state(q=q_init)

    term = _MutableResidualDenseTask(residual_dim=1)
    optimizer = WarpLMOptimizer(
        terms=[term],
        placeholder_var=state,
        device=device,
        config=WarpLMOptimizerConfig(max_iter=1, use_early_stopping=False),
    )

    term._residual_dim = 2
    with pytest.raises(ValueError, match="residual layout mismatch"):
        optimizer.solve(state)


def test_sparse_optimizer_fails_fast_on_residual_layout_drift() -> None:
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    q_init = wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device)
    state = robot.state(q=q_init)

    task = _MutableResidualSparseTask(residual_dim=1)
    optimizer = SparseWarpOptimizer(
        term=[task],
        batch_size=state.batch_size,
        total_tangent_dim=state.tangent_dim,
        device=device,
        config=SparseWarpOptimizerConfig(max_iter=1, use_early_stopping=False),
    )

    task._residual_dim = 2
    with pytest.raises(ValueError, match="residual layout mismatch"):
        optimizer.solve(state)
