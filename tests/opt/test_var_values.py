import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.lie.warp_se3 import WarpSE3
from robokit.opt.var_values import WarpVarValues
from robokit.opt.warp_optimizer import WarpLMOptimizer, WarpLMOptimizerConfig
from robokit.robo.robot import Robot
from robokit.terms import WarpFrameTask, WarpPositionLimit
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_vec6, wp_vec7


@wp.kernel
def _vec6_to_float32_kernel(
    src: wp.array(dtype=wp_vec6),
    dst: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    v = src[i]
    dst[i, 0] = v[0]
    dst[i, 1] = v[1]
    dst[i, 2] = v[2]
    dst[i, 3] = v[3]
    dst[i, 4] = v[4]
    dst[i, 5] = v[5]


@wp.kernel
def _copy_residual_kernel(
    residual: wp.array2d(dtype=wp.float32),
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
):
    batch_idx, col_idx = wp.tid()
    residual_buffer[batch_idx, row_offset + col_idx] = residual[batch_idx, col_idx]


# ---------------------------------------------------------------------------
# Simple SE3 target-matching task for multi-var tests (autodiff)
# ---------------------------------------------------------------------------
class SE3TargetTask(WarpTask):
    JACOBIAN_MODE = "autodiff"

    def __init__(self, target: WarpSE3, var_key: str):
        self.target = target
        self.var_key = var_key
        self.residual_weight = None
        self._batch_size = target.batch_size

    @property
    def residual_dim(self) -> int:
        return 6

    def compute_weighted_residual(
        self,
        var: WarpSE3,
        residual_buffer=None,
        row_offset: int = 0,
    ) -> wp.array:
        error_vec6 = self.target.inverse().multiply(var).log()
        error = wp.empty(
            (self._batch_size, 6),
            dtype=wp.float32,
            device=error_vec6.device,
            requires_grad=error_vec6.requires_grad,
        )
        wp.launch(
            kernel=_vec6_to_float32_kernel,
            dim=self._batch_size,
            inputs=[error_vec6],
            outputs=[error],
            device=error_vec6.device,
        )
        if residual_buffer is not None:
            wp.launch(
                kernel=_copy_residual_kernel,
                dim=(self._batch_size, 6),
                inputs=[error, row_offset],
                outputs=[residual_buffer],
                device=error.device,
            )
            return residual_buffer
        return error


# ---------------------------------------------------------------------------
# Unit tests for WarpVarValues
# ---------------------------------------------------------------------------
def test_tangent_dim_and_offsets():
    se3_a = WarpSE3.identity(shape=(2,))
    se3_b = WarpSE3.identity(shape=(2,))
    vals = WarpVarValues(pose_a=se3_a, pose_b=se3_b)
    assert vals.tangent_dim == 12
    assert vals.batch_size == 2
    assert vals.tangent_offset("pose_a") == 0
    assert vals.tangent_offset("pose_b") == 6


def test_clone():
    se3_a = WarpSE3.identity(shape=(2,))
    se3_b = WarpSE3.identity(shape=(2,))
    vals = WarpVarValues(pose_a=se3_a, pose_b=se3_b)
    cloned = vals.clone()
    assert cloned.tangent_dim == vals.tangent_dim
    assert cloned.batch_size == vals.batch_size
    assert cloned.tangent_offset("pose_a") == 0
    assert cloned.tangent_offset("pose_b") == 6

    original_xyz = vals.get("pose_a").xyz.numpy().copy()
    velocity = wp.zeros((2, 12), dtype=wp.float32)
    # Modify the original via integrate with non-zero velocity
    vel_np = np.zeros((2, 12), dtype=np.float32)
    vel_np[:, 0] = 1.0  # translate pose_a in x
    velocity = wp.from_numpy(vel_np, dtype=wp.float32)
    vals.integrate(velocity)
    # Clone should be unchanged
    assert np.allclose(cloned.get("pose_a").xyz.numpy(), original_xyz)


def test_integrate():
    se3_a = WarpSE3.identity(shape=(1,))
    se3_b = WarpSE3.identity(shape=(1,))
    vals = WarpVarValues(pose_a=se3_a, pose_b=se3_b)

    vel_np = np.zeros((1, 12), dtype=np.float32)
    vel_np[0, 0] = 0.5  # translate pose_a in x
    vel_np[0, 7] = 0.3  # translate pose_b in y
    velocity = wp.from_numpy(vel_np, dtype=wp.float32)

    result = vals.integrate(velocity)
    result_a = result.get("pose_a")
    result_b = result.get("pose_b")

    assert abs(result_a.xyz.numpy()[0, 0] - 0.5) < 1e-4
    assert abs(result_b.xyz.numpy()[0, 1] - 0.3) < 1e-4


def test_gather():
    se3_a = WarpSE3(
        wp.from_numpy(
            np.array(
                [
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            dtype=wp_vec7,
        )
    )
    se3_b = WarpSE3.identity(shape=(3,))
    vals = WarpVarValues(pose_a=se3_a, pose_b=se3_b)

    indices = wp.from_numpy(np.array([2, 0], dtype=np.int32))
    gathered = vals.gather(indices)
    assert gathered.batch_size == 2
    gathered_xyz = gathered.get("pose_a").xyz.numpy()
    assert abs(gathered_xyz[0, 0] - 3.0) < 1e-4
    assert abs(gathered_xyz[1, 0] - 1.0) < 1e-4


def test_single_var_getattr():
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
    state = robot.state(q=q_init)
    vals = WarpVarValues.from_var(state)

    assert vals.tangent_dim == state.tangent_dim
    assert vals.batch_size == state.batch_size
    # __getattr__ should delegate to the sole var
    assert np.allclose(vals.q.numpy(), state.q.numpy())


def test_from_var():
    se3 = WarpSE3.identity(shape=(2,))
    vals = WarpVarValues.from_var(se3, name="pose")
    assert vals.tangent_dim == 6
    assert vals.tangent_offset("pose") == 0
    assert vals.get("pose") is se3


# ---------------------------------------------------------------------------
# Backward compatibility: IK with VarValues wrapper
# ---------------------------------------------------------------------------
def test_ik_with_var_values():
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    target_pose = WarpSE3(wp.from_numpy(np.array([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]), dtype=wp_vec7))

    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
    state = robot.state(q=q_init)
    vals = WarpVarValues.from_var(state)

    frame_index = robot.link_names.index("ee_link")
    frame_task = WarpFrameTask(
        robot=robot,
        frame_index=frame_index,
        T_world_target=target_pose,
        position_weight=1.0,
        orientation_weight=1.0,
    )
    position_limit = WarpPositionLimit(robot=robot, weight=1.0)

    optimizer = WarpLMOptimizer(
        terms=[frame_task, position_limit],
        placeholder_var=vals,
        config=WarpLMOptimizerConfig(max_iter=100),
    )

    result_vals = optimizer.solve(vals)
    result_state = result_vals.get("robot")
    result_state = robot.forward_kinematics(result_state)
    achieved_pose = result_state.get_T_world_link(frame_index)

    assert np.allclose(achieved_pose.xyz.numpy(), target_pose.xyz.numpy(), atol=1e-3)
    assert np.allclose(achieved_pose.quat_wxyz.numpy(), target_pose.quat_wxyz.numpy(), atol=1e-3)


# ---------------------------------------------------------------------------
# Multi-variable optimization: two SE3 poses
# ---------------------------------------------------------------------------
def test_multi_var_se3_optimization():
    batch_size = 1
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

    pose_a = WarpSE3.identity(shape=(batch_size,), device=device)
    pose_b = WarpSE3.identity(shape=(batch_size,), device=device)

    target_a_np = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    target_b_np = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    target_a = WarpSE3(wp.from_numpy(target_a_np, dtype=wp_vec7, device=device))
    target_b = WarpSE3(wp.from_numpy(target_b_np, dtype=wp_vec7, device=device))

    task_a = SE3TargetTask(target=target_a, var_key="pose_a")
    task_b = SE3TargetTask(target=target_b, var_key="pose_b")

    vals = WarpVarValues(pose_a=pose_a, pose_b=pose_b)

    optimizer = WarpLMOptimizer(
        terms=[task_a, task_b],
        placeholder_var=vals,
        device=device,
        config=WarpLMOptimizerConfig(max_iter=100),
    )

    result = optimizer.solve(vals)
    result_a_xyz = result.get("pose_a").xyz.numpy()
    result_b_xyz = result.get("pose_b").xyz.numpy()

    assert np.allclose(result_a_xyz, [[1.0, 0.0, 0.0]], atol=1e-2), f"pose_a xyz: {result_a_xyz}"
    assert np.allclose(result_b_xyz, [[0.0, 1.0, 0.0]], atol=1e-2), f"pose_b xyz: {result_b_xyz}"


# ---------------------------------------------------------------------------
# Mixed: robot IK + independent SE3 pose
# ---------------------------------------------------------------------------
def test_robot_plus_se3_optimization():
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32, device=device)
    state = robot.state(q=q_init)

    ee_pose = WarpSE3.identity(shape=(1,), device=device)

    vals = WarpVarValues(robot=state, ee_pose=ee_pose)

    robot_target = WarpSE3(wp.from_numpy(np.array([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]), dtype=wp_vec7, device=device))
    ee_target_np = np.array([[0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ee_target = WarpSE3(wp.from_numpy(ee_target_np, dtype=wp_vec7, device=device))

    frame_index = robot.link_names.index("ee_link")
    frame_task = WarpFrameTask(
        robot=robot,
        frame_index=frame_index,
        T_world_target=robot_target,
        position_weight=1.0,
        orientation_weight=1.0,
    )
    ee_task = SE3TargetTask(target=ee_target, var_key="ee_pose")

    optimizer = WarpLMOptimizer(
        terms=[frame_task, ee_task],
        placeholder_var=vals,
        device=device,
        config=WarpLMOptimizerConfig(max_iter=100),
    )

    result = optimizer.solve(vals)

    # Verify robot IK converged
    result_state = result.get("robot")
    result_state = robot.forward_kinematics(result_state)
    achieved_pose = result_state.get_T_world_link(frame_index)
    assert np.allclose(achieved_pose.xyz.numpy(), robot_target.xyz.numpy(), atol=1e-3)

    # Verify SE3 pose converged
    result_ee = result.get("ee_pose")
    assert np.allclose(result_ee.xyz.numpy(), ee_target.xyz.numpy(), atol=1e-2)


# ---------------------------------------------------------------------------
# Benchmark: VarValues wrapper vs bare var should produce identical costs
# ---------------------------------------------------------------------------
def test_var_values_cost_equivalence():
    robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
    target_pose = WarpSE3(wp.from_numpy(np.array([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]), dtype=wp_vec7))

    frame_index = robot.link_names.index("ee_link")

    frame_task = WarpFrameTask(
        robot=robot,
        frame_index=frame_index,
        T_world_target=target_pose,
        position_weight=1.0,
        orientation_weight=1.0,
    )

    config = WarpLMOptimizerConfig(max_iter=50)

    # Solve with bare var
    state_bare = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
    opt_bare = WarpLMOptimizer(terms=[frame_task], placeholder_var=state_bare, config=config)
    result_bare = opt_bare.solve(state_bare)

    # Solve with VarValues wrapper
    state_wrapped = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
    vals = WarpVarValues.from_var(state_wrapped)
    opt_wrapped = WarpLMOptimizer(terms=[frame_task], placeholder_var=vals, config=config)
    result_wrapped = opt_wrapped.solve(vals)

    q_bare = result_bare.q.numpy()
    q_wrapped = result_wrapped.get("robot").q.numpy()
    assert np.allclose(q_bare, q_wrapped, atol=1e-4), f"Max diff: {np.max(np.abs(q_bare - q_wrapped))}"
