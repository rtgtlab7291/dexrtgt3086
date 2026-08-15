import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms import FrameTask, PositionLimit
from robokit.terms.dense.rest_task import RestTask
from robokit.utils.warp_utils import wp_vec7


_robot_cache: dict = {}


def _target_pose() -> wp.array:
    target = np.array([[[0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    return wp.from_numpy(target, dtype=wp_vec7)


def _get_robot(desc_name: str) -> Robot:
    if desc_name not in _robot_cache:
        _robot_cache[desc_name] = Robot.load(load_robot_description(desc_name))
    return _robot_cache[desc_name]


requires_cuda = pytest.mark.skipif(
    not wp.is_cuda_available(), reason="Analytic gradient IK convergence is compiler-dependent on CPU"
)


class TestLBFGSOptimizer:
    def test_armijo_rejects_non_decreasing_step(self):
        robot = _get_robot("ur10_description")
        q_init = robot.spec.zero_q.astype(np.float32).reshape(1, -1)
        optimizer = LBFGSOptimizer(
            terms=[RestTask(robot=robot, rest_q=robot.spec.midrange_q)],
            config=LBFGSOptimizerConfig(max_iter=3, line_search_alphas=(1.0e6,)),
            device="cpu",
        )
        var = VarValues(robot=robot.state(q=wp.from_numpy(q_init, dtype=wp.float32, device="cpu")))

        result, costs = optimizer.solve(var)

        np.testing.assert_array_equal(result.get("robot").q.numpy(), q_init)
        assert costs.shape == (1,)

    @requires_cuda
    def test_ik_converges(self):
        robot = _get_robot("ur10_description")
        target_pose = _target_pose()

        config = LBFGSOptimizerConfig(max_iter=100, use_early_stopping=True, cost_tol=1e-6)
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = LBFGSOptimizer(
            terms=[frame_task],
            config=config,
        )

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        state = robot.forward_kinematics(state)
        achieved_pose = state.get_T_world_link(frame_index)

        assert np.allclose(achieved_pose.numpy(), target_pose.numpy()[:, 0], atol=1e-2)

    @requires_cuda
    def test_ik_mobile(self):
        """Multi-var IK: optimize both robot q and T_world_base."""
        robot = _get_robot("panda_description")

        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

        T_world_base_np = np.array([0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device)

        q_np = robot.spec.midrange_q
        q = wp.from_numpy(q_np.reshape(1, -1), dtype=wp.float32, device=device)
        state = robot.state(q=q, T_world_base=T_world_base)

        target_pose_np = np.array([0.6, 0.0, 0.55, 0.0, 0.707, 0.0, -0.707], dtype=np.float32).reshape(1, 7)
        target_pose = wp.from_numpy(target_pose_np[:, None], dtype=wp_vec7, device=device)
        frame_index = robot.link_names.index("panda_hand")

        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )
        position_limit = PositionLimit(robot=robot, weight=2.0)

        config = LBFGSOptimizerConfig(max_iter=100, use_early_stopping=False)
        optimizer = LBFGSOptimizer(
            terms=[frame_task, position_limit],
            device=device,
            config=config,
        )

        state = robot.state(q=q, T_world_base=T_world_base)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        state = robot.forward_kinematics(state)

        achieved_pose = state.get_T_world_link(frame_index).numpy()
        target_pose_np = target_pose.numpy()[:, 0]
        assert np.allclose(achieved_pose[:, :3], target_pose_np[:, :3], atol=1e-2)
        assert np.all(np.abs(np.sum(achieved_pose[:, 3:] * target_pose_np[:, 3:], axis=1)) > 0.99)

    def test_ik_autodiff_gradient(self):
        """Test that autodiff gradient mode converges as well as analytic."""
        robot = _get_robot("ur10_description")
        target_pose = _target_pose()

        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target_pose,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = LBFGSOptimizerConfig(max_iter=100, use_early_stopping=True, cost_tol=1e-6, gradient_mode="autodiff")
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = LBFGSOptimizer(terms=[frame_task], config=config)

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues(robot=state))[0].get("robot")
        state = robot.forward_kinematics(state)
        achieved_pose = state.get_T_world_link(frame_index)

        assert np.allclose(achieved_pose.numpy(), target_pose.numpy()[:, 0], atol=1e-2)
