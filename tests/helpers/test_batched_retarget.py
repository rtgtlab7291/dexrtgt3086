"""Public offline retargeting handles batched clips and variable lengths."""

from typing import Sequence

import numpy as np
import pytest
import warp as wp

from robokit.helpers.humanoid_retarget import (
    HumanoidRetargetingOffline,
    HumanoidRetargetingOfflineConfig,
    HumanoidRetargetingOnline,
    HumanoidRetargetingOnlineConfig,
    LinkMapping,
)
from robokit.lie.se3 import se3_identity
from robokit.opt.multi_seed_solver import StageConfig
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


pytestmark = pytest.mark.slow


@pytest.fixture
def device() -> str:
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _retargeter(robot: Robot, link_names: Sequence[str], device: str) -> HumanoidRetargetingOffline:
    zero = np.zeros(3, dtype=np.float32)
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mapping = {name: LinkMapping(name, 10.0, 5.0, zero.copy(), identity.copy()) for name in link_names}
    retarget_config = HumanoidRetargetingOnlineConfig(
        urdf_path="<robot-passed-in>",
        human_root_name=link_names[0],
        scale_table={name: 1.0 for name in link_names},
        link_mapping=mapping,
    )
    offline_config = HumanoidRetargetingOfflineConfig(
        max_iter=20,
        cuda_graph_mode="full" if device.startswith("cuda") else "none",
    )
    return HumanoidRetargetingOffline(retarget_config, offline_config, robot=robot, device=device)


def _motion(robot: Robot, link_names: Sequence[str], num_frames: int, seed: int, device: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dofs = robot.spec.num_actuated_joints
    limits = robot.spec.actuated_joint_limits
    amplitude = 0.15 * (limits[:, 1] - limits[:, 0])
    phase = rng.uniform(0, 2 * np.pi, dofs)
    frequency = rng.uniform(0.5, 1.5, dofs)
    time = np.linspace(0.0, 1.0, num_frames)[:, None]
    q = robot.spec.midrange_q[None] + amplitude[None] * np.sin(2 * np.pi * frequency[None] * time + phase[None])
    q = np.clip(q, limits[:, 0], limits[:, 1]).astype(np.float32)
    state = robot.forward_kinematics(
        robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device=device),
            T_world_base=se3_identity(shape=(num_frames,), device=device),
        )
    )
    link_indices = [robot.spec.link_names.index(name) for name in link_names]
    return np.ascontiguousarray(state.T_world_link.numpy()[:, link_indices])


class TestBatchedOfflineRetarget:
    def test_targets_match_online(self, g1_robot: Robot, device: str):
        names = [
            g1_robot.spec.link_names[0],
            g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 3],
            g1_robot.spec.link_names[2 * len(g1_robot.spec.link_names) // 3],
        ]
        offline = _retargeter(g1_robot, names, device)
        motion = _motion(g1_robot, names, 4, 4, device)
        offline.warmup(1, len(motion))
        offline._compute_targets(
            wp.from_numpy(motion[None], dtype=wp_vec7, device=device),
            wp.from_numpy(np.array([1.6], dtype=np.float32), dtype=wp.float32, device=device),
        )

        offline.retarget_config.stages = [StageConfig(num_seeds=1, iters=0)]
        offline.retarget_config.cuda_graph_mode = "none"
        online = HumanoidRetargetingOnline(offline.retarget_config, robot=g1_robot, device=device)
        online.warmup(len(motion))
        online.solve_numpy(motion, np.full(len(motion), 1.6, dtype=np.float32))

        np.testing.assert_array_equal(
            offline._T_world_target_wp.numpy(),
            online._frame_tasks[0].T_world_target.numpy(),
        )
        np.testing.assert_array_equal(
            offline._T_world_base_target_wp.numpy(),
            online._base_target_wp.numpy(),
        )

    def test_batch_matches_per_clip(self, g1_robot: Robot, device: str):
        if not wp.is_cuda_available():
            pytest.skip("offline batch equality is float-sensitive on the Warp CPU backend")
        names = [
            g1_robot.spec.link_names[0],
            g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 3],
            g1_robot.spec.link_names[2 * len(g1_robot.spec.link_names) // 3],
        ]
        retargeter = _retargeter(g1_robot, names, device)
        motion_a = _motion(g1_robot, names, 16, 0, device)
        motion_b = _motion(g1_robot, names, 16, 1, device)

        qpos_a = retargeter.solve_numpy(motion_a[None])[0]
        single_optimizer = retargeter._optimizer
        retargeter.warmup(1, 16)
        assert retargeter._optimizer is single_optimizer
        qpos_b = retargeter.solve_numpy(motion_b[None])[0]
        assert retargeter._optimizer is single_optimizer
        batched_a, batched_b = retargeter.solve_numpy(np.stack([motion_a, motion_b]))
        assert retargeter._optimizer is not single_optimizer

        np.testing.assert_allclose(batched_a, qpos_a, atol=5e-3)
        np.testing.assert_allclose(batched_b, qpos_b, atol=5e-3)

    def test_variable_lengths_are_trimmed(self, g1_robot: Robot, device: str):
        names = [
            g1_robot.spec.link_names[0],
            g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 3],
            g1_robot.spec.link_names[2 * len(g1_robot.spec.link_names) // 3],
        ]
        retargeter = _retargeter(g1_robot, names, device)
        short = _motion(g1_robot, names, 8, 2, device)
        long = _motion(g1_robot, names, 12, 3, device)

        retargeter.warmup(2, 12)
        optimizer = retargeter._optimizer
        padded_short = np.concatenate([short, np.repeat(short[-1:], len(long) - len(short), axis=0)])
        qpos = retargeter.solve_numpy(
            np.stack([padded_short, long]),
            valid_lengths=np.array([len(short), len(long)], dtype=np.int32),
        )
        short_qpos, long_qpos = qpos[0, : len(short)], qpos[1]

        assert retargeter._optimizer is optimizer
        assert short_qpos.shape == (8, 7 + g1_robot.spec.num_actuated_joints)
        assert long_qpos.shape == (12, 7 + g1_robot.spec.num_actuated_joints)
        assert np.isfinite(short_qpos).all()
        assert np.isfinite(long_qpos).all()

    def test_caller_owned_output(self, g1_robot: Robot, device: str):
        names = [g1_robot.spec.link_names[0], g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 2]]
        retargeter = _retargeter(g1_robot, names, device)
        motion = _motion(g1_robot, names, 4, 5, device)
        retargeter.warmup(1, len(motion))
        out = wp.empty((1, len(motion), 7 + g1_robot.spec.num_actuated_joints), dtype=wp.float32, device=device)
        result = retargeter.solve(wp.from_numpy(motion[None], dtype=wp_vec7, device=device), out=out)
        assert result is out
        assert np.isfinite(out.numpy()).all()
