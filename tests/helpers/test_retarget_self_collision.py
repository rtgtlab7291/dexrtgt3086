"""Self-collision term in online humanoid retargeting.

Both wrists are commanded to the same point: without the term the hands must
interpenetrate; with it every non-adjacent sphere pair keeps ~margin clearance.
Uses the G1 preset's URDF + collision spheres from HF assets, so it is gated the
same way as tests/assets (needs HF auth, skipped in CI).
"""

import os

import numpy as np
import pytest
import warp as wp
from huggingface_hub import get_token

from robokit.helpers.humanoid_retarget import (
    HumanoidRetargetingOnline,
    HumanoidRetargetingOnlineConfig,
    LinkMapping,
)
from robokit.opt.multi_seed_solver import StageConfig
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


_RUNNABLE = get_token() is not None and not os.environ.get("GITHUB_ACTIONS")
_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_IDENT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_ZERO3 = np.zeros(3, dtype=np.float32)


def _config(urdf_path: str, spheres_path: str, weight: float) -> HumanoidRetargetingOnlineConfig:
    mapping = {
        "pelvis": LinkMapping("pelvis", 200.0, 50.0, _ZERO3, _IDENT),
        "left_wrist_yaw_link": LinkMapping("left_wrist", 50.0, 0.0, _ZERO3, _IDENT),
        "right_wrist_yaw_link": LinkMapping("right_wrist", 50.0, 0.0, _ZERO3, _IDENT),
    }
    return HumanoidRetargetingOnlineConfig(
        urdf_path=urdf_path,
        collision_spheres_path=spheres_path,
        link_mapping=mapping,
        smoothness_weight=0.0,
        base_smoothness_weight=0.0,
        velocity_limit_weight=0.0,
        velocity_clamp_scale=0.0,
        position_limit_weight=10.0,
        limit_warmup_frames=0,
        cuda_graph_mode="none",
        self_collision_weight=weight,
        self_collision_margin=0.01,
        stages=[
            StageConfig(num_seeds=4, iters=10, lm_lambda=1.0),
            StageConfig(num_seeds=1, iters=5, lm_lambda=1.0),
        ],
    )


def _left_right_min_dist(robot: Robot, qpos: np.ndarray) -> float:
    """Min surface distance over left-arm <-> right-arm collision sphere pairs."""
    from robokit.terms.dense.self_collision_task import compute_active_collision_pairs

    pairs = compute_active_collision_pairs(robot.spec)
    sphere_link = robot.spec.collision_spheres_link_indices
    left = np.array(["left_" in robot.spec.link_names[sphere_link[i]] for i in range(len(sphere_link))])
    right = np.array(["right_" in robot.spec.link_names[sphere_link[i]] for i in range(len(sphere_link))])
    lr = (left[pairs[:, 0]] & right[pairs[:, 1]]) | (right[pairs[:, 0]] & left[pairs[:, 1]])
    pairs = pairs[lr]

    base = wp.from_numpy(qpos[:7].reshape(1, 7).astype(np.float32), dtype=wp_vec7, device=_DEVICE)
    q = wp.from_numpy(qpos[7:].reshape(1, -1).astype(np.float32), dtype=wp.float32, device=_DEVICE)
    state = robot.state(q=q, T_world_base=base)
    robot.forward_kinematics(state)
    robot.transform_collision_spheres(state)
    centers = state.collision_sphere_centers_world.numpy()[0]
    radii = robot.spec.collision_sphere_radii
    d = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
    return float((d - radii[pairs[:, 0]] - radii[pairs[:, 1]]).min())


@pytest.mark.skipif(not _RUNNABLE, reason="needs HuggingFace auth (set HF_TOKEN)")
class TestRetargetSelfCollision:
    def _solve(self, robot: Robot, urdf_path: str, spheres_path: str, weight: float) -> float:
        helper = HumanoidRetargetingOnline(_config(urdf_path, spheres_path, weight), device=_DEVICE, robot=robot)
        frame = np.zeros((3, 7), dtype=np.float32)
        frame[:, 3] = 1.0
        frame[:, :3] = [[0.0, 0.0, 0.8], [0.25, 0.0, 0.9], [0.25, 0.0, 0.9]]
        data = np.repeat(frame[None], 10, axis=0)
        helper.warmup(1)
        helper.reset()
        for human_frame in data:
            qpos = helper.solve_numpy(human_frame[None])[0]
        return _left_right_min_dist(robot, qpos)

    def test_term_prevents_arm_interpenetration(self):
        from robokit.helpers.humanoid_retarget.presets.g1 import g1

        assert g1.urdf_path is not None and g1.collision_spheres_path is not None
        urdf_path = str(g1.urdf_path)
        spheres_path = g1.collision_spheres_path
        robot = Robot.load(urdf_path, load_collision_spheres=True, collision_spheres_path=spheres_path)

        dist_off = self._solve(robot, urdf_path, spheres_path, weight=0.0)
        dist_on = self._solve(robot, urdf_path, spheres_path, weight=100.0)

        assert dist_off < -0.02, f"scenario should force interpenetration without the term, got {dist_off}"
        assert dist_on > -0.002, f"self-collision term should keep clearance, got {dist_on}"
