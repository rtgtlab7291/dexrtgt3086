"""Optional vector-pair correspondence objective for humanoid retargeting.

Builds a minimal G1 config against the offline ``g1_description`` fixture and
checks the correspondence edge plumbs through: a term per stage, the per-frame
target equals the human joint-to-joint vector, and the main path is untouched
when ``correspondence_edges`` is ``None``.
"""

import numpy as np
import warp as wp

from robokit.helpers.humanoid_retarget import (
    CorrespondenceEdge,
    HumanoidRetargetingOnline,
    HumanoidRetargetingOnlineConfig,
    LinkMapping,
)
from robokit.opt.multi_seed_solver import StageConfig


_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_IDENT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_ZERO3 = np.zeros(3, dtype=np.float32)
_EDGE = CorrespondenceEdge("left_wrist_yaw_link", "right_wrist_yaw_link", "left_wrist", "right_wrist", weight=100.0)


def _config(edges):
    mapping = {
        "pelvis": LinkMapping("pelvis", 200.0, 50.0, _ZERO3, _IDENT),
        "left_wrist_yaw_link": LinkMapping("left_wrist", 50.0, 0.0, _ZERO3, _IDENT),
        "right_wrist_yaw_link": LinkMapping("right_wrist", 50.0, 0.0, _ZERO3, _IDENT),
    }
    return HumanoidRetargetingOnlineConfig(
        urdf_path="<robot-passed-in>",  # robot is supplied, so urdf_path is never loaded
        link_mapping=mapping,
        correspondence_edges=edges,
        smoothness_weight=0.0,
        base_smoothness_weight=0.0,
        velocity_limit_weight=0.0,
        position_limit_weight=0.0,
        cuda_graph_mode="none",
        stages=[StageConfig(num_seeds=1, iters=3, lm_lambda=1.0)],
    )


def _frame(wrist_gap):
    frame = np.zeros((1, 3, 7), dtype=np.float32)
    frame[..., 3] = 1.0
    frame[0, :, :3] = [[0.0, 0.0, 0.8], [0.0, wrist_gap / 2, 0.8], [0.0, -wrist_gap / 2, 0.8]]
    return frame


class TestHumanoidCorrespondence:
    def test_edges_none_adds_no_term(self, g1_robot):
        helper = HumanoidRetargetingOnline(_config(None), device=_DEVICE, robot=g1_robot)
        helper.warmup(1)
        assert helper._correspondence_tasks == []

    def test_edge_builds_one_task_per_stage(self, g1_robot):
        cfg = _config([_EDGE])
        helper = HumanoidRetargetingOnline(cfg, device=_DEVICE, robot=g1_robot)
        helper.warmup(1)
        assert len(helper._correspondence_tasks) == len(cfg.stages)

    def test_target_is_human_joint_difference(self, g1_robot):
        helper = HumanoidRetargetingOnline(_config([_EDGE]), device=_DEVICE, robot=g1_robot)
        data = _frame(0.5)
        helper.warmup(1)
        helper.solve_numpy(data)
        expected = data[0, 2, :3] - data[0, 1, :3]
        got = helper._correspondence_tasks[0].targets_wp.numpy()[0, 0]
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_solve_runs_with_edge(self, g1_robot):
        helper = HumanoidRetargetingOnline(_config([_EDGE]), device=_DEVICE, robot=g1_robot)
        helper.warmup(1)
        qpos = helper.solve_numpy(_frame(0.3))[0]
        assert qpos.shape == (7 + g1_robot.spec.num_actuated_joints,)
