"""Instance-batched online retargeting through caller-owned frame iteration.

Two bitwise-provable guarantees of the batch dimension:
  - Rows are independent: a clip's output does not depend on its batch neighbours.
  - Without the smoothness term, batch_size=B reproduces batch_size=1 exactly.

(With smoothness on, batched is NOT bitwise-identical to the per-clip loop: that term's
GPU kernel is float-sensitive to the total batch size, and multi-solution IK amplifies it on
some frames. That is a known limitation, not tested here.)
"""

import numpy as np
import pytest
import warp as wp

from robokit.helpers.humanoid_retarget import (
    HumanoidRetargetingOnline,
    HumanoidRetargetingOnlineConfig,
    LinkMapping,
)
from robokit.opt.multi_seed_solver import StageConfig
from robokit.utils.warp_utils import wp_vec7


_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_IDENT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_ZERO3 = np.zeros(3, dtype=np.float32)


def _config(smoothness):
    mapping = {
        "pelvis": LinkMapping("pelvis", 200.0, 50.0, _ZERO3, _IDENT),
        "left_wrist_yaw_link": LinkMapping("left_wrist", 50.0, 0.0, _ZERO3, _IDENT),
        "right_wrist_yaw_link": LinkMapping("right_wrist", 50.0, 0.0, _ZERO3, _IDENT),
    }
    return HumanoidRetargetingOnlineConfig(
        urdf_path="<robot-passed-in>",
        link_mapping=mapping,
        smoothness_weight=2.0 if smoothness else 0.0,
        base_smoothness_weight=2.0 if smoothness else 0.0,
        velocity_limit_weight=0.0,
        velocity_clamp_scale=0.0,
        position_limit_weight=10.0,
        limit_warmup_frames=0,
        cuda_graph_mode="none",
        stages=[StageConfig(num_seeds=1, iters=4, lm_lambda=1.0)],
    )


def _clip(seed, num_frames):
    """A short trajectory of moving wrists (so consecutive frames differ)."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((num_frames, 3, 7), dtype=np.float32)
    frames[..., 3] = 1.0
    for f in range(num_frames):
        gap = 0.3 + 0.1 * f + 0.05 * rng.standard_normal()
        frames[f, 0, :3] = [0.02 * f, 0.0, 0.8]
        frames[f, 1, :3] = [0.0, gap / 2, 0.8 + 0.03 * f]
        frames[f, 2, :3] = [0.0, -gap / 2, 0.8 - 0.02 * f]
    return frames


def _run_lockstep(helper, clips):
    """Advance clips in caller-owned lockstep, repeating each final frame."""
    helper.warmup(len(clips))
    helper.reset()
    output = np.stack(
        [
            helper.solve_numpy(np.stack([clip[min(frame, len(clip) - 1)] for clip in clips]))
            for frame in range(max(map(len, clips)))
        ],
        axis=1,
    )
    return [output[b, : len(clip)] for b, clip in enumerate(clips)]


class TestInstanceBatchedRetarget:
    def test_warmup_fixes_batch_size(self, g1_robot):
        helper = HumanoidRetargetingOnline(_config(False), device=_DEVICE, robot=g1_robot)
        assert helper._batch_size is None
        helper.warmup(2)
        solver = helper._solver
        helper.warmup(2)
        assert helper._solver is solver
        with pytest.raises(ValueError, match="batch size mismatch"):
            helper.warmup(3)

    def test_rows_independent_without_smoothness(self, g1_robot):
        # A clip's output is identical regardless of which clips share its batch.
        cfg = _config(smoothness=False)
        helper = HumanoidRetargetingOnline(cfg, device=_DEVICE, robot=g1_robot, seed=0)
        probe = _clip(seed=0, num_frames=4)
        out_a = _run_lockstep(helper, [probe, _clip(1, 4), _clip(2, 4)])[0]
        out_b = _run_lockstep(helper, [probe, _clip(7, 4), _clip(8, 4)])[0]
        assert np.isfinite(out_a).all()
        np.testing.assert_array_equal(out_a, out_b)  # bitwise: neighbours cannot leak in

    def test_batch_exact_without_smoothness(self, g1_robot):
        # With smoothness off, batch_size=B reproduces the per-clip batch_size=1 result exactly.
        cfg = _config(smoothness=False)
        h1 = HumanoidRetargetingOnline(cfg, device=_DEVICE, robot=g1_robot, seed=0)
        clips = [_clip(seed=s, num_frames=n) for s, n in [(0, 5), (1, 3), (2, 4)]]
        baseline = [_run_lockstep(h1, [clips[i]])[0] for i in range(3)]

        h3 = HumanoidRetargetingOnline(cfg, device=_DEVICE, robot=g1_robot, seed=0)
        batched = _run_lockstep(h3, clips)
        for i in range(3):
            np.testing.assert_array_equal(batched[i], baseline[i])

    def test_with_smoothness_close_not_exact(self, g1_robot):
        # With smoothness on, batched tracks the per-clip result closely but NOT bitwise
        # (the smoothness GPU kernel is float-sensitive to the total batch size).
        cfg = _config(smoothness=True)
        h1 = HumanoidRetargetingOnline(cfg, device=_DEVICE, robot=g1_robot, seed=0)
        clips = [_clip(seed=s, num_frames=5) for s in range(3)]
        baseline = [_run_lockstep(h1, [clips[i]])[0] for i in range(3)]

        h3 = HumanoidRetargetingOnline(cfg, device=_DEVICE, robot=g1_robot, seed=0)
        batched = _run_lockstep(h3, clips)
        for i in range(3):
            assert np.isfinite(batched[i]).all()
            assert np.abs(batched[i] - baseline[i]).mean() < 0.05  # close in the mean

    def test_warp_and_numpy_match(self, g1_robot):
        helper = HumanoidRetargetingOnline(_config(False), device=_DEVICE, robot=g1_robot)
        data = _clip(0, 1)
        helper.warmup(1)
        helper.reset()
        out = wp.empty((1, 7 + g1_robot.spec.num_actuated_joints), dtype=wp.float32, device=_DEVICE)
        warp_result = helper.solve(wp.from_numpy(data, dtype=wp_vec7, device=_DEVICE), out=out)
        assert warp_result is out
        warp_output = out.numpy().copy()
        helper.reset()
        numpy_output = helper.solve_numpy(data)
        np.testing.assert_array_equal(warp_output, numpy_output)

    def test_one_call_advances_one_frame(self, g1_robot):
        helper = HumanoidRetargetingOnline(_config(False), device=_DEVICE, robot=g1_robot)
        data = _clip(0, 2)
        helper.warmup(1)
        first = helper.solve_numpy(data[:1]).copy()
        second = helper.solve_numpy(data[1:]).copy()
        assert first.shape == second.shape == (1, 7 + g1_robot.spec.num_actuated_joints)
        assert helper._frame_count == 2
