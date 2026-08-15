"""Regression tests for the named online hand-retargeting pipeline."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from robokit.helpers.hand_retargeting.config import HandRetargetingOnlineConfig, HandSpec
from robokit.helpers.hand_retargeting.online import HandRetargetingOnline
from robokit.lie.se3 import se3_identity
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.utils.hand_coord_utils import MANOPTH_HAND_COORD_SPEC


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_GOLDEN = _FIXTURES / "hand_retargeting_legacy_golden.npz"
_URDF = _FIXTURES / "shadow_hand_right.urdf"
_TARGET_NAMES = (
    "middle_tip",
    "root",
    "thumb_tip",
    "index_base",
    "index_tip",
    "little_middle",
    "little_tip",
    "ring_middle",
    "ring_tip",
    "index_middle",
    "index_distal",
)
_SPEC = HandSpec(
    floating_base=True,
    target_names=_TARGET_NAMES,
    target_link_names={"root": "palm", "middle_tip": "mftip"},
    target_chains=(
        ("thumb_tip",),
        ("index_base", "index_middle", "index_distal", "index_tip"),
        ("middle_tip",),
        ("ring_middle", "ring_tip"),
        ("little_middle", "little_tip"),
    ),
    root_target_name="root",
    contact_target_names=("thumb_tip", "index_tip"),
    target_coord_spec=MANOPTH_HAND_COORD_SPEC,
    root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
)
_PINCH = ("thtip", "fftip", "thumb_tip", "index_tip")
_CONFIG = HandRetargetingOnlineConfig(
    position_weights={"root": 2.0, "middle_tip": 2.0},
    vector_weights={
        ("palm", "thtip", "root", "thumb_tip"): 1.5,
        ("palm", "fftip", "root", "index_tip"): 1.5,
        _PINCH: 1.5,
    },
    direction_weights={
        ("ffproximal", "ffmiddle", "index_base", "index_middle"): 1.0,
        ("ffmiddle", "ffdistal", "index_middle", "index_distal"): 1.0,
    },
    pinch_correspondences=(_PINCH,),
    vector_target_ema_alpha=0.6,
    direction_target_ema_alpha=0.7,
    output_ema_alpha=0.5,
    seed=17,
    sampling_distance=0.04,
    base_sampling_distance=0.02,
    pinch_threshold=0.02,
    pinch_release_threshold=0.05,
    pinch_target_norm=1e-4,
    q_smoothness_weight=0.02,
    regularization_weight=0.01,
    solver=MultiSeedSolverConfig(
        stages=[StageConfig(num_seeds=1, iters=2, lm_lambda=0.2)],
        cuda_graph_mode="none",
    ),
)


class TestHandRetargetingOnlineRedesign:
    """Protect streaming behavior without assuming a 21-point topology."""

    def test_irregular_stream_matches_legacy_golden(self) -> None:
        """Match streaming EMA, pinch hysteresis, and reset on 11 named points."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        helper = HandRetargetingOnline(robot, _SPEC, _CONFIG, device="cpu")
        helper.warmup(1)
        with np.load(_GOLDEN) as golden:
            points = golden["input_online_targets"]
            expected = np.concatenate([golden["output_online_stream_base"], golden["output_online_stream_q"]], axis=-1)
            expected_reset = np.concatenate(
                [golden["output_online_reset_base"], golden["output_online_reset_q"]], axis=-1
            )

        actual = np.concatenate([helper.solve_numpy(frame) for frame in points])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        helper.reset()
        np.testing.assert_allclose(helper.solve_numpy(points[0])[0], expected_reset, rtol=2e-5, atol=2e-5)

    def test_batch_and_caller_output(self) -> None:
        """Write two independent streams into a caller-owned Warp buffer."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = HandRetargetingOnlineConfig(
            **{
                **_CONFIG.__dict__,
                "output_ema_alpha": 1.0,
                "solver": MultiSeedSolverConfig(
                    stages=[StageConfig(num_seeds=1, iters=0, lm_lambda=0.2)], cuda_graph_mode="none"
                ),
            }
        )
        helper = HandRetargetingOnline(robot, _SPEC, config, device="cpu")
        helper.warmup(2)
        assert all(not updater.config.use_early_stopping for updater in helper._solver._updaters)
        with np.load(_GOLDEN) as golden:
            points = np.ascontiguousarray(golden["input_online_batch_targets"])
            expected = np.concatenate([golden["output_online_batch_base"], golden["output_online_batch_q"]], axis=-1)
        points_wp = wp.from_numpy(points, dtype=wp.float32, device="cpu")
        out = wp.empty(expected.shape, dtype=float, device="cpu")
        assert helper.solve(points_wp, out=out) is out
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-5, atol=2e-5)
        bad_init = robot.state(
            q=wp.zeros((2, robot.spec.num_actuated_joints), dtype=wp.float32, device="cpu"),
            T_world_base=se3_identity((1,), "cpu"),
        )
        with pytest.raises(ValueError, match="init_state must match"):
            helper.solve(points_wp, init_state=bad_init)
        with pytest.raises(ValueError, match="state must match"):
            helper.compute_costs(bad_init)
        helper.warmup(1)
        assert helper.solve(points_wp[:1]).shape[0] == 1

    def test_direction_only_and_numpy_output(self) -> None:
        """Accept a direction-only topology through the NumPy wrapper."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = HandRetargetingOnlineConfig(
            direction_weights={
                ("ffproximal", "ffmiddle", "index_base", "index_middle"): 1.0,
            },
            solver=MultiSeedSolverConfig(
                stages=[StageConfig(num_seeds=1, iters=0, lm_lambda=0.2)], cuda_graph_mode="none"
            ),
        )
        helper = HandRetargetingOnline(robot, _SPEC, config, device="cpu")
        helper.warmup(1)
        points = np.zeros((1, len(_TARGET_NAMES), 3), dtype=np.float32)
        points[:, _TARGET_NAMES.index("index_middle"), 0] = 1.0
        result = helper.solve_numpy(points)

        assert np.isfinite(result).all()

    def test_latched_pinch_gates_direction_group(self) -> None:
        """Disable affected direction pairs until a latched pinch is released."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = HandRetargetingOnlineConfig(
            **{
                **_CONFIG.__dict__,
                "gate_direction_on_pinch": True,
                "solver": MultiSeedSolverConfig(
                    stages=[StageConfig(num_seeds=1, iters=0, lm_lambda=0.2)], cuda_graph_mode="none"
                ),
            }
        )
        helper = HandRetargetingOnline(robot, _SPEC, config, device="cpu")
        helper.warmup(1)
        assert helper._direction_task is not None
        points = np.zeros((1, len(_TARGET_NAMES), 3), dtype=np.float32)
        points[:, _TARGET_NAMES.index("index_tip"), 0] = 0.01

        helper.solve_numpy(points)
        np.testing.assert_array_equal(helper._direction_task.weights_wp.numpy(), np.zeros((1, 2)))

        points[:, _TARGET_NAMES.index("index_tip"), 0] = 0.1
        helper.solve_numpy(points)
        released = np.broadcast_to(helper._direction_weights_np, (1, 2))
        np.testing.assert_allclose(helper._direction_task.weights_wp.numpy(), released, atol=1e-6)
