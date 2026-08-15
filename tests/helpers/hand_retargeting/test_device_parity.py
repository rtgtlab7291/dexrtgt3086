"""CPU/CUDA parity tests for named hand-retargeting point inputs."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from robokit.helpers.hand_retargeting import (
    HandRetargetingOffline,
    HandRetargetingOfflineConfig,
    HandRetargetingOnline,
    HandRetargetingOnlineConfig,
    HandSpec,
)
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizerConfig
from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.utils.hand_coord_utils import MANOPTH_HAND_COORD_SPEC


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_URDF = _FIXTURES / "shadow_hand_right.urdf"
_SPEC = HandSpec(
    floating_base=True,
    target_names=("index_tip", "root", "thumb_tip", "index_base", "index_middle"),
    target_link_names={
        "root": "palm",
        "thumb_tip": "thtip",
        "index_middle": "ffmiddle",
        "index_tip": "fftip",
    },
    target_chains=(("thumb_tip",), ("index_base", "index_middle", "index_tip"), ("root",)),
    root_target_name="root",
    contact_target_names=("thumb_tip", "index_tip"),
    target_coord_spec=MANOPTH_HAND_COORD_SPEC,
    root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
    init_q_by_name={"FFJ1": 0.0},
)
_POINTS = np.array(
    [
        [[0.08, 0.02, 0.01], [0.01, -0.02, 0.03], [-0.04, 0.03, 0.02], [0.03, 0.01, 0.00], [0.06, 0.02, 0.01]],
        [[0.07, 0.01, 0.02], [-0.01, 0.01, 0.02], [-0.03, 0.04, 0.01], [0.02, 0.00, 0.01], [0.05, 0.01, 0.02]],
    ],
    dtype=np.float32,
)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
class TestHandRetargetingDeviceParity:
    """Compare packed CPU and CUDA outputs for canonical point arrays."""

    def test_online_irregular_points(self) -> None:
        """Match batched online results with named position, vector, and direction targets."""
        config = HandRetargetingOnlineConfig(
            position_weights={"root": 1.0, "index_tip": 1.0},
            vector_weights={("thtip", "fftip", "thumb_tip", "index_tip"): 1.0},
            direction_weights={
                ("ffproximal", "ffmiddle", "index_base", "index_middle"): 1.0,
            },
            sampling_distance=0.0,
            base_sampling_distance=0.0,
            limit_weight=0.0,
            solver=MultiSeedSolverConfig(
                stages=[StageConfig(num_seeds=1, iters=1, lm_lambda=0.2)],
                cuda_graph_mode="none",
            ),
        )
        results = []
        for device in ("cpu", "cuda:0"):
            robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
            helper = HandRetargetingOnline(robot, _SPEC, config, device=device)
            helper.warmup(_POINTS.shape[0])
            points_wp = wp.from_numpy(_POINTS, dtype=wp.float32, device=device)
            results.append(helper.solve(points_wp).numpy())

        np.testing.assert_allclose(results[1], results[0], rtol=1e-6, atol=1e-6)

    def test_offline_irregular_points(self) -> None:
        """Match batched trajectory results from point-only offline input."""
        config = HandRetargetingOfflineConfig(
            root_orientation_weight=0.0,
            velocity_weight=0.0,
            acceleration_weight=0.0,
            root_position_velocity_weight=0.0,
            root_orientation_velocity_weight=0.0,
            limit_weight=0.0,
            rest_weight=0.0,
            solver=SparseLMOptimizerConfig(max_iter=1, lm_lambda=0.5, use_cuda_graph=False),
        )
        points = np.stack((_POINTS, _POINTS + 0.002, _POINTS + 0.004), axis=1).astype(np.float32)
        results = []
        for device in ("cpu", "cuda:0"):
            robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
            helper = HandRetargetingOffline(robot, _SPEC, config, device=device)
            helper.warmup(points.shape[0], points.shape[1])
            points_wp = wp.from_numpy(points, dtype=wp.float32, device=device)
            results.append(helper.solve(points_wp).numpy())

        np.testing.assert_allclose(results[1], results[0], rtol=1e-5, atol=1e-4)
