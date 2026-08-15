"""End-to-end test for the graduated GraspOptHelper (ported from graspkit).

Uses the inspire right hand against a synthetic cube. Gated on the asset already being cached
(no HF fetch in CI) and on CUDA for the solve.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
import warp as wp


_INSPIRE_URDF = Path(
    os.path.expanduser(
        "~/.robokit/cache/main/robots/robot_description/end_effectors/inspire_hand/inspire_hand_right.urdf"
    )
)
_REQUIRES_ASSET = pytest.mark.skipif(not _INSPIRE_URDF.exists(), reason="inspire hand asset not cached")
_REQUIRES_CUDA = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required for grasp solver")


class TestGraspOptHelper:
    @_REQUIRES_ASSET
    def test_load_contact_candidates(self) -> None:
        from robokit.assets.robots.hands import inspire_hand
        from robokit.helpers.combo_retarget.grasp_opt import load_contact_candidates
        from robokit.robo import Robot

        robot = Robot.load(str(inspire_hand.URDF_PATH))
        points, link_indices = load_contact_candidates(str(inspire_hand.CONTACT_POINTS_PATH), robot, "cpu")
        assert points.shape[0] == link_indices.shape[0] > 0
        assert int(link_indices.numpy().max()) < len(robot.spec.link_names)

    @_REQUIRES_ASSET
    @_REQUIRES_CUDA
    def test_contact_resampling_solve_reduces_energy(self, tmp_path: Path) -> None:
        from robokit.assets.robots.hands import inspire_hand
        from robokit.geom import MeshGeom, WarpScene
        from robokit.helpers.combo_retarget.grasp_opt import (
            GraspOptHelper,
            GraspOptHelperConfig,
            load_contact_candidates,
        )
        from robokit.opt.multi_seed_solver import StageConfig
        from robokit.robo import Robot
        from robokit.utils.warp_utils import wp_vec7

        device = "cuda:0"
        batch, n_contact = 2, 4
        robot = Robot.load(
            str(inspire_hand.URDF_PATH),
            load_collision_spheres=True,
            collision_spheres_path=str(inspire_hand.COLLISION_SPHERE_PATH),
        )
        candidates, cand_links = load_contact_candidates(str(inspire_hand.CONTACT_POINTS_PATH), robot, device)

        cube_path = str(tmp_path / "cube.obj")
        trimesh.creation.box((0.06, 0.06, 0.06)).export(cube_path)
        target = WarpScene(1, device).add(MeshGeom([cube_path], np.array([0, 1], dtype=np.int32)))
        scene = WarpScene(1, device).add(MeshGeom([cube_path], np.array([0, 1], dtype=np.int32)))

        helper = GraspOptHelper(
            robot=robot,
            batch_size=batch,
            num_contact_points=n_contact,
            target_meshes=target,
            scene_meshes=scene,
            contact_candidates=candidates,
            contact_candidate_link_indices=cand_links,
            config=GraspOptHelperConfig(
                stages=(
                    StageConfig(num_seeds=4, iters=24, lm_lambda=10.0),
                    StageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
                ),
                contact_resample_interval=8,
            ),
        )

        limits = robot.spec.actuated_joint_limits
        q0 = np.tile(limits.mean(axis=1).astype(np.float32), (batch, 1))
        base0 = np.zeros((batch, 7), dtype=np.float32)
        base0[:, 0] = 0.1
        base0[:, 3] = 1.0
        init_state = robot.state(
            q=wp.from_numpy(q0, dtype=wp.float32, device=device),
            T_world_base=wp.from_numpy(base0, dtype=wp_vec7, device=device),
        )
        rng = np.random.default_rng(0)
        init_idx_wp = wp.from_numpy(
            rng.integers(0, candidates.shape[0], size=(batch, n_contact)).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        np.random.seed(0)
        e_init = helper.compute_energy(init_state, contact_indices=init_idx_wp)
        best, best_idx = helper.solve_with_contact_resampling(init_state, init_idx_wp)
        e_final, per_term = helper.compute_energy(best, contact_indices=best_idx, return_per_term=True)

        assert float(e_final.numpy().mean()) < float(e_init.numpy().mean())
        assert float(per_term["E_dis"].numpy().mean()) < 0.5 * 100.0 * 0.06  # contacts pulled well inside one cube edge
        _, max_pen = helper.compute_penetration(best)
        assert max_pen < 0.02
