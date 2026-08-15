# ruff: noqa: E402
"""DexYCB and real-preset integration tests for unified offline retargeting."""

import os
from dataclasses import replace

import pytest
from huggingface_hub import get_token


os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

if get_token() is None:
    pytest.skip("hand retargeting tests need HuggingFace auth (set HF_TOKEN)", allow_module_level=True)
if os.environ.get("GITHUB_ACTIONS"):
    pytest.skip("skip in CI: pulls ~48MB DexYCB benchmark (429 rate-limited)", allow_module_level=True)

from typing import Tuple

import numpy as np
import warp as wp

from robokit.assets.robots.hands import xhand as xhand_asset
from robokit.helpers.hand_retargeting import HandRetargetingOffline
from robokit.helpers.hand_retargeting.dexycb_utils import DexYCBVideoDataset, load_dexycb_keypoints
from robokit.helpers.hand_retargeting.presets.xhand import offline, spec
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizerConfig
from robokit.robo import Robot
from robokit.xform.numpy import matrix_to_quaternion, quaternion_multiply


class TestHandRetargetingOffline:
    def _targets_from_q(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        robot = Robot.load(str(xhand_asset.URDF_PATH))
        state = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        T_world_link = state.T_world_link.numpy()
        link_index = {name: index for index, name in enumerate(robot.spec.link_names)}
        points = np.zeros((*q.shape[:2], len(spec.target_names), 3), dtype=np.float32)
        for target_name, link_name in spec.target_link_names.items():
            points[:, :, spec.target_names.index(target_name)] = T_world_link[:, :, link_index[link_name], :3]
        root_link = link_index[spec.target_link_names[spec.root_target_name]]
        return points, T_world_link[:, :, root_link, 3:].copy()

    def test_dexycb_bundle_loader_uses_plain_dicts(self) -> None:
        from robokit.assets.benchmarks import dexycb
        from robokit.assets.body_models import mano

        dexycb_dir = dexycb.DIR
        dataset = DexYCBVideoDataset(dexycb_dir)
        sequence = dataset[0]
        assert len(dataset) > 0
        assert sequence["hand_pose"].shape == (72, 1, 51)
        assert sequence["hand_shape"].shape == (10,)
        assert sequence["extrinsics"].shape == (4, 4)
        assert sequence["object_pose_camera_xyzw_xyz"].shape == (72, 7)
        assert sequence["object_mesh_path"].exists()
        assert sequence["capture_name"] == "capture_0000"

        data = load_dexycb_keypoints(dexycb_dir, mano.DIR, capture_index=0, device="cpu")
        assert data["keypoints"].shape == (63, 21, 3)
        assert data["wrist_quat_wxyz"].shape == (63, 4)
        assert data["object_pos"].shape == (63, 3)
        assert data["object_quat_wxyz"].shape == (63, 4)
        assert data["object_mesh_path"].exists()

        raw_object_pose = sequence["object_pose_camera_xyzw_xyz"][data["frame_indices"][0]]
        q_wxyz_camera_object = raw_object_pose[[3, 0, 1, 2]].astype(np.float32)
        q_wxyz_camera_object /= np.linalg.norm(q_wxyz_camera_object)
        q_wxyz_world_camera = matrix_to_quaternion(np.linalg.inv(sequence["extrinsics"])[:3, :3]).astype(np.float32)
        expected_object_quat = quaternion_multiply(q_wxyz_world_camera, q_wxyz_camera_object).astype(np.float32)
        assert np.allclose(data["object_quat_wxyz"][0], expected_object_quat)

    def test_normal_and_contact_shapes(self) -> None:
        robot = Robot.load(str(xhand_asset.URDF_PATH))
        num_frames = 7
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, num_frames, 1))
        points, _ = self._targets_from_q(q)
        config = replace(
            offline,
            root_orientation_weight=0.0,
            solver=SparseLMOptimizerConfig(max_iter=0, lm_lambda=0.5, use_cuda_graph=False),
        )
        retargeter = HandRetargetingOffline(robot, spec, config, device="cpu")
        packed = retargeter.solve_numpy(points)
        contact_indices = [spec.target_names.index(name) for name in spec.contact_target_names]
        contact_points = points[:, :, contact_indices]
        contact_mask = np.ones(contact_points.shape[:-1], dtype=np.bool_)
        contact = retargeter.solve_with_contact_numpy(points, contact_points, contact_mask)

        expected = (1, num_frames, 7 + robot.spec.num_actuated_joints)
        assert packed.shape == contact.shape == expected
        assert np.isfinite(packed).all()
        assert np.isfinite(contact).all()
