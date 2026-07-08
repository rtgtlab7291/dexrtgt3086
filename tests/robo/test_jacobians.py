import json
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.lie.pinocchio_se3 import PinocchioSE3
from robokit.lie.torch_se3 import TorchSE3
from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def test_pinocchio_jacobians():
    gt_path = Path(__file__).parent / "jacobian_ground_truth.json"

    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    for robot_data in gt_data["robots"]:
        robot_desc_name = robot_data["robot_description_name"]
        print(f"Testing {robot_desc_name}...")

        urdf_path = load_robot_description(robot_desc_name)
        robot = Robot.load(urdf_path, backend="numpy")

        assert robot.num_actuated_joints == robot_data["num_actuated_joints"], (
            f"Mismatch in num_actuated_joints for {robot_desc_name}"
        )

        for config in robot_data["configurations"]:
            config_name = config["name"]
            qpos = np.array(config["qpos"], dtype=np.float32)

            if config["T_world_base"] is None:
                T_world_base = None
            else:
                T_world_base_xyzwxyz = np.array(config["T_world_base"], dtype=np.float32)
                T_world_base = PinocchioSE3(T_world_base_xyzwxyz)

            state = robot.state(q=qpos, T_world_base=T_world_base)
            state = robot.compute_motion_subspace(state)

            for link_jac_data in config["link_jacobians"]:
                link_name = link_jac_data["link_name"]

                if link_name not in robot.link_names:
                    continue

                link_idx = robot.link_names.index(link_name)

                # Test both reference frames
                ref_frames: List[Literal["body", "spatial"]] = ["body", "spatial"]
                for ref_frame in ref_frames:
                    J_actual = state.get_link_jacobian(link_idx, ref_frame)

                    J_joints_expected = np.array(link_jac_data[f"{ref_frame}_J_joints"], dtype=np.float32)

                    if T_world_base is None:
                        assert np.allclose(J_actual, J_joints_expected, atol=1e-5), (
                            f"Jacobian mismatch for {robot_desc_name}, config {config_name}, "
                            f"link {link_name}, frame {ref_frame}:\n"
                            f"Expected shape: {J_joints_expected.shape}\n"
                            f"Actual shape: {J_actual.shape}\n"
                            f"Max difference: {np.max(np.abs(J_actual - J_joints_expected))}"
                        )
                    else:
                        J_base_expected = np.array(link_jac_data[f"{ref_frame}_J_base"], dtype=np.float32)
                        J_expected = np.concatenate([J_base_expected, J_joints_expected], axis=-1)

                        assert np.allclose(J_actual, J_expected, atol=1e-5), (
                            f"Jacobian mismatch for {robot_desc_name}, config {config_name}, "
                            f"link {link_name}, frame {ref_frame}:\n"
                            f"Expected shape: {J_expected.shape}\n"
                            f"Actual shape: {J_actual.shape}\n"
                            f"Max difference: {np.max(np.abs(J_actual - J_expected))}"
                        )

        print(f"  ✓ {robot_desc_name} passed all Jacobian tests")


def test_torch_jacobians():
    gt_path = Path(__file__).parent / "jacobian_ground_truth.json"

    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    for robot_data in gt_data["robots"]:
        robot_desc_name = robot_data["robot_description_name"]
        print(f"Testing {robot_desc_name}...")

        urdf_path = load_robot_description(robot_desc_name)
        robot = Robot.load(urdf_path, backend="torch")

        assert robot.num_actuated_joints == robot_data["num_actuated_joints"], (
            f"Mismatch in num_actuated_joints for {robot_desc_name}"
        )

        for config in robot_data["configurations"]:
            config_name = config["name"]
            qpos = torch.tensor(config["qpos"], dtype=torch.float32)

            if config["T_world_base"] is None:
                T_world_base = None
            else:
                T_world_base_xyzwxyz = torch.tensor(config["T_world_base"], dtype=torch.float32)
                T_world_base = TorchSE3(T_world_base_xyzwxyz)

            state = robot.state(q=qpos, T_world_base=T_world_base)
            state = robot.compute_motion_subspace(state)

            for link_jac_data in config["link_jacobians"]:
                link_name = link_jac_data["link_name"]

                if link_name not in robot.link_names:
                    continue

                link_idx = robot.link_names.index(link_name)

                # Test both reference frames
                ref_frames: List[Literal["body", "spatial"]] = ["body", "spatial"]
                for ref_frame in ref_frames:
                    J_joints_expected = torch.tensor(link_jac_data[f"{ref_frame}_J_joints"], dtype=torch.float32)
                    J_actual = state.get_link_jacobian(link_idx, ref_frame)
                    assert isinstance(J_actual, torch.Tensor)

                    if T_world_base is None:
                        assert torch.allclose(J_actual, J_joints_expected, atol=1e-5), (
                            f"Jacobian mismatch for {robot_desc_name}, config {config_name}, "
                            f"link {link_name}, frame {ref_frame}:\n"
                            f"Expected shape: {J_joints_expected.shape}\n"
                            f"Actual shape: {J_actual.shape}\n"
                            f"Max difference: {torch.max(torch.abs(J_actual - J_joints_expected))}"
                        )
                    else:
                        J_base_expected = torch.tensor(link_jac_data[f"{ref_frame}_J_base"], dtype=torch.float32)
                        J_expected = torch.cat([J_base_expected, J_joints_expected], dim=-1)

                        assert torch.allclose(J_actual, J_expected, atol=1e-5), (
                            f"Jacobian mismatch for {robot_desc_name}, config {config_name}, "
                            f"link {link_name}, frame {ref_frame}:\n"
                            f"Expected shape: {J_expected.shape}\n"
                            f"Actual shape: {J_actual.shape}\n"
                            f"Max difference: {torch.max(torch.abs(J_actual - J_expected))}"
                        )

        print(f"  ✓ {robot_desc_name} passed all Jacobian tests")


def test_warp_jacobians():
    gt_path = Path(__file__).parent / "jacobian_ground_truth.json"

    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    for robot_data in gt_data["robots"]:
        robot_desc_name = robot_data["robot_description_name"]
        print(f"Testing {robot_desc_name}...")

        urdf_path = load_robot_description(robot_desc_name)
        robot = Robot.load(urdf_path, backend="warp")

        assert robot.num_actuated_joints == robot_data["num_actuated_joints"], (
            f"Mismatch in num_actuated_joints for {robot_desc_name}"
        )

        for config in robot_data["configurations"]:
            config_name = config["name"]
            qpos = wp.from_numpy(np.array(config["qpos"], dtype=np.float32), dtype=wp.float32)

            if config["T_world_base"] is None:
                T_world_base = None
            else:
                T_world_base_xyzwxyz = wp.from_numpy(
                    np.array(config["T_world_base"], dtype=np.float32).reshape(1, 7), dtype=wp_vec7
                )
                T_world_base = WarpSE3(T_world_base_xyzwxyz)

            state = robot.state(q=qpos, T_world_base=T_world_base)
            state = robot.compute_motion_subspace(state)

            for link_jac_data in config["link_jacobians"]:
                link_name = link_jac_data["link_name"]

                if link_name not in robot.link_names:
                    continue

                link_idx = robot.link_names.index(link_name)

                ref_frames: List[Literal["body", "spatial"]] = ["body", "spatial"]
                for ref_frame in ref_frames:
                    J_joints_expected = np.array(link_jac_data[f"{ref_frame}_J_joints"], dtype=np.float32)
                    J_actual = state.get_link_jacobian(link_idx, ref_frame)
                    assert isinstance(J_actual, wp.array)
                    J_actual_np = J_actual.numpy().squeeze()

                    if T_world_base is None:
                        assert np.allclose(J_actual_np, J_joints_expected, atol=1e-5), (
                            f"Jacobian mismatch for {robot_desc_name}, config {config_name}, "
                            f"link {link_name}, frame {ref_frame}:\n"
                            f"Expected shape: {J_joints_expected.shape}\n"
                            f"Actual shape: {J_actual_np.shape}\n"
                            f"Max difference: {np.max(np.abs(J_actual_np - J_joints_expected))}"
                        )
                    else:
                        J_base_expected = np.array(link_jac_data[f"{ref_frame}_J_base"], dtype=np.float32)
                        J_expected = np.concatenate([J_base_expected, J_joints_expected], axis=-1)

                        assert np.allclose(J_actual_np, J_expected, atol=1e-5), (
                            f"Jacobian mismatch for {robot_desc_name}, config {config_name}, "
                            f"link {link_name}, frame {ref_frame}:\n"
                            f"Expected shape: {J_expected.shape}\n"
                            f"Actual shape: {J_actual_np.shape}\n"
                            f"Max difference: {np.max(np.abs(J_actual_np - J_expected))}"
                        )

        print(f"  ✓ {robot_desc_name} passed all Jacobian tests")


def test_batch_jacobians():
    gt_path = Path(__file__).parent / "jacobian_ground_truth.json"
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    robot_data = gt_data["robots"][0]
    config = robot_data["configurations"][0]
    urdf_path = load_robot_description(robot_data["robot_description_name"])

    qpos_np = np.array(config["qpos"], dtype=np.float32).reshape(1, 1, -1)
    T_base_np = np.array(config["T_world_base"], dtype=np.float32).reshape(1, 1, 7) if config["T_world_base"] else None
    link_name = config["link_jacobians"][0]["link_name"]

    ref_frames: List[Literal["body", "spatial"]] = ["body", "spatial"]

    print("Testing batch shape for torch...")
    robot = Robot.load(urdf_path, backend="torch")
    qpos = torch.from_numpy(qpos_np)
    T_world_base = TorchSE3(torch.from_numpy(T_base_np)) if T_base_np is not None else None
    state = robot.compute_motion_subspace(robot.state(q=qpos, T_world_base=T_world_base))
    link_idx = robot.link_names.index(link_name)

    for ref_frame in ref_frames:
        J_actual = state.get_link_jacobian(link_idx, ref_frame)
        J_joints_expected_np = np.array(config["link_jacobians"][0][f"{ref_frame}_J_joints"], dtype=np.float32)

        if T_base_np is not None:
            J_base_expected_np = np.array(config["link_jacobians"][0][f"{ref_frame}_J_base"], dtype=np.float32)
            J_expected_np = np.concatenate([J_base_expected_np, J_joints_expected_np], axis=-1)
        else:
            J_expected_np = J_joints_expected_np

        J_expected_np = J_expected_np.reshape(1, 1, *J_expected_np.shape)
        J_expected = torch.from_numpy(J_expected_np)
        assert torch.allclose(J_actual, J_expected, atol=1e-5), f"Jacobian mismatch for frame {ref_frame}"
    print("  ✓ torch passed batch shape Jacobian test")

    print("Testing batch shape for warp...")
    robot = Robot.load(urdf_path, backend="warp")
    qpos = wp.from_numpy(qpos_np, dtype=wp.float32)
    T_world_base = WarpSE3(wp.from_numpy(T_base_np, dtype=wp_vec7)) if T_base_np is not None else None
    state = robot.compute_motion_subspace(robot.state(q=qpos, T_world_base=T_world_base))
    link_idx = robot.link_names.index(link_name)

    for ref_frame in ref_frames:
        J_actual = state.get_link_jacobian(link_idx, ref_frame)
        J_joints_expected_np = np.array(config["link_jacobians"][0][f"{ref_frame}_J_joints"], dtype=np.float32)

        if T_base_np is not None:
            J_base_expected_np = np.array(config["link_jacobians"][0][f"{ref_frame}_J_base"], dtype=np.float32)
            J_expected_np = np.concatenate([J_base_expected_np, J_joints_expected_np], axis=-1)
        else:
            J_expected_np = J_joints_expected_np

        J_expected_np = J_expected_np.reshape(1, 1, *J_expected_np.shape)
        assert np.allclose(J_actual.numpy(), J_expected_np, atol=1e-5), f"Jacobian mismatch for frame {ref_frame}"
    print("  ✓ warp passed batch shape Jacobian test")
