#!/usr/bin/env python3
"""Retarget a single SMPL-X motion file to robot joint positions.

Usage:
    uv run python examples/humanoid_retarget/smplx_to_robot.py \
        --smplx_file motion_data/raw/ACCAD/walk.npz

    uv run python examples/humanoid_retarget/smplx_to_robot.py \
        --smplx_file motion_data/raw/ACCAD/walk.npz \
        --config ik_config/smplx_h1_2.yaml --visualize

    uv run python examples/humanoid_retarget/smplx_to_robot.py \
        --smplx_file motion_data/raw/ACCAD/walk.npz \
        --save_path output/walk.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import warp as wp
from tqdm import tqdm

from robokit.robo import Robot
from robokit.helpers.humanoid_retarget.humanoid_retargeting_online import (
    HumanoidRetargetingHelper,
    RetargetConfig,
)
from robokit.helpers.humanoid_retarget.utils.smplx_loader import (
    load_smplx_file,
    get_smplx_frames_offline_fast,
)


def retarget_motion(
    helper: HumanoidRetargetingHelper,
    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    helper.reset()

    qpos_list = []
    scaled_targets_list = []

    iterator = tqdm(frames, desc="Retargeting") if show_progress else frames

    for frame in iterator:
        qpos, scaled_targets = helper.retarget(frame, offset_to_ground=True)
        qpos_list.append(qpos)
        scaled_targets_list.append(scaled_targets)

    return np.array(qpos_list), scaled_targets_list


def save_motion(
    save_path: str,
    qpos_array: np.ndarray,
    fps: float,
    human_height: float,
    robot_name: str,
) -> None:
    root_pos = qpos_array[:, :3]
    root_rot_wxyz = qpos_array[:, 3:7]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    dof_pos = qpos_array[:, 7:]

    min_z = root_pos[:, 2].min()
    root_pos[:, 2] -= min_z

    root_pos[:, :2] -= root_pos[0, :2]

    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,  # xyzw format
        "dof_pos": dof_pos,
        "human_height": human_height,
        "robot": robot_name,
        "local_body_pos": None,
        "link_body_list": None,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Retarget SMPL-X motion to robot")
    parser.add_argument(
        "--smplx_file",
        type=str,
        required=True,
        help="Path to SMPL-X motion file (.npz)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to robot URDF (default: use config-specified robot)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ik_config/smplx_g1.yaml",
        help="Path to retarget config YAML, default: ik_config/smplx_g1.yaml",
    )
    parser.add_argument(
        "--smplx_body_model",
        type=str,
        default=None,
        help="Path to SMPL-X body model folder (default: assets/body_models)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save retargeted motion (.pkl)",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="Target FPS for output (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for retargeting (default: cuda:0)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the retargeted motion",
    )
    parser.add_argument(
        "--smoothness_weight",
        type=float,
        default=0.5,
        help="Smoothness weight for temporal consistency (default: 0.5, set 0 to disable)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = script_dir / args.config

    smplx_body_model_path = args.smplx_body_model
    if smplx_body_model_path is None:
        smplx_body_model_path = project_root / "assets" / "body_models"

    wp.init()

    print("=" * 60)
    print("SMPL-X to Robot Retargeting (GPU-accelerated)")
    print("=" * 60)

    print(f"\n[1/4] Loading SMPL-X data from {args.smplx_file}...")
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.smplx_file, str(smplx_body_model_path)
    )

    frames, fps, _ = get_smplx_frames_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.target_fps
    )
    print(f"   Loaded {len(frames)} frames @ {fps} fps")
    print(f"   Human height: {human_height:.2f}m")

    print(f"\n[2/4] Loading config from {config_path}...")
    config = RetargetConfig.from_yaml(config_path)
    config = config.with_actual_height(human_height)
    config.smoothness_weight = args.smoothness_weight
    if args.smoothness_weight > 0:
        print(f"   Smoothness weight: {args.smoothness_weight}")

    urdf_path = args.urdf
    if urdf_path is None:
        urdf_path = project_root / "assets" / "robot_description" / "unitree_g1" / "g1_custom_collision_29dof.urdf"

    print(f"\n[3/4] Loading robot from {urdf_path}...")
    robot = Robot.load(str(urdf_path), backend="warp")

    print(f"\n[4/4] Initializing retargeting helper on {args.device}...")
    helper = HumanoidRetargetingHelper(robot, config, device=args.device)

    print("\nRetargeting motion...")
    start_time = time.perf_counter()
    qpos_array, scaled_targets = retarget_motion(helper, frames)
    wp.synchronize()
    elapsed = time.perf_counter() - start_time

    print(f"\nRetargeting complete!")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Average: {elapsed / len(frames) * 1000:.2f}ms per frame")
    print(f"   Throughput: {len(frames) / elapsed:.1f} fps")

    if args.save_path:
        robot_name = Path(urdf_path).stem
        save_motion(args.save_path, qpos_array, fps, human_height, robot_name)

    if args.visualize:
        from robokit.helpers.humanoid_retarget.motion_viewer import MotionViewer

        print("\nLaunching viewer (press 'q' to quit)...")
        viewer = MotionViewer(
            urdf_path=str(urdf_path),
        )
        viewer.play_motion(qpos_array, scaled_targets, loop=True)
        viewer.close()


if __name__ == "__main__":
    main()
