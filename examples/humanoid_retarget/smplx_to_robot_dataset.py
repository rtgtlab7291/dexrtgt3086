#!/usr/bin/env python3
"""Batch retarget an entire SMPL-X dataset to robot joint positions.

Walks a source folder of .npz/.pkl SMPL-X files, retargets each to the
target robot, and saves per-motion .pkl files to the output folder.

Usage:
    uv run python examples/humanoid_retarget/smplx_to_robot_dataset.py \
        --src_folder motion_data/raw/ACCAD \
        --tgt_folder motion_data/retargeted/ACCAD

    uv run python examples/humanoid_retarget/smplx_to_robot_dataset.py \
        --robot h1_2 \
        --src_folder motion_data/raw/ACCAD \
        --tgt_folder motion_data/retargeted_h1_2/ACCAD \
        --num_workers 4 --batch_load 20
"""
import argparse
import gc
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import warp as wp
from natsort import natsorted
from tqdm import tqdm

from robokit.helpers.humanoid_retarget.humanoid_retargeting_online import (
    HumanoidRetargetingHelper,
    RetargetConfig,
)
from robokit.helpers.humanoid_retarget.utils.smplx_loader import (
    get_smplx_frames_offline_fast,
    load_smplx_file,
)
from robokit.robo import Robot


def load_smplx_data_worker(
    src_path: str,
    smplx_body_model_path: str,
    target_fps: int,
) -> Tuple[str, List, float, float]:
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(src_path, smplx_body_model_path)
    frames, fps, _ = get_smplx_frames_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=target_fps)
    return (src_path, frames, fps, human_height)


def collect_files(
    src_folder: str,
    tgt_folder: str,
    override: bool = False,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    if exclude_patterns is None:
        exclude_patterns = []

    file_pairs = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue

            if filename.endswith((".pkl", ".npz")):
                src_path = os.path.join(dirpath, filename)
                tgt_path = src_path.replace(src_folder, tgt_folder).replace(".npz", ".pkl")

                if any(pattern in src_path for pattern in exclude_patterns):
                    continue

                if not override and os.path.exists(tgt_path):
                    continue

                file_pairs.append((src_path, tgt_path))

    return file_pairs


def save_motion_data(
    tgt_path: str,
    qpos_array: np.ndarray,
    fps: float,
    human_height: float,
) -> None:
    root_pos = qpos_array[:, :3].copy()
    root_rot_wxyz = qpos_array[:, 3:7]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    dof_pos = qpos_array[:, 7:]

    min_z = root_pos[:, 2].min()
    root_pos[:, 2] -= min_z

    root_pos[:, :2] -= root_pos[0, :2]

    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "human_height": human_height,
        "local_body_pos": None,
        "link_body_list": None,
    }

    os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
    with open(tgt_path, "wb") as f:
        pickle.dump(motion_data, f)


def process_single_motion(
    helper: HumanoidRetargetingHelper,
    frames: List[Dict],
) -> np.ndarray:
    helper.reset()

    qpos_list = []
    for frame in frames:
        qpos, _ = helper.retarget(frame, offset_to_ground=True)
        qpos_list.append(qpos)

    return np.array(qpos_list)


def main():
    parser = argparse.ArgumentParser(description="Batch retarget SMPL-X dataset to robot")
    parser.add_argument(
        "--src_folder",
        type=str,
        required=True,
        help="Source folder containing SMPL-X files",
    )
    parser.add_argument(
        "--tgt_folder",
        type=str,
        required=True,
        help="Target folder for retargeted motions",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot shorthand (e.g. g1, h1_2, bhl); resolves to ik_config/smplx_{robot}.yaml",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to robot URDF (overrides config urdf_path)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to retarget config YAML (overrides --robot)",
    )
    parser.add_argument(
        "--smplx_body_model",
        type=str,
        default=None,
        help="Path to SMPL-X body model folder",
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
        "--override",
        action="store_true",
        help="Override existing files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for SMPL-X loading (default: 1)",
    )
    parser.add_argument(
        "--batch_load",
        type=int,
        default=10,
        help="Number of files to pre-load in parallel (default: 10)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = script_dir / args.config
    elif args.robot is not None:
        config_path = script_dir / "ik_config" / f"smplx_{args.robot}.yaml"
    else:
        config_path = script_dir / "ik_config" / "smplx_g1.yaml"

    smplx_body_model_path = args.smplx_body_model
    if smplx_body_model_path is None:
        smplx_body_model_path = str(project_root / "assets" / "body_models")

    base_config = RetargetConfig.from_yaml(config_path)

    if args.urdf is not None:
        urdf_path = args.urdf
    elif base_config.urdf_path is not None:
        urdf_path = base_config.urdf_path
    else:
        urdf_path = str(project_root / "assets" / "robot_description" / "unitree_g1" / "g1_custom_collision_29dof.urdf")

    print("=" * 60)
    print("SMPL-X Dataset Retargeting (GPU-accelerated)")
    print("=" * 60)

    print(f"\nSource folder: {args.src_folder}")
    print(f"Target folder: {args.tgt_folder}")

    exclude_patterns = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
    print(f"Excluding patterns: {exclude_patterns}")

    file_pairs = collect_files(
        args.src_folder,
        args.tgt_folder,
        override=args.override,
        exclude_patterns=exclude_patterns,
    )

    print(f"\nFiles to process: {len(file_pairs)}")

    if len(file_pairs) == 0:
        print("No files to process. Done!")
        return

    wp.init()

    print(f"\nLoading robot from {urdf_path}...")
    robot = Robot.load(urdf_path, backend="warp")

    print(f"Loading config from {config_path}...")

    print(f"Initializing retargeting helper on {args.device}...")
    helper = HumanoidRetargetingHelper(robot, base_config, device=args.device)

    total_frames = 0
    total_time = 0.0
    success_count = 0

    pbar = tqdm(total=len(file_pairs), desc="Processing")

    if args.num_workers > 1:
        for batch_start in range(0, len(file_pairs), args.batch_load):
            batch_pairs = file_pairs[batch_start : batch_start + args.batch_load]

            loaded_data = []
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(
                        load_smplx_data_worker,
                        src_path,
                        smplx_body_model_path,
                        args.target_fps,
                    ): (src_path, tgt_path)
                    for src_path, tgt_path in batch_pairs
                }

                for future in as_completed(futures):
                    src_path, tgt_path = futures[future]
                    result = future.result()
                    if result is not None:
                        loaded_data.append((tgt_path, result))

            for tgt_path, (src_path, frames, fps, human_height) in loaded_data:
                start_time = time.perf_counter()
                qpos_array = process_single_motion(helper, frames)
                wp.synchronize()
                elapsed = time.perf_counter() - start_time

                save_motion_data(tgt_path, qpos_array, fps, human_height)

                total_frames += len(frames)
                total_time += elapsed
                success_count += 1
                pbar.update(1)

            gc.collect()
    else:
        for src_path, tgt_path in file_pairs:
            _, frames, fps, human_height = load_smplx_data_worker(src_path, smplx_body_model_path, args.target_fps)

            start_time = time.perf_counter()
            qpos_array = process_single_motion(helper, frames)
            wp.synchronize()
            elapsed = time.perf_counter() - start_time

            save_motion_data(tgt_path, qpos_array, fps, human_height)

            total_frames += len(frames)
            total_time += elapsed
            success_count += 1
            pbar.update(1)

            if success_count % 50 == 0:
                gc.collect()

    pbar.close()

    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Processed: {success_count}")
    print(f"Total frames: {total_frames}")
    if total_time > 0:
        print(f"Total time: {total_time:.1f}s")
        print(f"Average: {total_time / total_frames * 1000:.2f}ms per frame")
        print(f"Throughput: {total_frames / total_time:.1f} fps")
    print(f"\nOutput saved to: {args.tgt_folder}")


if __name__ == "__main__":
    main()
