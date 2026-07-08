#!/usr/bin/env python3
"""Fake-streaming BVH retargeting entry-point.

Loads a BVH file, converts to SMPL-X joint frames, then streams through
``HumanoidRetargetingHelper.retarget()`` frame by frame at real-time pace.

Usage:
    uv run python examples/humanoid_retarget/bvh_retarget_stream.py \
        --bvh motion_data/bvh/xxx.bvh --visualize
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import warp as wp

from robokit.helpers.humanoid_retarget.humanoid_retargeting_online import (
    HumanoidRetargetingHelper,
    RetargetConfig,
)
from robokit.helpers.humanoid_retarget.utils.bvh_loader import (
    compute_root_velocity,
    get_bvh_frames,
    parse_bvh,
)
from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def _get_robot_link_positions(
    robot: Robot,
    robot_pose: np.ndarray,
    link_indices: Dict[str, int],
    device: str,
) -> Dict[str, np.ndarray]:
    """Run robot FK and return world-frame positions for the given links."""
    root_xyz_quat = robot_pose[:7].reshape(1, 7).astype(np.float32)
    q = robot_pose[7:].reshape(1, -1).astype(np.float32)

    wp_device = wp.get_device(device)
    base_se3 = WarpSE3(wp.from_numpy(root_xyz_quat, dtype=wp_vec7, device=wp_device))
    q_wp = wp.from_numpy(q, dtype=wp.float32, device=wp_device)
    state = robot.state(q=q_wp, T_world_base=base_se3)
    state = robot.forward_kinematics(state)

    positions: Dict[str, np.ndarray] = {}
    for label, idx in link_indices.items():
        pose = state.get_T_world_link(idx)
        positions[label] = pose.xyz.numpy()[0].astype(np.float32)
    return positions


def run_benchmark(
    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    helper: HumanoidRetargetingHelper,
    warmup_frames: int = 10,
) -> None:
    """Run all frames without sleep and print timing stats."""
    # Warmup
    for i in range(min(warmup_frames, len(frames))):
        helper.retarget(frames[i], offset_to_ground=True)
    helper.reset()
    wp.synchronize()

    # Timed run
    times: List[float] = []
    for human_frame in frames:
        t0 = time.perf_counter()
        helper.retarget(human_frame, offset_to_ground=True)
        wp.synchronize()
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000.0
    print(f"\n--- Benchmark ({len(frames)} frames) ---")
    print(f"  Mean:   {times_ms.mean():.2f} ms/frame")
    print(f"  Median: {np.median(times_ms):.2f} ms/frame")
    print(f"  Std:    {times_ms.std():.2f} ms")
    print(f"  Min:    {times_ms.min():.2f} ms  Max: {times_ms.max():.2f} ms")
    print(f"  P95:    {np.percentile(times_ms, 95):.2f} ms  P99: {np.percentile(times_ms, 99):.2f} ms")
    print(f"  Throughput: {1000.0 / times_ms.mean():.1f} fps")


def run_fake_stream(
    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    helper: HumanoidRetargetingHelper,
    fps: float,
    viewer: Optional[object] = None,
    robot_link_indices: Optional[Dict[str, int]] = None,
    device: str = "cuda:0",
    loop: bool = False,
) -> None:
    """Stream BVH frames through the retargeting helper at real-time pace."""
    dt = 1.0 / fps
    prev_pos: Optional[np.ndarray] = None
    prev_quat: Optional[np.ndarray] = None
    lin_vel = np.zeros(3, dtype=np.float32)

    frame_idx = 0
    while True:
        human_frame = frames[frame_idx]
        t0 = time.perf_counter()

        robot_pose, all_targets = helper.retarget(human_frame, offset_to_ground=False)
        root_pos = robot_pose[:3]
        root_quat = robot_pose[3:7]

        if prev_pos is not None and prev_quat is not None:
            lin_vel, _ang_vel = compute_root_velocity(prev_pos, root_pos, prev_quat, root_quat, dt)

        if viewer is not None:
            viewer._update_robot(robot_pose)
            # Human reference keypoints (red)
            human_kp = {name: pos for name, (pos, _) in all_targets.items()}
            viewer._update_keypoints(human_kp)
            # Robot actual keypoints (blue)
            if robot_link_indices:
                robot_kp = _get_robot_link_positions(helper.robot, robot_pose, robot_link_indices, device)
                viewer._update_robot_keypoints(robot_kp)

        prev_pos = root_pos.copy()
        prev_quat = root_quat.copy()

        elapsed = time.perf_counter() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        frame_idx += 1
        if frame_idx >= len(frames):
            if loop:
                frame_idx = 0
                helper.reset()
                prev_pos = None
                prev_quat = None
                print("--- loop ---")
            else:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="BVH fake-streaming retarget")
    parser.add_argument("--bvh", type=str, required=True, help="Path to BVH file")
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot shorthand (e.g. g1, h1_2, bhl); resolves to ik_config/smplx_{robot}.yaml",
    )
    parser.add_argument("--urdf", type=str, default=None, help="Path to robot URDF (overrides config urdf_path)")
    parser.add_argument("--config", type=str, default=None, help="Path to retarget config YAML (overrides --robot)")
    parser.add_argument("--fps", type=float, default=0, help="Target FPS (0 = native BVH rate)")
    parser.add_argument("--visualize", action="store_true", help="Launch Viser viewer")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--benchmark", action="store_true", help="Run all frames without sleep and print timing stats")
    parser.add_argument("--fast", action="store_true", help="Use reduced solver stages for speed")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    if args.config is not None:
        config_path = args.config
    elif args.robot is not None:
        config_path = str(script_dir / "ik_config" / f"smplx_{args.robot}.yaml")
    else:
        config_path = str(script_dir / "ik_config" / "smplx_g1.yaml")

    # 1. Parse BVH
    print(f"Parsing BVH: {args.bvh}")
    bvh_data = parse_bvh(args.bvh)
    print(f"  Joints: {len(bvh_data.joint_names)}, Frames: {bvh_data.num_frames}, FPS: {1.0 / bvh_data.frame_time:.1f}")

    # 2. Convert to SMPL-X frames
    tgt_fps = args.fps if args.fps > 0 else None
    frames, fps = get_bvh_frames(bvh_data, tgt_fps=tgt_fps)
    print(f"  Output frames: {len(frames)} @ {fps:.1f} fps")

    # 3. Estimate height
    human_height = 1.65
    print(f"  Estimated human height: {human_height:.3f}m")

    # 4. Load config with height scaling
    config = RetargetConfig.from_yaml(config_path)

    if args.urdf is not None:
        urdf_path = args.urdf
    elif config.urdf_path is not None:
        urdf_path = config.urdf_path
    else:
        urdf_path = str(project_root / "assets" / "robot_description" / "unitree_g1" / "g1_custom_collision_29dof.urdf")

    config = config.with_actual_height(human_height)

    if args.fast:
        from robokit.opt.warp_solver import WarpStageConfig

        config.stages = [
            WarpStageConfig(num_seeds=4, iters=3, lm_lambda=1.0),
            WarpStageConfig(num_seeds=1, iters=3, lm_lambda=1.0),
        ]
        print("  Using fast solver stages: (4x3, 1x3)")

    # 5. Build robot + helper
    print(f"Loading robot: {urdf_path}")
    robot = Robot.load(urdf_path, backend="warp")
    helper = HumanoidRetargetingHelper(robot, config, device=args.device)

    # 6. Optional visualization
    viewer = None
    robot_link_indices: Optional[Dict[str, int]] = None
    if args.visualize:
        from robokit.helpers.humanoid_retarget.motion_viewer import MotionViewer

        viewer = MotionViewer(urdf_path=urdf_path, port=args.port)

        # Human reference keypoints (red spheres) — keyed by human joint name
        human_joint_names = list({m.human_joint for m in config.link_mapping.values()})
        viewer._setup_keypoints(human_joint_names, color=(255, 50, 50))

        # Robot actual keypoints (blue spheres) — keyed by robot link name
        robot_link_indices = {}
        link_name_to_idx = {name: i for i, name in enumerate(robot.link_names)}
        for robot_link in config.link_mapping:
            if robot_link in link_name_to_idx:
                robot_link_indices[robot_link] = link_name_to_idx[robot_link]
        viewer._setup_robot_keypoints(list(robot_link_indices.keys()), color=(50, 120, 255))

    # 7. Run
    if args.benchmark:
        print(f"\nBenchmarking ({len(frames)} frames) ...")
        run_benchmark(frames, helper)
    else:
        print(f"\nStarting fake stream ({len(frames)} frames @ {fps:.1f} fps) ...")
        run_fake_stream(
            frames,
            helper,
            fps,
            viewer=viewer,
            robot_link_indices=robot_link_indices,
            device=args.device,
            loop=args.loop,
        )
    print("Done.")


if __name__ == "__main__":
    main()
