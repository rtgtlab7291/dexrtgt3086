#!/usr/bin/env python3
"""SMPL-X to Robot retargeting with live visualization.

Retargets a single SMPL-X motion file to a robot and shows an interactive
Viser viewer with keypoint comparison (human vs. robot).

Usage:
    uv run python examples/humanoid_retarget/humanoid_retarget_online.py

    uv run python examples/humanoid_retarget/humanoid_retarget_online.py \
        --robot g1 --smplx_file motion_data/raw/ACCAD/walk.npz

    uv run python examples/humanoid_retarget/humanoid_retarget_online.py \
        --robot h1_2 --config ik_config/smplx_h1_2.yaml --port 8081
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.helpers.humanoid_retarget.humanoid_retargeting_online import HumanoidRetargetingHelper, RetargetConfig
from robokit.helpers.humanoid_retarget.utils.smplx_loader import get_smplx_frames_offline_fast, load_smplx_file
from robokit.lie.pinocchio_se3 import PinocchioSE3
from robokit.robo import Robot


def get_body_part_color(joint_name: str) -> np.ndarray:
    joint_lower = joint_name.lower()
    if "left" in joint_lower and (
        "arm" in joint_lower
        or "hand" in joint_lower
        or "shoulder" in joint_lower
        or "elbow" in joint_lower
        or "wrist" in joint_lower
    ):
        return np.array([0.2, 0.5, 1.0])
    elif "right" in joint_lower and (
        "arm" in joint_lower
        or "hand" in joint_lower
        or "shoulder" in joint_lower
        or "elbow" in joint_lower
        or "wrist" in joint_lower
    ):
        return np.array([1.0, 0.2, 0.2])
    elif "left" in joint_lower and (
        "leg" in joint_lower
        or "foot" in joint_lower
        or "hip" in joint_lower
        or "knee" in joint_lower
        or "ankle" in joint_lower
    ):
        return np.array([0.2, 1.0, 0.5])
    elif "right" in joint_lower and (
        "leg" in joint_lower
        or "foot" in joint_lower
        or "hip" in joint_lower
        or "knee" in joint_lower
        or "ankle" in joint_lower
    ):
        return np.array([0.5, 1.0, 0.2])
    elif "spine" in joint_lower or "pelvis" in joint_lower or "torso" in joint_lower:
        return np.array([0.8, 0.8, 0.2])
    elif "head" in joint_lower or "neck" in joint_lower:
        return np.array([1.0, 0.5, 0.0])
    else:
        return np.array([0.7, 0.7, 0.7])


def extract_robot_keypoints(robot: Robot, qpos: np.ndarray) -> Dict[str, np.ndarray]:
    base_pos = qpos[:3]
    base_quat_wxyz = qpos[3:7]
    joint_q = qpos[7:]

    state = robot.state(
        q=joint_q,
        T_world_base=PinocchioSE3(np.concatenate([base_pos, base_quat_wxyz])),
    )
    state = robot.forward_kinematics(state)

    link_positions = {}
    for i, link_name in enumerate(robot.link_names):
        T_link = state.get_T_world_link(i)
        link_positions[link_name] = T_link.xyz

    return link_positions


def create_joint_mapping(ik_config_path: str) -> Dict[str, str]:
    import yaml

    with open(ik_config_path, "r") as f:
        ik_config = yaml.safe_load(f)

    mapping = {}
    for robot_link, match_info in ik_config.get("link_mapping", {}).items():
        smplx_joint = match_info["human_joint"]
        if smplx_joint not in mapping:
            mapping[smplx_joint] = robot_link

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="SMPL-X to Robot Retargeting with visualization")
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot shorthand (e.g. g1, h1_2, bhl); resolves to ik_config/smplx_{robot}.yaml",
    )
    parser.add_argument("--urdf", type=str, default=None, help="Path to robot URDF (overrides config urdf_path)")
    parser.add_argument("--config", type=str, default=None, help="Path to retarget config YAML (overrides --robot)")
    parser.add_argument("--smplx_file", type=str, default=None, help="Path to SMPL-X .npz motion file")
    parser.add_argument("--smplx_body_model", type=str, default=None, help="Path to SMPL-X body model directory")
    parser.add_argument("--device", type=str, default=None, help="Compute device (default: auto-detect)")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    wp.init()
    device = args.device if args.device is not None else ("cuda:0" if wp.is_cuda_available() else "cpu")

    repo_root = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).parent

    if args.config is not None:
        ik_config_path = Path(args.config)
    elif args.robot is not None:
        ik_config_path = script_dir / "ik_config" / f"smplx_{args.robot}.yaml"
    else:
        ik_config_path = script_dir / "ik_config" / "smplx_g1.yaml"

    config = RetargetConfig.from_yaml(str(ik_config_path))

    if args.urdf is not None:
        urdf_path = Path(args.urdf)
    elif config.urdf_path is not None:
        urdf_path = Path(config.urdf_path)
    else:
        urdf_path = repo_root / "assets" / "robot_description" / "unitree_g1" / "g1_custom_collision_29dof.urdf"

    smplx_file = (
        Path(args.smplx_file)
        if args.smplx_file is not None
        else (
            repo_root
            / "motion_data"
            / "raw"
            / "ACCAD"
            / "Male2MartialArtsKicks_c3d"
            / "G4_-spinning_back_kick_stageii.npz"
        )
    )
    smplx_models = (
        Path(args.smplx_body_model) if args.smplx_body_model is not None else (repo_root / "assets" / "body_models")
    )
    target_fps = 30

    print("=" * 80)
    print("SMPL-X to Robot Retargeting")
    print("=" * 80)

    print("\n[1/4] Loading SMPL-X data...")
    print(f"   Device: {device}")

    smplx_data, body_model, smplx_output, human_height = load_smplx_file(str(smplx_file), str(smplx_models))
    frames, aligned_fps, _ = get_smplx_frames_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=target_fps)

    num_frames = len(frames)
    print(f"   Loaded {num_frames} frames @ {aligned_fps:.1f} fps")
    print(f"   Human height: {human_height:.2f}m")

    print("\n[2/4] Initializing retargeting helper...")

    robot_warp = Robot.load(str(urdf_path), backend="warp")

    config = config.with_actual_height(human_height)

    helper = HumanoidRetargetingHelper(
        robot=robot_warp,
        config=config,
        device=device,
    )

    print(f"   Link mapping tasks: {len(config.link_mapping)}")

    print("   Warming up...")
    _ = helper.retarget(frames[0], offset_to_ground=True)
    helper.reset()
    print("   Warmup complete.")

    print("\n[3/4] Retargeting all frames...")
    print(f"   Processing {num_frames} frames sequentially...")

    qpos_list: List[np.ndarray] = []
    scaled_human_frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []

    start_time = time.time()
    for i, frame in enumerate(frames):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            fps_so_far = i / elapsed if elapsed > 0 else 0
            print(f"   Frame {i}/{num_frames} ({fps_so_far:.1f} fps)...")

        robot_pose, scaled_targets = helper.retarget(frame, offset_to_ground=True)
        qpos_list.append(robot_pose)
        scaled_human_frames.append(scaled_targets)

    total_time = time.time() - start_time
    qpos_arr = np.stack(qpos_list, axis=0)

    print("\n   Retargeting completed!")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Time per frame: {total_time / num_frames * 1000:.2f}ms")
    print(f"   Effective FPS: {num_frames / total_time:.1f}")
    print(f"   Output shape: {qpos_arr.shape}")

    print("\n[4/4] Starting visualization...")

    robot_np = Robot.load(str(urdf_path), backend="numpy")
    joint_mapping = create_joint_mapping(str(ik_config_path))

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/ground", width=5, height=5)
    base_frame = server.scene.add_frame("/robot_base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf_path, root_node_name="/robot_base")

    with server.gui.add_folder("Playback Control", expand_by_default=True):
        frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
        play_button = server.gui.add_button("Play")
        pause_button = server.gui.add_button("Pause")
        speed_slider = server.gui.add_slider("Speed (FPS)", min=1.0, max=60.0, step=1.0, initial_value=aligned_fps)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

    with server.gui.add_folder("Visualization Options"):
        show_smplx = server.gui.add_checkbox("Show Human Keypoints", initial_value=True)
        show_robot_kpts = server.gui.add_checkbox("Show Robot Keypoints", initial_value=True)
        show_robot = server.gui.add_checkbox("Show Robot", initial_value=True)
        show_connections = server.gui.add_checkbox("Show Connections", initial_value=True)
        keypoint_size = server.gui.add_slider("Keypoint Size", min=0.005, max=0.05, step=0.005, initial_value=0.02)

    with server.gui.add_folder("Performance Info", expand_by_default=True):
        current_frame_display = server.gui.add_number("Current Frame", initial_value=0, disabled=True)
        mean_error_display = server.gui.add_number("Mean Error (m)", initial_value=0.0, disabled=True)
        max_error_display = server.gui.add_number("Max Error (m)", initial_value=0.0, disabled=True)

    with server.gui.add_folder("Export Options"):
        export_button = server.gui.add_button("Export All Frames (NPZ)")

    is_playing = [False]
    current_frame = [0]
    last_update_time = [time.time()]

    smplx_spheres: Dict = {}
    robot_spheres: Dict = {}
    connection_lines: Dict = {}

    def update_visualization(frame_idx: int) -> None:
        qpos = qpos_arr[frame_idx]
        scaled_frame = scaled_human_frames[frame_idx]

        base_pos = qpos[:3]
        base_quat_wxyz = qpos[3:7]
        joint_q = qpos[7:]

        if show_robot.value:
            base_frame.position = tuple(base_pos)
            base_frame.wxyz = tuple(base_quat_wxyz)
            urdf_vis.update_cfg(joint_q)

        robot_kpts = extract_robot_keypoints(robot_np, qpos)
        smplx_kpts = {name: pos for name, (pos, _) in scaled_frame.items()}

        errors = {}
        for smplx_joint, robot_link in joint_mapping.items():
            if smplx_joint in smplx_kpts and robot_link in robot_kpts:
                pos_error = np.linalg.norm(smplx_kpts[smplx_joint] - robot_kpts[robot_link])
                errors[f"{smplx_joint}->{robot_link}"] = float(pos_error)

        if errors:
            error_values = list(errors.values())
            mean_error_display.value = float(np.mean(error_values))
            max_error_display.value = float(np.max(error_values))
            max_error = max(error_values)
        else:
            max_error = 1.0

        for smplx_joint in joint_mapping.keys():
            if smplx_joint in smplx_kpts:
                pos = smplx_kpts[smplx_joint]
                sphere_name = f"/smplx/{smplx_joint}"
                if show_smplx.value:
                    color = get_body_part_color(smplx_joint)
                    if sphere_name not in smplx_spheres:
                        smplx_spheres[sphere_name] = server.scene.add_icosphere(
                            sphere_name, radius=keypoint_size.value, color=tuple(color), position=tuple(pos)
                        )
                    else:
                        smplx_spheres[sphere_name].position = tuple(pos)
                        smplx_spheres[sphere_name].radius = keypoint_size.value
                else:
                    if sphere_name in smplx_spheres:
                        smplx_spheres[sphere_name].remove()
                        del smplx_spheres[sphere_name]

        active_robot_links = set(joint_mapping.values())
        for link_name, pos in robot_kpts.items():
            if link_name in active_robot_links:
                sphere_name = f"/robot/{link_name}"
                if show_robot_kpts.value:
                    color = np.array([1.0, 0.5, 0.0])
                    if sphere_name not in robot_spheres:
                        robot_spheres[sphere_name] = server.scene.add_icosphere(
                            sphere_name, radius=keypoint_size.value * 0.8, color=tuple(color), position=tuple(pos)
                        )
                    else:
                        robot_spheres[sphere_name].position = tuple(pos)
                        robot_spheres[sphere_name].radius = keypoint_size.value * 0.8
                else:
                    if sphere_name in robot_spheres:
                        robot_spheres[sphere_name].remove()
                        del robot_spheres[sphere_name]

        for smplx_joint, robot_link in joint_mapping.items():
            if smplx_joint in smplx_kpts and robot_link in robot_kpts:
                line_name = f"/connection/{smplx_joint}_{robot_link}"

                if show_connections.value:
                    error_key = f"{smplx_joint}->{robot_link}"
                    error = errors.get(error_key, 0.0)
                    normalized = min(error / max_error, 1.0) if max_error > 0 else 0.0
                    if normalized < 0.5:
                        color = np.array([normalized * 2, 1.0, 0.0])
                    else:
                        color = np.array([1.0, 2.0 * (1.0 - normalized), 0.0])

                    start = smplx_kpts[smplx_joint]
                    end = robot_kpts[robot_link]
                    points = np.array([start, end])

                    if line_name not in connection_lines:
                        connection_lines[line_name] = server.scene.add_line_segments(
                            line_name, points=points.reshape(1, 2, 3), colors=color, line_width=2.0
                        )
                    else:
                        connection_lines[line_name].points = points.reshape(1, 2, 3)
                        connection_lines[line_name].colors = color
                else:
                    if line_name in connection_lines:
                        connection_lines[line_name].remove()
                        del connection_lines[line_name]

        current_frame_display.value = frame_idx

    @play_button.on_click
    def _(_) -> None:
        is_playing[0] = True

    @pause_button.on_click
    def _(_) -> None:
        is_playing[0] = False

    @export_button.on_click
    def _(_) -> None:
        output_file = f"retarget_{num_frames}frames.npz"
        np.savez_compressed(
            output_file,
            qpos=qpos_arr,
            fps=aligned_fps,
            num_frames=num_frames,
        )
        print(f"\n[Export] Saved all {num_frames} frames to {output_file}")

    print(f"\nViser server started at: http://localhost:{args.port}")
    print("=" * 80)
    print("Retargeting Summary:")
    print(f"  Frames: {num_frames}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Per frame: {total_time / num_frames * 1000:.2f}ms")
    print(f"  Effective FPS: {num_frames / total_time:.1f}")
    print("=" * 80 + "\n")

    update_visualization(0)

    try:
        while True:
            current_time = time.time()

            if is_playing[0]:
                delta_time = current_time - last_update_time[0]
                target_frame_time = 1.0 / speed_slider.value

                if delta_time >= target_frame_time:
                    current_frame[0] = current_frame[0] + 1

                    if current_frame[0] >= num_frames:
                        if loop_checkbox.value:
                            current_frame[0] = 0
                        else:
                            is_playing[0] = False
                            current_frame[0] = num_frames - 1

                    frame_slider.value = current_frame[0]
                    update_visualization(current_frame[0])
                    last_update_time[0] = current_time
            else:
                if frame_slider.value != current_frame[0]:
                    current_frame[0] = frame_slider.value
                    update_visualization(current_frame[0])
                    last_update_time[0] = current_time

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nShutting down...")


if __name__ == "__main__":
    main()
