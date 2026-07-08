#!/usr/bin/env python3
"""View retargeted robot motions in a Viser browser viewer.

Plays back a single .pkl motion file or browses a dataset folder.
Optionally overlays the original SMPL-X mesh and tracking keypoints.

Usage:
    uv run python examples/humanoid_retarget/view_motion.py \
        --motion motion_data/retargeted/ACCAD/walk.pkl

    uv run python examples/humanoid_retarget/view_motion.py \
        --robot h1_2 --dataset motion_data/retargeted_h1_2/ACCAD

    uv run python examples/humanoid_retarget/view_motion.py \
        --motion motion_data/retargeted/ACCAD/walk.pkl \
        --smplx_file motion_data/raw/ACCAD/walk.npz \
        --smplx_body_model assets/body_models --show_keypoints
"""
import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from robokit.helpers.humanoid_retarget.motion_viewer import (
    RENDER_PRESETS,
    MotionViewer,
    load_smplx_mesh_data,
)


def compute_human_tracking_targets(
    smplx_file: str,
    smplx_body_model_path: str,
    ik_config_path: str,
    target_fps: float,
    human_joint_names: List[str],
    robot_root_pos: np.ndarray,
) -> List[Dict[str, np.ndarray]]:
    """Compute scaled human joint positions that were used as IK tracking targets.

    Replicates the retargeting pipeline: load SMPLX -> scale -> ground offset -> center XY.

    Args:
        smplx_file: Path to SMPLX .npz file.
        smplx_body_model_path: Path to SMPLX body model directory.
        ik_config_path: Path to IK config JSON.
        target_fps: Target FPS for downsampling.
        human_joint_names: Which human joints to extract.
        robot_root_pos: Robot root_pos from the .pkl (T, 3), used to align XY origin.

    Returns:
        Per-frame dicts of {human_joint_name: xyz position}.
    """
    from robokit.helpers.humanoid_retarget.humanoid_retargeting_online import RetargetConfig
    from robokit.helpers.humanoid_retarget.utils.smplx_loader import (
        get_smplx_frames_offline_fast,
        load_smplx_file,
    )

    smplx_data, body_model, smplx_output, human_height = load_smplx_file(smplx_file, smplx_body_model_path)
    frames, _, _ = get_smplx_frames_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=int(target_fps))

    config = RetargetConfig.from_yaml(ik_config_path)
    config = config.with_actual_height(human_height)
    scale_table = config.scale_table
    root_name = config.human_root_name

    target_set = set(human_joint_names)
    num_frames = min(len(frames), len(robot_root_pos))

    keypoint_positions: List[Dict[str, np.ndarray]] = []
    for i in range(num_frames):
        frame = frames[i]
        root_pos_orig = frame[root_name][0].copy()
        root_scale = scale_table.get(root_name, 1.0)

        scaled: Dict[str, np.ndarray] = {}
        for joint_name, (pos, _quat) in frame.items():
            scale = scale_table.get(joint_name, 1.0)
            if joint_name == root_name:
                scaled[joint_name] = pos.copy() * scale
            else:
                relative = pos - root_pos_orig
                scaled[joint_name] = root_pos_orig * root_scale + relative * scale

        foot_joints = [j for j in scaled if "foot" in j.lower()]
        if foot_joints:
            min_z = min(float(scaled[j][2]) for j in foot_joints)
            for j, pos in scaled.items():
                pos[2] -= min_z

        if i == 0:
            origin_xy = scaled[root_name][:2].copy() - robot_root_pos[0, :2]

        for j, pos in scaled.items():
            pos[:2] -= origin_xy

        frame_kps: Dict[str, np.ndarray] = {}
        for j in target_set:
            if j in scaled:
                frame_kps[j] = scaled[j].astype(np.float32)
        keypoint_positions.append(frame_kps)

    return keypoint_positions


def main():
    parser = argparse.ArgumentParser(description="View retargeted robot motions (Viser)")
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
        "--motion",
        type=str,
        default=None,
        help="Path to single motion .pkl file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset folder for browsing",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS (default: from motion file or 30)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="paper",
        choices=list(RENDER_PRESETS.keys()),
        help="Rendering preset (default: paper). Options: paper, clean_white, dark, sunset, simple",
    )
    parser.add_argument(
        "--robot_color",
        type=int,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Override robot mesh color (0-255 each), e.g. --robot_color 80 120 200",
    )
    parser.add_argument(
        "--smplx_file",
        type=str,
        default=None,
        help="Path to SMPL-X .npz motion file for human mesh overlay",
    )
    parser.add_argument(
        "--smplx_body_model",
        type=str,
        default=None,
        help="Path to SMPL-X body model directory (e.g. assets/body_models/smplx)",
    )
    parser.add_argument(
        "--robot_height",
        type=float,
        default=1.2,
        help="Robot standing height in meters for scaling SMPL-X mesh (default: 1.2 for G1)",
    )
    parser.add_argument(
        "--human_offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Offset for human mesh position (default: 0 0 0)",
    )
    parser.add_argument(
        "--ground_offset",
        type=float,
        default=-0.67,
        help="Ground plane Z position (default: -0.67)",
    )
    parser.add_argument(
        "--show_keypoints",
        action="store_true",
        help="Visualize human tracking target keypoints as red spheres (requires --smplx_file and --smplx_body_model)",
    )
    parser.add_argument(
        "--ik_config",
        type=str,
        default=None,
        help="Path to IK config YAML (default: auto-detect smplx_g1.yaml)",
    )
    args = parser.parse_args()

    if args.motion is None and args.dataset is None:
        parser.error("Either --motion or --dataset must be specified")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    if args.ik_config is not None:
        ik_config_path = args.ik_config
    elif args.robot is not None:
        ik_config_path = str(script_dir / "ik_config" / f"smplx_{args.robot}.yaml")
    else:
        ik_config_path = str(script_dir / "ik_config" / "smplx_g1.yaml")

    from robokit.helpers.humanoid_retarget.humanoid_retargeting_online import RetargetConfig

    ik_retarget_config = RetargetConfig.from_yaml(ik_config_path)

    if args.urdf is not None:
        urdf_path = args.urdf
    elif ik_retarget_config.urdf_path is not None:
        urdf_path = ik_retarget_config.urdf_path
    else:
        urdf_path = str(project_root / "assets" / "robot_description" / "unitree_g1" / "g1_custom_collision_29dof.urdf")

    robot_color = tuple(args.robot_color) if args.robot_color else None

    print("=" * 60)
    print("Motion Viewer (Viser)")
    print("=" * 60)
    print(f"Robot: {urdf_path}")
    print(f"Render preset: {args.preset}")

    viewer = MotionViewer(
        urdf_path=urdf_path,
        port=args.port,
        render_preset=args.preset,
        robot_color=robot_color,
        ground_z=args.ground_offset,
    )

    if args.motion is not None:
        print(f"Motion: {args.motion}")

        with open(args.motion, "rb") as f:
            motion_data = pickle.load(f)

        root_pos = motion_data["root_pos"]
        root_rot_xyzw = motion_data["root_rot"]
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
        dof_pos = motion_data["dof_pos"]

        qpos_array = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)
        fps = args.fps or motion_data.get("fps", 30.0)

        print(f"Frames: {len(qpos_array)}")
        print(f"Duration: {len(qpos_array) / fps:.1f}s")

        smplx_mesh = None
        if args.smplx_file is not None and args.smplx_body_model is not None:
            print(f"Loading SMPL-X mesh from: {args.smplx_file}")
            smplx_mesh = load_smplx_mesh_data(
                args.smplx_file, args.smplx_body_model, target_fps=fps, robot_height=args.robot_height
            )
            print(f"SMPL-X mesh: {smplx_mesh.vertices.shape[0]} frames, {smplx_mesh.vertices.shape[1]} vertices")

        keypoint_positions = None
        if args.show_keypoints:
            if args.smplx_file is None or args.smplx_body_model is None:
                parser.error("--show_keypoints requires --smplx_file and --smplx_body_model")

            import yaml

            with open(ik_config_path) as f:
                ik_config = yaml.safe_load(f)
            human_joints: List[str] = []
            for _robot_link, entry in ik_config.get("link_mapping", {}).items():
                human_joints.append(entry["human_joint"])
            human_joints = list(set(human_joints))

            print(f"Computing human tracking targets for {len(human_joints)} joints...")
            keypoint_positions = compute_human_tracking_targets(
                smplx_file=args.smplx_file,
                smplx_body_model_path=args.smplx_body_model,
                ik_config_path=ik_config_path,
                target_fps=fps,
                human_joint_names=human_joints,
                robot_root_pos=root_pos,
            )
            viewer._setup_keypoints(human_joints)
            print(f"Tracking targets: {human_joints}")

        viewer.play_motion(
            qpos_array,
            fps=fps,
            loop=True,
            smplx_data=smplx_mesh,
            human_offset=np.array(args.human_offset),
            keypoint_positions=keypoint_positions,
        )

    elif args.dataset is not None:
        print(f"Dataset: {args.dataset}")
        viewer.browse_dataset(args.dataset)

    viewer.close()


if __name__ == "__main__":
    main()
