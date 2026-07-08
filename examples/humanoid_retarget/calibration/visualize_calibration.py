#!/usr/bin/env python3
"""Visualize calibration results for any robot: SMPL-X targets vs robot FK.

Usage (from repo root):
    # Compare original vs calibrated
    uv run python examples/humanoid_retarget/visualize_calibration_generic.py \
        --config examples/humanoid_retarget/ik_config/smplx_h1_2.yaml \
        --calibrated examples/humanoid_retarget/ik_config/smplx_h1_2_cali.yaml

    # Save to file
    uv run python examples/humanoid_retarget/visualize_calibration_generic.py \
        --config examples/humanoid_retarget/ik_config/smplx_h1_2.yaml \
        --calibrated examples/humanoid_retarget/ik_config/smplx_h1_2_cali.yaml \
        --save h1_2_calibration.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from calibrate_g1 import (
    BODY_MODEL_DIR,
    G1_POSE_OVERRIDES,
    _SMPLX_POSE_LOADERS,
    apply_scale_and_offset_np,
    find_mujoco_xml,
    get_link_mapping,
    load_config,
    prepare_pose,
)


def _auto_skeleton_bones(
    link_mapping: List[Tuple[str, str, np.ndarray, np.ndarray, float, float]],
) -> List[Tuple[str, str]]:
    """Auto-generate skeleton bone connectivity from link_mapping.

    Connects: root→hips, hips→knees, knees→feet, root→torso,
    torso→shoulders, shoulders→elbows, elbows→wrists.
    """
    human_to_robot = {}
    for robot_link, human_joint, _, _, _, _ in link_mapping:
        human_to_robot[human_joint] = robot_link

    # Define connectivity via human joint names
    human_bones = [
        ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot"),
        ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_foot"),
        ("pelvis", "spine3"),
        ("spine3", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("spine3", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("spine3", "head"),
    ]

    bones = []
    for h1, h2 in human_bones:
        r1, r2 = human_to_robot.get(h1), human_to_robot.get(h2)
        if r1 and r2:
            bones.append((r1, r2))
    return bones


def _draw_skeleton(
    ax, positions: Dict[str, np.ndarray], bones: List[Tuple[str, str]],
    color: str, alpha: float = 0.5,
):
    for b1, b2 in bones:
        if b1 in positions and b2 in positions:
            p1, p2 = positions[b1], positions[b2]
            ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]],
                    color=color, alpha=alpha, linewidth=1.5)


def visualize(
    configs: List[Tuple[str, dict, List]],  # [(label, config, link_mapping)]
    xml_path: str,
    body_model_path: str,
    skeleton_bones: List[Tuple[str, str]],
    save_path: str = None,
):
    pose_names = list(_SMPLX_POSE_LOADERS.keys())

    n_poses = len(pose_names)
    n_configs = len(configs)
    fig = plt.figure(figsize=(5 * n_poses, 5 * n_configs))

    for ci, (label, config, link_mapping) in enumerate(configs):
        root_name = config.get("human_root_name", "pelvis")
        scale_table = config.get("scale_table", {})

        for pi, pose_name in enumerate(pose_names):
            loader = _SMPLX_POSE_LOADERS[pose_name]
            smplx_pos, smplx_quat, _ = loader(body_model_path)
            joint_overrides = G1_POSE_OVERRIDES.get(pose_name, {})

            pose_data = prepare_pose(
                pose_name, smplx_pos, smplx_quat, xml_path, config, link_mapping,
                g1_joint_overrides=joint_overrides,
            )

            transformed = apply_scale_and_offset_np(
                pose_data["smplx_pos"], pose_data["smplx_quat"],
                root_name, scale_table, link_mapping,
            )

            g1_pos = pose_data["g1_pos"]

            ax = fig.add_subplot(n_configs, n_poses, ci * n_poses + pi + 1, projection="3d")

            # Plot target points (red) and robot FK (blue)
            errors = {}
            for robot_link, _, _, _, _, _ in link_mapping:
                if robot_link in transformed and robot_link in g1_pos:
                    t, g = transformed[robot_link], g1_pos[robot_link]
                    err = np.linalg.norm(t - g)
                    errors[robot_link] = err

                    # Color by error magnitude
                    err_color = plt.cm.RdYlGn_r(min(err / 0.15, 1.0))
                    ax.scatter(t[0], t[2], t[1], c="red", s=30, marker="o", alpha=0.8)
                    ax.scatter(g[0], g[2], g[1], c="blue", s=30, marker="^", alpha=0.8)
                    # Error line
                    ax.plot([t[0], g[0]], [t[2], g[2]], [t[1], g[1]],
                            color=err_color, linewidth=2, alpha=0.7)

            _draw_skeleton(ax, transformed, skeleton_bones, "red", 0.3)
            _draw_skeleton(ax, g1_pos, skeleton_bones, "blue", 0.3)

            mean_err = np.mean(list(errors.values())) if errors else 0
            ax.set_title(f"{pose_name}\n{mean_err:.3f}m", fontsize=9)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y")

            # Equal aspect ratio
            all_pts = np.array(list(transformed.values()) + list(g1_pos.values()))
            if len(all_pts) > 0:
                center = all_pts.mean(axis=0)
                max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[2] - max_range, center[2] + max_range)
                ax.set_zlim(center[1] - max_range, center[1] + max_range)

        # Row label
        fig.text(0.02, 1.0 - (ci + 0.5) / n_configs, label,
                 fontsize=12, fontweight="bold", va="center", rotation=90)

    fig.suptitle("Red=SMPL-X target, Blue=Robot FK, Lines=error", fontsize=12)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Original config YAML")
    parser.add_argument("--calibrated", default=None, help="Calibrated config YAML")
    parser.add_argument("--xml", default=None, help="MuJoCo XML (auto if not set)")
    parser.add_argument("--body-model-path", default=str(BODY_MODEL_DIR))
    parser.add_argument("--save", default=None, help="Save plot to file")
    args = parser.parse_args()

    xml_path = args.xml or find_mujoco_xml(args.config)

    configs = []

    orig_config = load_config(args.config)
    orig_mapping = get_link_mapping(orig_config)
    configs.append((f"Original ({Path(args.config).stem})", orig_config, orig_mapping))

    if args.calibrated:
        cal_config = load_config(args.calibrated)
        cal_mapping = get_link_mapping(cal_config)
        configs.append((f"Calibrated ({Path(args.calibrated).stem})", cal_config, cal_mapping))

    skeleton_bones = _auto_skeleton_bones(orig_mapping)

    print(f"MuJoCo XML: {xml_path}")
    print(f"Skeleton bones: {len(skeleton_bones)}")
    print(f"Configs: {len(configs)}")

    visualize(configs, xml_path, args.body_model_path, skeleton_bones, args.save)


if __name__ == "__main__":
    main()
