"""Visualize calibration results for any robot: SMPL-X targets vs robot FK.

Usage (from repo root):
    # Compare original vs calibrated (both are YAML files output by calibration)
    uv run python examples/humanoid_retarget/calibration/visualize_calibration.py \
        --config /tmp/smplx_h1_2.yaml \
        --calibrated /tmp/smplx_h1_2_calibrated.yaml

    # Save to file
    uv run python examples/humanoid_retarget/calibration/visualize_calibration.py \
        --config /tmp/smplx_h1_2.yaml \
        --calibrated /tmp/smplx_h1_2_calibrated.yaml \
        --save h1_2_calibration.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401  -- adds the calibration dir to sys.path; keep first
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from calibrate import (
    _body_model_dir,
    _load_smplx_pose,
    apply_scale_and_offset_np,
    get_link_mapping,
    load_config,
    load_pose_pairs,
    prepare_pose,
)

from robokit.robo import Robot


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

    # define connectivity via human joint names
    human_bones = [
        ("pelvis", "left_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_foot"),
        ("pelvis", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_foot"),
        ("pelvis", "spine3"),
        ("spine3", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("spine3", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("spine3", "head"),
    ]

    bones = []
    for h1, h2 in human_bones:
        r1, r2 = human_to_robot.get(h1), human_to_robot.get(h2)
        if r1 and r2:
            bones.append((r1, r2))
    return bones


def _draw_skeleton(
    ax,
    positions: Dict[str, np.ndarray],
    bones: List[Tuple[str, str]],
    color: str,
    alpha: float = 0.5,
):
    for b1, b2 in bones:
        if b1 in positions and b2 in positions:
            p1, p2 = positions[b1], positions[b2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, linewidth=1.5)


def visualize(
    configs: List[Tuple[str, dict, List]],  # [(label, config, link_mapping)]
    robot: Robot,
    body_model_path: str,
    skeleton_bones: List[Tuple[str, str]],
    save_path: Optional[str] = None,
    robot_name: str = "g1",
):
    pairs = load_pose_pairs(None, robot_name)
    pose_names = list(pairs)

    n_poses = len(pose_names)
    n_configs = len(configs)
    fig = plt.figure(figsize=(5 * n_poses, 5 * n_configs))

    for ci, (label, config, link_mapping) in enumerate(configs):
        root_name = config.get("human_root_name", "pelvis")
        scale_table = config.get("scale_table", {})

        for pi, pose_name in enumerate(pose_names):
            bp, joint_overrides = pairs[pose_name]
            smplx_pos, smplx_quat, _ = _load_smplx_pose(body_model_path, bp)

            pose_data = prepare_pose(
                pose_name,
                smplx_pos,
                smplx_quat,
                robot,
                config,
                link_mapping,
                joint_overrides=joint_overrides,
            )

            transformed = apply_scale_and_offset_np(
                pose_data["smplx_pos"],
                pose_data["smplx_quat"],
                root_name,
                scale_table,
                link_mapping,
            )

            robot_pos = pose_data["robot_pos"]

            ax = fig.add_subplot(n_configs, n_poses, ci * n_poses + pi + 1, projection="3d")

            # plot target points (red) and robot FK (blue)
            errors = {}
            for robot_link, _, _, _, _, _ in link_mapping:
                if robot_link in transformed and robot_link in robot_pos:
                    t, g = transformed[robot_link], robot_pos[robot_link]
                    err = np.linalg.norm(t - g)
                    errors[robot_link] = err

                    # color by error magnitude
                    err_color = plt.cm.RdYlGn_r(min(err / 0.15, 1.0))
                    ax.scatter(t[0], t[1], zs=t[2], c="red", s=30, marker="o", alpha=0.8)
                    ax.scatter(g[0], g[1], zs=g[2], c="blue", s=30, marker="^", alpha=0.8)
                    # error line
                    ax.plot([t[0], g[0]], [t[1], g[1]], [t[2], g[2]], color=err_color, linewidth=2, alpha=0.7)

            _draw_skeleton(ax, transformed, skeleton_bones, "red", 0.3)
            _draw_skeleton(ax, robot_pos, skeleton_bones, "blue", 0.3)

            mean_err = np.mean(list(errors.values())) if errors else 0
            ax.set_title(f"{pose_name}\n{mean_err:.3f}m", fontsize=9)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            # equal aspect ratio
            all_pts = np.array(list(transformed.values()) + list(robot_pos.values()))
            if len(all_pts) > 0:
                center = all_pts.mean(axis=0)
                max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[2] - max_range, center[2] + max_range)
                ax.set_zlim(center[1] - max_range, center[1] + max_range)

        # row label
        fig.text(0.02, 1.0 - (ci + 0.5) / n_configs, label, fontsize=12, fontweight="bold", va="center", rotation=90)

    fig.suptitle("Red=SMPL-X target, Blue=Robot FK, Lines=error", fontsize=12)
    plt.tight_layout(rect=(0.03, 0, 1, 0.97))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Original config YAML")
    parser.add_argument("--calibrated", default=None, help="Calibrated config YAML")
    parser.add_argument("--robot", default="g1", help="Robot whose pose pairs to load (poses/<robot>.yaml)")
    parser.add_argument("--body-model-path", default=str(_body_model_dir()))
    parser.add_argument("--save", default=None, help="Save plot to file")
    args = parser.parse_args()

    configs = []

    orig_config = load_config(args.config)
    wp.init()
    robot = Robot.load(orig_config["urdf_path"], load_meshes=True)
    orig_mapping = get_link_mapping(orig_config)
    configs.append((f"Original ({Path(args.config).stem})", orig_config, orig_mapping))

    if args.calibrated:
        cal_config = load_config(args.calibrated)
        cal_mapping = get_link_mapping(cal_config)
        configs.append((f"Calibrated ({Path(args.calibrated).stem})", cal_config, cal_mapping))

    skeleton_bones = _auto_skeleton_bones(orig_mapping)

    print(f"Robot URDF: {orig_config['urdf_path']}")
    print(f"Skeleton bones: {len(skeleton_bones)}")
    print(f"Configs: {len(configs)}")

    visualize(configs, robot, args.body_model_path, skeleton_bones, args.save, args.robot)


if __name__ == "__main__":
    main()
