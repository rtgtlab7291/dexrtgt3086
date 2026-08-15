"""Skeleton alignment before vs after calibration ("pose alignment") - static figure.

For each calibration pose, two panels side by side:
  left  = Uncalibrated (scale=1, no offset): SMPL-X targets (red) vs robot FK links, error lines
  right = Calibrated:                         same, after pose alignment

Red target keypoints sit loosely off the robot on the left and snap onto it on the
right - a direct visual read of how well calibration aligned the SMPL-X→robot link
mapping. Works for any robot via --robot (uncalibrated baseline) + --calibrated YAML.

Usage (from repo root):
    uv run python examples/humanoid_retarget/calibration/viz/viz_skeleton_alignment.py \
        --robot h1_2 --calibrated /tmp/h1_2_cali.yaml --out outputs/h1_2_skeleton_alignment.png
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import _bootstrap  # noqa: F401  -- adds the calibration dir to sys.path; keep first
import matplotlib.pyplot as plt
import numpy as np
import paper_viz as pv
import warp as wp
from cali_configs import uncalibrated
from calibrate import (
    PRESETS,
    _body_model_dir,
    _load_smplx_pose,
    apply_scale_and_offset_np,
    get_link_mapping,
    load_config,
    load_pose_pairs,
    prepare_pose,
)

from robokit.helpers.humanoid_retarget.presets.g1_cali import g1_cali
from robokit.robo import Robot


def pose_arrays(config: dict, link_mapping, pose: dict) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Return (bone index pairs, target keypoints[K,3], robot FK links[K,3]) for one pose."""
    transformed = apply_scale_and_offset_np(
        pose["smplx_pos"], pose["smplx_quat"], config["human_root_name"], config.get("scale_table", {}), link_mapping
    )
    links = [rl for rl, *_ in link_mapping if rl in transformed and rl in pose["robot_pos"]]
    targets = np.array([transformed[rl] for rl in links], dtype=np.float32)
    robot = np.array([pose["robot_pos"][rl] for rl in links], dtype=np.float32)

    hj_of = {rl: hj for rl, hj, *_ in link_mapping}
    hj_to_i = {hj_of[rl]: i for i, rl in enumerate(links)}
    bones = [(hj_to_i[a], hj_to_i[b]) for a, b in pv.robot_skeleton_bones() if a in hj_to_i and b in hj_to_i]
    return bones, targets, robot


def render_panel(ax, bones, targets, robot, color, up, azim, label):
    mean_err = pv.draw_error_lines(ax, targets, robot, up, max_err=0.12, lw=2.0)
    pv.draw_skeleton(ax, robot, bones, up, color, lw=2.6, joint_size=22)
    pv.draw_points(ax, targets, up, pv.TARGET, size=40)
    all_pts = np.vstack([targets, robot])
    pv.style_ax(ax, all_pts, up, elev=10, azim=azim, title=f"{label}\nmean err {mean_err * 1000:.0f} mm")
    return mean_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot", default="g1", choices=sorted(PRESETS), help="Robot for the uncalibrated baseline + pose pairs"
    )
    parser.add_argument("--calibrated", default=None, help="Calibrated config YAML (default: g1_cali preset)")
    parser.add_argument("--out", default="/tmp/cali_out/skeleton_alignment.png")
    parser.add_argument("--azim", type=float, default=-90.0, help="Single view azimuth")
    parser.add_argument("--body-model-path", default=str(_body_model_dir()))
    args = parser.parse_args()

    wp.init()
    pre_cfg = uncalibrated(PRESETS[args.robot]).to_dict()
    post_cfg = load_config(args.calibrated) if args.calibrated else g1_cali.to_dict()
    post_label = Path(args.calibrated).stem if args.calibrated else "g1_cali"
    robot = Robot.load(pre_cfg["urdf_path"], load_meshes=True)
    pre_lm, post_lm = get_link_mapping(pre_cfg), get_link_mapping(post_cfg)

    pairs = load_pose_pairs(None, args.robot)
    n = len(pairs)
    fig = plt.figure(figsize=(8.5, 4.2 * n), dpi=110)
    pre_errs: List[float] = []
    post_errs: List[float] = []

    for pi, (pose_name, (bp, ov)) in enumerate(pairs.items()):
        smplx_pos, smplx_quat, _ = _load_smplx_pose(args.body_model_path, bp)
        pre_pose = prepare_pose(pose_name, smplx_pos, smplx_quat, robot, pre_cfg, pre_lm, ov)
        post_pose = prepare_pose(pose_name, smplx_pos, smplx_quat, robot, post_cfg, post_lm, ov)
        pre_b, pre_t, pre_r = pose_arrays(pre_cfg, pre_lm, pre_pose)
        post_b, post_t, post_r = pose_arrays(post_cfg, post_lm, post_pose)

        ax1 = fig.add_subplot(n, 2, 2 * pi + 1, projection="3d")
        ax2 = fig.add_subplot(n, 2, 2 * pi + 2, projection="3d")
        e_pre = render_panel(ax1, pre_b, pre_t, pre_r, pv.PRE, "z", args.azim, f"{pose_name} - uncalibrated")
        e_post = render_panel(
            ax2, post_b, post_t, post_r, pv.POST, "z", args.azim, f"{pose_name} - calibrated ({post_label})"
        )
        pre_errs.append(e_pre)
        post_errs.append(e_post)
        print(f"  {pose_name:12s} pre {e_pre * 1000:5.0f}mm  ->  post {e_post * 1000:5.0f}mm")

    mp, mq = np.mean(pre_errs) * 1000, np.mean(post_errs) * 1000
    fig.suptitle(
        f"Pose alignment - {args.robot}   (red = SMPL-X target, lines = error)\nmean error {mp:.0f} mm  →  {mq:.0f} mm",
        fontsize=13,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    plt.close(fig)
    print(f"mean: pre {mp:.0f}mm -> post {mq:.0f}mm   saved {args.out}")


if __name__ == "__main__":
    main()
