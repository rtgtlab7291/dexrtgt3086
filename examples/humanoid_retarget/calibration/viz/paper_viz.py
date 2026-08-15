"""Headless matplotlib 3D helpers for the calibration alignment figures.

Pure-function rendering: draw a skeleton (human SMPL-X or robot FK), draw target
keypoints + error lines, and style a clean 3D axis. SMPL-X data is Y-up,
robot-world data is Z-up - pass `up` accordingly so the vertical axis renders
upward.
"""

from typing import List, Optional, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# palette (colorblind-friendly, high contrast on white)
PRE = "#3B7DD8"  # blue - uncalibrated robot
POST = "#2CA02C"  # green - calibrated robot
TARGET = "#E24A33"  # red - target keypoints

# Vertical-axis remap: matplotlib's 3rd plotted axis is "up".
_UP = {"y": (0, 2, 1), "z": (0, 1, 2)}


def style_ax(
    ax,
    all_pts: np.ndarray,
    up: str,
    elev: float = 12.0,
    azim: float = -70.0,
    title: Optional[str] = None,
    pad: float = 1.08,
    zoom: float = 1.35,
    title_color: str = "black",
):
    """Equal-aspect cube around `all_pts` with a clean, label-free look."""
    o = _UP[up]
    p = all_pts[:, o]
    center = (p.max(0) + p.min(0)) / 2
    radius = (p.max(0) - p.min(0)).max() / 2 * pad
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    if title is not None:
        ax.set_title(title, fontsize=15, color=title_color, pad=2)


def draw_skeleton(
    ax,
    pts: np.ndarray,
    bones: List[Tuple[int, int]],
    up: str,
    color: str,
    lw: float = 2.4,
    alpha: float = 1.0,
    joint_size: float = 18.0,
):
    """Draw a skeleton from a `(J, 3)` point array and integer bone pairs."""
    o = _UP[up]
    q = pts[:, o]
    for a, b in bones:
        ax.plot(
            [q[a, 0], q[b, 0]],
            [q[a, 1], q[b, 1]],
            [q[a, 2], q[b, 2]],
            color=color,
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
        )
    if joint_size > 0:
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=color, s=joint_size, alpha=alpha, depthshade=False, edgecolors="none")


def draw_points(ax, pts: np.ndarray, up: str, color: str, size: float = 34.0, marker: str = "o", alpha: float = 1.0):
    """Scatter a `(K, 3)` point array (e.g. target keypoints)."""
    o = _UP[up]
    q = pts[:, o]
    ax.scatter(
        q[:, 0],
        q[:, 1],
        q[:, 2],
        c=color,
        s=size,
        marker=marker,
        alpha=alpha,
        depthshade=False,
        edgecolors="white",
        linewidths=0.4,
    )


def draw_error_lines(
    ax,
    src: np.ndarray,
    dst: np.ndarray,
    up: str,
    max_err: float = 0.12,
    lw: float = 2.0,
) -> float:
    """Draw per-keypoint error segments src→dst, colored by magnitude. Returns mean error (m)."""
    o = _UP[up]
    s, d = src[:, o], dst[:, o]
    errs = np.linalg.norm(src - dst, axis=1)
    cmap = plt.cm.RdYlGn_r
    for i in range(len(s)):
        ax.plot(
            [s[i, 0], d[i, 0]],
            [s[i, 1], d[i, 1]],
            [s[i, 2], d[i, 2]],
            color=cmap(min(errs[i] / max_err, 1.0)),
            lw=lw,
            alpha=0.9,
        )
    return float(errs.mean()) if len(errs) else 0.0


def robot_skeleton_bones() -> List[Tuple[str, str]]:
    """Human-joint connectivity used to draw the mapped-link robot/target skeleton."""
    return [
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
    ]
