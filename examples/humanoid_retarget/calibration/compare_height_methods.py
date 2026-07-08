#!/usr/bin/env python3
"""Compare different approaches for estimating human height from SMPL-X data.

Measures ground truth height from mesh vertices and compares against:
  A) Current formula: 1.66 + 0.1 * betas[0]
  B) Mesh-based: max(Y) - min(Y) from T-pose vertices
  C) Mesh-based: max(Y) - min(Y) from first-frame vertices
  D) Joint-based: head joint Y - min foot joint Y from T-pose

Usage:
    cd robokit-internal
    uv run python examples/humanoid_retarget/calibration/compare_height_methods.py
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import smplx
import torch


BODY_MODEL_PATH = Path(__file__).resolve().parents[3] / "assets" / "body_models"
DATA_DIR = Path(__file__).resolve().parents[3] / "motion_data" / "raw" / "SFU"


def load_npz(path: str) -> dict:
    raw = np.load(path, allow_pickle=True)
    data = dict(raw)
    if "poses" in data and "pose_body" not in data:
        poses = data["poses"]
        data["root_orient"] = poses[:, :3]
        data["pose_body"] = poses[:, 3:66]
    if "mocap_framerate" in data and "mocap_frame_rate" not in data:
        data["mocap_frame_rate"] = data["mocap_framerate"]
    return data


def get_body_model(gender: str, body_model_path: str):
    return smplx.create(body_model_path, "smplx", gender=gender, use_pca=False)


def get_betas_tensor(smplx_data: dict, body_model) -> torch.Tensor:
    betas_raw = smplx_data["betas"]
    if betas_raw.ndim > 1:
        betas_raw = betas_raw.reshape(-1)

    shapedirs = getattr(body_model, "shapedirs", None)
    shape_dim = int(shapedirs.shape[-1]) if shapedirs is not None else 10
    betas_trimmed = (
        betas_raw[:shape_dim]
        if len(betas_raw) > shape_dim
        else np.pad(betas_raw, (0, max(0, shape_dim - len(betas_raw))))
    )
    return torch.tensor(betas_trimmed).float().view(1, -1)


# ──────────────────────────────────────────────────────────────
# Method A: Current linear approximation
# ──────────────────────────────────────────────────────────────
def height_formula_current(smplx_data: dict) -> float:
    betas = smplx_data["betas"]
    b0 = float(betas[0] if betas.ndim == 1 else betas[0, 0])
    return 1.66 + 0.1 * b0


# ──────────────────────────────────────────────────────────────
# Method B: T-pose mesh vertices (ground truth)
# ──────────────────────────────────────────────────────────────
def height_tpose_mesh(smplx_data: dict, body_model) -> float:
    """Run SMPL-X forward pass in T-pose (zero pose) and measure vertex extent."""
    betas = get_betas_tensor(smplx_data, body_model)

    output = body_model(
        betas=betas,
        global_orient=torch.zeros(1, 3).float(),
        body_pose=torch.zeros(1, 63).float(),
        transl=torch.zeros(1, 3).float(),
        expression=torch.zeros(1, body_model.num_expression_coeffs).float(),
        left_hand_pose=torch.zeros(1, 45).float(),
        right_hand_pose=torch.zeros(1, 45).float(),
        jaw_pose=torch.zeros(1, 3).float(),
        leye_pose=torch.zeros(1, 3).float(),
        reye_pose=torch.zeros(1, 3).float(),
    )
    verts = output.vertices.detach().cpu().numpy().squeeze()  # [10475, 3]
    # SMPL-X is Y-up
    return float(verts[:, 1].max() - verts[:, 1].min())


# ──────────────────────────────────────────────────────────────
# Method C: First-frame mesh vertices
# ──────────────────────────────────────────────────────────────
def height_first_frame_mesh(smplx_data: dict, body_model) -> float:
    """Run SMPL-X forward pass with first frame's pose and measure vertex extent.

    This is less ideal because pose affects height (e.g., squatting).
    Included for comparison.
    """
    betas = get_betas_tensor(smplx_data, body_model)

    output = body_model(
        betas=betas,
        global_orient=torch.tensor(smplx_data["root_orient"][:1]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"][:1]).float(),
        transl=torch.zeros(1, 3).float(),  # zero transl to get body height only
        expression=torch.zeros(1, body_model.num_expression_coeffs).float(),
        left_hand_pose=torch.zeros(1, 45).float(),
        right_hand_pose=torch.zeros(1, 45).float(),
        jaw_pose=torch.zeros(1, 3).float(),
        leye_pose=torch.zeros(1, 3).float(),
        reye_pose=torch.zeros(1, 3).float(),
    )
    verts = output.vertices.detach().cpu().numpy().squeeze()
    return float(verts[:, 1].max() - verts[:, 1].min())


# ──────────────────────────────────────────────────────────────
# Method D: T-pose joint-based height (head - foot)
# ──────────────────────────────────────────────────────────────
def height_tpose_joints(smplx_data: dict, body_model) -> float:
    """Use joint positions in T-pose: head top Y - min(foot joints Y)."""
    from smplx.joint_names import JOINT_NAMES

    betas = get_betas_tensor(smplx_data, body_model)

    output = body_model(
        betas=betas,
        global_orient=torch.zeros(1, 3).float(),
        body_pose=torch.zeros(1, 63).float(),
        transl=torch.zeros(1, 3).float(),
        expression=torch.zeros(1, body_model.num_expression_coeffs).float(),
        left_hand_pose=torch.zeros(1, 45).float(),
        right_hand_pose=torch.zeros(1, 45).float(),
        jaw_pose=torch.zeros(1, 3).float(),
        leye_pose=torch.zeros(1, 3).float(),
        reye_pose=torch.zeros(1, 3).float(),
    )
    joints = output.joints.detach().cpu().numpy().squeeze()  # [N_joints, 3]
    joint_names = JOINT_NAMES[: joints.shape[0]]

    # Find head and foot joints
    head_joints = [i for i, n in enumerate(joint_names) if "head" in n.lower()]
    foot_joints = [i for i, n in enumerate(joint_names) if "foot" in n.lower() or "toe" in n.lower()]

    if not head_joints or not foot_joints:
        # Fallback: use all joints
        return float(joints[:, 1].max() - joints[:, 1].min())

    head_y = max(joints[i, 1] for i in head_joints)
    foot_y = min(joints[i, 1] for i in foot_joints)
    return float(head_y - foot_y)


# ──────────────────────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────────────────────
def main():
    npz_files = sorted(glob.glob(str(DATA_DIR / "**" / "*_poses.npz"), recursive=True))

    if not npz_files:
        print(f"No NPZ files found in {DATA_DIR}")
        return

    print(f"Found {len(npz_files)} sequences in {DATA_DIR.name}")
    print(f"Body model: {BODY_MODEL_PATH}")

    # Cache body models by gender
    body_models: Dict[str, object] = {}

    results: List[Dict] = []

    # Collect unique subjects (by betas) to avoid redundant computation
    seen_subjects: Dict[str, str] = {}  # betas_hash -> first file path

    print(f"\n{'='*100}")
    print(f"{'File':<45s} {'betas[0]':>8s} | {'A:formula':>9s} {'B:T-mesh':>9s} {'C:1st-frm':>9s} {'D:joints':>9s} | {'A err':>7s} {'C err':>7s} {'D err':>7s}")
    print(f"{'='*100}")

    for npz_path in npz_files:
        smplx_data = load_npz(npz_path)

        gender = str(smplx_data.get("gender", "neutral"))
        if gender not in body_models:
            body_models[gender] = get_body_model(gender, str(BODY_MODEL_PATH))
        bm = body_models[gender]

        betas = smplx_data["betas"]
        b0 = float(betas[0] if betas.ndim == 1 else betas[0, 0])

        # Compute all methods
        h_a = height_formula_current(smplx_data)
        h_b = height_tpose_mesh(smplx_data, bm)        # ground truth
        h_c = height_first_frame_mesh(smplx_data, bm)
        h_d = height_tpose_joints(smplx_data, bm)

        err_a = h_a - h_b
        err_c = h_c - h_b
        err_d = h_d - h_b

        short_name = "/".join(Path(npz_path).parts[-2:])
        print(
            f"{short_name:<45s} {b0:>8.3f} | "
            f"{h_a:>9.4f} {h_b:>9.4f} {h_c:>9.4f} {h_d:>9.4f} | "
            f"{err_a:>+7.4f} {err_c:>+7.4f} {err_d:>+7.4f}"
        )

        results.append({
            "file": short_name, "gender": gender, "betas0": b0,
            "h_formula": h_a, "h_tpose_mesh": h_b, "h_first_frame": h_c, "h_joints": h_d,
        })

    # Summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY (errors relative to Method B: T-pose mesh = ground truth)")
    print(f"{'='*100}")

    errs_a = [r["h_formula"] - r["h_tpose_mesh"] for r in results]
    errs_c = [r["h_first_frame"] - r["h_tpose_mesh"] for r in results]
    errs_d = [r["h_joints"] - r["h_tpose_mesh"] for r in results]

    print(f"\n  Method A (1.66 + 0.1*betas[0]):")
    print(f"    Mean error: {np.mean(errs_a):>+.4f}m")
    print(f"    Std error:  {np.std(errs_a):>.4f}m")
    print(f"    Max |error|: {np.max(np.abs(errs_a)):>.4f}m")
    print(f"    Range: [{np.min(errs_a):>+.4f}, {np.max(errs_a):>+.4f}]m")

    print(f"\n  Method B (T-pose mesh vertices) = GROUND TRUTH")
    gt_heights = [r["h_tpose_mesh"] for r in results]
    print(f"    Height range: [{np.min(gt_heights):.4f}, {np.max(gt_heights):.4f}]m")

    print(f"\n  Method C (first-frame mesh vertices):")
    print(f"    Mean error: {np.mean(errs_c):>+.4f}m")
    print(f"    Std error:  {np.std(errs_c):>.4f}m")
    print(f"    Max |error|: {np.max(np.abs(errs_c)):>.4f}m")
    print(f"    Range: [{np.min(errs_c):>+.4f}, {np.max(errs_c):>+.4f}]m")

    print(f"\n  Method D (T-pose joint-based):")
    print(f"    Mean error: {np.mean(errs_d):>+.4f}m")
    print(f"    Std error:  {np.std(errs_d):>.4f}m")
    print(f"    Max |error|: {np.max(np.abs(errs_d)):>.4f}m")
    print(f"    Range: [{np.min(errs_d):>+.4f}, {np.max(errs_d):>+.4f}]m")

    # Fit a better linear model from data
    betas0_arr = np.array([r["betas0"] for r in results])
    gt_arr = np.array(gt_heights)

    if len(set(betas0_arr)) > 1:
        # Linear regression: height = a + b * betas[0]
        coeffs = np.polyfit(betas0_arr, gt_arr, 1)
        fitted = np.polyval(coeffs, betas0_arr)
        fit_errs = fitted - gt_arr
        print(f"\n  Fitted linear model: height = {coeffs[1]:.4f} + {coeffs[0]:.4f} * betas[0]")
        print(f"    (Current formula:  height = 1.6600 + 0.1000 * betas[0])")
        print(f"    Fit residual std: {np.std(fit_errs):.4f}m")
        print(f"    Fit max |error|:  {np.max(np.abs(fit_errs)):.4f}m")

    # Recommendation
    print(f"\n{'='*100}")
    print("RECOMMENDATION")
    print(f"{'='*100}")
    print("""
  Method B (T-pose mesh) is the gold standard — it runs the full SMPL-X forward
  pass with zero pose and measures vertex extent. This is what SMPL-Anthropometry
  and other standard tools use.

  Pros/cons of each approach:

  A) Formula (1.66 + 0.1*betas[0]):
     + Zero compute cost (no forward pass needed)
     - Ignores betas[1:] which also affect height
     - Coefficients may not generalize across SMPL-X versions
     - Linear approximation of a nonlinear relationship

  B) T-pose mesh vertices (RECOMMENDED):
     + Most accurate — uses full body model
     + Standard approach in the community
     - Requires one extra forward pass (T-pose, 1 frame)
     - ~10ms overhead, negligible for retargeting

  C) First-frame mesh vertices:
     + No extra forward pass (reuses existing computation)
     - Pose-dependent: squatting/bending gives wrong height
     - Unreliable

  D) T-pose joint-based:
     + Same forward pass as B
     + Slightly faster (fewer points)
     - Joints don't include top of head / bottom of feet
     - Systematically shorter than true height
""")


if __name__ == "__main__":
    main()
