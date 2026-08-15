"""Verify that calibration math matches runtime math, and that height
adjustment works correctly.

Tests:
  1. At calibration height: calibration and runtime must produce identical positions
  2. At different heights: inverted ratio correctly compensates for SMPL-X size
  3. Consistency: targets should be height-invariant (map to robot body regardless)

Usage:
    cd robokit-internal
    uv run python examples/humanoid_retarget/calibration/verify_calibration.py \\
        --config /tmp/calibrated.yaml
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from robokit.helpers.humanoid_retarget import HumanoidRetargetingOnlineConfig
from robokit.xform.numpy import quaternion_apply as _rotate_vector_by_quat
from robokit.xform.numpy import quaternion_multiply as _quat_multiply


# ---------------------------------------------------------------------------
# Calibration code path (from calibrate.py)
# ---------------------------------------------------------------------------


def _quat_multiply_np(q1, q2):
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    result = r1 * r2
    xyzw = result.as_quat(canonical=False)  # type: ignore[call-arg]
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _rotate_vector_np(v, q_wxyz):
    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot.apply(v).astype(np.float32)


def calibration_apply_scale_and_offset(
    smplx_positions,
    smplx_quaternions,
    root_name,
    scale_table,
    link_mapping,
):
    """EXACT copy of calibrate.apply_scale_and_offset_np()."""
    root_pos_orig = smplx_positions[root_name].copy()
    root_scale = scale_table.get(root_name, 1.0)

    scaled_positions = {}
    for joint_name, pos in smplx_positions.items():
        scale = scale_table.get(joint_name, 1.0)
        if joint_name == root_name:
            scaled_positions[joint_name] = pos * scale
        else:
            relative = pos - root_pos_orig
            scaled_root = root_pos_orig * root_scale
            scaled_positions[joint_name] = scaled_root + relative * scale

    result = {}
    for robot_link, human_joint, pos_offset, rot_offset, pw, ow in link_mapping:
        if human_joint not in scaled_positions:
            continue
        pos = scaled_positions[human_joint].copy()
        quat = smplx_quaternions[human_joint].copy()

        if np.linalg.norm(rot_offset) > 1e-6:
            rot_offset_n = rot_offset / np.linalg.norm(rot_offset)
            quat = _quat_multiply_np(quat, rot_offset_n)
        if np.linalg.norm(pos_offset) > 1e-6:
            global_offset = _rotate_vector_np(pos_offset, quat)
            pos = pos + global_offset
        result[robot_link] = pos
    return result


# ---------------------------------------------------------------------------
# Runtime code path (replicated from kinematic/online.py)
# ---------------------------------------------------------------------------


def runtime_scale_and_offset(human_frame, config, human_height):
    """Replicate runtime scale_human_frame + apply_offsets."""
    root_name = config.human_root_name
    root_pos_orig = human_frame[root_name][0].copy()
    height_scale = config.human_height_assumption / human_height
    root_scale = config.scale_table.get(root_name, 1.0) * height_scale

    scaled_frame = {}
    for joint_name, (pos, quat) in human_frame.items():
        pos = pos.copy().astype(np.float32)
        quat = quat.copy().astype(np.float32)
        scale = config.scale_table.get(joint_name, 1.0) * height_scale
        if joint_name == root_name:
            scaled_pos = pos * scale
        else:
            relative_pos = pos - root_pos_orig
            scaled_root = root_pos_orig * root_scale
            scaled_pos = scaled_root + relative_pos * scale
        scaled_frame[joint_name] = (scaled_pos, quat)

    targets = {}
    for robot_link, entry in config.link_mapping.items():
        human_joint = entry.human_joint
        if human_joint not in scaled_frame:
            continue
        pos, quat = scaled_frame[human_joint]
        pos = pos.copy()
        quat = quat.copy()

        rot_offset = entry.rotation_offset
        if np.linalg.norm(rot_offset) > 1e-6:
            rot_offset = rot_offset / np.linalg.norm(rot_offset)
            quat = _quat_multiply(quat, rot_offset)

        pos_offset = entry.position_offset
        if np.linalg.norm(pos_offset) > 1e-6:
            global_offset = _rotate_vector_by_quat(quat, pos_offset)
            pos = pos + global_offset
        pos[2] -= config.ground_height
        targets[human_joint] = (pos, quat)
    return targets


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------


def make_frame_at_height(height_ratio: float = 1.0):
    """Create a synthetic standing frame. Positions scale with height_ratio.

    height_ratio=1.0 gives the reference person (at human_height_assumption).
    height_ratio=0.92 gives a shorter person (positions 8% closer together).
    """
    base_joints = {
        "pelvis": np.array([0.0, 0.9, 0.0]),
        "left_hip": np.array([0.08, 0.85, 0.0]),
        "right_hip": np.array([-0.08, 0.85, 0.0]),
        "left_knee": np.array([0.08, 0.47, 0.02]),
        "right_knee": np.array([-0.08, 0.47, 0.02]),
        "left_foot": np.array([0.08, 0.05, 0.04]),
        "right_foot": np.array([-0.08, 0.05, 0.04]),
        "spine3": np.array([0.0, 1.25, 0.0]),
        "left_shoulder": np.array([0.17, 1.35, 0.0]),
        "right_shoulder": np.array([-0.17, 1.35, 0.0]),
        "left_elbow": np.array([0.40, 1.35, 0.0]),
        "right_elbow": np.array([-0.40, 1.35, 0.0]),
        "left_wrist": np.array([0.62, 1.35, 0.0]),
        "right_wrist": np.array([-0.62, 1.35, 0.0]),
    }

    # Scale positions proportionally to height (SMPL-X does this via betas)
    pelvis = base_joints["pelvis"]
    identity_q = np.array([1, 0, 0, 0], dtype=np.float32)
    frame = {}
    for name, pos in base_joints.items():
        if name == "pelvis":
            scaled_pos = pos * height_ratio
        else:
            relative = pos - pelvis
            scaled_pos = pelvis * height_ratio + relative * height_ratio
        frame[name] = (scaled_pos.astype(np.float32), identity_q.copy())
    return frame


def get_cal_link_mapping(config_data):
    entries = []
    for robot_link, entry in config_data.get("link_mapping", {}).items():
        entries.append(
            (
                robot_link,
                entry["human_joint"],
                np.array(entry["position_offset"], dtype=np.float32),
                np.array(entry["rotation_offset"], dtype=np.float32),
                float(entry["position_weight"]),
                float(entry["orientation_weight"]),
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_math_match(raw: dict, rt_config: HumanoidRetargetingOnlineConfig, height: float, assumption: float) -> float:
    """Test that calibration and runtime produce the same positions."""
    cal_mapping = get_cal_link_mapping(raw)
    root_name = raw.get("human_root_name", "pelvis")
    height_ratio = height / assumption
    frame = make_frame_at_height(height_ratio)

    positions = {n: p.copy() for n, (p, _) in frame.items()}
    quaternions = {n: q.copy() for n, (_, q) in frame.items()}
    cal_result = calibration_apply_scale_and_offset(
        positions,
        quaternions,
        root_name,
        raw.get("scale_table", {}),
        cal_mapping,
    )

    rt_targets = runtime_scale_and_offset(frame, rt_config, height)

    max_diff = 0.0
    for robot_link, human_joint, _, _, _, _ in cal_mapping:
        cal_pos = cal_result.get(robot_link)
        rt_data = rt_targets.get(human_joint)
        if cal_pos is not None and rt_data is not None:
            max_diff = max(max_diff, np.linalg.norm(cal_pos - rt_data[0]))
    return max_diff


def test_height_invariance(raw: dict, rt_config: HumanoidRetargetingOnlineConfig, heights: list) -> list:
    """Test that targets are height-invariant after height adjustment."""
    assumption = raw.get("human_height_assumption", 1.8)
    reference_targets = None
    results = []
    for h in heights:
        frame = make_frame_at_height(h / assumption)
        targets = runtime_scale_and_offset(frame, rt_config, h)
        if reference_targets is None:
            reference_targets = targets
            results.append((h, 0.0))
        else:
            max_diff = max(
                np.linalg.norm(ref_pos - targets[hj][0])
                for hj, (ref_pos, _) in reference_targets.items()
                if hj in targets
            )
            results.append((h, max_diff))
    return results


def _load_raw_and_config(path: "str | None", robot: str = "g1") -> "tuple[dict, HumanoidRetargetingOnlineConfig]":
    if path is not None:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return raw, HumanoidRetargetingOnlineConfig.from_yaml(path)
    from calibrate import PRESETS

    preset = PRESETS[robot]
    return preset.to_dict(), preset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Calibrated config YAML to verify")
    parser.add_argument("--original", type=str, default=None, help="Reference config YAML (default: --robot preset)")
    parser.add_argument("--robot", type=str, default="g1", help="Baseline preset when --original is not given")
    args = parser.parse_args()

    cal_raw, cal_config = _load_raw_and_config(args.config)
    orig_raw, orig_config = _load_raw_and_config(args.original, args.robot)

    cal_assumption = cal_raw.get("human_height_assumption", 1.8)
    orig_assumption = orig_raw.get("human_height_assumption", 1.8)

    print(f"Calibrated config: {Path(args.config).name} (assumption={cal_assumption}m)")
    print(f"Original config:   {args.original or f'presets.{args.robot}'} (assumption={orig_assumption}m)")

    # ---- TEST 1: Math match at calibration height ----
    print(f"\n{'=' * 70}")
    print("TEST 1: Calibration vs runtime math match at reference height")
    print(f"{'=' * 70}")

    diff = test_math_match(cal_raw, cal_config, cal_assumption, cal_assumption)
    status = "PASS" if diff < 1e-6 else "FAIL"
    print(f"  Calibrated @ h={cal_assumption}m: max diff = {diff:.8f}m  [{status}]")

    diff = test_math_match(orig_raw, orig_config, orig_assumption, orig_assumption)
    status = "PASS" if diff < 1e-6 else "FAIL"
    print(f"  Original   @ h={orig_assumption}m: max diff = {diff:.8f}m  [{status}]")

    # ---- TEST 2: Height invariance ----
    print(f"\n{'=' * 70}")
    print("TEST 2: Height invariance — targets should be ~same for all heights")
    print("  (With correct ratio, robot targets don't depend on human height)")
    print(f"{'=' * 70}")

    heights = [1.50, 1.60, 1.66, 1.70, 1.75, 1.80, 1.85, 1.90, 2.00]

    print(f"\n  Calibrated config (assumption={cal_assumption}m):")
    print(f"  {'Height':>8s} {'Max diff from ref':>18s} {'Status':>8s}")
    cal_results = test_height_invariance(cal_raw, cal_config, heights)
    for h, diff in cal_results:
        status = "PASS" if diff < 0.001 else "WARN" if diff < 0.01 else "FAIL"
        marker = " (ref)" if h == cal_assumption else ""
        print(f"  {h:8.2f}m {diff:18.6f}m  {status:>8s}{marker}")

    print(f"\n  Original config (assumption={orig_assumption}m):")
    print(f"  {'Height':>8s} {'Max diff from ref':>18s} {'Status':>8s}")
    orig_results = test_height_invariance(orig_raw, orig_config, heights)
    for h, diff in orig_results:
        status = "PASS" if diff < 0.001 else "WARN" if diff < 0.01 else "FAIL"
        marker = " (ref)" if h == orig_assumption else ""
        print(f"  {h:8.2f}m {diff:18.6f}m  {status:>8s}{marker}")

    # ---- TEST 3: Show actual targets for a couple of heights ----
    print(f"\n{'=' * 70}")
    print("TEST 3: Target positions at different heights (should be ~identical)")
    print(f"{'=' * 70}")

    test_heights = [1.60, cal_assumption, 1.80, 2.00]
    cal_mapping = get_cal_link_mapping(cal_raw)

    for h in test_heights:
        frame = make_frame_at_height(h / cal_assumption)
        targets = runtime_scale_and_offset(frame, cal_config, h)

        print(f"\n  h={h:.2f}m (ratio to ref: {h / cal_assumption:.3f}):")
        for robot_link, human_joint, _, _, _, _ in cal_mapping[:3]:  # show first 3
            if human_joint in targets:
                pos = targets[human_joint][0]
                print(f"    {robot_link:<30s} [{pos[0]:8.5f}, {pos[1]:8.5f}, {pos[2]:8.5f}]")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    all_pass = all(d < 0.001 for _, d in cal_results)
    if all_pass:
        print("  All height invariance tests PASS — ratio fix is working correctly!")
    else:
        max_err = max(d for _, d in cal_results)
        print(f"  WARNING: Max height invariance error = {max_err:.4f}m")
        print("  Some error is expected due to position_offset rotation effects.")


if __name__ == "__main__":
    main()
