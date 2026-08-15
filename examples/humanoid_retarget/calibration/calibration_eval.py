"""Quantitative evaluation for a humanoid retargeting calibration (any robot).

Runs the online retargeting pipeline on real SMPL-X clips and measures how well a
config's keypoints are tracked, so a calibration can be checked against deployment
(not just the static calibration poses). No visualization dependencies - this is
the metric the calibration tooling and tests rely on.

Run directly to compare uncalibrated / calibrated (and optionally a YAML) on SFU clips:

    uv run python examples/humanoid_retarget/calibration/calibration_eval.py \\
        --robot g1 --yaml /tmp/g1_grounded.yaml
"""

from typing import Dict, List, Tuple

import numpy as np
import warp as wp

from robokit.helpers.humanoid_retarget import HumanoidRetargetingOnline, HumanoidRetargetingOnlineConfig
from robokit.helpers.humanoid_retarget.loaders import get_smplx_motion, load_smplx_file
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def _foot_links_and_clearance(config: HumanoidRetargetingOnlineConfig) -> Tuple[List[Tuple[str, str]], float]:
    """The (robot_foot_link, human_foot_joint) pairs and the foot-link-origin
    height above the sole when the robot stands grounded in its zero pose.

    Both are derived per robot: feet from the link_mapping, the clearance from
    the same all-links geometry grounding the calibration uses.
    """
    from calibrate import load_robot_grounded

    from robokit.xform.numpy import quaternion_multiply

    feet = [
        (rl, m.human_joint) for rl, m in config.link_mapping.items() if m.human_joint in ("left_foot", "right_foot")
    ]
    robot = Robot.load(config.urdf_path, load_meshes=True)
    pelvis_rot = config.link_mapping[next(rl for rl, m in config.link_mapping.items() if m.human_joint == "pelvis")]
    # Upright standing base quat = T-pose smplx pelvis quat (Rx90, Z-up frame) * rotation_offset.
    rx90 = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)
    base_quat = quaternion_multiply(rx90, pelvis_rot.rotation_offset)
    positions, _ = load_robot_grounded(robot, np.zeros(2, np.float32), base_quat)
    clearance = float(min(positions[rl][2] for rl, _ in feet))
    return feet, clearance


def run_retarget(
    helper: HumanoidRetargetingOnline,
    T_world_human: np.ndarray,
    human_height: float,
) -> Tuple[np.ndarray, List[Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    """Retarget every frame; return qpos array `[N, 7+dof]` and per-frame target dicts."""
    human_heights = np.array([human_height], dtype=np.float32)
    helper.warmup(1)
    helper.reset()
    helper.solve_numpy(T_world_human[:1], human_heights)  # warm up CUDA graph
    helper.reset()
    joints = [entry.human_joint for entry in helper.config.link_mapping.values()]
    qpos = []
    targets = []
    for frame in T_world_human:
        qpos.append(helper.solve_numpy(frame[None], human_heights)[0])
        target = helper._frame_tasks[0].T_world_target.numpy()[0]
        targets.append({joint: (target[k, :3], target[k, 3:]) for k, joint in enumerate(joints)})
    return np.stack(qpos), targets


def per_link_errors(
    config: HumanoidRetargetingOnlineConfig, clips, body_model: str, device: str, fps: int, max_frames: int
) -> Dict[str, np.ndarray]:
    """Per-tracked-link retarget error (mm) aggregated over clips, keyed by human joint.

    Only position-tracked links (`position_weight > 0`) are measured - orientation-only
    links like the torso are excluded. Error is `||robot_link_FK - IK_target||`.
    """
    robot = Robot.load(config.urdf_path)
    names = list(robot.spec.link_names)
    links = [rl for rl in config.link_mapping if config.link_mapping[rl].position_weight > 0 and rl in names]
    li = [names.index(rl) for rl in links]
    hj = [config.link_mapping[rl].human_joint for rl in links]
    acc: Dict[str, List[float]] = {h: [] for h in hj}

    for clip in clips:
        motion = load_smplx_file(clip, body_model, device=device, compute_vertices=False)
        helper = HumanoidRetargetingOnline(config, device=device, robot=Robot.load(config.urdf_path))
        T_world_human, _ = get_smplx_motion(motion, helper.human_joint_names, tgt_fps=fps)
        T_world_human = T_world_human[:max_frames]
        qpos, targets = run_retarget(helper, T_world_human, motion.human_height)
        base7, q = qpos[:, :7].astype(np.float32), qpos[:, 7:].astype(np.float32)
        state = robot.forward_kinematics(
            robot.state(q=q, T_world_base=wp.from_numpy(base7, dtype=wp_vec7, device=device))
        )
        T = state.T_world_link.numpy()
        for k, rl in enumerate(links):
            tgt = np.array([t[hj[k]][0] for t in targets], np.float32)
            err = np.linalg.norm(T[:, li[k], :3] - tgt, axis=1) * 1000.0
            acc[hj[k]].extend(err.tolist())
    return {h: np.array(v) for h, v in acc.items()}


def foot_float_metrics(
    config: HumanoidRetargetingOnlineConfig,
    clips,
    body_model: str,
    device: str,
    fps: int,
    max_frames: int,
    stance_thresh: float = 0.08,
) -> Dict[str, float]:
    """Robot foot-sole height above the ground (m) while the human foot is in stance.

    Stance frames are detected on the HUMAN side (foot joint z < `stance_thresh`),
    so flight phases (jumps, swing legs) are excluded. Positive mean/p95 = the robot
    floats; ~0 = feet on the ground. The foot-link origin is converted to sole height
    via the robot's standing clearance (derived from its geometry).
    """
    robot = Robot.load(config.urdf_path)
    names = list(robot.spec.link_names)
    feet, clearance = _foot_links_and_clearance(config)
    heights: List[float] = []
    for clip in clips:
        motion = load_smplx_file(clip, body_model, device=device, compute_vertices=False)
        helper = HumanoidRetargetingOnline(config, device=device, robot=Robot.load(config.urdf_path))
        T_world_human, _ = get_smplx_motion(motion, helper.human_joint_names, tgt_fps=fps)
        T_world_human = T_world_human[:max_frames]
        qpos, _ = run_retarget(helper, T_world_human, motion.human_height)
        base7, q = qpos[:, :7].astype(np.float32), qpos[:, 7:].astype(np.float32)
        state = robot.forward_kinematics(
            robot.state(q=q, T_world_base=wp.from_numpy(base7, dtype=wp_vec7, device=device))
        )
        T = state.T_world_link.numpy()
        for foot_link, hj in feet:
            stance = T_world_human[:, helper.human_joint_names.index(hj), 2] < stance_thresh
            sole_z = T[stance, names.index(foot_link), 2] - clearance
            heights.extend(sole_z.tolist())
    arr = np.array(heights)
    return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "min": float(arr.min())}


def main():
    import argparse

    from cali_configs import uncalibrated
    from calibrate import PRESETS, _body_model_dir

    from robokit.helpers.humanoid_retarget.presets.g1_cali import g1_cali

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="g1", choices=sorted(PRESETS), help="Robot preset to evaluate")
    parser.add_argument("--yaml", default=None, help="Extra config YAML to evaluate (e.g. a fresh calibration)")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--body-model-path", default=str(_body_model_dir()))
    parser.add_argument("--motions", nargs="*", default=None)
    args = parser.parse_args()

    motions_root = _body_model_dir().parent / "motions/SFU"
    clips = args.motions or [
        str(motions_root / "0007/0007_Walking001_poses.npz"),
        str(motions_root / "0017/0017_WushuKicks001_poses.npz"),
        str(motions_root / "0005/0005_Jogging001_poses.npz"),
    ]
    configs = [("uncalibrated", uncalibrated(PRESETS[args.robot]))]
    if args.robot == "g1":
        configs.append(("g1_cali", g1_cali))
    if args.yaml:
        configs.append((args.yaml.split("/")[-1], HumanoidRetargetingOnlineConfig.from_yaml(args.yaml)))

    wp.init()
    print(f"{len(clips)} clips, {args.max_frames} frames each\n")
    print(f"{'config':<24s} {'track mean (mm)':>16s} {'float mean (m)':>15s} {'float p95 (m)':>14s} {'min (m)':>9s}")
    for name, cfg in configs:
        errs = per_link_errors(cfg, clips, args.body_model_path, args.device, args.fps, args.max_frames)
        track = float(np.concatenate(list(errs.values())).mean())
        ff = foot_float_metrics(cfg, clips, args.body_model_path, args.device, args.fps, args.max_frames)
        print(f"{name:<24s} {track:>16.0f} {ff['mean']:>15.3f} {ff['p95']:>14.3f} {ff['min']:>9.3f}")


if __name__ == "__main__":
    main()
