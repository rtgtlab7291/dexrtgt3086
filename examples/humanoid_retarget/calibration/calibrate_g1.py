#!/usr/bin/env python3
"""Calibrate Robokit retargeting config for Unitree G1 (scale_table + position_offset).

Compares SMPL-X body model poses against G1 robot FK in the SMPL-X native frame
(Y-up). The robot is placed in the SMPL-X frame using the pelvis rotation_offset,
matching exactly how the runtime IK pipeline works.

Usage (run from repo root: robokit-internal/):

    # View current errors only (no optimization)
    uv run python examples/humanoid_retarget/calibration/calibrate_g1.py \
        --config examples/humanoid_retarget/ik_config/smplx_g1.yaml

    # Full calibration with auto-save
    uv run python examples/humanoid_retarget/calibration/calibrate_g1.py \
        --config examples/humanoid_retarget/ik_config/smplx_g1.yaml \
        --optimize-all --multi-pose --iters 3000 \
        --save examples/humanoid_retarget/ik_config/smplx_g1_calibrated.yaml

Options:
    --config PATH         Input YAML config (required)
    --optimize-scales     Optimize scale_table only
    --optimize-offsets    Optimize position_offset only
    --optimize-all        Optimize both scale_table and position_offset (recommended)
    --multi-pose          Use all 9 calibration poses (recommended, otherwise T-pose only)
    --iters N             Optimization iterations (default: 1000, recommended: 3000)
    --lr FLOAT            Learning rate (default: 0.005)
    --ee-weight FLOAT     End-effector weight multiplier (default: 3.0)
    --save PATH           Save calibrated config to file
    --body-model-path     SMPL-X body model directory
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
import smplx
import torch
import yaml
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BODY_MODEL_DIR = SCRIPT_DIR / "../../../assets/body_models"
DEFAULT_CONFIG = SCRIPT_DIR / "../ik_config/smplx_g1.yaml"

# G1 end-effector links (for extra loss weighting)
END_EFFECTOR_LINKS = {
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
}

# G1-specific joint overrides per calibration pose.
# These are tuned for the G1's zero configuration (arms forward, elbows bent 90°).
G1_POSE_OVERRIDES: Dict[str, Dict[str, float]] = {
    "T-pose": {
        "left_shoulder_roll_joint": 1.5708,
        "right_shoulder_roll_joint": -1.5708,
        "left_elbow_joint": 1.5708,
        "right_elbow_joint": 1.5708,
    },
    "A-pose": {
        "left_shoulder_roll_joint": 0.785,
        "right_shoulder_roll_joint": -0.785,
        "left_elbow_joint": 1.2,
        "right_elbow_joint": 1.2,
    },
    "Arms-down": {
        "left_elbow_joint": 1.5708,
        "right_elbow_joint": 1.5708,
    },
    "Arms-forward": {
        "left_shoulder_pitch_joint": -1.5708,
        "right_shoulder_pitch_joint": -1.5708,
        "left_elbow_joint": 1.5708,
        "right_elbow_joint": 1.5708,
    },
    "Squat": {
        "left_hip_pitch_joint": -1.0,
        "right_hip_pitch_joint": -1.0,
        "left_knee_joint": 1.5,
        "right_knee_joint": 1.5,
        "left_shoulder_roll_joint": 0.785,
        "right_shoulder_roll_joint": -0.785,
        "left_elbow_joint": 1.2,
        "right_elbow_joint": 1.2,
    },
    "Arms-up": {
        "left_shoulder_roll_joint": 2.4,
        "right_shoulder_roll_joint": -2.4,
        "left_elbow_joint": 1.5708,
        "right_elbow_joint": 1.5708,
    },
    "Half-T": {
        "left_shoulder_roll_joint": 1.2,
        "right_shoulder_roll_joint": -1.2,
        "left_elbow_joint": 1.5708,
        "right_elbow_joint": 1.5708,
    },
    "Walking": {
        "left_shoulder_roll_joint": 0.3,
        "right_shoulder_roll_joint": -0.3,
        "left_shoulder_pitch_joint": 0.4,
        "right_shoulder_pitch_joint": -0.4,
        "left_elbow_joint": 1.2,
        "right_elbow_joint": 1.2,
        "left_hip_pitch_joint": -0.5,
        "right_hip_pitch_joint": 0.3,
        "left_knee_joint": 0.8,
    },
    "Lunge": {
        "left_hip_pitch_joint": -1.2,
        "right_hip_pitch_joint": 0.4,
        "left_knee_joint": 1.3,
        "left_shoulder_roll_joint": 0.785,
        "right_shoulder_roll_joint": -0.785,
        "left_elbow_joint": 1.2,
        "right_elbow_joint": 1.2,
    },
}


# ---------------------------------------------------------------------------
# SMPL-X utilities
# ---------------------------------------------------------------------------


def _load_smplx_pose(
    body_model_path: str,
    body_pose: Optional[torch.Tensor] = None,
    gender: str = "neutral",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X body model with given body_pose.

    Returns positions and quaternions in **Y-up** frame (native SMPL-X).
    """
    body_model = smplx.create(
        body_model_path, model_type="smplx", gender=gender, use_pca=False
    )
    if body_pose is None:
        body_pose = torch.zeros(1, 63).float()

    shapedirs = getattr(body_model, "shapedirs", None)
    num_betas = int(shapedirs.shape[-1]) if shapedirs is not None else 10

    out = body_model(
        betas=torch.zeros(1, num_betas).float(),
        global_orient=torch.zeros(1, 3).float(),
        body_pose=body_pose,
        transl=torch.zeros(1, 3).float(),
        expression=torch.zeros(1, body_model.num_expression_coeffs).float(),
        left_hand_pose=torch.zeros(1, 45).float(),
        right_hand_pose=torch.zeros(1, 45).float(),
        jaw_pose=torch.zeros(1, 3).float(),
        leye_pose=torch.zeros(1, 3).float(),
        reye_pose=torch.zeros(1, 3).float(),
        return_full_pose=True,
    )
    joints = out.joints[0].detach().numpy()

    from smplx.joint_names import JOINT_NAMES

    parents = body_model.parents
    names = JOINT_NAMES[: len(parents)]

    full_pose = out.full_pose.reshape(1, -1, 3)[0].detach().numpy()
    global_rots: List[R] = []
    for j in range(len(names)):
        if j == 0:
            rot = R.from_rotvec(full_pose[j])
        else:
            rot = global_rots[parents[j]] * R.from_rotvec(full_pose[j])
        global_rots.append(rot)

    positions = {}
    quaternions = {}  # wxyz
    for i, name in enumerate(names):
        if i >= 22:
            break
        positions[name] = joints[i].copy()
        xyzw = global_rots[i].as_quat()
        quaternions[name] = np.array(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32
        )

    return positions, quaternions, names[:22]


def load_smplx_tpose(body_model_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X T-pose (all zeros). Returns Y-up positions and wxyz quats."""
    return _load_smplx_pose(body_model_path)


def load_smplx_apose(
    body_model_path: str, shoulder_angle: float = 0.6,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X A-pose (shoulders rotated down ~34deg). Returns Y-up."""
    body_pose = torch.zeros(1, 63).float()
    body_pose[0, 45:48] = torch.tensor([0, 0, -shoulder_angle])
    body_pose[0, 48:51] = torch.tensor([0, 0, shoulder_angle])
    return _load_smplx_pose(body_model_path, body_pose=body_pose)


def load_smplx_arms_down(
    body_model_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X arms-down pose (natural standing). Returns Y-up."""
    body_pose = torch.zeros(1, 63).float()
    body_pose[0, 45:48] = torch.tensor([0, 0, -1.57])
    body_pose[0, 48:51] = torch.tensor([0, 0, 1.57])
    return _load_smplx_pose(body_model_path, body_pose=body_pose)


def load_smplx_arms_forward(
    body_model_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X arms-forward pose (reaching forward). Returns Y-up."""
    body_pose = torch.zeros(1, 63).float()
    body_pose[0, 45:48] = torch.tensor([0, -1.57, 0])
    body_pose[0, 48:51] = torch.tensor([0, 1.57, 0])
    return _load_smplx_pose(body_model_path, body_pose=body_pose)


def load_smplx_squat(
    body_model_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X squat pose (knees bent). Returns Y-up."""
    body_pose = torch.zeros(1, 63).float()
    body_pose[0, 0:3] = torch.tensor([-1.0, 0, 0])
    body_pose[0, 3:6] = torch.tensor([-1.0, 0, 0])
    body_pose[0, 9:12] = torch.tensor([1.5, 0, 0])
    body_pose[0, 12:15] = torch.tensor([1.5, 0, 0])
    body_pose[0, 45:48] = torch.tensor([0, 0, -0.6])
    body_pose[0, 48:51] = torch.tensor([0, 0, 0.6])
    return _load_smplx_pose(body_model_path, body_pose=body_pose)


def load_smplx_arms_up(
    body_model_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X arms-up pose (overhead reach). Returns Y-up."""
    body_pose = torch.zeros(1, 63).float()
    body_pose[0, 45:48] = torch.tensor([0, 0, 1.4])
    body_pose[0, 48:51] = torch.tensor([0, 0, -1.4])
    return _load_smplx_pose(body_model_path, body_pose=body_pose)


def load_smplx_walking(
    body_model_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X walking pose (asymmetric). Returns Y-up."""
    body_pose = torch.zeros(1, 63).float()
    body_pose[0, 0:3] = torch.tensor([-0.5, 0, 0])
    body_pose[0, 3:6] = torch.tensor([0.3, 0, 0])
    body_pose[0, 9:12] = torch.tensor([0.8, 0, 0])
    body_pose[0, 45:48] = torch.tensor([0, 0.4, -0.8])
    body_pose[0, 48:51] = torch.tensor([0, 0.4, 0.8])
    body_pose[0, 51:54] = torch.tensor([0.6, 0, 0])
    body_pose[0, 54:57] = torch.tensor([0.6, 0, 0])
    return _load_smplx_pose(body_model_path, body_pose=body_pose)


def load_smplx_lunge(
    body_model_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X lunge pose (deep split stance). Returns Y-up."""
    body_pose = torch.zeros(1, 63).float()
    body_pose[0, 0:3] = torch.tensor([-1.2, 0, 0])
    body_pose[0, 3:6] = torch.tensor([0.4, 0, 0])
    body_pose[0, 9:12] = torch.tensor([1.3, 0, 0])
    body_pose[0, 45:48] = torch.tensor([0, 0, -0.6])
    body_pose[0, 48:51] = torch.tensor([0, 0, 0.6])
    return _load_smplx_pose(body_model_path, body_pose=body_pose)


def load_smplx_half_tpose(
    body_model_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SMPL-X half-T pose (arms at ~45deg up from horizontal). Returns Y-up."""
    body_pose = torch.zeros(1, 63).float()
    body_pose[0, 45:48] = torch.tensor([0, 0, 0.7])
    body_pose[0, 48:51] = torch.tensor([0, 0, -0.7])
    return _load_smplx_pose(body_model_path, body_pose=body_pose)


# SMPL-X pose loaders indexed by name
_SMPLX_POSE_LOADERS = {
    "T-pose": load_smplx_tpose,
    "A-pose": load_smplx_apose,
    "Arms-down": load_smplx_arms_down,
    "Arms-forward": load_smplx_arms_forward,
    "Squat": load_smplx_squat,
    "Arms-up": load_smplx_arms_up,
    "Half-T": load_smplx_half_tpose,
    "Walking": load_smplx_walking,
    "Lunge": load_smplx_lunge,
}


# ---------------------------------------------------------------------------
# G1 FK via MuJoCo — placed in the SMPL-X native frame
# ---------------------------------------------------------------------------


def _set_joint(m, d, joint_name: str, value: float):
    """Set a joint's qpos value by name (no-op if not found)."""
    for i in range(m.njnt):
        if m.joint(i).name == joint_name:
            d.qpos[m.joint(i).qposadr] = value
            return


def _extract_body_positions(m, d) -> Dict[str, np.ndarray]:
    """Extract all body positions from MuJoCo after forward kinematics."""
    positions = {}
    for i in range(m.nbody):
        name = m.body(i).name
        if name:
            positions[name] = d.xpos[i].copy().astype(np.float32)
    return positions


def load_g1_in_smplx_frame(
    xml_path: str,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Load G1 FK with root placed in the SMPL-X frame.

    This matches exactly how the IK solver sees the robot at runtime:
    - T_world_base.xyz = root_pos (from SMPL-X, after scale+offset)
    - T_world_base.quat = smplx_pelvis_quat * rotation_offset_pelvis

    For T-pose calibration with identity smplx_pelvis_quat:
      root_quat = rotation_offset_pelvis
    """
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat_wxyz

    if joint_overrides:
        for joint_name, value in joint_overrides.items():
            _set_joint(m, d, joint_name, value)

    mujoco.mj_forward(m, d)
    return _extract_body_positions(m, d)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_link_mapping(
    config: dict,
) -> List[Tuple[str, str, np.ndarray, np.ndarray, float, float]]:
    """Returns list of (robot_link, human_joint, pos_offset, rot_offset, pos_w, ori_w)."""
    entries = []
    for robot_link, entry in config.get("link_mapping", {}).items():
        entries.append((
            robot_link,
            entry["human_joint"],
            np.array(entry["position_offset"], dtype=np.float32),
            np.array(entry["rotation_offset"], dtype=np.float32),
            float(entry["position_weight"]),
            float(entry["orientation_weight"]),
        ))
    return entries


# ---------------------------------------------------------------------------
# Robokit math — numpy (exact copy of runtime logic)
# ---------------------------------------------------------------------------


def _quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    result = r1 * r2
    xyzw = result.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _rotate_vector_np(v: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz)."""
    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot.apply(v).astype(np.float32)


def apply_scale_and_offset_np(
    smplx_positions: Dict[str, np.ndarray],
    smplx_quaternions: Dict[str, np.ndarray],
    root_name: str,
    scale_table: Dict[str, float],
    link_mapping: List[Tuple[str, str, np.ndarray, np.ndarray, float, float]],
) -> Dict[str, np.ndarray]:
    """Apply Robokit's scale_human_frame + apply_offsets.

    This is the EXACT same math as the runtime pipeline.
    Returns {robot_link: target_position} in the SMPL-X native frame.
    """
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
# PyTorch differentiable forward (matches runtime math exactly)
# ---------------------------------------------------------------------------


def _rotate_vector_by_matrix(v: torch.Tensor, R_mat: torch.Tensor) -> torch.Tensor:
    return R_mat @ v


def build_rotation_matrices(
    smplx_quats: Dict[str, np.ndarray],
    rot_offsets: Dict[str, np.ndarray],
) -> Dict[str, torch.Tensor]:
    """Pre-compute rotation matrices: combined = smplx_quat * rot_offset."""
    result = {}
    for human_joint, rot_offset_wxyz in rot_offsets.items():
        if human_joint not in smplx_quats:
            continue
        q_wxyz = smplx_quats[human_joint]
        if np.linalg.norm(rot_offset_wxyz) < 1e-6:
            r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        else:
            rot_n = rot_offset_wxyz / np.linalg.norm(rot_offset_wxyz)
            r = (
                R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                * R.from_quat([rot_n[1], rot_n[2], rot_n[3], rot_n[0]])
            )
        result[human_joint] = torch.tensor(r.as_matrix(), dtype=torch.float32)
    return result


def differentiable_forward(
    smplx_positions: Dict[str, torch.Tensor],
    rot_matrices: Dict[str, torch.Tensor],
    root_name: str,
    scale_params: Dict[str, torch.Tensor],
    offset_params: Dict[str, torch.Tensor],
    link_mapping: List[Tuple[str, str, np.ndarray, np.ndarray, float, float]],
) -> Dict[str, torch.Tensor]:
    """Differentiable version of scale + offset. Returns {robot_link: pos [3]}."""
    root_pos = smplx_positions[root_name]
    root_scale = scale_params.get(root_name, torch.tensor(1.0))

    result = {}
    for robot_link, human_joint, _, _, pw, ow in link_mapping:
        if human_joint not in smplx_positions:
            continue
        pos = smplx_positions[human_joint]
        scale = scale_params.get(human_joint, torch.tensor(1.0))

        if human_joint == root_name:
            scaled_pos = pos * scale
        else:
            relative = pos - root_pos
            scaled_root = root_pos * root_scale
            scaled_pos = scaled_root + relative * scale

        if robot_link in offset_params:
            offset = offset_params[robot_link]
            if human_joint in rot_matrices:
                global_offset = _rotate_vector_by_matrix(offset, rot_matrices[human_joint])
            else:
                global_offset = offset
            scaled_pos = scaled_pos + global_offset

        result[robot_link] = scaled_pos

    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def print_comparison(
    label: str,
    transformed: Dict[str, np.ndarray],
    g1_positions: Dict[str, np.ndarray],
    link_mapping: List[Tuple[str, str, np.ndarray, np.ndarray, float, float]],
):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  {'Robot Link':<30s} {'Error (m)':>10s}  {'Target pos':>30s}  {'G1 pos':>30s}")
    print(f"  {'-'*30} {'-'*10}  {'-'*30}  {'-'*30}")

    total_err = 0.0
    count = 0
    for robot_link, human_joint, _, _, pw, _ in link_mapping:
        if robot_link not in transformed or robot_link not in g1_positions:
            continue
        s = transformed[robot_link]
        g = g1_positions[robot_link]
        err = np.linalg.norm(s - g)
        total_err += err
        count += 1
        s_str = f"[{s[0]:7.4f}, {s[1]:7.4f}, {s[2]:7.4f}]"
        g_str = f"[{g[0]:7.4f}, {g[1]:7.4f}, {g[2]:7.4f}]"
        print(f"  {robot_link:<30s} {err:10.4f}  {s_str:>30s}  {g_str:>30s}")

    if count > 0:
        print(f"\n  Mean error: {total_err / count:.4f} m | Total: {total_err:.4f} m")


# ---------------------------------------------------------------------------
# End-effector weight map
# ---------------------------------------------------------------------------


def build_loss_weights(
    link_mapping: List[Tuple[str, str, np.ndarray, np.ndarray, float, float]],
    ee_weight: float = 1.0,
) -> Dict[str, float]:
    """Build per-link loss weights from config position_weight, with ee multiplier.

    Normalizes so the mean weight is 1.0, then multiplies end-effectors by ee_weight.
    """
    raw = {}
    for robot_link, _, _, _, pw, _ in link_mapping:
        raw[robot_link] = max(float(pw), 1.0)

    mean_w = sum(raw.values()) / len(raw) if raw else 1.0
    weights = {k: v / mean_w for k, v in raw.items()}

    for link in END_EFFECTOR_LINKS:
        if link in weights:
            weights[link] *= ee_weight

    return weights


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class ScaleOptimizer:
    """Optimize scale_table values only."""

    def __init__(self, poses, config, link_mapping, root_name="pelvis",
                 lr=0.01, reg_weight=0.01, ee_weight=1.0):
        self.link_mapping = link_mapping
        self.root_name = root_name
        self.reg_weight = reg_weight
        self.loss_weights = build_loss_weights(link_mapping, ee_weight)

        existing_scales = config.get("scale_table", {})
        self.scale_params: Dict[str, torch.Tensor] = {}
        joint_names = {hj for _, hj, _, _, _, _ in link_mapping}
        joint_names.add(root_name)
        for name in joint_names:
            self.scale_params[name] = torch.tensor(
                existing_scales.get(name, 1.0), dtype=torch.float32, requires_grad=True
            )

        self.pose_data = []
        for pose in poses:
            smplx_pos_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in pose["smplx_pos"].items()}
            g1_pos_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in pose["g1_pos"].items()}
            rot_offsets = {hj: ro for _, hj, _, ro, _, _ in link_mapping}
            rot_matrices = build_rotation_matrices(pose["smplx_quat"], rot_offsets)
            self.pose_data.append((smplx_pos_t, g1_pos_t, rot_matrices))

        self.optimizer = torch.optim.Adam(list(self.scale_params.values()), lr=lr)

    def optimize(self, num_iters: int = 500) -> Dict[str, float]:
        empty_offsets: Dict[str, torch.Tensor] = {}
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for smplx_pos_t, g1_pos_t, rot_matrices in self.pose_data:
                pred = differentiable_forward(smplx_pos_t, rot_matrices, self.root_name,
                                              self.scale_params, empty_offsets, self.link_mapping)
                for rl in pred:
                    if rl in g1_pos_t:
                        w = self.loss_weights.get(rl, 1.0)
                        loss = loss + w * torch.sum((pred[rl] - g1_pos_t[rl]) ** 2)
            for param in self.scale_params.values():
                loss = loss + self.reg_weight * (param - 1.0) ** 2
            loss.backward()
            self.optimizer.step()
            if (it + 1) % 100 == 0 or it == 0:
                print(f"  Iter {it+1:4d}: loss = {loss.item():.6f}")
        return {k: v.item() for k, v in self.scale_params.items()}


class JointOptimizer:
    """Jointly optimize scale_table + position_offset."""

    def __init__(self, poses, config, link_mapping, root_name="pelvis",
                 lr=0.005, scale_reg=0.01, offset_reg=0.1, symmetry_weight=0.5,
                 ee_weight=1.0):
        self.link_mapping = link_mapping
        self.root_name = root_name
        self.scale_reg = scale_reg
        self.offset_reg = offset_reg
        self.symmetry_weight = symmetry_weight
        self.loss_weights = build_loss_weights(link_mapping, ee_weight)

        existing_scales = config.get("scale_table", {})
        self.scale_params: Dict[str, torch.Tensor] = {}
        joint_names = {hj for _, hj, _, _, _, _ in link_mapping}
        joint_names.add(root_name)
        for name in joint_names:
            self.scale_params[name] = torch.tensor(
                existing_scales.get(name, 1.0), dtype=torch.float32, requires_grad=True)

        self.offset_params: Dict[str, torch.Tensor] = {}
        for robot_link, _, pos_off, _, _, _ in link_mapping:
            self.offset_params[robot_link] = torch.tensor(
                pos_off, dtype=torch.float32, requires_grad=True)

        self.symmetric_scale_pairs = _find_symmetric_pairs(list(self.scale_params.keys()))
        self.symmetric_offset_pairs = _find_symmetric_pairs(list(self.offset_params.keys()))

        self.pose_data = []
        for pose in poses:
            smplx_pos_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in pose["smplx_pos"].items()}
            g1_pos_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in pose["g1_pos"].items()}
            rot_offsets = {hj: ro for _, hj, _, ro, _, _ in link_mapping}
            rot_matrices = build_rotation_matrices(pose["smplx_quat"], rot_offsets)
            self.pose_data.append((smplx_pos_t, g1_pos_t, rot_matrices))

        all_params = list(self.scale_params.values()) + list(self.offset_params.values())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    def optimize(self, num_iters=1000):
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = torch.tensor(0.0)

            for smplx_pos_t, g1_pos_t, rot_matrices in self.pose_data:
                pred = differentiable_forward(smplx_pos_t, rot_matrices, self.root_name,
                                              self.scale_params, self.offset_params, self.link_mapping)
                for rl in pred:
                    if rl in g1_pos_t:
                        w = self.loss_weights.get(rl, 1.0)
                        loss = loss + w * torch.sum((pred[rl] - g1_pos_t[rl]) ** 2)

            for param in self.scale_params.values():
                loss = loss + self.scale_reg * (param - 1.0) ** 2
            for param in self.offset_params.values():
                loss = loss + self.offset_reg * torch.sum(param ** 2)
            for l, r in self.symmetric_scale_pairs:
                loss = loss + self.symmetry_weight * (self.scale_params[l] - self.scale_params[r]) ** 2
            for l, r in self.symmetric_offset_pairs:
                lo, ro = self.offset_params[l], self.offset_params[r]
                loss = loss + self.symmetry_weight * torch.sum((lo + ro) ** 2)

            loss.backward()
            self.optimizer.step()
            if (it + 1) % 200 == 0 or it == 0:
                print(f"  Iter {it+1:4d}: loss = {loss.item():.6f}")

        scales = {k: v.item() for k, v in self.scale_params.items()}
        offsets = {k: v.detach().numpy() for k, v in self.offset_params.items()}
        return scales, offsets


def _find_symmetric_pairs(names):
    pairs, seen = [], set()
    for name in names:
        if name in seen:
            continue
        mirror = None
        if "left" in name:
            mirror = name.replace("left", "right")
        elif "right" in name:
            mirror = name.replace("right", "left")
        if mirror and mirror in names and mirror != name:
            pairs.append((name, mirror))
            seen.update([name, mirror])
    return pairs


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def print_yaml_output(config, optimized_scales=None, optimized_offsets=None):
    print("\n" + "=" * 70)
    print("  YAML Config Output (copy into your config)")
    print("=" * 70)

    if optimized_scales:
        print("\nscale_table:")
        for name, val in sorted(optimized_scales.items()):
            print(f"  {name}: {val:.4f}")

    if optimized_offsets:
        print("\nlink_mapping:")
        for robot_link, entry in config.get("link_mapping", {}).items():
            offset = optimized_offsets.get(robot_link, entry["position_offset"])
            if isinstance(offset, np.ndarray):
                offset = offset.tolist()
            rot_off = entry["rotation_offset"]
            print(f"  {robot_link}:")
            print(f"    human_joint: {entry['human_joint']}")
            print(f"    position_weight: {entry['position_weight']}")
            print(f"    orientation_weight: {entry['orientation_weight']}")
            print(f"    position_offset: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
            print(f"    rotation_offset: [{rot_off[0]}, {rot_off[1]}, {rot_off[2]}, {rot_off[3]}]")


def save_calibrated_config(config_path, output_path, optimized_scales=None, optimized_offsets=None):
    config = load_config(config_path)
    if optimized_scales:
        config["scale_table"] = {k: round(float(v), 4) for k, v in optimized_scales.items()}
    if optimized_offsets:
        for rl, offset in optimized_offsets.items():
            if rl in config.get("link_mapping", {}):
                if isinstance(offset, np.ndarray):
                    offset = offset.tolist()
                config["link_mapping"][rl]["position_offset"] = [round(float(x), 4) for x in offset]
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved calibrated config to: {output_path}")


# ---------------------------------------------------------------------------
# Pose preparation
# ---------------------------------------------------------------------------


def prepare_pose(
    label: str,
    smplx_pos: Dict[str, np.ndarray],
    smplx_quat: Dict[str, np.ndarray],
    xml_path: str,
    config: dict,
    link_mapping: List[Tuple[str, str, np.ndarray, np.ndarray, float, float]],
    g1_joint_overrides: Optional[Dict[str, float]] = None,
) -> dict:
    """Prepare a calibration pose by placing G1 in the SMPL-X frame.

    Key insight: at runtime, the IK solver places the robot at the SMPL-X
    root with quat = smplx_pelvis_quat * rotation_offset_pelvis. For T-pose
    (smplx_pelvis_quat = identity), the base quat IS the rotation_offset.
    """
    root_name = config.get("human_root_name", "pelvis")

    root_rot_offset = np.array([1, 0, 0, 0], dtype=np.float32)
    for _, hj, _, rot_off, _, _ in link_mapping:
        if hj == root_name:
            root_rot_offset = rot_off
            break

    scale_table = config.get("scale_table", {})
    root_scale = scale_table.get(root_name, 1.0)
    scaled_root_pos = smplx_pos[root_name] * root_scale

    root_quat = _quat_multiply_np(smplx_quat[root_name], root_rot_offset)

    g1_positions = load_g1_in_smplx_frame(
        xml_path, scaled_root_pos, root_quat, joint_overrides=g1_joint_overrides,
    )

    return {
        "label": label,
        "smplx_pos": smplx_pos,
        "smplx_quat": smplx_quat,
        "g1_pos": g1_positions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_mujoco_xml(config_path: str) -> str:
    """Find G1 MuJoCo XML from the config's URDF path."""
    config = load_config(config_path)
    if "urdf_path" not in config:
        raise ValueError("Config must have urdf_path")

    urdf_resolved = (Path(config_path).parent / config["urdf_path"]).resolve()
    urdf_dir = urdf_resolved.parent

    # Look for g1_mocap_29dof.xml in same dir or sibling mjcf/ dir
    for search_dir in [urdf_dir, urdf_dir.parent]:
        for xml_path in sorted(search_dir.rglob("g1_mocap_29dof.xml")):
            return str(xml_path)

    # Fallback: any XML with same stem as URDF
    candidate = urdf_dir / f"{urdf_resolved.stem}.xml"
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(
        f"No G1 MuJoCo XML found near '{urdf_resolved}'. "
        f"Expected g1_mocap_29dof.xml in the robot description directory."
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate Robokit retargeting config for Unitree G1")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--optimize-scales", action="store_true")
    parser.add_argument("--optimize-offsets", action="store_true")
    parser.add_argument("--optimize-all", action="store_true")
    parser.add_argument("--multi-pose", action="store_true",
                        help="Use all 9 calibration poses (otherwise T-pose only)")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--ee-weight", type=float, default=3.0,
                        help="Extra weight multiplier for end-effectors (wrists+feet)")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--body-model-path", type=str, default=str(BODY_MODEL_DIR))
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    root_name = config.get("human_root_name", "pelvis")
    link_mapping = get_link_mapping(config)
    xml_path = find_mujoco_xml(config_path)

    print(f"Config: {config_path}")
    print(f"MuJoCo XML: {xml_path}")
    print(f"Body model: {args.body_model_path}")
    print(f"Root joint: {root_name}")
    print(f"Link mappings: {len(link_mapping)}")
    print(f"EE weight: {args.ee_weight}")
    print(f"\nFrame: SMPL-X native (Y-up) — robot placed via rotation_offset")

    # ---- Load poses ----
    poses = []
    pose_names = ["T-pose"] if not args.multi_pose else list(_SMPLX_POSE_LOADERS.keys())

    for pose_name in pose_names:
        print(f"Loading SMPL-X {pose_name}...")
        loader = _SMPLX_POSE_LOADERS[pose_name]
        smplx_pos, smplx_quat, _ = loader(args.body_model_path)
        joint_overrides = G1_POSE_OVERRIDES.get(pose_name, {})
        poses.append(prepare_pose(
            pose_name, smplx_pos, smplx_quat, xml_path, config, link_mapping,
            g1_joint_overrides=joint_overrides,
        ))

    # ---- Show current errors ----
    print("\n--- Current config errors ---")
    for pose in poses:
        transformed = apply_scale_and_offset_np(
            pose["smplx_pos"], pose["smplx_quat"], root_name,
            config.get("scale_table", {}), link_mapping,
        )
        print_comparison(f"{pose['label']} (before)", transformed, pose["g1_pos"], link_mapping)

    # ---- Optimize ----
    optimized_scales = None
    optimized_offsets = None

    if args.optimize_scales:
        print("\n--- Optimizing scale_table ---")
        opt = ScaleOptimizer(poses, config, link_mapping, root_name, lr=args.lr,
                             ee_weight=args.ee_weight)
        optimized_scales = opt.optimize(args.iters)

    elif args.optimize_offsets:
        print("\n--- Optimizing position_offset ---")
        opt = JointOptimizer(poses, config, link_mapping, root_name, lr=args.lr,
                             scale_reg=0, offset_reg=0.1, ee_weight=args.ee_weight)
        for p in opt.scale_params.values():
            p.requires_grad_(False)
        _, optimized_offsets = opt.optimize(args.iters)

    elif args.optimize_all:
        print("\n--- Jointly optimizing scale_table + position_offset ---")
        opt = JointOptimizer(poses, config, link_mapping, root_name, lr=args.lr,
                             ee_weight=args.ee_weight)
        optimized_scales, optimized_offsets = opt.optimize(args.iters)

    # ---- Show optimized errors ----
    if optimized_scales or optimized_offsets:
        new_scale_table = config.get("scale_table", {}).copy()
        if optimized_scales:
            new_scale_table.update(optimized_scales)

        new_mapping = []
        for rl, hj, po, ro, pw, ow in link_mapping:
            if optimized_offsets and rl in optimized_offsets:
                po = optimized_offsets[rl]
            new_mapping.append((rl, hj, po, ro, pw, ow))

        print("\n--- Optimized errors ---")
        for pose in poses:
            transformed = apply_scale_and_offset_np(
                pose["smplx_pos"], pose["smplx_quat"], root_name,
                new_scale_table, new_mapping,
            )
            print_comparison(f"{pose['label']} (after)", transformed, pose["g1_pos"], new_mapping)

        print_yaml_output(config, optimized_scales, optimized_offsets)

        if args.save:
            save_calibrated_config(config_path, args.save, optimized_scales, optimized_offsets)
    else:
        print("\nNo optimization requested. Use --optimize-scales, --optimize-offsets, or --optimize-all.")


if __name__ == "__main__":
    main()
