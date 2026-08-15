"""Calibrate a Robokit humanoid retargeting config (scale_table + position_offset).

Universal across robots: everything robot-specific comes from the retarget preset
(--robot g1|gr3|h1_2|bhl: urdf_path, link_mapping, rotation offsets) or from the
pose-pair data (poses/<robot>.yaml, built interactively with build_poses.py / --edit).

Compares SMPL-X body model poses against robot FK in the RUNTIME frame: Z-up,
standing on the ground plane z=0 (the same convention real motion data uses).
Both the human (mesh min-z grounded) and the robot (foot-sole collision geometry
grounded) stand on the floor, so the learned root scale and offsets encode the
true robot-vs-human stature ratio — the retargeted robot keeps its feet on the
ground instead of floating at the human's pelvis height. The calibration also
measures the calibration body's T-pose height and exports it as
``human_height_assumption``, which the runtime compares with each motion's
``human_heights`` input.

Usage (run from repo root: robokit-internal/):

    # View current errors only (no optimization); uses presets.g1 as default config
    uv run python examples/humanoid_retarget/calibration/calibrate.py

    # Full calibration with auto-save
    uv run python examples/humanoid_retarget/calibration/calibrate.py \
        --optimize-all --multi-pose --iters 3000 \
        --save /tmp/smplx_g1_calibrated.yaml

    # Another robot: build its pose pairs in the browser, then calibrate, in one run
    uv run python examples/humanoid_retarget/calibration/calibrate.py \
        --robot gr3 --edit --optimize-all --multi-pose --iters 3000 --save /tmp/gr3.yaml

Options:
    --robot NAME          Robot preset: g1, gr3, h1_2, bhl (default: g1)
    --config PATH         Input YAML config (overrides --robot)
    --optimize-scales     Optimize scale_table only
    --optimize-offsets    Optimize position_offset only
    --optimize-all        Optimize both scale_table and position_offset (recommended)
    --multi-pose          Use all 9 calibration poses (recommended, otherwise T-pose only)
    --iters N             Optimization iterations (default: 1000, recommended: 3000)
    --lr FLOAT            Learning rate (default: 0.005)
    --ee-weight FLOAT     End-effector weight multiplier (default: 3.0)
    --save PATH           Save calibrated config to file
    --body-model-path     SMPL-X body model directory
    --poses PATH          Pose-pair YAML from build_poses.py (default: poses/<robot>.yaml)
    --precheck PATH.png   Render all pose pairs + audit table, then exit (no optimization)
    --edit                Open the viser pose editor first; continue after 'Apply & continue'
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import warp as wp
import yaml
from scipy.spatial.transform import Rotation as R

from robokit.helpers.humanoid_retarget.presets.bhl import bhl
from robokit.helpers.humanoid_retarget.presets.g1 import g1
from robokit.helpers.humanoid_retarget.presets.gr3 import gr3
from robokit.helpers.humanoid_retarget.presets.h1_2 import h1_2
from robokit.robo import Robot
from robokit.smplx import (
    SMPLX_JOINT_NAMES,
    BodyModelState,
    body_lbs_torch,
    load_smplx,
)
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_apply, quaternion_multiply


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PRESETS_DIR = SCRIPT_DIR.parents[2] / "src/robokit/helpers/humanoid_retarget/presets"


def _body_model_dir() -> Path:
    from robokit.assets import fetch

    return fetch(["body_models/**"]) / "body_models"


# base retarget presets a calibration can start from (same names as gallery --robot)
PRESETS = {"g1": g1, "gr3": gr3, "h1_2": h1_2, "bhl": bhl}

# human joints whose mapped robot links count as end-effectors (extra loss weighting)
EE_HUMAN_JOINTS = {"left_wrist", "right_wrist", "left_foot", "right_foot"}


def _bp(*edits: Tuple[int, List[float]]) -> np.ndarray:
    """Build a [63] SMPL-X body_pose axis-angle vector from sparse (start_idx, xyz) edits.

    Indices: hips 0/3, knees 9/12, shoulders 45/48, elbows 51/54 (left/right).
    """
    bp = np.zeros(63, dtype=np.float32)
    for start, vals in edits:
        bp[start : start + 3] = vals
    return bp


# The 9 calibration poses — SMPL-X side only, robot-independent. Robot joint
# overrides pair with these per robot (poses/<robot>.yaml from build_poses.py, or
# the curated G1 set below). A pair must put BOTH bodies in the same physical
# configuration — the loss matches human keypoints onto the robot's FIXED FK, so
# a mismatched pair biases the learned scales/offsets. The robot's joint limits
# are the binding constraint (e.g. G1 shoulder_roll tops out at 2.2515, so the
# lateral raises are specced to what the robot can reach).
SMPLX_POSES: Dict[str, np.ndarray] = {
    "T-pose": _bp(),
    "A-pose": _bp((45, [0, 0, -0.6]), (48, [0, 0, 0.6])),
    "Arms-down": _bp((45, [0, 0, -1.57]), (48, [0, 0, 1.57])),
    "Arms-forward": _bp((45, [0, -1.57, 0]), (48, [0, 1.57, 0])),
    "Squat": _bp(
        (0, [-1, 0, 0]), (3, [-1, 0, 0]), (9, [1.5, 0, 0]), (12, [1.5, 0, 0]), (45, [0, 0, -0.6]), (48, [0, 0, 0.6])
    ),
    "Arms-up": _bp((45, [0, 0, 0.6]), (48, [0, 0, -0.6])),
    "Half-T": _bp((45, [0, 0, 0.35]), (48, [0, 0, -0.35])),
    "Walking": _bp((0, [-0.5, 0, 0]), (3, [0.3, 0, 0]), (9, [0.8, 0, 0]), (45, [0, 0.4, -0.8]), (48, [0, 0.4, 0.8])),
    "Lunge": _bp((0, [-1.2, 0, 0]), (3, [0.4, 0, 0]), (9, [1.3, 0, 0]), (45, [0, 0, -0.6]), (48, [0, 0, 0.6])),
}

# limb segments compared between the human skeleton and the robot links, with the
# structural floor (deg) each comparison carries even for a perfectly matched pair
# (SMPL joint centers sit inside the body, robot link origins on the structure;
# floors were measured on the G1 — treat them as heuristics for other robots).
# A pair is suspicious when its angle exceeds floor + 15°. Bones whose human
# joints are not in the robot's link_mapping are skipped automatically.
BONE_PAIRS: Tuple[Tuple[str, str, str, float], ...] = (
    ("L uarm", "left_shoulder", "left_elbow", 26.0),
    ("L farm", "left_elbow", "left_wrist", 6.0),
    ("R uarm", "right_shoulder", "right_elbow", 19.0),
    ("R farm", "right_elbow", "right_wrist", 6.0),
    ("L thigh", "left_hip", "left_knee", 9.0),
    ("L shin", "left_knee", "left_foot", 9.0),
    ("R thigh", "right_hip", "right_knee", 9.0),
    ("R shin", "right_knee", "right_foot", 9.0),
)


def mapped_bones(hj2rl: Dict[str, str]) -> Tuple[Tuple[str, str, str, float], ...]:
    """BONE_PAIRS entries whose human joints are both in the robot's link_mapping."""
    return tuple(bone for bone in BONE_PAIRS if bone[1] in hj2rl and bone[2] in hj2rl)


def bone_angles(pose: dict, hj2rl: Dict[str, str]) -> Dict[str, float]:
    """Human-vs-robot limb direction mismatch (deg) per mapped BONE_PAIRS entry.

    Calibration-independent: unlike the keypoint residual, scales/offsets cannot
    absorb a direction error, so this is the metric that exposes a mismatched
    pose pair (e.g. human arms raised while the robot's are lowered).
    """
    out = {}
    for label, a, b, _ in mapped_bones(hj2rl):
        hd = pose["smplx_pos"][b] - pose["smplx_pos"][a]
        rd = pose["robot_pos"][hj2rl[b]] - pose["robot_pos"][hj2rl[a]]
        cosang = np.dot(hd / np.linalg.norm(hd), rd / np.linalg.norm(rd))
        out[label] = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    return out


def load_pose_pairs(path: Optional[str], robot_name: str = "g1") -> Dict[str, Tuple[np.ndarray, Dict[str, float]]]:
    """Calibration pose pairs for a robot.

    Resolution order: explicit YAML ``path`` > ``poses/<robot_name>.yaml`` (the
    robot side of the pairs, built by ``build_poses.py``) > the SMPL-X poses with
    EMPTY robot overrides (bootstrap a new robot with ``--edit`` or ``--ik-bootstrap``).

    YAML format: ``poses: {name: {body_pose_aa: [63 floats], joint_overrides: {joint: val}}}``.
    """
    if path is None:
        robot_yaml = SCRIPT_DIR / "poses" / f"{robot_name}.yaml"
        if robot_yaml.is_file():
            path = str(robot_yaml)
        else:
            print(
                f"No pose pairs for '{robot_name}' yet — starting from zero joint overrides. "
                f"Build them with --edit (auto-saves to poses/{robot_name}.yaml)."
            )
            return {name: (bp, {}) for name, bp in SMPLX_POSES.items()}
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {
        name: (np.array(e["body_pose_aa"], dtype=np.float32), dict(e["joint_overrides"]))
        for name, e in raw["poses"].items()
    }


# ---------------------------------------------------------------------------
# SMPL-X utilities
# ---------------------------------------------------------------------------


def _resolve_smplx_npz(body_model_path: str, gender: str) -> Path:
    """Accept either a ``SMPLX_*.npz`` file path or a directory containing one."""
    p = Path(body_model_path)
    if p.is_file():
        return p
    npz_name = f"SMPLX_{gender.upper()}.npz"
    candidates = [p / npz_name, p / "smplx" / npz_name]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"SMPL-X model not found. Looked at: {candidates}")


# SMPL-X native frame is Y-up; the runtime (AMASS motion data) is Z-up. Rotating
# every calibration pose by Rx(90°) puts the calibration in the runtime frame.
# Left-multiplying the global joint rotations is consistent with the runtime
# because all rotation_offsets are RIGHT-multiplied onto the human quaternions.
_R_ZUP = R.from_euler("x", 90, degrees=True)


def _load_smplx_pose(
    body_model_path: str,
    body_pose_aa: np.ndarray,
    gender: str = "neutral",
    return_mesh: bool = False,
):
    """Load SMPL-X body model with the given [63] body_pose axis-angle vector.

    Returns positions and quaternions in the runtime frame: **Z-up**, grounded so
    the lowest mesh vertex sits at z=0 (mesh grounding handles pitched feet and
    picks the stance foot in single-stance poses). When ``return_mesh`` is set,
    also returns the posed ``(vertices [V,3], faces [F,3])``.
    """
    npz_path = _resolve_smplx_npz(body_model_path, gender)
    spec = load_smplx(npz_path, gender=gender, num_betas=10)

    num_joints = spec.parents.shape[0]
    full_pose_aa = torch.zeros(1, num_joints, 3, dtype=torch.float32)
    full_pose_aa[:, 1:22] = torch.from_numpy(body_pose_aa.astype(np.float32)).reshape(1, 21, 3)

    state = BodyModelState(
        betas=torch.zeros(1, spec.num_betas, dtype=torch.float32),
        full_pose_aa=full_pose_aa,
        transl=torch.zeros(1, 3, dtype=torch.float32),
    )
    out = body_lbs_torch(spec, state, return_landmarks=False)
    joints = out["joints"][0].detach().numpy()
    verts = out["vertices"][0].detach().numpy().astype(np.float32)

    # Y-up -> Z-up: (x, y, z) -> (x, -z, y), then ground on the mesh.
    joints = np.stack([joints[:, 0], -joints[:, 2], joints[:, 1]], axis=1)
    verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)
    ground_z = verts[:, 2].min()
    joints[:, 2] -= ground_z
    verts[:, 2] -= ground_z

    parents = spec.parents
    names = list(SMPLX_JOINT_NAMES[: len(parents)])

    full_pose_np = full_pose_aa[0].detach().numpy()
    global_rots: List[R] = []
    for j in range(len(names)):
        if j == 0:
            rot = _R_ZUP * R.from_rotvec(full_pose_np[j])
        else:
            rot = global_rots[parents[j]] * R.from_rotvec(full_pose_np[j])
        global_rots.append(rot)

    positions = {}
    quaternions = {}  # wxyz
    for i, name in enumerate(names):
        if i >= 22:
            break
        positions[name] = joints[i].astype(np.float32).copy()
        xyzw = global_rots[i].as_quat(canonical=False)  # type: ignore[call-arg]
        quaternions[name] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)

    if return_mesh:
        return positions, quaternions, names[:22], verts, spec.faces
    return positions, quaternions, names[:22]


def measure_calibration_height(body_model_path: str) -> float:
    """T-pose mesh height of the calibration body (betas=0).

    Same number as ``smplx_loader._compute_tpose_height`` measures for a motion's
    betas — this is the reference used with the runtime ``human_heights`` input.
    """
    verts, _ = load_smplx_pose_mesh(body_model_path, "T-pose")
    return float(verts[:, 2].max() - verts[:, 2].min())


def load_smplx_pose_mesh(body_model_path: str, pose_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the posed SMPL-X ``(vertices [V,3], faces [F,3])`` for a built-in pose (Z-up, grounded)."""
    _, _, _, verts, faces = _load_smplx_pose(body_model_path, SMPLX_POSES[pose_name], return_mesh=True)
    return verts, faces


# ---------------------------------------------------------------------------
# robot FK via RoboKit Warp Robot — grounded in the runtime (Z-up) frame
# ---------------------------------------------------------------------------


def load_robot_grounded(
    robot: Robot,
    root_xy: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute robot link world positions with the robot standing on the z=0 plane.

    Uses robokit's own Warp forward kinematics (the same FK the runtime IK solver
    uses): FK once with the base at (root_xy, z=0) and quat =
    smplx_pelvis_quat * rotation_offset_pelvis, then shift the whole robot so its
    lowest geometry point sits at z=0. The lowest point of a standing/squatting
    humanoid IS the foot sole, so this needs no per-robot foot-link config; per
    link the collision scene is used when it has any geometry, else the visual
    scene (some URDFs lack collision on a foot — e.g. GR3's right foot).
    Requires ``Robot.load(..., load_meshes=True)``.

    Returns ({robot_link: world_position[3]}, base_pos[3]).
    """
    q = np.zeros(robot.spec.num_actuated_joints, dtype=np.float32)
    if joint_overrides:
        name_to_idx = {name: i for i, name in enumerate(robot.spec.actuated_joint_names)}
        for joint_name, value in joint_overrides.items():
            q[name_to_idx[joint_name]] = value

    base7 = np.concatenate([root_xy, [0.0], root_quat_wxyz]).astype(np.float32).reshape(1, 7)
    T_world_base = wp.from_numpy(base7, dtype=wp_vec7, device="cpu")
    state = robot.forward_kinematics(robot.state(q=q.reshape(1, -1), T_world_base=T_world_base))
    T = state.T_world_link.numpy()[0]  # (num_links, 7): xyz + wxyz

    sole_z = np.inf
    for i, name in enumerate(robot.spec.link_names):
        scene = robot.spec.link_collision_geometries[name]
        if scene.is_empty:
            scene = robot.spec.link_visual_geometries[name]
        if scene.is_empty:
            continue
        pose = T[i]
        mesh = scene.dump(concatenate=True)
        rot = R.from_quat([pose[4], pose[5], pose[6], pose[3]])  # wxyz -> xyzw
        world_verts = rot.apply(mesh.vertices) + pose[:3]
        sole_z = min(sole_z, float(world_verts[:, 2].min()))

    dz = np.array([0.0, 0.0, -sole_z], dtype=np.float32)
    positions = {name: T[i, :3].astype(np.float32) + dz for i, name in enumerate(robot.spec.link_names)}
    base_pos = base7[0, :3] + dz
    return positions, base_pos


# ---------------------------------------------------------------------------
# configuration loading
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
# RoboKit math — NumPy (exact copy of runtime logic)
# ---------------------------------------------------------------------------


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
            quat = quaternion_multiply(quat, rot_offset_n)

        if np.linalg.norm(pos_offset) > 1e-6:
            global_offset = quaternion_apply(quat, pos_offset)
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
            r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]) * R.from_quat(
                [rot_n[1], rot_n[2], rot_n[3], rot_n[0]]
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
# comparison
# ---------------------------------------------------------------------------


def print_comparison(
    label: str,
    transformed: Dict[str, np.ndarray],
    robot_positions: Dict[str, np.ndarray],
    link_mapping: List[Tuple[str, str, np.ndarray, np.ndarray, float, float]],
):
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"  {'Robot Link':<30s} {'Error (m)':>10s}  {'Target pos':>30s}  {'G1 pos':>30s}")
    print(f"  {'-' * 30} {'-' * 10}  {'-' * 30}  {'-' * 30}")

    total_err = 0.0
    count = 0
    for robot_link, human_joint, _, _, pw, _ in link_mapping:
        if robot_link not in transformed or robot_link not in robot_positions:
            continue
        s = transformed[robot_link]
        g = robot_positions[robot_link]
        err = np.linalg.norm(s - g)
        total_err += err
        count += 1
        s_str = f"[{s[0]:7.4f}, {s[1]:7.4f}, {s[2]:7.4f}]"
        g_str = f"[{g[0]:7.4f}, {g[1]:7.4f}, {g[2]:7.4f}]"
        print(f"  {robot_link:<30s} {err:10.4f}  {s_str:>30s}  {g_str:>30s}")

    if count > 0:
        print(f"\n  Mean error: {total_err / count:.4f} m | Total: {total_err:.4f} m")


# ---------------------------------------------------------------------------
# end-effector weight map
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

    for robot_link, human_joint, _, _, _, _ in link_mapping:
        if human_joint in EE_HUMAN_JOINTS and robot_link in weights:
            weights[robot_link] *= ee_weight

    return weights


# ---------------------------------------------------------------------------
# optimizers
# ---------------------------------------------------------------------------


class ScaleOptimizer:
    """Optimize scale_table values only."""

    def __init__(self, poses, config, link_mapping, root_name="pelvis", lr=0.01, reg_weight=0.01, ee_weight=1.0):
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
            robot_pos_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in pose["robot_pos"].items()}
            rot_offsets = {hj: ro for _, hj, _, ro, _, _ in link_mapping}
            rot_matrices = build_rotation_matrices(pose["smplx_quat"], rot_offsets)
            self.pose_data.append((smplx_pos_t, robot_pos_t, rot_matrices))

        from torch.optim.adam import Adam

        self.optimizer = Adam(list(self.scale_params.values()), lr=lr)

    def optimize(self, num_iters: int = 500) -> Dict[str, float]:
        empty_offsets: Dict[str, torch.Tensor] = {}
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for smplx_pos_t, robot_pos_t, rot_matrices in self.pose_data:
                pred = differentiable_forward(
                    smplx_pos_t, rot_matrices, self.root_name, self.scale_params, empty_offsets, self.link_mapping
                )
                for rl in pred:
                    if rl in robot_pos_t:
                        w = self.loss_weights.get(rl, 1.0)
                        loss = loss + w * torch.sum((pred[rl] - robot_pos_t[rl]) ** 2)
            for param in self.scale_params.values():
                loss = loss + self.reg_weight * (param - 1.0) ** 2
            loss.backward()
            self.optimizer.step()
            if (it + 1) % 100 == 0 or it == 0:
                print(f"  Iter {it + 1:4d}: loss = {loss.item():.6f}")
        return {k: v.item() for k, v in self.scale_params.items()}


class JointOptimizer:
    """Jointly optimize scale_table + position_offset."""

    def __init__(
        self,
        poses,
        config,
        link_mapping,
        root_name="pelvis",
        lr=0.005,
        scale_reg=0.01,
        offset_reg=0.1,
        symmetry_weight=0.5,
        ee_weight=1.0,
    ):
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
                existing_scales.get(name, 1.0), dtype=torch.float32, requires_grad=True
            )

        self.offset_params: Dict[str, torch.Tensor] = {}
        for robot_link, _, pos_off, _, _, _ in link_mapping:
            self.offset_params[robot_link] = torch.tensor(pos_off, dtype=torch.float32, requires_grad=True)

        self.symmetric_scale_pairs = _find_symmetric_pairs(list(self.scale_params.keys()))
        self.symmetric_offset_pairs = _find_symmetric_pairs(list(self.offset_params.keys()))

        self.pose_data = []
        for pose in poses:
            smplx_pos_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in pose["smplx_pos"].items()}
            robot_pos_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in pose["robot_pos"].items()}
            rot_offsets = {hj: ro for _, hj, _, ro, _, _ in link_mapping}
            rot_matrices = build_rotation_matrices(pose["smplx_quat"], rot_offsets)
            self.pose_data.append((smplx_pos_t, robot_pos_t, rot_matrices))

        from torch.optim.adam import Adam

        all_params = list(self.scale_params.values()) + list(self.offset_params.values())
        self.optimizer = Adam(all_params, lr=lr)

    def optimize(self, num_iters=1000, record_every: int = 0):
        # When record_every > 0, snapshot (iter, loss, scales, offsets) for the
        # gradient-descent visualization. The snapshot at each step lets a viewer
        # recompute the predicted targets and watch them converge onto the robot.
        self.history: List[dict] = []
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = torch.tensor(0.0)

            for smplx_pos_t, robot_pos_t, rot_matrices in self.pose_data:
                pred = differentiable_forward(
                    smplx_pos_t, rot_matrices, self.root_name, self.scale_params, self.offset_params, self.link_mapping
                )
                for rl in pred:
                    if rl in robot_pos_t:
                        w = self.loss_weights.get(rl, 1.0)
                        loss = loss + w * torch.sum((pred[rl] - robot_pos_t[rl]) ** 2)

            for param in self.scale_params.values():
                loss = loss + self.scale_reg * (param - 1.0) ** 2
            for param in self.offset_params.values():
                loss = loss + self.offset_reg * torch.sum(param**2)
            for l, r in self.symmetric_scale_pairs:
                loss = loss + self.symmetry_weight * (self.scale_params[l] - self.scale_params[r]) ** 2
            for l, r in self.symmetric_offset_pairs:
                lo, ro = self.offset_params[l], self.offset_params[r]
                loss = loss + self.symmetry_weight * torch.sum((lo + ro) ** 2)

            if record_every > 0 and (it % record_every == 0 or it == num_iters - 1):
                self.history.append(
                    {
                        "iter": it,
                        "loss": loss.item(),
                        "scales": {k: v.item() for k, v in self.scale_params.items()},
                        "offsets": {k: v.detach().numpy().copy() for k, v in self.offset_params.items()},
                    }
                )

            loss.backward()
            self.optimizer.step()
            if (it + 1) % 200 == 0 or it == 0:
                print(f"  Iter {it + 1:4d}: loss = {loss.item():.6f}")

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


def print_yaml_output(config, optimized_scales=None, optimized_offsets=None, measured_height: float = 0.0):
    print("\n" + "=" * 70)
    print("  YAML Config Output (copy into your config)")
    print("=" * 70)

    if measured_height > 0.0:
        print(f"\nhuman_height_assumption: {measured_height:.4f}")

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


def save_calibrated_config(
    config: dict, output_path: str, optimized_scales=None, optimized_offsets=None, measured_height: float = 0.0
):
    import copy

    config = copy.deepcopy(config)
    if measured_height > 0.0:
        config["human_height_assumption"] = round(measured_height, 4)
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
    return config


def emit_preset(robot_name: str, config: dict) -> str:
    """Render a ``<robot>_cali.py`` preset source from a calibrated config dict.

    Calibration learns ``scale_table`` + per-link ``position_offset``; the preset
    inherits everything else from the base preset via ``dataclasses.replace``.
    """
    offsets = "".join(
        f'    "{rl}": np.array({[round(float(x), 4) for x in m["position_offset"]]}, dtype=np.float32),\n'
        for rl, m in config["link_mapping"].items()
    )
    scales = "".join(f'        "{k}": {round(float(v), 4)},\n' for k, v in config["scale_table"].items())
    return (
        f'"""{robot_name} calibrated retargeting preset (generated by calibrate.py --emit-preset).\n\n'
        f"Calibration learns scale_table + per-link position_offset; everything else is\n"
        f'inherited from the base ``{robot_name}`` preset.\n"""\n\n'
        "import dataclasses\n\n"
        "import numpy as np\n\n"
        f"from robokit.helpers.humanoid_retarget.presets.{robot_name} import {robot_name}\n\n\n"
        f"_OFFSETS = {{\n{offsets}}}\n\n\n"
        f"{robot_name}_cali = dataclasses.replace(\n"
        f"    {robot_name},\n"
        f"    human_height_assumption={round(float(config['human_height_assumption']), 4)},\n"
        f"    scale_table={{\n{scales}    }},\n"
        "    link_mapping={\n"
        f"        link: dataclasses.replace({robot_name}.link_mapping[link], position_offset=offset)\n"
        "        for link, offset in _OFFSETS.items()\n"
        "    },\n"
        ")\n"
    )


# ---------------------------------------------------------------------------
# pose preparation
# ---------------------------------------------------------------------------


def prepare_pose(
    label: str,
    smplx_pos: Dict[str, np.ndarray],
    smplx_quat: Dict[str, np.ndarray],
    robot: Robot,
    config: dict,
    link_mapping: List[Tuple[str, str, np.ndarray, np.ndarray, float, float]],
    joint_overrides: Optional[Dict[str, float]] = None,
) -> dict:
    """Prepare a calibration pose: grounded human and grounded G1, both Z-up.

    The G1 base quat follows the runtime convention
    (smplx_pelvis_quat * rotation_offset_pelvis), its base xy sits at the human
    pelvis xy, and its z is whatever puts the foot soles on the ground — the
    optimization then learns the root scale/offset that map the human pelvis onto
    this grounded robot pelvis.
    """
    root_name = config.get("human_root_name", "pelvis")

    root_rot_offset = np.array([1, 0, 0, 0], dtype=np.float32)
    for _, hj, _, rot_off, _, _ in link_mapping:
        if hj == root_name:
            root_rot_offset = rot_off
            break

    root_quat = quaternion_multiply(smplx_quat[root_name], root_rot_offset)

    robot_positions, robot_base_pos = load_robot_grounded(
        robot,
        smplx_pos[root_name][:2],
        root_quat,
        joint_overrides=joint_overrides,
    )

    return {
        "label": label,
        "smplx_pos": smplx_pos,
        "smplx_quat": smplx_quat,
        "robot_pos": robot_positions,
        "robot_base_pos": robot_base_pos,
        "robot_base_quat": root_quat,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Calibrate a Robokit humanoid retargeting config")
    parser.add_argument("--robot", type=str, default="g1", choices=sorted(PRESETS), help="Robot preset to calibrate")
    parser.add_argument("--config", type=str, default=None, help="Input config YAML (overrides --robot)")
    parser.add_argument("--optimize-scales", action="store_true")
    parser.add_argument("--optimize-offsets", action="store_true")
    parser.add_argument("--optimize-all", action="store_true")
    parser.add_argument("--multi-pose", action="store_true", help="Use all 9 calibration poses (otherwise T-pose only)")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--ee-weight", type=float, default=3.0, help="Extra weight multiplier for end-effectors (wrists+feet)"
    )
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--body-model-path", type=str, default=str(_body_model_dir()))
    parser.add_argument(
        "--poses", type=str, default=None, help="Pose-pair YAML from build_poses.py (default: poses/<robot>.yaml)"
    )
    parser.add_argument(
        "--precheck", type=str, default=None, help="Render all pose pairs to this PNG and exit (no optimization)"
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Open the viser pose-pair editor first; calibration continues with your edits after 'Apply & continue'",
    )
    parser.add_argument(
        "--ik-bootstrap",
        action="store_true",
        help="Seed robot pose pairs from the IK-init solve instead of hand-built overrides (auto-on when none exist)",
    )
    parser.add_argument(
        "--emit-preset",
        action="store_true",
        help="After --save, write src/.../presets/<robot>_cali.py from the calibration result",
    )
    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
    else:
        config = PRESETS[args.robot].to_dict()
    root_name = config.get("human_root_name", "pelvis")
    link_mapping = get_link_mapping(config)

    wp.init()
    robot = Robot.load(config["urdf_path"], load_meshes=True)
    measured_height = measure_calibration_height(args.body_model_path)

    print(f"Config: {args.config or f'presets.{args.robot}'}")
    print(f"Robot URDF: {config['urdf_path']}")
    print(f"Body model: {args.body_model_path}")
    print(f"Root joint: {root_name}")
    print(f"Link mappings: {len(link_mapping)}")
    print(f"EE weight: {args.ee_weight}")
    print(f"Calibration body T-pose height: {measured_height:.4f} m")
    print("\nFrame: runtime Z-up, human and robot grounded on z=0")

    # --- load poses ---
    poses = []
    pairs = load_pose_pairs(args.poses, args.robot)
    if not args.multi_pose:
        pairs = {"T-pose": pairs["T-pose"]}

    # IK init as the default seed: when a robot has no built pose pairs (empty
    # overrides), or --ik-bootstrap is given, solve each pose with the IK-init
    # config and use that as the robot side. --edit takes over (live IK button).
    if not args.edit and (args.ik_bootstrap or all(not ov for _bp, ov in pairs.values())):
        from build_poses import IK_PRESETS, ik_bootstrap_pairs, save_pose_pairs

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        print("\nNo hand-built pairs — seeding robot poses from IK init (--ik-bootstrap)...")
        pairs = ik_bootstrap_pairs(IK_PRESETS[args.robot], robot, args.body_model_path, pairs, device)
        save_path = args.poses or str(SCRIPT_DIR / "poses" / f"{args.robot}.yaml")
        save_pose_pairs(pairs, save_path)
        print(f"Saved IK-bootstrapped pairs to {save_path}")

    if args.edit:
        from build_poses import IK_PRESETS, edit_pose_pairs

        save_path = args.poses or str(SCRIPT_DIR / "poses" / f"{args.robot}.yaml")
        print("\n--- Pose editor: review/fix the pairs in the browser, then click 'Apply & continue calibration' ---")
        pairs = edit_pose_pairs(
            pairs,
            config,
            robot,
            args.body_model_path,
            save_path,
            finish_label="Apply & continue calibration",
            ik_config=IK_PRESETS[args.robot],
        )

    for pose_name, (body_pose_aa, joint_overrides) in pairs.items():
        print(f"Loading SMPL-X {pose_name}...")
        smplx_pos, smplx_quat, _ = _load_smplx_pose(args.body_model_path, body_pose_aa)
        poses.append(
            prepare_pose(
                pose_name,
                smplx_pos,
                smplx_quat,
                robot,
                config,
                link_mapping,
                joint_overrides=joint_overrides,
            )
        )

    # --- pose-pair audit: catch mismatched pairs before they bias the optimization ---
    hj2rl = {hj: rl for rl, hj, _, _, _, _ in link_mapping}
    bones = mapped_bones(hj2rl)
    print("\n--- Pose-pair audit: human-vs-robot limb direction (deg, ⚠ = exceeds structural floor + 15°) ---")
    print(f"  {'pose':<14s}" + "".join(f"{label:>10s}" for label, _, _, _ in bones))
    flagged = []
    for pose in poses:
        angs = bone_angles(pose, hj2rl)
        cells = []
        for label, _, _, floor in bones:
            bad = angs[label] > floor + 15.0
            cells.append(f"{angs[label]:>8.1f}{'⚠' if bad else ' '} ")
            if bad:
                flagged.append((pose["label"], label, angs[label]))
        print(f"  {pose['label']:<14s}" + "".join(cells))
    if flagged:
        print(
            f"\n  ⚠ {len(flagged)} suspicious pair segment(s) — inspect/fix them in build_poses.py before calibrating."
        )

    if args.precheck:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        edges = [
            ("pelvis", "left_hip"),
            ("pelvis", "right_hip"),
            ("pelvis", "spine3"),
            ("spine3", "left_shoulder"),
            ("spine3", "right_shoulder"),
        ]
        edges = [e for e in edges if e[0] in hj2rl and e[1] in hj2rl] + [(a, b) for _, a, b, _ in bones]
        n = len(poses)
        fig = plt.figure(figsize=(3.6 * n, 4.6), dpi=110)
        for i, pose in enumerate(poses):
            ax = fig.add_subplot(1, n, i + 1, projection="3d")
            for a, b in edges:
                hp = np.stack([pose["smplx_pos"][a], pose["smplx_pos"][b]])
                rp = np.stack([pose["robot_pos"][hj2rl[a]], pose["robot_pos"][hj2rl[b]]])
                ax.plot(hp[:, 0], hp[:, 1], hp[:, 2], color="#3da5ff", lw=3.0)
                ax.plot(rp[:, 0], rp[:, 1], rp[:, 2], color="#555555", lw=3.0)
            angs = bone_angles(pose, hj2rl)
            worst = max(angs[label] - floor for label, _, _, floor in bones)
            ok = worst <= 15.0
            ax.set_title(f"{pose['label']}\n{'OK' if ok else f'⚠ worst +{worst:.0f}°'}", color="green" if ok else "red")
            ax.set_box_aspect((1, 1, 1))
            for setter in (ax.set_xlim, ax.set_ylim):
                setter(-0.9, 0.9)
            ax.set_zlim(0.0, 1.8)
            ax.view_init(elev=8, azim=-70)
            ax.set_axis_off()
        fig.suptitle("Pose-pair pre-check — blue = SMPL-X human, grey = robot (both grounded)", fontsize=13)
        fig.tight_layout()
        fig.savefig(args.precheck, bbox_inches="tight")
        print(f"\nWrote pre-check render to {args.precheck} — inspect it, then re-run without --precheck to optimize.")
        return

    # --- show current errors ---
    print("\n--- Current config errors ---")
    for pose in poses:
        transformed = apply_scale_and_offset_np(
            pose["smplx_pos"],
            pose["smplx_quat"],
            root_name,
            config.get("scale_table", {}),
            link_mapping,
        )
        print_comparison(f"{pose['label']} (before)", transformed, pose["robot_pos"], link_mapping)

    # --- optimize ---
    optimized_scales = None
    optimized_offsets = None

    if args.optimize_scales:
        print("\n--- Optimizing scale_table ---")
        opt = ScaleOptimizer(poses, config, link_mapping, root_name, lr=args.lr, ee_weight=args.ee_weight)
        optimized_scales = opt.optimize(args.iters)

    elif args.optimize_offsets:
        print("\n--- Optimizing position_offset ---")
        opt = JointOptimizer(
            poses, config, link_mapping, root_name, lr=args.lr, scale_reg=0, offset_reg=0.1, ee_weight=args.ee_weight
        )
        for p in opt.scale_params.values():
            p.requires_grad_(False)
        _, optimized_offsets = opt.optimize(args.iters)

    elif args.optimize_all:
        print("\n--- Jointly optimizing scale_table + position_offset ---")
        opt = JointOptimizer(poses, config, link_mapping, root_name, lr=args.lr, ee_weight=args.ee_weight)
        optimized_scales, optimized_offsets = opt.optimize(args.iters)

    # --- show optimized errors ---
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
                pose["smplx_pos"],
                pose["smplx_quat"],
                root_name,
                new_scale_table,
                new_mapping,
            )
            print_comparison(f"{pose['label']} (after)", transformed, pose["robot_pos"], new_mapping)

        print_yaml_output(config, optimized_scales, optimized_offsets, measured_height)

        if args.save:
            final = save_calibrated_config(config, args.save, optimized_scales, optimized_offsets, measured_height)
            if args.emit_preset:
                preset_path = PRESETS_DIR / f"{args.robot}_cali.py"
                preset_path.write_text(emit_preset(args.robot, final))
                print(f"Wrote preset: {preset_path}")
                print(f'Register "{args.robot}_cali" in examples/humanoid_retarget/10_gallery.py (PRESETS)')
    else:
        print("\nNo optimization requested. Use --optimize-scales, --optimize-offsets, or --optimize-all.")


if __name__ == "__main__":
    main()
