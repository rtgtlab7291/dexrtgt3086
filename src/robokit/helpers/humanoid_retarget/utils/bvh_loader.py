"""BVH file loader with forward kinematics and SMPL-X joint mapping.

All quaternions are in wxyz order internally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciR


@dataclass
class BvhJoint:
    name: str
    parent: int  # parent index, -1 for root
    offset: np.ndarray  # (3,) rest-pose offset in cm
    channels: str  # euler order string, e.g. "yxz"
    channel_offset: int  # start index in per-frame channel data


@dataclass
class BvhData:
    joints: List[BvhJoint]
    joint_names: List[str]
    parents: np.ndarray  # (J,) int, -1 for root
    offsets: np.ndarray  # (J, 3) rest offsets in cm
    end_site_offsets: Dict[str, np.ndarray]  # joint_name -> end site offset in cm
    num_frames: int
    frame_time: float
    local_quats: np.ndarray  # (T, J, 4) wxyz
    root_positions: np.ndarray  # (T, 3) in cm


# ---------------------------------------------------------------------------
# BVH joint name -> SMPL-X joint name mapping
# ---------------------------------------------------------------------------
BVH_TO_SMPLX_MAP: Dict[str, str] = {
    "Hips": "pelvis",
    "LeftUpLeg": "left_hip",
    "RightUpLeg": "right_hip",
    "LeftLeg": "left_knee",
    "RightLeg": "right_knee",
    "LeftFoot": "left_ankle",
    "RightFoot": "right_ankle",
    "Spine": "spine1",
    "Spine1": "spine2",
    "Spine2": "spine3",
    "LeftArm": "left_shoulder",
    "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow",
    "RightForeArm": "right_elbow",
    "LeftHand": "left_wrist",
    "RightHand": "right_wrist",
    "Neck": "neck",
    "Head": "head",
}


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz convention, vectorized over arbitrary leading dims)
# ---------------------------------------------------------------------------


def _quat_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternion arrays in wxyz order."""
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]
    return np.concatenate(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )


def _quat_mul_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by quaternion(s). q: wxyz, v: xyz."""
    t = 2.0 * np.cross(q[..., 1:], v)
    return v + q[..., 0:1] * t + np.cross(q[..., 1:], t)


def _angle_axis_to_quat(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    return np.concatenate([c, s * axis], axis=-1)


def _euler_to_quat(angles_rad: np.ndarray, order: str) -> np.ndarray:
    """Convert Euler angles (..., 3) in radians to wxyz quaternions.

    ``order`` is a 3-char string like ``"yxz"`` specifying the rotation axes
    applied left-to-right (i.e. first character is applied first).
    """
    axis_map = {
        "x": np.asarray([1, 0, 0], dtype=np.float32),
        "y": np.asarray([0, 1, 0], dtype=np.float32),
        "z": np.asarray([0, 0, 1], dtype=np.float32),
    }
    q0 = _angle_axis_to_quat(angles_rad[..., 0], axis_map[order[0]])
    q1 = _angle_axis_to_quat(angles_rad[..., 1], axis_map[order[1]])
    q2 = _angle_axis_to_quat(angles_rad[..., 2], axis_map[order[2]])
    return _quat_mul(q0, _quat_mul(q1, q2))


def _remove_quat_discontinuities(rotations: np.ndarray) -> np.ndarray:
    """Fix sign flips along the time axis. Shape (T, J, 4)."""
    rots_inv = -rotations
    for i in range(1, rotations.shape[0]):
        dot_orig = np.sum(rotations[i - 1 : i] * rotations[i : i + 1], axis=-1)
        dot_flip = np.sum(rotations[i - 1 : i] * rots_inv[i : i + 1], axis=-1)
        mask = (dot_orig < dot_flip)[..., np.newaxis]
        rotations[i] = mask * rots_inv[i] + (1.0 - mask) * rotations[i]
    return rotations


# ---------------------------------------------------------------------------
# BVH parser
# ---------------------------------------------------------------------------

_CHANNEL_MAP = {
    "Xrotation": "x",
    "Yrotation": "y",
    "Zrotation": "z",
}


def parse_bvh(path: str | Path) -> BvhData:
    """Parse a BVH file into ``BvhData``.

    Euler angles are converted to wxyz quaternions and discontinuities are
    removed so that the resulting ``local_quats`` can be used directly for FK.
    """
    path = Path(path)

    joints: List[BvhJoint] = []
    parents: List[int] = []
    offsets: List[np.ndarray] = []
    end_site_offsets: Dict[str, np.ndarray] = {}

    active = -1
    end_site = False
    euler_order: str = "yxz"
    channel_offset = 0
    root_channels = 0

    num_frames = 0
    frame_time = 0.0
    in_motion = False
    raw_frames: List[np.ndarray] = []

    with open(path) as f:
        for line in f:
            stripped = line.strip()

            if stripped in ("HIERARCHY", ""):
                continue

            if stripped == "MOTION":
                in_motion = True
                continue

            # ---- header section ----
            if not in_motion:
                rmatch = re.match(r"ROOT\s+(\S+)", stripped)
                if rmatch:
                    name = rmatch.group(1)
                    joints.append(BvhJoint(name=name, parent=-1, offset=np.zeros(3), channels="", channel_offset=0))
                    parents.append(-1)
                    offsets.append(np.zeros(3, dtype=np.float64))
                    active = len(joints) - 1
                    continue

                jmatch = re.match(r"JOINT\s+(\S+)", stripped)
                if jmatch:
                    name = jmatch.group(1)
                    joints.append(BvhJoint(name=name, parent=active, offset=np.zeros(3), channels="", channel_offset=0))
                    parents.append(active)
                    offsets.append(np.zeros(3, dtype=np.float64))
                    active = len(joints) - 1
                    continue

                if "End Site" in stripped:
                    end_site = True
                    continue

                if "{" in stripped:
                    continue

                if "}" in stripped:
                    if end_site:
                        end_site = False
                    else:
                        active = parents[active]
                    continue

                offmatch = re.match(r"OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", stripped)
                if offmatch:
                    off = np.array([float(x) for x in offmatch.groups()], dtype=np.float64)
                    if end_site:
                        end_site_offsets[joints[active].name] = off
                    else:
                        offsets[active] = off
                        joints[active].offset = off
                    continue

                chanmatch = re.match(r"CHANNELS\s+(\d+)(.*)", stripped)
                if chanmatch:
                    nchan = int(chanmatch.group(1))
                    parts = stripped.split()[2:]
                    if nchan == 6:
                        # Root: 3 position + 3 rotation
                        root_channels = 6
                        rot_parts = parts[3:6]
                        order = "".join(_CHANNEL_MAP[p] for p in rot_parts if p in _CHANNEL_MAP)
                        joints[active].channels = order
                        joints[active].channel_offset = 0
                        channel_offset = 6
                    else:
                        rot_parts = parts[:3]
                        order = "".join(_CHANNEL_MAP[p] for p in rot_parts if p in _CHANNEL_MAP)
                        joints[active].channels = order
                        joints[active].channel_offset = channel_offset
                        channel_offset += 3

                    euler_order = order
                    continue

            # ---- motion section ----
            fmatch = re.match(r"Frames:\s+(\d+)", stripped)
            if fmatch:
                num_frames = int(fmatch.group(1))
                continue

            ftmatch = re.match(r"Frame Time:\s+([\d\.]+)", stripped)
            if ftmatch:
                frame_time = float(ftmatch.group(1))
                continue

            # data line
            values = stripped.split()
            if values and in_motion:
                raw_frames.append(np.array([float(v) for v in values], dtype=np.float64))

    num_frames = len(raw_frames)
    num_joints = len(joints)
    offsets_arr = np.array(offsets, dtype=np.float64)

    # Build rotation and position arrays
    root_positions = np.zeros((num_frames, 3), dtype=np.float64)
    all_euler = np.zeros((num_frames, num_joints, 3), dtype=np.float64)

    for t, data in enumerate(raw_frames):
        if root_channels == 6:
            root_positions[t] = data[0:3]
        for j, joint in enumerate(joints):
            co = joint.channel_offset
            if j == 0 and root_channels == 6:
                all_euler[t, j] = data[3:6]
            else:
                all_euler[t, j] = data[co : co + 3]

    # Per-joint euler order -> quaternion conversion
    # Check if all joints share the same order (common case)
    orders = [j.channels for j in joints]
    unique_orders = set(orders)
    if len(unique_orders) == 1 and euler_order:
        local_quats = _euler_to_quat(np.radians(all_euler), euler_order)
    else:
        local_quats = np.zeros((num_frames, num_joints, 4), dtype=np.float64)
        for j, joint in enumerate(joints):
            order = joint.channels if joint.channels else euler_order
            local_quats[:, j] = _euler_to_quat(np.radians(all_euler[:, j]), order)

    local_quats = local_quats.astype(np.float32)
    local_quats = _remove_quat_discontinuities(local_quats)

    return BvhData(
        joints=joints,
        joint_names=[j.name for j in joints],
        parents=np.array([j.parent for j in joints], dtype=np.int32),
        offsets=offsets_arr.astype(np.float32),
        end_site_offsets=end_site_offsets,
        num_frames=num_frames,
        frame_time=frame_time,
        local_quats=local_quats,
        root_positions=root_positions.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def _quat_fk(
    local_quats: np.ndarray,
    root_positions: np.ndarray,
    offsets: np.ndarray,
    parents: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized forward kinematics.

    Args:
        local_quats: (T, J, 4) wxyz local rotations.
        root_positions: (T, 3) root world positions in cm.
        offsets: (J, 3) rest-pose offsets in cm.
        parents: (J,) parent indices, -1 for root.

    Returns:
        global_positions: (T, J, 3) in cm, BVH Y-up frame.
        global_quats: (T, J, 4) wxyz in BVH Y-up frame.
    """
    num_frames = local_quats.shape[0]
    num_joints = local_quats.shape[1]

    # Build local positions: root uses root_positions, children use offsets
    local_pos = np.tile(offsets[np.newaxis], (num_frames, 1, 1)).copy()
    local_pos[:, 0] = root_positions

    gp = [local_pos[:, 0:1, :]]
    gr = [local_quats[:, 0:1, :]]

    for i in range(1, num_joints):
        pi = parents[i]
        gr_i = _quat_mul(gr[pi], local_quats[:, i : i + 1, :])
        gp_i = _quat_mul_vec(gr[pi], local_pos[:, i : i + 1, :]) + gp[pi]
        gr.append(gr_i)
        gp.append(gp_i)

    global_quats = np.concatenate(gr, axis=1)
    global_positions = np.concatenate(gp, axis=1)
    return global_positions, global_quats


# ---------------------------------------------------------------------------
# Coordinate conversion: Y-up (BVH) -> Z-up, cm -> m
# ---------------------------------------------------------------------------

_Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_y2z_xyzw = SciR.from_matrix(_Y_UP_TO_Z_UP).as_quat()
_Y_UP_TO_Z_UP_QUAT = np.array(
    [_y2z_xyzw[3], _y2z_xyzw[0], _y2z_xyzw[1], _y2z_xyzw[2]], dtype=np.float32
)


def _convert_to_zup(
    positions_cm: np.ndarray,
    quats_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert from BVH Y-up cm to Z-up meters.

    Args:
        positions_cm: (..., 3) positions in cm, Y-up.
        quats_wxyz: (..., 4) quaternions in wxyz, Y-up.

    Returns:
        positions_m: (..., 3) in meters, Z-up.
        quats_wxyz: (..., 4) in wxyz, Z-up.
    """
    positions_m = (positions_cm @ _Y_UP_TO_Z_UP.T) / 100.0
    quats_zup = _quat_mul(_Y_UP_TO_Z_UP_QUAT, quats_wxyz)
    return positions_m, quats_zup


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_bvh_frames(
    data: BvhData,
    tgt_fps: Optional[float] = None,
) -> Tuple[List[Dict[str, Tuple[np.ndarray, np.ndarray]]], float]:
    """Run FK, convert to Z-up meters, map to SMPL-X names.

    Args:
        data: Parsed BVH data.
        tgt_fps: Target FPS for downsampling. None or 0 keeps native rate.

    Returns:
        frames: Per-frame dicts ``{smplx_name: (pos_xyz_m, quat_wxyz)}``.
        fps: Actual output FPS.
    """
    global_pos_cm, global_quats = _quat_fk(
        data.local_quats,
        data.root_positions,
        data.offsets,
        data.parents,
    )

    global_pos_m, global_quats_zup = _convert_to_zup(global_pos_cm, global_quats)

    src_fps = 1.0 / data.frame_time if data.frame_time > 0 else 30.0
    step = 1
    if tgt_fps and tgt_fps > 0 and tgt_fps < src_fps:
        step = max(1, round(src_fps / tgt_fps))
    actual_fps = src_fps / step

    # Build name index for BVH joints
    name_to_idx = {name: i for i, name in enumerate(data.joint_names)}

    # Precompute end-site global positions for synthetic foot joints
    # left_foot = LeftFoot pos + LeftFoot_rot * end_site_offset (converted)
    foot_synthetics: Dict[str, Tuple[int, np.ndarray]] = {}
    for ankle_bvh, foot_smplx in [("LeftFoot", "left_foot"), ("RightFoot", "right_foot")]:
        if ankle_bvh in name_to_idx and ankle_bvh in data.end_site_offsets:
            foot_synthetics[foot_smplx] = (
                name_to_idx[ankle_bvh],
                data.end_site_offsets[ankle_bvh].astype(np.float32),
            )

    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    for t in range(0, data.num_frames, step):
        frame: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for bvh_name, smplx_name in BVH_TO_SMPLX_MAP.items():
            if bvh_name not in name_to_idx:
                continue
            idx = name_to_idx[bvh_name]
            pos = global_pos_m[t, idx].astype(np.float32)
            quat = global_quats_zup[t, idx].astype(np.float32)
            frame[smplx_name] = (pos, quat)

        # Synthetic foot joints: ankle position + ankle_rot * end_offset
        for foot_smplx, (ankle_idx, end_off_cm) in foot_synthetics.items():
            ankle_quat_yup = global_quats[t, ankle_idx]
            foot_pos_cm = (
                global_pos_cm[t, ankle_idx]
                + _quat_mul_vec(
                    ankle_quat_yup[np.newaxis],
                    end_off_cm[np.newaxis],
                )[0]
            )
            foot_pos_m, foot_quat = _convert_to_zup(
                foot_pos_cm[np.newaxis],
                ankle_quat_yup[np.newaxis],
            )
            # Use toe orientation (the ankle orientation rotated to Z-up)
            frame[foot_smplx] = (foot_pos_m[0].astype(np.float32), foot_quat[0].astype(np.float32))

        frames.append(frame)

    return frames, actual_fps


def estimate_height(data: BvhData) -> float:
    """Estimate human height from frame 0 global positions (Z-up, meters)."""
    global_pos_cm, _ = _quat_fk(
        data.local_quats[0:1],
        data.root_positions[0:1],
        data.offsets,
        data.parents,
    )
    global_pos_m, _ = _convert_to_zup(global_pos_cm, data.local_quats[0:1])
    z_vals = global_pos_m[0, :, 2]
    return float(np.max(z_vals) - np.min(z_vals))


def compute_root_velocity(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    prev_quat_wxyz: np.ndarray,
    curr_quat_wxyz: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference root velocity.

    Returns:
        lin_vel: (3,) linear velocity in m/s.
        ang_vel: (3,) angular velocity in rad/s.
    """
    lin_vel = (curr_pos - prev_pos) / dt

    r_prev = SciR.from_quat([prev_quat_wxyz[1], prev_quat_wxyz[2], prev_quat_wxyz[3], prev_quat_wxyz[0]])
    r_curr = SciR.from_quat([curr_quat_wxyz[1], curr_quat_wxyz[2], curr_quat_wxyz[3], curr_quat_wxyz[0]])
    delta_r = r_curr * r_prev.inv()
    ang_vel = delta_r.as_rotvec().astype(np.float32) / dt

    return lin_vel.astype(np.float32), ang_vel
