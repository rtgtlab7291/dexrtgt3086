"""ParaHome sequences as ``ComboClip`` intervals.

ParaHome is 30 fps, metres, z-up, and shares one world frame between SMPL-X and objects. Each hand's
object part is inferred from sustained surface proximity; contacts are the nearby mesh projections.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from robokit.assets.body_models import smplx as smplx_assets
from robokit.assets.objects import parahome as parahome_assets
from robokit.helpers.combo_retarget.clip import (
    CONTACT_DISTANCE,
    CONTACT_MANO_INDICES,
    HANDS,
    ComboClip,
    HandTrack,
    SceneGeometry,
    mano_index,
)
from robokit.smplx import (
    SMPLX_JOINT_NAMES,
    SMPLX_STATIC_LANDMARK_NAMES,
    BodyModelSpecTensors,
    BodyModelState,
    body_lbs_warp,
    load_smplx,
)
from robokit.xform.numpy import tf_mat_to_pose7


PARAHOME_FPS = 30
GRASPABLE_DIAGONAL = 0.6  # m; parts bigger than this are furniture, not grasp targets
ASSOCIATION_PROBES = 12
MIN_MULTI_CONTACT_RATIO = 0.25
MIN_SINGLE_CONTACT_RATIO = 0.5
# ParaHome hand skeleton in MANO-21 order; right starts at 48, left at 23.
NATIVE_MANO21 = (0, 1, 22, 23, 24, 18, 19, 20, 21, 14, 15, 16, 17, 10, 11, 12, 13, 6, 7, 8, 9)
NATIVE_HAND_OFFSET = {"right": 48, "left": 23}


@dataclass(frozen=True)
class ParaHomeInterval:
    """One annotated interaction: a frame span of a sequence, with a target part per engaged hand."""

    sequence: str  # "s1"
    start: int  # inclusive, in native 30 fps frames
    end: int  # exclusive
    text: str  # the action annotation
    parts: Dict[str, str] = field(default_factory=dict)
    distances: Dict[str, float] = field(default_factory=dict)

    @property
    def sides(self) -> List[str]:
        return [s for s in HANDS if s in self.parts]

    @property
    def label(self) -> str:
        parts = "+".join(f"{s[0].upper()}:{self.parts[s]}" for s in self.sides)
        return f"{parts} {self.text}"


def find_parahome_intervals(  # noqa: PLR0917
    data_root: str,
    sequence: str,
    max_distance: float = 0.05,
    min_frames: int = 30,
    hands: Sequence[str] = HANDS,
    mesh_dir: Optional[str] = None,
) -> List[ParaHomeInterval]:
    """Mine annotated spans for the object part used by each hand."""
    root = Path(data_root)
    seq_dir = root / "seq" / sequence
    annotations = json.loads((seq_dir / "text_annotation.json").read_text())
    joints = pickle.loads((seq_dir / "joint_positions.pkl").read_bytes())
    transforms = pickle.loads((seq_dir / "object_transformations.pkl").read_bytes())

    baked_dir = Path(mesh_dir) if mesh_dir is not None else parahome_assets.mesh_dir()
    trees: Dict[str, cKDTree] = {}
    for part in {k for frame in transforms.values() for k in frame}:
        mesh = trimesh.load(str(baked_dir / f"{part}.obj"), force="mesh")
        assert isinstance(mesh, trimesh.Trimesh)
        if float(np.linalg.norm(mesh.extents)) <= GRASPABLE_DIAGONAL:
            trees[part] = cKDTree(np.asarray(mesh.vertices, dtype=np.float64))

    native_slots = np.asarray(NATIVE_MANO21)[np.asarray(CONTACT_MANO_INDICES)]
    slots = {side: NATIVE_HAND_OFFSET[side] + native_slots for side in hands}
    intervals = []
    for span, text in annotations.items():
        start, end = (int(x) for x in span.split())
        end = min(end, len(joints))
        if end - start < min_frames:
            continue
        probes = [f for f in range(start, end, max(1, (end - start) // ASSOCIATION_PROBES)) if f in transforms]
        best: Dict[str, tuple] = {}
        for part, tree in trees.items():
            frames = [f for f in probes if part in transforms[f]]
            if len(frames) < len(probes) // 2 or not frames:
                continue
            for side, rows in slots.items():
                local = []
                for frame in frames:
                    matrix = np.asarray(transforms[frame][part], dtype=np.float32)
                    local.append((joints[frame, rows] - matrix[:3, 3]) @ matrix[:3, :3])
                distance = tree.query(np.concatenate(local))[0].reshape(len(frames), -1)
                multi_distance = float(np.median(np.sort(distance, axis=1)[:, :3].mean(axis=1)))
                single_distance = float(np.median(distance.min(axis=1)))
                contact_count = (distance < CONTACT_DISTANCE).sum(axis=1)
                multi_ratio = float(np.mean(contact_count >= 2))
                single_ratio = float(np.mean(contact_count >= 1))
                association_distance = min(multi_distance, single_distance + CONTACT_DISTANCE)
                score = association_distance + 0.5 * CONTACT_DISTANCE * (1.0 - multi_ratio)
                if side not in best or score < best[side][1]:
                    best[side] = (part, score, association_distance, multi_ratio, single_ratio)
        engaged = {
            side: value
            for side, value in best.items()
            if value[2] <= max_distance
            and (value[3] >= MIN_MULTI_CONTACT_RATIO or value[4] >= MIN_SINGLE_CONTACT_RATIO)
        }
        if engaged:
            intervals.append(
                ParaHomeInterval(
                    sequence=sequence,
                    start=start,
                    end=end,
                    text=text,
                    parts={s: v[0] for s, v in engaged.items()},
                    distances={s: v[2] for s, v in engaged.items()},
                )
            )
    return intervals


def load_parahome_clip(  # noqa: PLR0917
    data_root: str,
    interval: ParaHomeInterval,
    robot_height: float = 1.32,
    fps: int = 30,
    device: str = "cuda:0",
    mesh_dir: Optional[str] = None,
    load_scene: bool = False,
) -> ComboClip:
    """Load one interval, its hand tracks, and optionally the room."""
    root = Path(data_root)
    seq_dir = root / "seq" / interval.sequence
    transforms = pickle.loads((seq_dir / "object_transformations.pkl").read_bytes())

    step = PARAHOME_FPS // fps
    targets = set(interval.parts.values())
    frames = [f for f in range(interval.start, interval.end, step) if targets <= set(transforms.get(f, {}))]
    assert frames, f"{sorted(targets)} are not all present for any frame of the span"
    T = len(frames)

    params = pickle.loads((root / "smplx_seq" / interval.sequence / "smplx_params.pkl").read_bytes())
    pose = pickle.loads((root / "smplx_seq" / interval.sequence / "smplx_pose.pkl").read_bytes())
    gender = str(params["gender"])
    betas = params["beta"].cpu().numpy().astype(np.float32)
    model_npz = {"male": smplx_assets.MALE_NPZ_PATH, "female": smplx_assets.FEMALE_NPZ_PATH}[gender]
    spec = load_smplx(str(model_npz), gender, num_betas=betas.shape[1])  # type: ignore[arg-type]
    rows = np.asarray(frames)
    full_pose = np.concatenate(
        [
            pose["global_orient"].cpu().numpy()[rows].reshape(T, 1, 3),
            pose["body_pose"].cpu().numpy()[rows].reshape(T, 21, 3),
            np.zeros((T, 3, 3), np.float32),
            pose["hand_pose"].cpu().numpy()[rows].reshape(T, 30, 3),
        ],
        axis=1,
    ).astype(np.float32)
    state = BodyModelState(
        betas=np.repeat(betas, T, axis=0),
        full_pose_aa=full_pose,
        transl=pose["transl"].cpu().numpy()[rows].astype(np.float32),
    )
    spec_tensors = BodyModelSpecTensors(spec=spec, device=device)
    out = body_lbs_warp(spec_tensors, state, return_landmarks=True)
    joint_pose7 = out["T_world_joint"].numpy()
    landmarks = out["landmarks"].numpy()

    foot_idx = [
        SMPLX_STATIC_LANDMARK_NAMES.index(n) for n in ["left_heel", "right_heel", "left_big_toe", "right_big_toe"]
    ]
    z0 = float(landmarks[:, foot_idx, 2].min())
    tpose = BodyModelState(
        betas=betas,
        full_pose_aa=np.zeros((1, 55, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    tpose_verts = body_lbs_warp(spec_tensors, tpose, return_landmarks=False)["vertices"].numpy()[0]
    height = float(tpose_verts[:, 1].max() - tpose_verts[:, 1].min())
    scale = robot_height / height

    joints = joint_pose7[..., :3] - [0.0, 0.0, z0]
    landmarks = landmarks - [0.0, 0.0, z0]
    baked_dir = Path(mesh_dir) if mesh_dir is not None else parahome_assets.mesh_dir()

    track_cache: Dict[str, Tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray]] = {}

    def object_tracks(part: str) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray]:
        if part in track_cache:  # both hands often grasp the same part; load it once
            return track_cache[part]
        mesh = trimesh.load(str(baked_dir / f"{part}.obj"), force="mesh")
        assert isinstance(mesh, trimesh.Trimesh)
        matrices = np.stack([np.asarray(transforms[f][part], dtype=np.float32) for f in frames])
        native = tf_mat_to_pose7(matrices)
        native[:, 2] -= z0
        world = native.copy()
        world[:, :3] *= scale
        track_cache[part] = (mesh, native, world, matrices[:, :3, :3])
        return track_cache[part]

    slots = np.asarray(CONTACT_MANO_INDICES)
    tracks: Dict[str, HandTrack] = {}
    for side in HANDS:
        joint_idx, tip_idx = mano_index(side)
        mano = np.stack(
            [landmarks[:, t] if j < 0 else joints[:, j] for j, t in zip(joint_idx, tip_idx)], axis=1
        ).astype(np.float32)
        wrist = SMPLX_JOINT_NAMES.index(f"{side}_wrist")
        wrist_pos = joints[:, wrist].astype(np.float32)
        wrist_quat = joint_pose7[:, wrist, 3:].astype(np.float32)
        if side in interval.parts:
            part = interval.parts[side]
            mesh, native, world, rot_obj = object_tracks(part)
            local = np.einsum("fij,fsj->fsi", np.swapaxes(rot_obj, 1, 2), mano[:, slots] - native[:, None, :3])
            closest, distance, triangle = trimesh.proximity.closest_point(mesh, local.reshape(-1, 3))
            mask = (distance < CONTACT_DISTANCE).reshape(T, len(slots))
            points = (
                np.einsum("fij,fsj->fsi", rot_obj, closest.reshape(T, -1, 3).astype(np.float32)) + native[:, None, :3]
            )
            normals = np.asarray(mesh.face_normals, dtype=np.float32)[triangle].reshape(T, -1, 3)
            object_name, object_mesh = part, mesh
            contact_points = (points * mask[..., None]).astype(np.float32)
            contact_normals = (np.einsum("fij,fsj->fsi", rot_obj, normals) * mask[..., None]).astype(np.float32)
        else:
            native = np.concatenate([wrist_pos, wrist_quat], axis=1)
            world = native.copy()
            world[:, :3] *= scale
            object_name, object_mesh = None, None
            mask = np.zeros((T, len(slots)), dtype=bool)
            contact_points = np.zeros((T, len(slots), 3), dtype=np.float32)
            contact_normals = np.zeros((T, len(slots), 3), dtype=np.float32)
        tracks[side] = HandTrack(
            mano=mano,
            mano_overlay=(mano - native[:, None, :3] + world[:, None, :3]).astype(np.float32),
            wrist_pos=wrist_pos,
            wrist_quat_wxyz=wrist_quat,
            object_name=object_name,
            object_mesh=object_mesh,
            object_pose_native=native,
            object_pose_world=world,
            contact=np.asarray(mask.any(axis=1)),
            contact_points=contact_points,
            contact_normals=contact_normals,
            contact_mask=mask,
        )

    scene = None
    if load_scene:
        names, meshes, poses = [], [], []
        for part in sorted({k for f in frames for k in transforms[f]} - targets):
            if any(part not in transforms[f] for f in frames):
                continue
            part_pose = tf_mat_to_pose7(np.stack([np.asarray(transforms[f][part], dtype=np.float32) for f in frames]))
            part_pose[:, :3] = (part_pose[:, :3] - [0.0, 0.0, z0]) * scale
            part_mesh = trimesh.load(str(baked_dir / f"{part}.obj"), force="mesh")
            assert isinstance(part_mesh, trimesh.Trimesh)
            part_mesh.apply_scale(scale)
            names.append(part)
            meshes.append(part_mesh)
            poses.append(part_pose)
        stacked = np.stack(poses) if poses else np.zeros((0, T, 7), dtype=np.float32)
        scene = SceneGeometry(
            names=names,
            meshes=meshes,
            poses=stacked,
            static=np.ptp(stacked, axis=1).max(axis=-1) < 1e-6 if names else np.zeros(0, dtype=bool),
        )

    return ComboClip(
        human_pose7=np.concatenate([joints, joint_pose7[..., 3:]], axis=-1).astype(np.float32),
        human_height=height,
        scale=scale,
        fps=fps,
        hands=tracks,
        scene=scene,
    )


__all__ = ["GRASPABLE_DIAGONAL", "PARAHOME_FPS", "ParaHomeInterval", "find_parahome_intervals", "load_parahome_clip"]
