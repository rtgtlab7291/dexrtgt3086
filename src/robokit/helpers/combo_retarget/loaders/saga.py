"""Load SAGA FullGraspPose (GRAB-derived) motion as a ``ComboClip``.

SAGA is z-up and object-centered. Hand poses are 24-component MANO PCA, object rotations are
negated rotvecs, and per-vertex labels identify finger contacts.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from robokit.assets.body_models import smplx as smplx_assets
from robokit.assets.objects import saga as saga_assets
from robokit.helpers.combo_retarget.clip import (
    CONTACT_DISTANCE,
    CONTACT_MANO_INDICES,
    FINGERS,
    ComboClip,
    HandTrack,
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
from robokit.xform.numpy import axis_angle_to_matrix, matrix_to_quaternion


GRAB_FPS = 120


def load_saga_clip(
    npz_path: str,
    robot_height: float = 1.32,
    fps: int = 30,
    device: str = "cuda:0",
    mesh_dir: Optional[str] = None,
) -> ComboClip:
    d = np.load(npz_path, allow_pickle=True)
    body = d["body"][()]
    gender = str(d["gender"])
    model_npz = {"male": smplx_assets.MALE_NPZ_PATH, "female": smplx_assets.FEMALE_NPZ_PATH}[gender]
    spec = load_smplx(str(model_npz), gender)  # type: ignore[arg-type]
    model_raw = np.load(str(model_npz))

    step = GRAB_FPS // fps
    idx = np.arange(0, body["transl"].shape[0], step)
    T = len(idx)
    aa_l = (
        body["left_hand_pose"][idx] @ model_raw["hands_componentsl"][:24].astype(np.float32)
        + spec.metadata["hands_meanl"]
    )
    aa_r = (
        body["right_hand_pose"][idx] @ model_raw["hands_componentsr"][:24].astype(np.float32)
        + spec.metadata["hands_meanr"]
    )
    full_pose = np.concatenate(
        [
            body["global_orient"][idx],
            body["body_pose"][idx],
            body["jaw_pose"][idx],
            body["leye_pose"][idx],
            body["reye_pose"][idx],
            aa_l,
            aa_r,
        ],
        axis=1,
    ).astype(np.float32)
    state = BodyModelState(
        betas=np.repeat(d["betas"], T, axis=0).astype(np.float32),
        full_pose_aa=full_pose.reshape(T, 55, 3),
        transl=body["transl"][idx].astype(np.float32),
    )
    spec_tensors = BodyModelSpecTensors(spec=spec, device=device)
    out = body_lbs_warp(spec_tensors, state, return_landmarks=True)
    joint_pose7 = out["T_world_joint"].numpy()  # (T, 55, 7) xyz_wxyz
    landmarks = out["landmarks"].numpy()  # (T, 21, 3)

    transf = d["transf_transl"][idx].astype(np.float32)
    joints = joint_pose7[..., :3] + transf[:, None]
    landmarks = landmarks + transf[:, None]

    verts = d["verts_object"][idx].astype(np.float32)
    rot = axis_angle_to_matrix(-d["global_orient_object"][idx]).astype(np.float32)
    t_obj = verts.mean(axis=1) + transf
    object_name = Path(npz_path).name.split("_")[0]
    baked_dir = Path(mesh_dir) if mesh_dir is not None else saga_assets.mesh_dir()
    baked = baked_dir / f"{object_name}.obj"
    mesh = trimesh.load(str(baked), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)

    foot_idx = [
        SMPLX_STATIC_LANDMARK_NAMES.index(n) for n in ["left_heel", "right_heel", "left_big_toe", "right_big_toe"]
    ]
    z0 = landmarks[:, foot_idx, 2].min()
    tpose = BodyModelState(
        betas=d["betas"].astype(np.float32),
        full_pose_aa=np.zeros((1, 55, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    tpose_verts = body_lbs_warp(spec_tensors, tpose, return_landmarks=False)["vertices"].numpy()[0]
    height = float(tpose_verts[:, 1].max() - tpose_verts[:, 1].min())  # canonical T-pose is y-up
    scale = robot_height / height

    ground = [0.0, 0.0, z0]
    quat_obj = matrix_to_quaternion(rot)
    object_pose_native = np.concatenate([t_obj, quat_obj], axis=1).astype(np.float32)
    object_pose = object_pose_native.copy()
    object_pose[:, :3] = (t_obj - ground) * scale

    joint_idx, tip_idx = mano_index("right")
    mano = np.stack([landmarks[:, t] if j < 0 else joints[:, j] for j, t in zip(joint_idx, tip_idx)], axis=1)

    verts_world = verts + transf[:, None]
    labels = d["contact_object"][idx]
    finger_label_ids = np.array(
        [[SMPLX_JOINT_NAMES.index(f"right_{finger}{joint}") + 1 for joint in (1, 2, 3)] for finger in FINGERS],
        dtype=labels.dtype,
    )
    contact_points = np.zeros((T, len(CONTACT_MANO_INDICES), 3), dtype=np.float32)
    contact_normals = np.zeros((T, len(CONTACT_MANO_INDICES), 3), dtype=np.float32)
    contact_mask = np.zeros((T, len(CONTACT_MANO_INDICES)), dtype=np.bool_)
    frames = np.arange(T)
    for slot, mano_idx in enumerate(CONTACT_MANO_INDICES):
        labeled = np.isin(labels, finger_label_ids[slot // 2])
        deltas = verts_world - mano[:, mano_idx, None]
        distance2 = np.einsum("fvi,fvi->fv", deltas, deltas)
        distance2[~labeled] = np.inf
        nearest = distance2.argmin(axis=1)
        valid = labeled.any(axis=1) & (np.linalg.norm(deltas[frames, nearest], axis=1) < CONTACT_DISTANCE)
        contact_points[valid, slot] = verts_world[frames[valid], nearest[valid]]
        contact_mask[:, slot] = valid

    distance = np.linalg.norm(mano[:, CONTACT_MANO_INDICES] - contact_points, axis=-1)
    for slot in range(0, len(CONTACT_MANO_INDICES), 2):
        both = contact_mask[:, slot] & contact_mask[:, slot + 1]
        if both.any():
            first = np.median(distance[contact_mask[:, slot], slot])
            second = np.median(distance[contact_mask[:, slot + 1], slot + 1])
            drop = slot + 1 if first <= second else slot
            contact_mask[both, drop] = False
    if contact_mask.any():
        local = np.einsum("fji,fsj->fsi", rot, contact_points - t_obj[:, None])
        closest, _, triangle = trimesh.proximity.closest_point(mesh, local[contact_mask])
        contact_frames = np.nonzero(contact_mask)[0]
        contact_points[contact_mask] = np.einsum("fij,fj->fi", rot[contact_frames], closest) + t_obj[contact_frames]
        contact_normals[contact_mask] = np.einsum("fij,fj->fi", rot[contact_frames], mesh.face_normals[triangle])

    human_pose7 = joint_pose7.copy()
    human_pose7[..., :3] = joints - ground

    return ComboClip(
        human_pose7=human_pose7.astype(np.float32),
        human_height=height,
        scale=scale,
        fps=fps,
        hands={
            "right": HandTrack(
                mano=mano.astype(np.float32),
                mano_overlay=(mano - t_obj[:, None] + object_pose[:, None, :3]).astype(np.float32),
                wrist_pos=joints[:, SMPLX_JOINT_NAMES.index("right_wrist")].astype(np.float32),
                wrist_quat_wxyz=joint_pose7[:, SMPLX_JOINT_NAMES.index("right_wrist"), 3:].astype(np.float32),
                object_name=object_name,
                object_mesh=mesh,
                object_pose_native=object_pose_native,
                object_pose_world=object_pose,
                contact=labels.sum(axis=1) > 0,
                contact_points=contact_points,
                contact_normals=contact_normals,
                contact_mask=contact_mask,
            )
        },
    )


__all__ = ["GRAB_FPS", "load_saga_clip"]
