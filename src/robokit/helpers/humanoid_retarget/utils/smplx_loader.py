from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R


def _prepare_betas(smplx_data: dict, body_model) -> torch.Tensor:
    """Trim or pad raw betas to match the body model's shape dimension."""
    betas_raw = smplx_data["betas"]
    if betas_raw.ndim > 1:
        betas_raw = betas_raw.reshape(-1)

    shapedirs_tensor = getattr(body_model, "shapedirs", None)
    shape_dim = int(shapedirs_tensor.shape[-1]) if shapedirs_tensor is not None else 10
    betas_trimmed = (
        betas_raw[:shape_dim]
        if len(betas_raw) > shape_dim
        else np.pad(betas_raw, (0, int(max(0, shape_dim - len(betas_raw)))))
    )
    return torch.tensor(betas_trimmed).float().view(1, -1)


def compute_height_from_tpose(body_model, betas: torch.Tensor) -> float:
    """Compute human height from T-pose mesh vertices (Y-up).

    Runs a single-frame SMPL-X forward pass with zero pose and measures
    the vertical extent of the mesh. This is the standard approach used
    by SMPL-Anthropometry and other body measurement tools.
    """
    with torch.no_grad():
        tpose_output = body_model(
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
    verts = tpose_output.vertices.detach().cpu().numpy().squeeze()  # [10475, 3]
    return float(verts[:, 1].max() - verts[:, 1].min())


def load_smplx_file(smplx_file: str, smplx_body_model_path: str):
    smplx_data_raw = np.load(smplx_file, allow_pickle=True)
    smplx_data = dict(smplx_data_raw)

    if "poses" in smplx_data and "pose_body" not in smplx_data:
        poses = smplx_data["poses"]
        smplx_data["root_orient"] = poses[:, :3]
        smplx_data["pose_body"] = poses[:, 3:66]

    if "mocap_framerate" in smplx_data and "mocap_frame_rate" not in smplx_data:
        smplx_data["mocap_frame_rate"] = smplx_data["mocap_framerate"]

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data.get("gender", "neutral")),
        use_pca=False,
    )

    num_frames = smplx_data["pose_body"].shape[0]
    expression = torch.zeros(num_frames, body_model.num_expression_coeffs).float()

    betas = _prepare_betas(smplx_data, body_model)

    smplx_output = body_model(
        betas=betas,
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        expression=expression,
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )

    human_height = compute_height_from_tpose(body_model, betas)

    return smplx_data, body_model, smplx_output, human_height


def _slerp_rotvec(rotvec1: np.ndarray, rotvec2: np.ndarray, alpha: float) -> np.ndarray:
    r1 = R.from_rotvec(rotvec1)
    r2 = R.from_rotvec(rotvec2)
    return R.slerp(0.0, 1.0, [r1, r2])(alpha).as_rotvec()


def get_smplx_frames_offline_fast(
    smplx_data, body_model, smplx_output, tgt_fps: int = 30
) -> Tuple[List[Dict[str, Tuple[np.ndarray, np.ndarray]]], float, float]:
    from smplx.joint_names import JOINT_NAMES

    src_fps = smplx_data["mocap_frame_rate"].item() if "mocap_frame_rate" in smplx_data else 60
    num_frames = smplx_data["pose_body"].shape[0]
    parents = body_model.parents

    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().cpu().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(parents)]

    if tgt_fps < src_fps and src_fps % tgt_fps == 0:
        step = int(src_fps // tgt_fps)
        idx = np.arange(0, num_frames, step)
        global_orient = global_orient[idx]
        full_body_pose = full_body_pose[idx]
        joints = joints[idx]
        aligned_fps = tgt_fps
    else:
        aligned_fps = src_fps

    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    for t in range(global_orient.shape[0]):
        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        joint_orientations: List[R] = []
        go = global_orient[t]
        bp = full_body_pose[t]
        jp = joints[t]

        for j, name in enumerate(joint_names):
            if j == 0:
                rot = R.from_rotvec(go)
            else:
                rot = joint_orientations[parents[j]] * R.from_rotvec(bp[j])
            joint_orientations.append(rot)
            _xyzw = rot.as_quat()
            result[name] = (jp[j], np.array([_xyzw[3], _xyzw[0], _xyzw[1], _xyzw[2]], dtype=np.float32))

        frames.append(result)

    betas = _prepare_betas(smplx_data, body_model)
    human_height = compute_height_from_tpose(body_model, betas)

    return frames, float(aligned_fps), human_height
