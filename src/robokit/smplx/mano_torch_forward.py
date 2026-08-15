"""Differentiable MANO forward pass."""

from typing import Tuple

import torch

from robokit.smplx.mano_constants import MANO_NUM_JOINTS, MANOPTH_KEYPOINT_PERMUTATION
from robokit.smplx.spec import BodyModelSpec
from robokit.smplx.state import BodyModelState
from robokit.smplx.torch_lbs import body_lbs_torch


def mano_forward_pca_mm(
    spec: BodyModelSpec,
    pose_48: torch.Tensor,
    betas_10: torch.Tensor,
    transl: torch.Tensor,
    flat_hand_mean: bool,
    use_pca: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MANO LBS in **millimeters** with manopth-compatible inputs.

    Pose layout `[B, 48]` = `[global_orient(3), finger(45)]`. The 45
    finger components are PCA-encoded when `use_pca=True` (manopth's
    default) and raw axis-angle when `use_pca=False`. Returns
    `(vertices_mm, joints_21_mm)` with joints in OpenPose hand-keypoint
    order (wrist, thumb×3+tip, …). Differentiable end-to-end.
    """
    B = pose_48.shape[0]
    global_orient = pose_48[:, :3]
    finger_input = pose_48[:, 3:48]
    if use_pca:
        hands_components = torch.from_numpy(spec.metadata["hands_components"]).to(pose_48)
        finger_input = finger_input @ hands_components
    if not flat_hand_mean:
        hands_mean = torch.from_numpy(spec.metadata["hands_mean"]).to(pose_48)
        finger_input = finger_input + hands_mean
    full_pose_aa = torch.cat([global_orient.unsqueeze(1), finger_input.reshape(B, MANO_NUM_JOINTS - 1, 3)], dim=1)

    state = BodyModelState(betas=betas_10, full_pose_aa=full_pose_aa, transl=transl)
    out = body_lbs_torch(spec, state, return_landmarks=True)
    joints_native = torch.cat([out["joints"], out["landmarks"]], dim=1)
    perm = torch.as_tensor(MANOPTH_KEYPOINT_PERMUTATION, device=joints_native.device, dtype=torch.long)
    joints_21 = joints_native.index_select(dim=1, index=perm)
    return out["vertices"] * 1000.0, joints_21 * 1000.0
