"""Torch reference LBS for parametric body models (SMPL / SMPL-X / MANO / FLAME)."""

from typing import Dict

import torch

from robokit.smplx.spec import BodyModelSpec
from robokit.smplx.state import BodyModelState
from robokit.xform.warp.torch_wrappers import matrix_to_quaternion


def body_lbs_torch(
    spec: BodyModelSpec,
    state: BodyModelState,
    return_landmarks: bool = True,
) -> Dict[str, torch.Tensor]:
    """Run forward kinematics + linear blend skinning on Torch.

    Returns a dict with:

    - `"vertices"` `[B, V, 3]` - posed mesh vertices in world coords.
    - `"joints"` `[B, J, 3]` - posed joint positions in world coords.
    - `"global_T_world_joint_xyz_wxyz"` `[B, J, 7]` - per-joint world pose
      in `wp_vec7` layout `(x, y, z, qw, qx, qy, qz)`.
    - `"landmarks"` `[B, L, 3]` - only when `return_landmarks=True` and
      the spec defines `static_landmark_vertex_ids`.

    Generic over `V`, `J`, `P`: shapes are read from `spec`.
    """
    # `state` may carry numpy (default runtime path) or torch tensors
    # (differentiable callers such as `mano_forward_pca_mm`); `torch.as_tensor`
    # handles both without copying tensors or breaking autograd.
    betas_t = torch.as_tensor(state.betas, dtype=torch.float32)
    full_pose_t = torch.as_tensor(state.full_pose_aa, dtype=torch.float32)
    transl_t = torch.as_tensor(state.transl, dtype=torch.float32)

    device = full_pose_t.device
    dtype = full_pose_t.dtype
    B = state.batch_size
    V = spec.v_template.shape[0]
    J = spec.J_regressor.shape[0]

    v_template = torch.from_numpy(spec.v_template).to(device=device, dtype=dtype)
    shapedirs = torch.from_numpy(spec.shapedirs).to(device=device, dtype=dtype)
    posedirs = torch.from_numpy(spec.posedirs).to(device=device, dtype=dtype)
    J_regressor = torch.from_numpy(spec.J_regressor).to(device=device, dtype=dtype)
    lbs_weights = torch.from_numpy(spec.lbs_weights).to(device=device, dtype=dtype)
    parents = spec.parents

    v_shaped = v_template + torch.einsum("bl,vkl->bvk", betas_t, shapedirs)
    J_rest = torch.einsum("bvk,jv->bjk", v_shaped, J_regressor)

    # Rodrigues with additive 1e-8 nudge - matches smplx.lbs epsilon handling so
    # the Warp path tracks float32 precision near the identity rotation.
    pose_flat = full_pose_t.reshape(-1, 3)
    angle = torch.norm(pose_flat + 1e-8, dim=1, keepdim=True)
    rot_dir = pose_flat / angle
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    rx, ry, rz = rot_dir.unbind(dim=1)
    zero = torch.zeros_like(rx)
    K = torch.stack(
        [
            torch.stack([zero, -rz, ry], dim=1),
            torch.stack([rz, zero, -rx], dim=1),
            torch.stack([-ry, rx, zero], dim=1),
        ],
        dim=1,
    )
    ident = torch.eye(3, device=device, dtype=dtype).expand_as(K)
    rot_mats = (ident + sin * K + (1.0 - cos) * torch.bmm(K, K)).reshape(B, J, 3, 3)

    ident3 = torch.eye(3, device=device, dtype=dtype)
    pose_feature = (rot_mats[:, 1:] - ident3).reshape(B, -1)
    pose_offsets = (pose_feature @ posedirs).reshape(B, V, 3)
    v_posed = v_shaped + pose_offsets

    rel_joints = J_rest.clone()
    rel_joints[:, 1:] = rel_joints[:, 1:] - J_rest[:, parents[1:]]

    T_local = torch.zeros(B, J, 4, 4, device=device, dtype=dtype)
    T_local[:, :, :3, :3] = rot_mats
    T_local[:, :, :3, 3] = rel_joints
    T_local[:, :, 3, 3] = 1.0

    T_world_joint_list = [T_local[:, 0]]
    for i in range(1, J):
        T_world_joint_list.append(T_world_joint_list[int(parents[i])] @ T_local[:, i])
    T_world_joint = torch.stack(T_world_joint_list, dim=1)

    J_posed = T_world_joint[:, :, :3, 3]

    # cancel out the rest-pose joint center so LBS rotations pivot correctly
    J_rest_hzero = torch.cat([J_rest, torch.zeros(B, J, 1, device=device, dtype=dtype)], dim=-1).unsqueeze(-1)
    offset = (T_world_joint @ J_rest_hzero).squeeze(-1)
    A = T_world_joint.clone()
    A[:, :, :, 3] = A[:, :, :, 3] - offset

    T_skin = torch.einsum("vj,bjnm->bvnm", lbs_weights, A)
    v_posed_h = torch.cat([v_posed, torch.ones(B, V, 1, device=device, dtype=dtype)], dim=-1)
    verts = (T_skin @ v_posed_h.unsqueeze(-1)).squeeze(-1)[:, :, :3]

    transl_b1x3 = transl_t.unsqueeze(1)
    verts_world = verts + transl_b1x3
    joints_world = J_posed + transl_b1x3

    quat_wxyz = matrix_to_quaternion(T_world_joint[:, :, :3, :3])
    global_T_world_joint = torch.cat([joints_world, quat_wxyz], dim=-1)

    out: Dict[str, torch.Tensor] = {
        "vertices": verts_world,
        "joints": joints_world,
        "global_T_world_joint_xyz_wxyz": global_T_world_joint,
    }
    if return_landmarks and spec.static_landmark_vertex_ids is not None:
        lmk_idx = torch.from_numpy(spec.static_landmark_vertex_ids).to(device=device)
        out["landmarks"] = verts_world.index_select(dim=1, index=lmk_idx)
    return out
