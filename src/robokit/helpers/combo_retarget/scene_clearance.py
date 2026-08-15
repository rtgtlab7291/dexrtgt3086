"""Whole-body scene clearance - push the solved body out of penetrated scene parts.

A single whole-body IK per clip: `SceneCollisionTask` on the core-body spheres vs every scene
part, `PositionTask` pinning feet and hands at their achieved poses, and `RestTask` holding the
current pose. Links whose human counterpart is weight-bearing on the scene (sitting, leaning)
are excluded from the collision guard. Works with any scene, but link names assume G1 + Inspire.
"""

from typing import Dict, List

import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree

from robokit.geom import WarpScene
from robokit.geom.geoms import MeshGeom
from robokit.helpers.combo_retarget.clip import ComboClip
from robokit.helpers.ik import IK
from robokit.helpers.ik.config import IKConfig
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.robo import Robot
from robokit.smplx import SMPLX_JOINT_NAMES
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.xform.numpy import pose7_to_tf_mat, tf_mat_to_pose7


SCENE_MARGIN = 0.005  # m, target clearance between core spheres and the scene
SUPPORT_DIST = 0.03  # m, human-to-scene distance that counts as weight-bearing
SUPPORT_FRAC = 0.5  # fraction of frames within SUPPORT_DIST that marks a support link
MAX_CORE_SPHERES = 240  # cap guarded spheres so the LM Jacobian fits GPU shared memory
# robot link name -> SMPL-X joint of its human region, for the weight-bearing gate (G1 naming)
_SIDED_SEGMENTS = {
    "hip": "hip",
    "knee": "knee",
    "ankle_pitch": "ankle",
    "ankle_roll": "foot",
    "shoulder": "shoulder",
    "elbow": "elbow",
    "wrist": "wrist",
}
_CENTRAL_SEGMENTS = {"pelvis": "pelvis", "waist": "pelvis", "torso": "spine2", "head": "head"}


def apply_scene_clearance(robot: Robot, combo: ComboClip, qpos: np.ndarray, device: str) -> np.ndarray:
    """Eject the core body from the scene it over-penetrates; see the module docstring."""
    scene = combo.scene
    if scene is None or not scene.names:
        return qpos
    num_frames = qpos.shape[0]
    spec = robot.spec
    nd = spec.num_actuated_joints
    n_sph = spec.local_collision_sphere_centers.shape[0]
    sphere_links = [spec.link_names[i] for i in spec.collision_spheres_link_indices]

    # Human weight-bearing links (sitting/leaning) are left out of the guard, so support survives.
    name_to_idx = {n: i for i, n in enumerate(SMPLX_JOINT_NAMES)}
    hj = combo.human_pose7[:num_frames, :, :3] * combo.scale
    trees = [cKDTree(np.asarray(scene.meshes[p].vertices, dtype=np.float32)) for p in range(len(scene.names))]
    part_rots = [pose7_to_tf_mat(scene.poses[p][:num_frames])[:, :3, :3] for p in range(len(scene.names))]
    support: set = set()
    for ln in set(sphere_links):
        side, _, rest = ln.partition("_")
        segments = _SIDED_SEGMENTS if side in ("left", "right") else _CENTRAL_SEGMENTS
        joint = next(
            (v for k, v in segments.items() if (rest if side in ("left", "right") else ln).startswith(k)), None
        )
        j = None if joint is None else name_to_idx.get(f"{side}_{joint}" if side in ("left", "right") else joint)
        if j is None:
            continue
        d = np.full(num_frames, np.inf, dtype=np.float32)
        for p in range(len(scene.names)):
            pose = scene.poses[p][:num_frames]
            d = np.minimum(d, trees[p].query(np.einsum("tji,tj->ti", part_rots[p], hj[:, j] - pose[:, :3]))[0])
        if float((d < SUPPORT_DIST).mean()) > SUPPORT_FRAC:
            support.add(ln)

    def is_core(ln: str) -> bool:
        return (
            not ln.startswith(("R_", "L_"))
            and all(k not in ln for k in ("ankle", "wrist", "hand"))
            and ln not in support
        )

    core = [i for i in range(n_sph) if is_core(sphere_links[i])]
    if not core:
        return qpos
    if len(core) > MAX_CORE_SPHERES:  # per-link subsample so every guarded link keeps representatives
        by_link: Dict[int, List[int]] = {}
        for i in core:
            by_link.setdefault(int(spec.collision_spheres_link_indices[i]), []).append(i)
        per = max(1, MAX_CORE_SPHERES // len(by_link))
        core = [i for sphs in by_link.values() for i in sphs[:: max(1, len(sphs) // per)][:per]]

    # the whole scene, one BVH per part instanced at every frame pose; scene t owns query row t
    P = len(scene.names)
    wp_meshes = [
        wp.Mesh(
            points=wp.array(np.asarray(m.vertices, np.float32), dtype=wp.vec3, device=device),
            indices=wp.array(np.asarray(m.faces, np.int32).reshape(-1), device=device),
        )
        for m in scene.meshes
    ]
    inst_meshes = [wp_meshes[p] for _ in range(num_frames) for p in range(P)]
    # instance order is (t, p): frame-major, matching scene_offsets = arange(T + 1) * P
    inst_poses = pose7_to_tf_mat(scene.poses[:, :num_frames].transpose(1, 0, 2).reshape(-1, 7))
    warp_scene = WarpScene(num_scenes=num_frames, device=device).add(
        MeshGeom(
            inst_meshes,
            scene_offsets=np.arange(num_frames + 1) * P,
            poses=wp.array(inst_poses, dtype=wp.mat44, device=device),
        )
    )

    hold = ["left_ankle_roll_link", "right_ankle_roll_link", "L_hand_base_link", "R_hand_base_link"]
    missing = [ln for ln in hold if ln not in spec.link_names]
    assert not missing, f"scene clearance assumes G1 + Inspire link names; missing {missing}"
    ik_cfg = (
        IKConfig(
            init_sample_range=0.0,
            base_init_sample_range=0.0,
            enable_T_world_base=True,
            base_lock_mask=(0.0,) * 6,
            solver=MultiSeedSolverConfig(
                stages=[StageConfig(num_seeds=1, iters=20, lm_lambda=1.0)], cuda_graph_mode="none"
            ),
        )
        .add(PositionTask(weight=400.0))  # feet + hands are functional contacts: pin hard
        .add(
            SceneCollisionTask(
                scene=warp_scene,
                sphere_indices=core,
                weight=600.0,
                margin=SCENE_MARGIN,
                scene_indices=wp.from_numpy(np.arange(num_frames, dtype=np.int32), dtype=wp.int32, device=device),
            )
        )
        .add(RestTask(robot, weight=15.0, base_weight=15.0))  # regularize toward the CURRENT pose (rest_q below)
        .add(PositionLimit(weight=50.0))
    )
    ik = IK(ik_cfg, robot=robot, link=hold, device=device)
    finger = {i for i, n in enumerate(spec.actuated_joint_names) if n.startswith(("R_", "L_"))}
    ik.set_active_joint_mask([0.0 if i in finger else 1.0 for i in range(nd)])

    base = pose7_to_tf_mat(qpos[:, :7]).astype(np.float32)
    links = (
        robot.forward_kinematics_via_matrix_torch(
            torch.from_numpy(qpos[:, 7 : 7 + nd].astype(np.float32)).to(device), torch.from_numpy(base).to(device)
        )
        .cpu()
        .numpy()
    )
    hold_tgt = links[:, [spec.link_names.index(ln) for ln in hold]]  # (T, H, 4, 4) world poses to preserve
    q_cur = torch.from_numpy(qpos[:, 7 : 7 + nd].astype(np.float32)).to(device)
    base_cur = torch.from_numpy(base).to(device)
    snap = ik.solve_torch(
        torch.from_numpy(hold_tgt.astype(np.float32)).to(device),
        init_q=q_cur,
        init_T_world_base=base_cur,
        rest_q=q_cur,
        rest_T_world_base=base_cur,  # RestTask holds the CURRENT pose, not the robot's default rest
    )
    qpos = qpos.copy()
    qpos[:, 7 : 7 + nd] = snap.q.cpu().numpy()
    qpos[:, :7] = tf_mat_to_pose7(snap.T_world_base.cpu().numpy())
    return qpos


__all__ = ["apply_scene_clearance"]
