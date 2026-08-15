"""Combo driver: `ComboClip` -> qpos `[T, 7 + dof + 7 * num_objects]`.

Every stage runs per engaged hand, so one- and two-handed clips take the same path:

1. hand pass - captured MANO keypoints onto the standalone hand, in the native world;
2. body pass - whole-body tracking with each grasp wrist injected as a high-weight target track;
3. arm snap - an arm-only IK puts each palm exactly at the hand-pass pose;
4. contact refine - fingers re-solve at the achieved wrist against that hand's object, with
   sphere-vs-mesh collision. Stages 2-4 iterate twice so the body absorbs the wrist correction.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import warp as wp
from scipy.ndimage import uniform_filter1d

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.combo_retarget.clip import CONTACT_MANO_INDICES, ComboClip, HandTrack
from robokit.helpers.combo_retarget.config import ComboPreset
from robokit.helpers.combo_retarget.scene_clearance import apply_scene_clearance
from robokit.helpers.hand_retargeting import HandRetargetingOffline
from robokit.helpers.hand_retargeting.config import HandRetargetingOfflineConfig, HandSpec
from robokit.helpers.hand_retargeting.presets import TARGET_NAMES
from robokit.helpers.hand_retargeting.presets import inspire as inspire_preset
from robokit.helpers.humanoid_retarget.offline import HumanoidRetargetingOffline
from robokit.helpers.humanoid_retarget.presets.g1_offline import g1_offline as kin_g1_offline
from robokit.helpers.humanoid_retarget.presets.g1_offline_mapping import (
    g1_offline_mapping as kin_g1_offline_mapping,
)
from robokit.helpers.ik import IK
from robokit.helpers.ik.config import IKConfig
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizerConfig
from robokit.robo import Robot
from robokit.smplx import SMPLX_JOINT_NAMES
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import inverse_tf_mat, pose7_to_tf_mat, tf_mat_to_pose7


@dataclass
class HandTrajectory:
    """Externally optimized standalone-hand trajectory: ``q (T, D)``, base pose ``(T, 7)``."""

    q: np.ndarray
    T_world_base: np.ndarray


class _KinematicOffline(HumanoidRetargetingOffline):
    """Offline body solver + per-link target-track overrides (the grasp wrist injection)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_overrides: Dict[int, np.ndarray] = {}  # target column -> (T, 7) pose track

    def _compute_targets(self, T_world_human: wp.array, human_heights: wp.array) -> wp.array:
        out = super()._compute_targets(T_world_human, human_heights)
        if self.target_overrides:
            buf = self._target_buf_wp.numpy()  # (B, T, num_targets, 7)
            for col, track in self.target_overrides.items():
                buf[0, : track.shape[0], col] = track
            self._target_buf_wp.assign(buf)
        return out


def _smooth_pose7(pose7: np.ndarray, window: int) -> np.ndarray:
    """Low-pass a ``(T, 7)`` pose track: quaternions are hemisphere-aligned to the previous frame
    first, so the filter never averages across a sign flip."""
    out = pose7.copy()
    out[:, :3] = uniform_filter1d(out[:, :3], window, axis=0, mode="nearest")
    quats = out[:, 3:]
    quats[1:] *= np.cumprod(np.where(np.einsum("fi,fi->f", quats[1:], quats[:-1]) < 0.0, -1.0, 1.0))[:, None]
    quats[:] = uniform_filter1d(quats, window, axis=0, mode="nearest")
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    return out


def solve_hand_pass(
    robot: Robot,
    spec: HandSpec,
    track: HandTrack,
    num_frames: int,
    device: str,
    *,
    hand_span: float,
    **weight_overrides: float,
) -> np.ndarray:
    """Captured MANO keypoints -> packed `(T, 7 + dof)` on the floating standalone hand.

    Keypoints are scaled about the wrist by robot-hand-span / subject-hand-span, so small hands
    don't trade wrist accuracy for finger fit.
    """
    human_span = float(np.median(np.linalg.norm(track.mano[:num_frames, 12] - track.mano[:num_frames, 0], axis=-1)))
    config = replace(
        inspire_preset.offline, target_scale=float(np.clip(hand_span / human_span, 0.9, 1.4)), **weight_overrides
    )
    # The solver reads the root position off keypoint 0; pin it to the wrist track.
    target_points = track.mano[:num_frames].copy()
    target_points[:, 0] = track.wrist_pos[:num_frames]
    return HandRetargetingOffline(robot, spec, config, device=device).solve_numpy(
        target_points, root_quat_wxyz=track.wrist_quat_wxyz[:num_frames]
    )


def compute_inward_contact_points(
    robot: Robot, slot_indices: np.ndarray, T_obj_link: np.ndarray, normals_obj: np.ndarray
) -> np.ndarray:
    """Per-slot link-mesh surface point facing furthest into the object; the origin for meshless
    links, whose frame is already the surface point."""
    inward_local = np.einsum("fsji,fsj->fsi", T_obj_link[..., :3, :3], -normals_obj)
    points = np.zeros(inward_local.shape, dtype=np.float32)
    for s, link in enumerate(slot_indices):
        geom = robot.spec.link_visual_geometries.get(robot.spec.link_names[link])
        vertices = None if geom is None else np.asarray(geom.to_mesh().vertices, np.float32)
        if vertices is not None and vertices.shape[0]:
            points[:, s] = vertices[np.argmax(inward_local[:, s] @ vertices.T, axis=1)]
    return points


def build_contact_refine_config(
    *,
    contact_weight: float,
    contact_margin: float,
    collision_weight: float,
    collision_margin: float,
    anchor_q_weight: float,
    anchor_base_weight: float,
    optimize_base: bool,
) -> HandRetargetingOfflineConfig:
    """Anchored contact-only refine: tracking weights off, so the solution answers to the contact
    labels and the anchor. Base smoothness is stiff because the optimized base can become the next
    body pass's wrist target."""
    return HandRetargetingOfflineConfig(
        global_position_weight=0.0,
        root_position_weight=0.0,
        root_orientation_weight=0.0,
        vector_weight=0.0,
        direction_weight=0.0,
        velocity_weight=10.0,
        acceleration_weight=20.0,
        root_position_velocity_weight=100.0,
        root_orientation_velocity_weight=50.0,
        limit_weight=100.0,
        rest_weight=0.0,
        contact_weight=contact_weight,
        contact_margin=contact_margin,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
        anchor_q_weight=anchor_q_weight,
        anchor_base_weight=anchor_base_weight,
        optimize_base=optimize_base,
        locked_prefix_frames=2,  # the first frames set the trajectory's reference velocity
        solver=SparseLMOptimizerConfig(lm_lambda=0.5, max_iter=30, use_cuda_graph=True),
    )


class ComboRetargeter:
    """Whole-body pass + captured-finger hand retargeting, with an optional contact refine.

    See the module docstring for the stages; grasp synthesis lives in `grasp.py`.
    """

    def __init__(
        self,
        preset: ComboPreset,
        device: str = "cuda:0",
        load_meshes: bool = False,
        contact_refine: bool = True,
        scene_clearance: bool = False,  # eject the core body from furniture it over-penetrates
    ):
        self.preset = preset
        self.device = device
        self.contact_refine = contact_refine
        self.scene_clearance = scene_clearance
        assert preset.body.urdf_path is not None
        self.body_robot = preset.body.load_robot(load_meshes=load_meshes)

        # Per-side standalone hands, assets repaired first (HandBinding.prepare_assets).
        self.hand_robots: Dict[str, Robot] = {}
        self._hand_spec = {}
        self._T_base_wrist: Dict[str, np.ndarray] = {}
        self._hand_span: Dict[str, float] = {}
        self._arm_ik: Optional[IK] = None  # solver is batch- and side-set-bound; rebuilt on change
        self._arm_ik_key: Tuple = ()
        # Warm refine solvers, (side, optimize_base) -> (mesh, num_frames, solver): building one
        # compiles three optimizers + CUDA graphs (seconds), so re-solves of the same clip reuse it.
        self._refiners: Dict[Tuple[str, bool], Tuple[object, int, HandRetargetingOffline]] = {}
        # The refine's contact slots: the 10 labelled finger keypoints, in CONTACT_MANO_INDICES order.
        # Naming them on the spec is what selects their links AND the finger-chain collision spheres.
        self._contact_target_names = tuple(TARGET_NAMES[m] for m in CONTACT_MANO_INDICES)
        for side, binding in preset.hands.items():
            urdf, spheres, _ = binding.prepare_assets(
                preset.body.urdf_path, device, sphere_source=preset.hands.get("right")
            )
            robot = Robot.load(
                urdf,
                load_meshes=True,  # the refine attracts fingertip MESH surface points
                mesh_dir=str(Path(binding.urdf).parent),
                load_collision_spheres=True,
                collision_spheres_path=spheres,
            )
            self.hand_robots[side] = robot
            # The hand pass runs scene-free, so the sphere set only matters for the refine (which
            # uses ``robot``); the per-side spec just carries this side's wrist convention.
            self._hand_spec[side] = replace(
                inspire_preset.spec,
                root_link_coord_spec=binding.wrist_coord_spec or inspire_preset.spec.root_link_coord_spec,
            )
            eye = torch.eye(4, device=device)[None]
            q_rest = torch.zeros((1, robot.spec.num_actuated_joints), device=device)
            rest_links = robot.forward_kinematics_via_matrix_torch(q_rest, eye).cpu().numpy()[0]
            self._T_base_wrist[side] = rest_links[robot.spec.link_names.index(binding.hand_wrist_link)]
            # Robot hand span (wrist -> middle fingertip at rest) for per-subject keypoint calibration.
            middle_tip = inspire_preset.spec.target_link_names["middle_tip"]
            self._hand_span[side] = float(
                np.linalg.norm(
                    rest_links[robot.spec.link_names.index(middle_tip), :3, 3] - self._T_base_wrist[side][:3, 3]
                )
            )

        # Per-limb tracking on the plain G1 with each grasp wrist injected as a high-weight target
        # track; the 29 body joints transfer into the assembly by name.
        self._T_handbase_wristyaw: Dict[str, np.ndarray] = {}
        link_mapping = dict(kin_g1_offline_mapping.link_mapping)
        for binding in preset.hands.values():
            link_mapping[binding.wrist_yaw_link] = replace(
                link_mapping[binding.wrist_yaw_link], position_weight=500, orientation_weight=150
            )
            link_mapping[binding.elbow_link] = replace(link_mapping[binding.elbow_link], position_weight=60)
        mapping = replace(kin_g1_offline_mapping, link_mapping=link_mapping)
        self._kin_robot = mapping.load_robot()
        assert isinstance(kin_g1_offline.rest_weight, list)
        rest = list(kin_g1_offline.rest_weight)
        kin_names = self._kin_robot.spec.actuated_joint_names
        for binding in preset.hands.values():
            # that side's shoulder: stop fighting the reach (pitch/roll/yaw, decreasing hold)
            for axis, weight in zip(("pitch", "roll", "yaw"), (1.0, 0.5, 0.1)):
                rest[kin_names.index(f"{binding.arm_prefix}shoulder_{axis}_joint")] = weight
        self._kin = _KinematicOffline(
            mapping, replace(kin_g1_offline, rest_weight=rest), robot=self._kin_robot, device=device
        )
        # Human input columns and override target columns, both in the mapping's canonical orders.
        self._kin_input_cols = [SMPLX_JOINT_NAMES.index(n) for n in self._kin.human_joint_names]
        target_order = list(mapping.link_mapping)
        self._kin_target_col = {
            side: target_order.index(binding.wrist_yaw_link) for side, binding in preset.hands.items()
        }
        asm_names = self.body_robot.spec.actuated_joint_names
        self._kin_cols = np.array([asm_names.index(n) for n in kin_names])
        # Fixed hand mount: desired <side>_hand_base_link pose -> desired wrist-yaw-link pose.
        q0 = torch.zeros((1, self.body_robot.spec.num_actuated_joints), device=device)
        fk0 = (
            self.body_robot.forward_kinematics_via_matrix_torch(q0, torch.eye(4, device=device)[None]).cpu().numpy()[0]
        )
        for side, binding in preset.hands.items():
            wl = self.body_robot.spec.link_names.index(binding.wrist_yaw_link)
            hb = self.body_robot.spec.link_names.index(binding.wrist_link)
            self._T_handbase_wristyaw[side] = inverse_tf_mat(fk0[hb][None])[0] @ fk0[wl]

    def _arm_columns(self, side: str) -> List[int]:
        """Assembly joint indices for one side's shoulder/elbow/wrist chain."""
        prefix = self.preset.hands[side].arm_prefix
        return [
            i
            for i, n in enumerate(self.body_robot.spec.actuated_joint_names)
            if prefix in n and ("shoulder" in n or "elbow" in n or "wrist" in n)
        ]

    def solve(
        self,
        combo: ComboClip,
        /,
        max_frames: int = 0,
        *,
        hand_trajectories: Optional[Dict[str, HandTrajectory]] = None,
    ) -> np.ndarray:
        num_frames = combo.num_frames if max_frames == 0 else min(max_frames, combo.num_frames)
        sides = [s for s in combo.sides if s in self.preset.hands]
        assert sides, f"clip has hands {combo.sides} but the preset binds {list(self.preset.hands)}"
        hand_trajectories = hand_trajectories or {}
        assert set(hand_trajectories) <= set(sides), f"unknown hand trajectories: {set(hand_trajectories) - set(sides)}"
        # Sides that grasp an object (get a body anchor + contact refine) vs free hands (fingers only).
        object_sides = [s for s in sides if combo.hands[s].object_mesh is not None]
        assert object_sides, f"no grasped object among hands {sides}"

        # Hand pass, weights favouring the wrist: its trajectory drives the body anchors, while
        # the refine owns fingertip contact.
        finger_q: Dict[str, np.ndarray] = {}
        hand_links: Dict[str, np.ndarray] = {}
        for side in sides:
            track = combo.hands[side]
            if side in hand_trajectories:
                trajectory = hand_trajectories[side]
                assert trajectory.q.shape[0] >= num_frames
                assert trajectory.q.shape[1:] == (self.hand_robots[side].spec.num_actuated_joints,)
                assert trajectory.T_world_base.shape[0] >= num_frames
                assert trajectory.T_world_base.shape[1:] == (7,)
                finger_q[side] = trajectory.q[:num_frames].astype(np.float32)
                base_world = pose7_to_tf_mat(trajectory.T_world_base[:num_frames]).astype(np.float32)
                if num_frames >= 3:
                    finger_q[side] = uniform_filter1d(finger_q[side], 3, axis=0, mode="nearest")
                    T_world_obj = pose7_to_tf_mat(track.object_pose_world[:num_frames])
                    base_obj = _smooth_pose7(tf_mat_to_pose7(inverse_tf_mat(T_world_obj) @ base_world), 3)
                    base_world = T_world_obj @ pose7_to_tf_mat(base_obj)
            else:
                packed = solve_hand_pass(
                    self.hand_robots[side],
                    self._hand_spec[side],
                    track,
                    num_frames,
                    self.device,
                    hand_span=self._hand_span[side],
                    root_position_weight=30.0,
                    global_position_weight=10.0,
                )
                finger_q[side] = packed[:, 7:]  # (T, hand_dof) standalone actuated order
                # Native -> scaled body world through THAT HAND's object frame (shared rotations,
                # scaled translation), so each hand rides its own object with native offsets intact.
                bridge = pose7_to_tf_mat(track.object_pose_world[:num_frames]) @ inverse_tf_mat(
                    pose7_to_tf_mat(track.object_pose_native[:num_frames])
                )
                base_world = (bridge @ pose7_to_tf_mat(packed[:, :7])).astype(np.float32)
            hand_links[side] = (
                self.hand_robots[side]
                .forward_kinematics_via_matrix_torch(
                    torch.from_numpy(finger_q[side].astype(np.float32)).to(self.device),
                    torch.from_numpy(base_world).to(self.device),
                )
                .cpu()
                .numpy()
            )

        # qpos tail: one 7-pose per distinct grasped object, in canonical hand order (free hands add none).
        world_by_object = {
            combo.hands[s].object_name: combo.hands[s].object_pose_world[:num_frames] for s in object_sides
        }
        object_tail = np.concatenate([world_by_object[n] for n in combo.object_names], axis=1).astype(np.float32)

        wrist_link_idx = {
            s: self.hand_robots[s].spec.link_names.index(self.preset.hands[s].hand_wrist_link) for s in sides
        }
        # Standalone-hand indices of the palm-rigid orientation links (the arm snap's extra anchors).
        hand_idx = {
            s: [
                self.hand_robots[s].spec.link_names.index(ln.removeprefix(self.preset.hands[s].joint_prefix))
                for ln in self.preset.hands[s].orientation_links
            ]
            for s in sides
        }

        # Iterated: refine 1 runs with a SOFT base to discover the wrist correction each grasp
        # wants, pass 2 re-welds the arms to it, refine 2 finishes with the base FROZEN.
        finger_final = dict(finger_q)
        refine_sides = [
            s
            for s in sides
            if s not in hand_trajectories
            and self.contact_refine
            and bool(combo.hands[s].contact_mask[:num_frames].any())
        ]
        qpos = np.zeros((num_frames, 0), dtype=np.float32)
        kin_frames = np.ascontiguousarray(combo.human_pose7[:num_frames][:, self._kin_input_cols])
        for it in range(2 if refine_sides else 1):
            # One whole-body solve per iteration, with every grasp wrist injected as a high-weight
            # target track.
            self._kin.target_overrides = {
                self._kin_target_col[s]: tf_mat_to_pose7(
                    hand_links[s][:, wrist_link_idx[s]] @ self._T_handbase_wristyaw[s]
                ).astype(np.float32)
                for s in sides
            }
            kin = self._kin.solve_numpy(
                kin_frames[None], human_heights=np.array([combo.human_height], dtype=np.float32)
            )[0]
            q_asm = np.zeros((num_frames, self.body_robot.spec.num_actuated_joints), dtype=np.float32)
            q_asm[:, self._kin_cols] = kin[:, 7:]
            if num_frames >= 3:
                q_asm = uniform_filter1d(q_asm, 3, axis=0, mode="nearest")
            qpos = np.concatenate([kin[:, :7], q_asm, object_tail], axis=1, dtype=np.float32)
            if num_frames >= 3:
                qpos[:, :7] = _smooth_pose7(qpos[:, :7], 3)

            # Arm snap: position targets on the palm-rigid links pin the wrist without cross-URDF
            # conventions. Both arms must share one IK; a per-arm solve writes back a full q and
            # would clobber the other arm.
            nd = self.body_robot.spec.num_actuated_joints
            base_mat = pose7_to_tf_mat(qpos[:, :7])
            T_base_world = inverse_tf_mat(base_mat)
            key = (tuple(sides), num_frames, tuple(hand_trajectories))
            if self._arm_ik_key != key:
                # Single warm-started seed: sampling would perturb (then freeze) masked joints.
                ik_cfg = (
                    IKConfig(
                        init_sample_range=0.0,
                        solver=MultiSeedSolverConfig(
                            stages=[
                                StageConfig(
                                    num_seeds=1,
                                    iters=40 if hand_trajectories else 15,
                                    lm_lambda=1.0,
                                )
                            ],
                            cuda_graph_mode="full",
                        ),
                    )
                    .add(PositionTask(weight=50.0))
                    .add(PositionLimit(weight=50.0))
                )
                links: List[str] = []
                for side in sides:
                    binding = self.preset.hands[side]
                    links += [binding.wrist_link] + list(binding.orientation_links)
                self._arm_ik = IK(ik_cfg, robot=self.body_robot, link=links, device=self.device)
                active = {c for side in sides for c in self._arm_columns(side)}
                if hand_trajectories:
                    active.update(
                        i
                        for i, name in enumerate(self.body_robot.spec.actuated_joint_names)
                        if name.startswith("waist_")
                    )
                self._arm_ik.set_active_joint_mask([1.0 if i in active else 0.0 for i in range(nd)])
                self._arm_ik_key = key
            assert self._arm_ik is not None
            ik_targets = []
            for side in sides:
                for i in [wrist_link_idx[side]] + hand_idx[side]:
                    ik_targets.append(
                        torch.from_numpy((T_base_world @ hand_links[side][:, i]).astype(np.float32)).to(self.device)
                    )
            snap = self._arm_ik.solve_torch(
                torch.stack(ik_targets, dim=1),
                init_q=torch.from_numpy(qpos[:, 7 : 7 + nd].astype(np.float32)).to(self.device),
            )
            qpos[:, 7 : 7 + nd] = snap.q.cpu().numpy()

            final_iter = it == 1
            if refine_sides:  # one FK of the achieved body serves both refine sides
                asm_links = (
                    self.body_robot.forward_kinematics_via_matrix_torch(
                        torch.from_numpy(qpos[:, 7 : 7 + nd]).to(self.device),
                        torch.from_numpy(base_mat).to(self.device),  # the snap leaves the base as-is
                    )
                    .cpu()
                    .numpy()
                )
            for side in refine_sides:
                track = combo.hands[side]
                refined_q, base_r = self._refine(
                    combo,
                    side,
                    asm_links=asm_links,
                    finger_q=finger_q[side],
                    num_frames=num_frames,
                    optimize_base=not final_iter,
                )
                merged = np.where(
                    np.asarray(track.contact_mask[:num_frames].any(axis=1))[:, None], refined_q, finger_q[side]
                ).astype(np.float32)
                if final_iter:
                    finger_final[side] = merged
                    continue
                # Feed the corrected hand back into the body anchors, low-passed in the object
                # frame (a stable grasp is near-constant there even while the object moves fast).
                finger_q[side] = merged
                if num_frames >= 5:
                    base_r = _smooth_pose7(base_r, 5)
                base_world = pose7_to_tf_mat(track.object_pose_world[:num_frames]) @ pose7_to_tf_mat(base_r)
                hand_links[side] = (
                    self.hand_robots[side]
                    .forward_kinematics_via_matrix_torch(
                        torch.from_numpy(finger_q[side]).to(self.device),
                        torch.from_numpy(base_world.astype(np.float32)).to(self.device),
                    )
                    .cpu()
                    .numpy()
                )

        asm_names = self.body_robot.spec.actuated_joint_names
        for side in sides:
            prefix = self.preset.hands[side].joint_prefix
            asm_cols = 7 + np.array(
                [asm_names.index(prefix + n) for n in self.hand_robots[side].spec.actuated_joint_names]
            )
            limits = self.body_robot.spec.actuated_joint_limits[asm_cols - 7]
            qpos[:, asm_cols] = np.clip(finger_final[side], limits[:, 0], limits[:, 1])
        if self.scene_clearance and combo.scene is not None:
            qpos = apply_scene_clearance(self.body_robot, combo, qpos, self.device)
        return qpos

    def _refine(
        self,
        combo: ComboClip,
        side: str,
        *,
        asm_links: np.ndarray,
        finger_q: np.ndarray,
        num_frames: int,
        optimize_base: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Contact refine in one hand's object frame, anchored at the achieved body wrist
        (`asm_links` is the assembly FK of the current qpos). Contacts attract the fingertip mesh
        surface point to the labeled object surface; sphere-vs-mesh collision guards the rest of
        the hand. With `optimize_base` the base is softly anchored instead of frozen.

        Returns `(refined_q (T, dof), T_obj_base (T, 7))`.
        """
        track = combo.hands[side]
        hand_robot = self.hand_robots[side]
        T_world_wrist = asm_links[:, self.body_robot.spec.link_names.index(self.preset.hands[side].wrist_link)]
        T_obj_world = inverse_tf_mat(pose7_to_tf_mat(track.object_pose_world[:num_frames]))
        base_a = tf_mat_to_pose7(T_obj_world @ T_world_wrist @ inverse_tf_mat(self._T_base_wrist[side][None])[0])

        # Labels and normals into the object frame, via the native pose (labels are native-world).
        T_obj_native = inverse_tf_mat(pose7_to_tf_mat(track.object_pose_native[:num_frames]))
        r_obj_world = T_obj_native[:, :3, :3]
        points_obj = (
            np.einsum("fij,fsj->fsi", r_obj_world, track.contact_points[:num_frames]) + T_obj_native[:, None, :3, 3]
        )
        normals_obj = np.einsum("fij,fsj->fsi", r_obj_world, track.contact_normals[:num_frames])
        slot_idx = np.array(
            [
                hand_robot.spec.link_names.index(inspire_preset.spec.target_link_names[n])
                for n in self._contact_target_names
            ],
            dtype=np.int32,
        )
        link_a = (
            hand_robot.forward_kinematics_via_matrix_torch(
                torch.from_numpy(finger_q.astype(np.float32)).to(self.device),
                torch.from_numpy(pose7_to_tf_mat(base_a)).to(self.device),
            )
            .cpu()
            .numpy()
        )
        local_contact_points = compute_inward_contact_points(hand_robot, slot_idx, link_a[:, slot_idx], normals_obj)

        # Anchored contact solve: contacts + finger-chain collision against the object, regularized
        # to the given trajectory (see build_contact_refine_config).
        key = (side, optimize_base)
        cached = self._refiners.get(key)
        if cached is None or cached[0] is not track.object_mesh or cached[1] != num_frames:
            assert track.object_mesh is not None
            refiner = HandRetargetingOffline(
                hand_robot,
                replace(self._hand_spec[side], contact_target_names=self._contact_target_names),
                build_contact_refine_config(
                    contact_weight=300.0,
                    contact_margin=0.0,
                    collision_weight=100.0,
                    collision_margin=0.0,
                    anchor_q_weight=10.0,
                    anchor_base_weight=20.0,
                    optimize_base=optimize_base,
                ),
                scene=WarpScene(1, self.device).add(MeshGeom([track.object_mesh], np.asarray([0, 1], dtype=np.int32))),
                device=self.device,
            )
            refiner.warmup(1, num_frames)
            self._refiners[key] = (track.object_mesh, num_frames, refiner)
        refiner = self._refiners[key][2]
        packed = refiner.solve_with_contact_numpy(
            np.zeros((num_frames, len(inspire_preset.spec.target_names), 3), dtype=np.float32),
            points_obj,
            track.contact_mask[:num_frames],
            init_state=hand_robot.state(
                q=wp.from_numpy(finger_q.astype(np.float32)[None], dtype=wp.float32, device=self.device),
                T_world_base=wp.from_numpy(base_a[None].astype(np.float32), dtype=wp_vec7, device=self.device),
            ),
            local_contact_points=local_contact_points,
        )
        return packed[:, 7:], packed[:, :7]


__all__ = ["HandTrajectory", "ComboRetargeter"]
