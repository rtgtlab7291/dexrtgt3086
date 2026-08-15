"""Object-fixed grasp synthesis initialized by captured MANO motion.

The captured hand selects a local mesh region and initializes one force-closure/collision LM solve.
That single grasp stays fixed in the object frame during the sustained carry interval and is blended
into the MANO trajectory at the interval edges. Vessel interiors can be removed from contact targets
at runtime without changing their collision meshes. Non-carrying hands keep MANO, with an optional
labelled-contact refine, and ``ComboRetargeter`` makes the whole body follow the resulting trajectory.

SAGA uses three fingers; ParaHome uses five and can optionally prune fingers that bend far beyond
the captured pose.
"""

import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import warp as wp
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter1d,
    label,
    uniform_filter1d,
)

from robokit.assets.robots.hands import inspire_hand
from robokit.geom import MeshGeom, SdfVolume, VolumeGeom, WarpScene
from robokit.helpers.combo_retarget.clip import CONTACT_MANO_INDICES, ComboClip, HandTrack
from robokit.helpers.combo_retarget.config import ComboPreset, link_transfer_maps
from robokit.helpers.combo_retarget.grasp_opt import GraspOptHelper, GraspOptHelperConfig, load_contact_candidates
from robokit.helpers.combo_retarget.retarget import (
    ComboRetargeter,
    HandTrajectory,
    build_contact_refine_config,
    compute_inward_contact_points,
    solve_hand_pass,
)
from robokit.helpers.hand_retargeting import HandRetargetingOffline
from robokit.helpers.hand_retargeting.presets import TARGET_NAMES
from robokit.helpers.hand_retargeting.presets import inspire as inspire_preset
from robokit.opt.multi_seed_solver import StageConfig
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import (
    axis_angle_to_matrix,
    inverse_tf_mat,
    matrix_to_axis_angle,
    pose7_to_tf_mat,
    tf_mat_to_pose7,
)


Samples = Tuple[np.ndarray, np.ndarray]  # (P, 3) link-frame surface points and their (P,) link index

PALM_MANO_INDICES = (0, 5, 9, 13, 17)  # wrist + the four finger knuckles
MIDDLE_TIP_MANO_INDEX = 12
FIVE_FINGER_CONTACTS = (
    "thumb_intermediate",
    "index_intermediate",
    "middle_intermediate",
    "ring_intermediate",
    "pinky_intermediate",
)

# --- motion / contact gating -------------------------------------------------------------------
MOVE_DISTANCE = 0.02  # p95 object displacement over the clip that counts as "carried"
MOVE_ANGLE = np.deg2rad(10.0)
MOTION_STEP_DISTANCE = 0.002  # per-frame object step that counts as "moving right now"
MOTION_STEP_ANGLE = np.deg2rad(1.0)
MOTION_FILTER_SECONDS = 0.3  # majority window for both contact and motion, and the blend ramp

# --- how far the grasp may drift to follow the captured motion ----------------------------------
RESIDUAL_FILTER_SECONDS = 0.08
RESIDUAL_TRANSLATION_LIMIT = 0.008
RESIDUAL_ROTATION_LIMIT = np.deg2rad(6.0)

# --- whole-hand refine (stage 2) ----------------------------------------------------------------
REFINE_CONTACT_WEIGHT = 100.0
REFINE_COLLISION_WEIGHT = 300.0
REFINE_COLLISION_MARGIN = 0.002
REFINE_ANCHOR_Q_WEIGHT = 3.0
REFINE_ANCHOR_BASE_WEIGHT = 300.0

# --- GraspKit (stage 3) -------------------------------------------------------------------------
DISTANCE_WEIGHT = 600.0
FORCE_CLOSURE_WEIGHT = 20.0
ANCHOR_Q_WEIGHT = 0.5
ANCHOR_TRANSLATION_WEIGHT = 120.0
ANCHOR_ROTATION_WEIGHT = 40.0
MIN_ROI_FACES = 256
PADS_PER_LINK = 4  # pad candidates kept per contact link before the topology product
VALIDATION_SAMPLES_PER_LINK = 4096  # denser than the optimizer's set: gates must not be fooled
VALIDATION_MARGIN = 0.0001  # solve against a slightly tighter bound than the gate checks


@dataclass(frozen=True)
class PruneConfig:
    """Second GraspKit pass: drop the fingers the captured hand never bent that far, then re-solve.

    A five-finger force-closure grasp often curls a finger far past what the human did; comparing
    each finger's proximal flexion against the MANO solution exposes those, and the remaining
    fingers get a full-iteration solve of their own. The result is kept only when it is
    collision-safe AND lands its pads closer to the surface than the first pass did.
    """

    bend_limit: float = np.deg2rad(24.0)
    iters: int = 48
    num_seeds: int = 16
    collision_radius: float = 0.001  # sample spheres, not points: pass 2 keeps a hard skin
    collision_samples_per_link: int = 128
    collision_weight: float = 300.0
    max_penetration: float = 0.004


@dataclass(frozen=True)
class GraspConfig:
    """Everything the two datasets tune differently; see ``SAGA_GRASP`` / ``PARAHOME_GRASP``."""

    contact_links: Tuple[str, ...] = FIVE_FINGER_CONTACTS
    contact_candidates_per_link: int = 2  # 0 = pair pads by rank instead of taking their product
    iters: int = 48
    num_seeds: int = 16
    roi_radius: float = 0.07
    self_collision: bool = True
    collision_samples_per_link: int = 2048
    collision_weight: float = 600.0
    max_penetration: float = 0.0015
    exterior_contacts: bool = False
    object_contact: bool = False
    single_grasp: bool = False
    prune: Optional[PruneConfig] = None


SAGA_GRASP = GraspConfig(
    contact_links=("thumb_intermediate", "index_intermediate", "middle_intermediate"),
    contact_candidates_per_link=2,
    collision_samples_per_link=1024,
    collision_weight=150.0,
    max_penetration=0.005,
    exterior_contacts=True,
    object_contact=True,
    single_grasp=True,
)

PARAHOME_GRASP = GraspConfig(iters=8, prune=PruneConfig())


@dataclass
class ComboGrasp:
    """Solved standalone-hand trajectories for one clip, plus whole-body qpos when asked for."""

    combo: ComboClip
    num_frames: int
    robots: Dict[str, Robot]  # side -> standalone hand
    joints: Dict[str, np.ndarray]  # side -> (T, D) hand joints
    bases_world: Dict[str, np.ndarray]  # side -> (T, 7) hand base pose, body world
    targets_world: Dict[str, np.ndarray]  # side -> (T, C, 3) object-surface contact targets
    contacts_world: Dict[str, np.ndarray]  # side -> (T, C, 3) the pads chasing them
    contact_masks: Dict[str, np.ndarray]  # side -> (T, C) bool, slot is an enforced contact
    moving_objects: Tuple[str, ...]
    qpos: Optional[np.ndarray]  # (T, 7 + dof + 7 * num_objects)


@dataclass
class _Pads:
    """Inspire contact-pad candidates on the links a grasp is allowed to touch (link frames)."""

    points: np.ndarray  # (C, 3)
    links: np.ndarray  # (C,)
    normals: np.ndarray  # (C, 3) outward
    points_wp: wp.array
    links_wp: wp.array


@dataclass
class _Hand:
    """One side's assets and solve state; every pose lives in the OBJECT frame until the driver
    lifts the finished trajectory into the body world."""

    label: str  # "<side>/<object>", the prefix of every log line
    track: HandTrack
    urdf: str
    robot: Robot
    collision_robot: Robot  # `robot` with the optimizer's surface samples as collision spheres
    window: int  # odd majority window, MOTION_FILTER_SECONDS of frames
    object_mesh: Optional[trimesh.Trimesh]
    object_scene: Optional[WarpScene]
    grasp_scene: Optional[WarpScene]
    contact_face_mask: np.ndarray
    val_samples: Samples  # denser samples, trusted by the gates
    collision_weight: float
    T_world_obj: np.ndarray  # (T, 4, 4) captured object, native world
    T_obj_world: np.ndarray  # (T, 4, 4)
    T_obj_wrist: np.ndarray  # (T, 4, 4) captured wrist
    q_mano: np.ndarray  # (T, D) tracked joints BEFORE the refine
    T_obj_base_mano: np.ndarray  # (T, 4, 4) tracked base BEFORE the refine
    q: np.ndarray  # (T, D) solve state
    T_obj_base: np.ndarray  # (T, 4, 4) solve state
    targets_obj: np.ndarray  # (T, C, 3)
    contacts_obj: np.ndarray  # (T, C, 3)
    contact_mask: np.ndarray  # (T, C)
    num_phases: int = 0


@dataclass(frozen=True)
class _Phase:
    """Frames over which the object moves under one stable grasp."""

    start: int
    stop: int
    key: int


@dataclass
class _Grasp:
    """One solved canonical grasp, in the object frame."""

    phase: _Phase
    q: np.ndarray  # (D,)
    base_obj: np.ndarray  # (7,)
    target_obj: np.ndarray  # (C, 3) object-surface points the pads were driven onto
    pad_points: np.ndarray  # (C, 3) the pads themselves, link frames
    pad_links: np.ndarray  # (C,)


def _odd_window(fps: float) -> int:
    """Odd-length majority window spanning ``MOTION_FILTER_SECONDS`` of the clip."""
    window = max(3, round(fps * MOTION_FILTER_SECONDS))
    return window + 1 if window % 2 == 0 else window


def _majority(flags: np.ndarray, window: int) -> np.ndarray:
    """True where more than half of the surrounding ``window`` frames are True."""
    counts = np.convolve(flags.astype(np.int16), np.ones(window, dtype=np.int16), mode="same")
    start = (len(counts) - len(flags)) // 2
    return counts[start : start + len(flags)] >= window // 2 + 1


def _mesh_scene(mesh: trimesh.Trimesh, device: str) -> WarpScene:
    return WarpScene(1, device).add(MeshGeom([mesh], np.asarray([0, 1], dtype=np.int32)))


def _exterior_contact_surface(mesh: trimesh.Trimesh, device: str) -> Tuple[np.ndarray, Optional[SdfVolume]]:
    """Exterior contact faces and a collision volume filling the detected vessel cavity."""
    exterior = np.ones(len(mesh.faces), dtype=bool)
    vertices = np.asarray(mesh.vertices)
    centers = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)
    extents = np.asarray(mesh.extents)
    if not len(centers) or extents.max() <= 0.0:
        return exterior, None

    center = vertices.mean(axis=0)
    axes = np.linalg.eigh(np.cov(vertices - center, rowvar=False))[1].T
    scores = []
    for axis in axes:
        radial = centers - center
        radial -= np.outer(radial @ axis, axis)
        alignment = np.einsum("ij,ij->i", radial, normals) / np.maximum(np.linalg.norm(radial, axis=1), 1e-9)
        scores.append(np.average(alignment < -0.5, weights=mesh.area_faces))
    axis = axes[int(np.argmax(scores))]
    if max(scores) < 0.3:
        return exterior, None

    pitch = float(extents.max() / 128)
    voxels = mesh.voxelized(pitch).fill()
    padding = 14
    occupied = np.pad(np.asarray(voxels.matrix, dtype=bool), padding)
    origin = voxels.transform[:3, 3] - padding * pitch
    shape = np.asarray(occupied.shape)
    grid = [origin[i] + np.arange(shape[i]) * pitch for i in range(3)]
    projection = (vertices - center) @ axis
    span = np.ptp(projection)
    gx = grid[0][:, None, None] - center[0]
    gy = grid[1][None, :, None] - center[1]
    gz = grid[2][None, None, :] - center[2]
    along = gx * axis[0] + gy * axis[1] + gz * axis[2]
    caps = []
    for sign in (-1.0, 1.0):
        end = projection.max() if sign > 0.0 else projection.min()
        near = sign * (projection - end) > -max(12 * pitch, 0.06 * span)
        cap_center = vertices[near].mean(axis=0)
        delta = vertices[near] - cap_center
        radial = delta - np.outer(delta @ axis, axis)
        radius = np.linalg.norm(radial, axis=1).max() + 8 * pitch
        dx = grid[0][:, None, None] - cap_center[0]
        dy = grid[1][None, :, None] - cap_center[1]
        dz = grid[2][None, None, :] - cap_center[2]
        cap_along = dx * axis[0] + dy * axis[1] + dz * axis[2]
        radial2 = dx * dx + dy * dy + dz * dz - cap_along * cap_along
        caps.append((np.abs(along - (end - sign * 8 * pitch)) <= 4 * pitch) & (radial2 <= radius * radius))

    minimum = int(0.01 * np.prod(np.ceil(extents / pitch)))
    cavity = None
    sealed = None
    repair = 0
    for repair in range(9):
        best_size = 0
        for cap in caps:
            solid = occupied | cap
            filled = np.asarray(binary_fill_holes(solid), dtype=bool)
            regions = np.empty(solid.shape, dtype=np.int32)
            label(filled & ~solid, output=regions)
            sizes = np.bincount(regions.ravel())[1:]
            if len(sizes) and sizes.max() > best_size:
                component = int(np.argmax(sizes)) + 1
                best_size = int(sizes[component - 1])
                cavity = regions == component
                sealed = filled
        if best_size >= minimum:
            break
        cavity = sealed = None
        occupied = binary_dilation(occupied)
    if cavity is None or sealed is None:
        return exterior, None

    probe = centers + (repair + 2) * pitch * normals
    index = np.rint((probe - origin) / pitch).astype(np.int64)
    inside = np.all((index >= 0) & (index < shape), axis=1)
    inner = np.zeros(len(centers), dtype=bool)
    inner[inside] = cavity[index[inside, 0], index[inside, 1], index[inside, 2]]
    if repair > 2:
        cavity = np.asarray(binary_dilation(cavity, iterations=repair - 2), dtype=bool) & sealed
    outside = np.asarray(distance_transform_edt(~cavity), dtype=np.float32)
    inside = np.asarray(distance_transform_edt(cavity), dtype=np.float32)
    sdf = (outside - inside).astype(np.float32) * pitch + pitch
    volume = wp.Volume.load_from_numpy(
        sdf,
        min_world=tuple(float(value) for value in origin),
        voxel_size=pitch,
        bg_value=float(sdf.max()),
        device=device,
    )
    maximum = origin + (shape - 1) * pitch
    return ~inner, SdfVolume(volume, origin.astype(np.float32), maximum.astype(np.float32), padding * pitch)


def _limit(delta: np.ndarray, limit: float) -> np.ndarray:
    """Soft-clamp per-row vector norms to ``limit`` (tanh, so small corrections pass untouched)."""
    norm = np.linalg.norm(delta, axis=-1)
    return delta * (limit * np.tanh(norm / limit) / np.maximum(norm, 1e-8))[:, None]


def _sample_link_points(
    robot: Robot,
    per_link: int,
    seed_base: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Surface samples over every collision mesh: ``(P, 3)`` link-frame points and their link index."""
    meshes = robot.spec.get_link_meshes(mode="collision")
    points, links = [], []
    for index, name in enumerate(robot.spec.link_names):
        if name not in meshes:
            continue
        sample = trimesh.sample.sample_surface(meshes[name], per_link, seed=seed_base + index)[0]
        points.append(sample.astype(np.float32))
        links.append(np.full(len(sample), index, dtype=np.int32))
    return np.concatenate(points), np.concatenate(links)


def _penetration(
    robot: Robot,
    q: np.ndarray,
    T_obj_base: np.ndarray,
    object_scene: WarpScene,
    samples: Samples,
    *,
    device: str,
    radius: float = 0.0,
    chunk: int = 1024,
) -> np.ndarray:
    """Per-sample penetration of the hand's surface samples into the object, ``(B, P)``, >= 0."""
    if len(q) > chunk:
        return np.concatenate(
            [
                _penetration(
                    robot,
                    q[i : i + chunk],
                    T_obj_base[i : i + chunk],
                    object_scene,
                    samples,
                    device=device,
                    radius=radius,
                )
                for i in range(0, len(q), chunk)
            ]
        )
    points, point_links = samples
    q = np.clip(q, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])
    links = robot.forward_kinematics_via_matrix_torch(
        torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)).to(device),
        torch.from_numpy(np.ascontiguousarray(T_obj_base, dtype=np.float32)).to(device),
    )
    centers = robot.transform_link_points_torch(
        links,
        torch.from_numpy(points).to(device),
        torch.from_numpy(point_links).to(device).long(),
    )
    sdf, _, _ = object_scene.query_sdf_torch(
        centers.reshape(-1, 3),
        torch.tensor([0, centers.numel() // 3], dtype=torch.int32, device=device),
    )
    return torch.clamp(radius - sdf.reshape(centers.shape[:2]), min=0.0).cpu().numpy()


def _medoid_frame(T_obj_wrist: np.ndarray, start: int, stop: int) -> int:
    """Frame in ``[start, stop)`` whose object-relative wrist pose is the most typical of the span."""
    frames = np.arange(start, stop)
    frames = frames[:: max(1, len(frames) // 64)]
    relative = T_obj_wrist[frames]
    translation = np.linalg.norm(relative[:, None, :3, 3] - relative[None, :, :3, 3], axis=-1)
    rotation = np.arccos(
        np.clip((np.einsum("aij,bij->ab", relative[:, :3, :3], relative[:, :3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    )
    return int(frames[np.argmin(np.median(translation / 0.03 + rotation / 0.5, axis=1))])


def _prepare_hand(
    side: str,
    combo: ComboClip,
    preset: ComboPreset,
    config: GraspConfig,
    *,
    num_frames: int,
    refine_contacts: np.ndarray,
    refine: bool,
    device: str,
) -> _Hand:
    """Track one captured hand, optionally refining its non-carrying contact trajectory."""
    binding = preset.hands[side]
    track = combo.hands[side]
    assert preset.body.urdf_path is not None
    urdf, _, spheres = binding.prepare_assets(preset.body.urdf_path, device, sphere_source=preset.hands.get("right"))
    robot = Robot.load(
        urdf,
        load_meshes=True,
        mesh_dir=str(Path(urdf).parent),
        load_collision_spheres=True,
        collision_spheres_path=spheres,
    )
    hand_spec = replace(
        inspire_preset.spec,
        root_link_coord_spec=binding.wrist_coord_spec or inspire_preset.spec.root_link_coord_spec,
    )

    # A light surface sample set for optimization and a dense one for validation.
    opt_samples = _sample_link_points(robot, config.collision_samples_per_link)
    val_samples = _sample_link_points(robot, VALIDATION_SAMPLES_PER_LINK, 1000)
    opt_points, opt_links = opt_samples
    collision_robot = Robot(
        replace(
            robot.spec,
            local_collision_sphere_centers=opt_points,
            collision_sphere_radii=np.zeros(len(opt_points), dtype=np.float32),
            collision_spheres_link_indices=opt_links,
        )
    )

    # 1. MANO: per-subject keypoint calibration, then the offline joint solve.
    rest_links = (
        robot.forward_kinematics_via_matrix_torch(
            torch.zeros((1, robot.spec.num_actuated_joints), device=device), torch.eye(4, device=device)[None]
        )
        .cpu()
        .numpy()[0]
    )
    robot_span = float(
        np.linalg.norm(
            rest_links[
                robot.spec.link_names.index(hand_spec.target_link_names[TARGET_NAMES[MIDDLE_TIP_MANO_INDEX]]), :3, 3
            ]
            - rest_links[robot.spec.link_names.index(hand_spec.target_link_names[hand_spec.root_target_name]), :3, 3]
        )
    )
    packed = solve_hand_pass(
        robot, hand_spec, track, num_frames, device, hand_span=robot_span, root_position_weight=10.0
    )
    T_world_obj = pose7_to_tf_mat(track.object_pose_native[:num_frames])
    T_obj_world = inverse_tf_mat(T_world_obj)
    q_mano = packed[:, 7:].astype(np.float32)
    T_obj_base_mano = T_obj_world @ pose7_to_tf_mat(packed[:, :7])
    object_scene = None if track.object_mesh is None else _mesh_scene(track.object_mesh, device)
    grasp_scene = object_scene
    contact_face_mask = np.ones(0 if track.object_mesh is None else len(track.object_mesh.faces), dtype=bool)
    if track.object_mesh is not None and config.exterior_contacts and not refine:
        contact_face_mask, cavity = _exterior_contact_surface(track.object_mesh, device)
        if cavity is not None:
            grasp_scene = _mesh_scene(track.object_mesh, device)
            grasp_scene.add(VolumeGeom([cavity], np.asarray([0, 1], dtype=np.int32)))
        print(
            f"{side}/{track.object_name}: exterior contacts removed "
            f"{np.count_nonzero(~contact_face_mask)}/{len(contact_face_mask)} interior faces"
        )

    hand = _Hand(
        label=f"{side}/{track.object_name or 'free'}",
        track=track,
        urdf=urdf,
        robot=robot,
        collision_robot=collision_robot,
        window=_odd_window(combo.fps),
        object_mesh=track.object_mesh,
        object_scene=object_scene,
        grasp_scene=grasp_scene,
        contact_face_mask=contact_face_mask,
        val_samples=val_samples,
        collision_weight=config.collision_weight,
        T_world_obj=T_world_obj,
        T_obj_world=T_obj_world,
        T_obj_wrist=T_obj_world
        @ pose7_to_tf_mat(np.concatenate([track.wrist_pos[:num_frames], track.wrist_quat_wxyz[:num_frames]], axis=-1)),
        q_mano=q_mano,
        T_obj_base_mano=T_obj_base_mano,
        q=q_mano.copy(),
        T_obj_base=T_obj_base_mano.copy(),
        targets_obj=np.zeros((num_frames, 0, 3), dtype=np.float32),
        contacts_obj=np.zeros((num_frames, 0, 3), dtype=np.float32),
        contact_mask=np.zeros((num_frames, 0), dtype=bool),
    )
    if hand.object_scene is None or not refine:
        return hand
    assert hand.object_mesh is not None

    # 2. Refine: whole-hand collision push-out + attraction of the labelled pads onto the surface.
    slot_names = [inspire_preset.spec.target_link_names[TARGET_NAMES[index]] for index in CONTACT_MANO_INDICES]
    slot_indices = np.asarray([robot.spec.link_names.index(name) for name in slot_names], dtype=np.int32)
    points_obj = (
        np.einsum("fij,fsj->fsi", T_obj_world[:, :3, :3], track.contact_points[:num_frames])
        + T_obj_world[:, None, :3, 3]
    )
    normals_obj = np.einsum("fij,fsj->fsi", T_obj_world[:, :3, :3], track.contact_normals[:num_frames])
    T_obj_link = (
        robot.forward_kinematics_via_matrix_torch(
            torch.from_numpy(q_mano).to(device), torch.from_numpy(T_obj_base_mano.astype(np.float32)).to(device)
        )
        .cpu()
        .numpy()[:, slot_indices]
    )
    local_pads = compute_inward_contact_points(robot, slot_indices, T_obj_link, normals_obj)
    before = _penetration(robot, q_mano, T_obj_base_mano, hand.object_scene, val_samples, device=device).max(axis=1)
    # A never-enforced wrist slot rides along with the 10 labelled ones: naming the root among the
    # contacts is what widens the collision guard from the finger chains to the WHOLE hand, which is
    # what this stage wants (it pushes the entire hand out of the object, not just the fingers).
    idle = np.zeros((num_frames, 1, 3), dtype=np.float32)
    base_a = tf_mat_to_pose7(T_obj_base_mano)
    refiner = HandRetargetingOffline(
        robot,
        replace(
            hand_spec,
            contact_target_names=tuple(TARGET_NAMES[index] for index in CONTACT_MANO_INDICES)
            + (hand_spec.root_target_name,),
        ),
        build_contact_refine_config(
            contact_weight=REFINE_CONTACT_WEIGHT,
            contact_margin=0.01,
            collision_weight=REFINE_COLLISION_WEIGHT,
            collision_margin=REFINE_COLLISION_MARGIN,
            anchor_q_weight=REFINE_ANCHOR_Q_WEIGHT,
            anchor_base_weight=REFINE_ANCHOR_BASE_WEIGHT,
            optimize_base=True,
        ),
        scene=hand.object_scene,
        device=device,
    )
    refiner.warmup(1, num_frames)
    packed = refiner.solve_with_contact_numpy(
        np.zeros((num_frames, len(hand_spec.target_names), 3), dtype=np.float32),
        np.concatenate([points_obj, idle], axis=1),
        np.concatenate([refine_contacts, np.zeros((num_frames, 1), dtype=bool)], axis=1),
        init_state=robot.state(
            q=wp.from_numpy(q_mano[None], dtype=wp.float32, device=device),
            T_world_base=wp.from_numpy(base_a[None].astype(np.float32), dtype=wp_vec7, device=device),
        ),
        local_contact_points=np.concatenate([local_pads, idle], axis=1),
    )
    q_refined, base_refined = packed[:, 7:], packed[:, :7]

    # The refine may walk the wrist anywhere; only a small, low-pass-filtered correction is kept.
    delta = inverse_tf_mat(T_obj_base_mano) @ pose7_to_tf_mat(base_refined)
    sigma = max(1.0, combo.fps * RESIDUAL_FILTER_SECONDS)
    delta_t = _limit(
        gaussian_filter1d(delta[:, :3, 3], sigma=sigma, axis=0, mode="nearest"), RESIDUAL_TRANSLATION_LIMIT
    )
    delta_r = _limit(
        gaussian_filter1d(matrix_to_axis_angle(delta[:, :3, :3]), sigma=sigma, axis=0, mode="nearest"),
        RESIDUAL_ROTATION_LIMIT,
    )
    delta[:, :3, :3] = axis_angle_to_matrix(delta_r)
    delta[:, :3, 3] = delta_t
    hand.q = np.clip(q_refined, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1]).astype(
        np.float32
    )
    hand.T_obj_base = T_obj_base_mano @ delta
    after = _penetration(robot, hand.q, hand.T_obj_base, hand.object_scene, val_samples, device=device).max(axis=1)
    shift = np.linalg.norm(delta_t, axis=-1)
    angle = np.linalg.norm(delta_r, axis=-1)
    print(
        f"{hand.label}: common collision penetration median/p95/max "
        f"{np.median(before) * 1000:.1f}/{np.percentile(before, 95) * 1000:.1f}/{before.max() * 1000:.1f} -> "
        f"{np.median(after) * 1000:.1f}/{np.percentile(after, 95) * 1000:.1f}/{after.max() * 1000:.1f} mm, "
        f"wrist shift median/p95={np.median(shift) * 1000:.1f}/{np.percentile(shift, 95) * 1000:.1f} mm, "
        f"{np.rad2deg(np.median(angle)):.1f}/{np.rad2deg(np.percentile(angle, 95)):.1f} deg"
    )

    return hand


def _prepare_pads(hand: _Hand, preset: ComboPreset, side: str, config: GraspConfig, device: str) -> _Pads:
    """The shipped Inspire pad candidates on the links a grasp may touch, with outward normals.

    The candidates are authored for the RIGHT hand, so a mirrored side re-expresses each of them
    through the measured per-link transform between the two hands (see ``config.link_transfer_maps``).
    """
    robot = hand.robot
    points_wp, links_wp = load_contact_candidates(str(inspire_hand.CONTACT_POINTS_PATH), robot, device)
    points, links = points_wp.numpy(), links_wp.numpy()
    if preset.hands[side].mirror_spheres:
        maps = link_transfer_maps(preset.hands["right"].urdf, hand.urdf, device)
        for link in np.unique(links):
            transform = maps[robot.spec.link_names[link]]
            points[links == link] = points[links == link] @ transform[:3, :3].T + transform[:3, 3]
    # The thumb may oppose from either of its two pads, so both are candidates whatever the topology.
    allowed = np.asarray(
        [robot.spec.link_names.index(name) for name in {*config.contact_links, "thumb_distal"}], dtype=np.int32
    )
    keep = np.isin(links, allowed)
    points, links = points[keep], links[keep]

    meshes = robot.spec.get_link_meshes(mode="collision")
    normals = np.empty_like(points)
    for link in np.unique(links):
        indices = np.flatnonzero(links == link)
        mesh = meshes[robot.spec.link_names[link]]
        normals[indices] = mesh.face_normals[trimesh.proximity.closest_point(mesh, points[indices])[2]]
    return _Pads(
        points=points,
        links=links,
        normals=normals,
        points_wp=wp.from_numpy(points, dtype=wp.vec3, device=device),
        links_wp=wp.from_numpy(links, dtype=wp.int32, device=device),
    )


def _pad_frames(robot: Robot, q: np.ndarray, T_obj_base: np.ndarray, links: np.ndarray, device: str) -> np.ndarray:
    """Object-frame transforms of the contact links, ``(B, C, 4, 4)``; ``links`` is ``(C,)`` or ``(B, C)``."""
    link_T = (
        robot.forward_kinematics_via_matrix_torch(
            torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)).to(device),
            torch.from_numpy(np.ascontiguousarray(T_obj_base, dtype=np.float32)).to(device),
        )
        .cpu()
        .numpy()
    )
    return link_T[np.arange(len(link_T))[:, None], links] if links.ndim == 2 else link_T[:, links]


def _apply_se3(T: np.ndarray, local: np.ndarray, translate: bool = True) -> np.ndarray:
    """Points - or, with ``translate=False``, directions - through a broadcastable ``[..., 4, 4]``."""
    moved = np.einsum("...ij,...j->...i", T[..., :3, :3], local)
    return moved + T[..., :3, 3] if translate else moved


def _contact_seeds(
    hand: _Hand,
    pads: _Pads,
    key_frame: int,
    q_probe: np.ndarray,
    T_obj_base: np.ndarray,
    *,
    topologies: Tuple[np.ndarray, ...],
    per_link: int,
    roi_radius: float,
    device: str,
) -> Tuple[WarpScene, np.ndarray, np.ndarray, int]:
    """Palm-centred contact ROI plus the batch of pad-index rows GraspKit is seeded with.

    Per contact link the four pads facing the object and nearest its surface are kept; rows are then
    either the full product of the best ``per_link`` of them, or (``per_link=0``) rank-aligned tuples.
    """
    assert hand.object_mesh is not None
    T_obj_native = hand.T_obj_world[key_frame]
    palm_obj = T_obj_native[:3, :3] @ hand.track.mano[key_frame, PALM_MANO_INDICES].mean(axis=0) + T_obj_native[:3, 3]
    face_distance = np.linalg.norm(np.asarray(hand.object_mesh.triangles_center) - palm_obj, axis=-1)
    roi_mask = (face_distance < roi_radius) & hand.contact_face_mask
    if roi_mask.sum() < MIN_ROI_FACES:
        available = np.flatnonzero(hand.contact_face_mask)
        roi_mask[available[np.argsort(face_distance[available])[:MIN_ROI_FACES]]] = True
    target_mesh = hand.object_mesh.copy()
    target_mesh.update_faces(roi_mask)
    target_mesh.remove_unreferenced_vertices()
    target_scene = _mesh_scene(target_mesh, device)

    link_T = _pad_frames(hand.robot, q_probe[None], T_obj_base[None], pads.links, device)[0]
    distance, normal, _ = target_scene.query_sdf(
        wp.from_numpy(_apply_se3(link_T, pads.points).astype(np.float32), dtype=wp.vec3, device=device),
        scene_offsets=wp.array([0, len(pads.points)], dtype=wp.int32, device=device),
    )
    distance = distance.numpy()
    alignment = np.einsum("ij,ij->i", _apply_se3(link_T, pads.normals, translate=False), normal.numpy())
    nearest = {}
    for link in np.unique(np.concatenate(topologies)):
        indices = np.flatnonzero(pads.links == link)
        facing = indices[alignment[indices] < 0.0]
        if len(facing) < PADS_PER_LINK:
            facing = indices[np.argsort(alignment[indices])[:PADS_PER_LINK]]
        nearest[link] = facing[np.argsort(np.abs(distance[facing]))[:PADS_PER_LINK]]
    if per_link:
        rows = [row for topology in topologies for row in product(*(nearest[link][:per_link] for link in topology))]
    else:
        rows = [
            [nearest[link][rank] for link in topology]
            for topology in topologies
            for rank in range(min(len(nearest[link]) for link in topology))
        ]
    return target_scene, np.asarray(rows, dtype=np.int32), distance, int(roi_mask.sum())


def _canonical_grasp(
    hand: _Hand, pads: _Pads, config: GraspConfig, phase: _Phase, q_init: np.ndarray, *, device: str
) -> Optional[_Grasp]:
    """Solve, validate and report one local force-closure grasp."""
    assert hand.object_scene is not None and hand.grasp_scene is not None
    robot, contacts = hand.robot, config.contact_links
    topology_names = (contacts, ("thumb_distal", *contacts[1:]))
    topologies = tuple(
        np.asarray([robot.spec.link_names.index(name) for name in names], dtype=np.int32) for names in topology_names
    )
    T_obj_base_a = hand.T_obj_base[phase.key]
    base_a = tf_mat_to_pose7(T_obj_base_a[None])[0]
    exterior_mode = not hand.contact_face_mask.all()
    num_seeds = config.num_seeds * (2 if exterior_mode else 1)
    anchor_rotation_weight = 5.0 if exterior_mode else ANCHOR_ROTATION_WEIGHT
    target_scene, rows, pad_distance, roi_faces = _contact_seeds(
        hand,
        pads,
        phase.key,
        q_init,
        T_obj_base_a,
        topologies=topologies,
        per_link=config.contact_candidates_per_link,
        roi_radius=config.roi_radius,
        device=device,
    )

    helper = GraspOptHelper(
        robot=hand.collision_robot,
        batch_size=len(rows),
        num_contact_points=len(contacts),
        target_meshes=target_scene,
        scene_meshes=hand.grasp_scene,
        contact_candidates=pads.points_wp,
        contact_candidate_link_indices=pads.links_wp,
        config=GraspOptHelperConfig(
            stages=(
                StageConfig(
                    num_seeds=num_seeds,
                    iters=config.iters,
                    lm_lambda=2.0 if exterior_mode else 10.0,
                ),
                StageConfig(num_seeds=min(2, num_seeds), iters=config.iters, lm_lambda=2.0),
                StageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
            ),
            distance_weight=DISTANCE_WEIGHT,
            collision_weight=hand.collision_weight,
            force_closure_weight=FORCE_CLOSURE_WEIGHT,
            force_closure_contact_weights=(1.0, *((1.0 / (len(contacts) - 1),) * (len(contacts) - 1))),
            collision_margin=0.0,
            base_step_scale=3e-4,
            init_sample_range_base=0.02,
            init_sample_range_base_rotation=np.deg2rad(30.0) if exterior_mode else 0.0,
            init_sample_range_q=0.15,
            self_collision=config.self_collision,
            contact_resample_interval=config.iters,
            anchor_q_weight=ANCHOR_Q_WEIGHT,
            anchor_base_translation_weight=ANCHOR_TRANSLATION_WEIGHT,
            anchor_base_rotation_weight=anchor_rotation_weight,
        ),
    )
    helper.set_contact_points(
        wp.from_numpy(pads.points[rows], dtype=wp.vec3, device=device),
        wp.from_numpy(pads.links[rows], dtype=wp.int32, device=device),
    )
    init = hand.collision_robot.state(
        q=wp.from_numpy(np.repeat(q_init[None], len(rows), axis=0), dtype=wp.float32, device=device),
        T_world_base=wp.from_numpy(np.repeat(base_a[None], len(rows), axis=0), dtype=wp_vec7, device=device),
    )
    state = helper.solve_all_seeds(init) if exterior_mode else helper.solve(init)
    if exterior_mode:
        rows = np.repeat(rows, num_seeds, axis=0)
    batch = len(rows)
    energy, terms = helper.compute_energy(
        state,
        wp.from_numpy(rows, dtype=wp.int32, device=device),
        return_per_term=True,
        w_fc=FORCE_CLOSURE_WEIGHT,
        w_dis=DISTANCE_WEIGHT,
        w_pen=config.collision_weight,
    )
    q_all = state.q.numpy()
    base_all = state.T_world_base.numpy()
    T_all = pose7_to_tf_mat(base_all)

    wrist_shift = np.linalg.norm(base_all[:, :3] - base_a[:3], axis=-1)
    wrist_turn = 2.0 * np.arccos(np.clip(np.abs(np.einsum("ij,j->i", base_all[:, 3:], base_a[3:])), 0.0, 1.0))
    anchor_energy = 0.5 * (
        np.sum((ANCHOR_Q_WEIGHT * (q_all - q_init)) ** 2, axis=-1)
        + (ANCHOR_TRANSLATION_WEIGHT * wrist_shift) ** 2
        + (anchor_rotation_weight * wrist_turn) ** 2
    )
    depth = _penetration(robot, q_all, T_all, hand.object_scene, hand.val_samples, device=device)
    max_depth = depth.max(axis=1)
    pad_T = _pad_frames(robot, q_all, T_all, pads.links[rows], device)
    pads_obj = _apply_se3(pad_T, pads.points[rows])
    exterior = np.ones(batch, dtype=bool)
    if exterior_mode:
        _, _, contact_faces = trimesh.proximity.closest_point(hand.object_mesh, pads_obj.reshape(-1, 3))
        exterior = hand.contact_face_mask[contact_faces].reshape(batch, -1).all(axis=1)
    surface, object_normals, _ = target_scene.query_sdf(
        wp.from_numpy(pads_obj.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=device),
        scene_offsets=wp.array([0, pads_obj.size // 3], dtype=wp.int32, device=device),
    )
    contact_distance = np.abs(surface.numpy().reshape(batch, -1)).max(axis=1)
    object_normals = object_normals.numpy().reshape(batch, len(contacts), 3)
    finger_normals = object_normals[:, 1:].mean(axis=1)
    finger_normals /= np.linalg.norm(finger_normals, axis=1, keepdims=True)
    opposition = np.einsum("bi,bi->b", object_normals[:, 0], finger_normals)
    pad_alignment = np.einsum(
        "bci,bci->bc", _apply_se3(pad_T, pads.normals[rows], translate=False), object_normals
    ).mean(axis=1)
    safe_depth = config.max_penetration - VALIDATION_MARGIN
    valid = (max_depth <= safe_depth) & exterior
    if exterior_mode:
        valid &= (wrist_shift <= 0.03) & (wrist_turn <= np.deg2rad(20.0)) & (opposition <= 0.0)
    accepted = np.flatnonzero(valid)
    if not len(accepted):
        nearest = int(np.argmin(max_depth))
        exterior_reject = (
            f"and on the exterior (exterior={np.count_nonzero(exterior)}/{batch}, " if exterior_mode else "("
        )
        print(
            f"{hand.label}: reject grasp; no candidate below {safe_depth * 1000:.1f} mm penetration "
            f"{exterior_reject}"
            f"nearest={max_depth[nearest] * 1000:.1f} mm, "
            f"link={robot.spec.link_names[hand.val_samples[1][np.argmax(depth[nearest])]]}, "
            f"contact={contact_distance[nearest] * 1000:.1f} mm, "
            f"wrist={wrist_shift[nearest] * 1000:.1f} mm/{np.rad2deg(wrist_turn[nearest]):.1f} deg, "
            f"opposition={opposition[nearest]:.2f}, pad={pad_alignment[nearest]:.2f})"
        )
        helper.close()
        return None

    score = energy.numpy() + anchor_energy
    if exterior_mode:
        score -= terms["E_pen"].numpy()
    best = int(accepted[np.argmin(score[accepted])])
    q_best = np.clip(
        q_all[best], robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1]
    ).astype(np.float32)
    pad_points, pad_links = pads.points[rows[best]], pads.links[rows[best]]
    contacts_obj = _apply_se3(_pad_frames(robot, q_best[None], T_all[best : best + 1], pad_links, device), pad_points)[
        0
    ]
    final_distance, _, closest = target_scene.query_sdf(
        wp.from_numpy(contacts_obj.astype(np.float32), dtype=wp.vec3, device=device),
        scene_offsets=wp.array([0, len(contacts_obj)], dtype=wp.int32, device=device),
    )
    exterior_report = f", exterior={np.count_nonzero(exterior)}/{batch}" if exterior_mode else ""
    print(
        f"{hand.label}: frames={phase.start}-{phase.stop - 1}, key={phase.key}, "
        f"safe={len(accepted)}/{batch}{exterior_report} candidates, "
        f"ROI={roi_faces}/{len(hand.object_mesh.faces)} faces, "
        f"contacts={'+'.join(robot.spec.link_names[link] for link in pad_links)}, "
        f"surface={np.mean(np.abs(pad_distance[rows[best]])) * 1000:.1f}->"
        f"{np.mean(np.abs(final_distance.numpy())) * 1000:.1f} mm, "
        f"wrist shift={wrist_shift[best] * 1000:.1f} mm/{np.rad2deg(wrist_turn[best]):.1f} deg, "
        f"thumb/fingers normal dot={opposition[best]:.3f}, pad/object normal dot={pad_alignment[best]:.3f}, "
        f"hand mesh penetration p99/max={np.percentile(depth[best], 99) * 1000:.1f}/{max_depth[best] * 1000:.1f} mm "
        f"({robot.spec.link_names[hand.val_samples[1][np.argmax(depth[best])]]}), "
        + ", ".join(f"{name}={value.numpy()[best]:.3f}" for name, value in terms.items())
    )
    helper.close()
    return _Grasp(phase, q_best, base_all[best].astype(np.float32), closest.numpy(), pad_points, pad_links)


def _blend_grasps(hand: _Hand, grasps: List[_Grasp], device: str) -> None:
    """Fade each object-fixed grasp in and out over its detected motion phase."""
    assert hand.object_scene is not None
    robot, window = hand.robot, hand.window
    num_frames, num_contacts = len(hand.q), len(grasps[0].pad_points)
    q_path, T_path = hand.q.copy(), hand.T_obj_base.copy()
    targets = np.repeat(grasps[0].target_obj[None], num_frames, axis=0)
    contacts = np.zeros_like(targets)
    mask = np.zeros((num_frames, num_contacts), dtype=bool)

    for grasp in grasps:
        phase = grasp.phase
        frames = np.arange(phase.start, min(phase.stop + window + 1, num_frames))
        alpha = np.minimum(
            np.clip((frames - phase.start) / window, 0.0, 1.0),
            np.clip((phase.stop + window - frames) / window, 0.0, 1.0),
        )
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        alpha[alpha > 0.95] = 1.0

        q_grasp = np.clip(grasp.q, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])
        q_path[frames] = (1.0 - alpha[:, None]) * hand.q[frames] + alpha[:, None] * q_grasp
        T_free = hand.T_obj_base[frames]
        T_grasp = pose7_to_tf_mat(grasp.base_obj[None])[0]
        rotation_delta = matrix_to_axis_angle(T_free[:, :3, :3].transpose(0, 2, 1) @ T_grasp[:3, :3])
        T_blend = T_free.copy()
        T_blend[:, :3, :3] = T_free[:, :3, :3] @ axis_angle_to_matrix(alpha[:, None] * rotation_delta)
        T_blend[:, :3, 3] = (1.0 - alpha[:, None]) * T_free[:, :3, 3] + alpha[:, None] * T_grasp[:3, 3]
        T_path[frames] = T_blend
        mask[frames] = (alpha == 1.0)[:, None]
        contacts[frames] = _apply_se3(
            _pad_frames(robot, q_path[frames], T_blend, grasp.pad_links, device), grasp.pad_points
        )
        _, _, closest = hand.object_scene.query_sdf(
            wp.from_numpy(contacts[frames].reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=device),
            scene_offsets=wp.array([0, contacts[frames].size // 3], dtype=wp.int32, device=device),
        )
        targets[frames] = closest.numpy().reshape(len(frames), num_contacts, 3)
    hand.q, hand.T_obj_base = q_path, T_path
    hand.targets_obj, hand.contacts_obj, hand.contact_mask = targets, contacts, mask


def _prune_grasps(hand: _Hand, pads: _Pads, config: GraspConfig, device: str) -> None:
    """Re-solve each grasp without the fingers the captured hand never bent that far (see PruneConfig)."""
    prune = config.prune
    assert prune is not None and hand.object_scene is not None and hand.grasp_scene is not None
    robot, window = hand.robot, hand.window
    num_frames = len(hand.q)
    fingers = tuple(link.split("_", maxsplit=1)[0] for link in config.contact_links)
    q_index = {finger: robot.spec.actuated_joint_names.index(f"{finger}_proximal_joint") for finger in fingers[1:]}
    samples = _sample_link_points(robot, prune.collision_samples_per_link)
    collision_robot = Robot(
        replace(
            robot.spec,
            local_collision_sphere_centers=samples[0],
            collision_sphere_radii=np.full(len(samples[0]), prune.collision_radius, dtype=np.float32),
            collision_spheres_link_indices=samples[1],
        )
    )

    # Phases are the enforced-contact spans, with short gaps closed.
    enforced = np.asarray(hand.contact_mask.any(axis=1))
    active = enforced.copy()
    for gap_start, gap_stop in np.flatnonzero(np.diff(np.r_[True, active, True])).reshape(-1, 2):
        if gap_start and gap_stop < num_frames and gap_stop - gap_start <= window:
            active[gap_start:gap_stop] = True
    phases = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)

    for index, (start, stop) in enumerate(phases):
        key = _medoid_frame(hand.T_obj_wrist, start, stop)
        q_first = hand.q[key].copy()
        shift = {finger: float(q_first[column] - hand.q_mano[key, column]) for finger, column in q_index.items()}
        kept = [finger for finger in fingers[1:] if abs(shift[finger]) <= prune.bend_limit]
        kept = kept or [min(shift, key=lambda finger: abs(shift[finger]))]
        dropped = [finger for finger in fingers[1:] if finger not in kept]
        print(
            f"{hand.label}: prune phase={index + 1}/{len(phases)}, key={key}, "
            f"bend_shift={', '.join(f'{finger}={np.rad2deg(value):+.1f}' for finger, value in shift.items())} deg, "
            f"limit={np.rad2deg(prune.bend_limit):.1f} deg -> keep=thumb+{'+'.join(kept)}"
        )
        if not dropped:
            continue

        contact_names = ("thumb_intermediate", *(f"{finger}_intermediate" for finger in kept))
        topologies = tuple(
            np.asarray([robot.spec.link_names.index(name) for name in names], dtype=np.int32)
            for names in (contact_names, ("thumb_distal", *contact_names[1:]))
        )
        T_obj_base_a = hand.T_obj_base[key]
        target_scene, rows, _, _ = _contact_seeds(
            hand,
            pads,
            key,
            q_first,
            T_obj_base_a,
            topologies=topologies,
            per_link=0,
            roi_radius=config.roi_radius,
            device=device,
        )
        helper = GraspOptHelper(
            robot=collision_robot,
            batch_size=len(rows),
            num_contact_points=len(contact_names),
            target_meshes=target_scene,
            scene_meshes=hand.grasp_scene,
            contact_candidates=pads.points_wp,
            contact_candidate_link_indices=pads.links_wp,
            config=GraspOptHelperConfig(
                stages=(
                    StageConfig(num_seeds=prune.num_seeds, iters=prune.iters, lm_lambda=10.0),
                    StageConfig(num_seeds=min(2, prune.num_seeds), iters=prune.iters, lm_lambda=2.0),
                    StageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
                ),
                distance_weight=DISTANCE_WEIGHT,
                collision_weight=prune.collision_weight,
                force_closure_weight=FORCE_CLOSURE_WEIGHT,
                force_closure_contact_weights=(1.0, *((1.0 / len(kept),) * len(kept))),
                collision_margin=0.0,
                base_step_scale=3e-4,
                init_sample_range_base=0.02,
                init_sample_range_q=0.15,
                self_collision=True,
                anchor_q_weight=ANCHOR_Q_WEIGHT,
                anchor_base_translation_weight=ANCHOR_TRANSLATION_WEIGHT,
                anchor_base_rotation_weight=ANCHOR_ROTATION_WEIGHT,
            ),
        )
        helper.set_contact_points(
            wp.from_numpy(pads.points[rows], dtype=wp.vec3, device=device),
            wp.from_numpy(pads.links[rows], dtype=wp.int32, device=device),
        )
        dropped_columns = np.asarray([q_index[finger] for finger in dropped])
        q_init = q_first.copy()
        q_init[dropped_columns] = hand.q_mano[key, dropped_columns]
        base_a = tf_mat_to_pose7(T_obj_base_a[None])[0]
        state = helper.solve(
            collision_robot.state(
                q=wp.from_numpy(np.repeat(q_init[None], len(rows), axis=0), dtype=wp.float32, device=device),
                T_world_base=wp.from_numpy(np.repeat(base_a[None], len(rows), axis=0), dtype=wp_vec7, device=device),
            )
        )
        energy, terms = helper.compute_energy(
            state,
            wp.from_numpy(rows, dtype=wp.int32, device=device),
            return_per_term=True,
            w_fc=FORCE_CLOSURE_WEIGHT,
            w_dis=DISTANCE_WEIGHT,
            w_pen=prune.collision_weight,
        )
        q_all = state.q.numpy()
        base_all = state.T_world_base.numpy()
        q_all[:, dropped_columns] = hand.q_mano[key, dropped_columns]  # the dropped fingers stay captured
        score = energy.numpy() + 0.5 * (
            np.sum((ANCHOR_Q_WEIGHT * (q_all - q_init)) ** 2, axis=-1)
            + (ANCHOR_TRANSLATION_WEIGHT * np.linalg.norm(base_all[:, :3] - base_a[:3], axis=-1)) ** 2
            + (
                ANCHOR_ROTATION_WEIGHT
                * 2.0
                * np.arccos(np.clip(np.abs(np.einsum("ij,j->i", base_all[:, 3:], base_a[3:])), 0.0, 1.0))
            )
            ** 2
        )
        depth = _penetration(
            robot,
            q_all,
            pose7_to_tf_mat(base_all),
            hand.object_scene,
            samples,
            device=device,
            radius=prune.collision_radius,
        ).max(axis=1)
        valid = np.flatnonzero(depth <= prune.max_penetration)
        valid = valid if len(valid) else np.flatnonzero(depth <= depth.min() + 0.001)
        best = int(valid[np.argmin(score[valid])])
        q_best = np.clip(q_all[best], robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])
        T_best = pose7_to_tf_mat(base_all[best : best + 1])
        pad_points, pad_links = pads.points[rows[best]], pads.links[rows[best]]
        selected = [robot.spec.link_names[link].split("_", maxsplit=1)[0] for link in pad_links]
        slots = np.asarray([fingers.index(finger) for finger in selected])

        before_depth = _penetration(
            robot,
            q_first[None],
            T_obj_base_a[None],
            hand.object_scene,
            samples,
            device=device,
            radius=prune.collision_radius,
        ).max()
        before_surface = np.linalg.norm(hand.contacts_obj[key, slots] - hand.targets_obj[key, slots], axis=-1).mean()
        after_surface, _, _ = target_scene.query_sdf(
            wp.from_numpy(
                _apply_se3(_pad_frames(robot, q_best[None], T_best, pad_links, device), pad_points)[0].astype(
                    np.float32
                ),
                dtype=wp.vec3,
                device=device,
            ),
            scene_offsets=wp.array([0, len(pad_points)], dtype=wp.int32, device=device),
        )
        after_surface = np.abs(after_surface.numpy()).mean()
        accepted = depth[best] <= prune.max_penetration and after_surface <= before_surface
        print(
            f"{hand.label}: pass2 contacts={'+'.join(selected)}, "
            f"surface={before_surface * 1000:.1f}->{after_surface * 1000:.1f} mm, "
            f"penetration={before_depth * 1000:.1f}->{depth[best] * 1000:.1f} mm, "
            + ", ".join(f"{name}={value.numpy()[best]:.3f}" for name, value in terms.items())
            + f" -> {'accept' if accepted else 'reject'}"
        )
        helper.close()
        if not accepted:
            continue

        # Fade the correction in over the phase; the dropped fingers fade back to their captured pose.
        frames = np.arange(max(0, int(start) - window), min(num_frames, int(stop) + window + 1))
        alpha = np.minimum(
            np.clip((frames - frames[0]) / window, 0.0, 1.0), np.clip((frames[-1] - frames) / window, 0.0, 1.0)
        )
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        kept_columns = np.asarray(
            [
                column
                for column, name in enumerate(robot.spec.actuated_joint_names)
                if name.startswith("thumb_") or name.split("_", maxsplit=1)[0] in kept
            ]
        )
        correction = q_best - hand.q[key]
        hand.q[np.ix_(frames, kept_columns)] += alpha[:, None] * correction[kept_columns]
        hand.q[np.ix_(frames, dropped_columns)] = (1.0 - alpha[:, None]) * hand.q[
            np.ix_(frames, dropped_columns)
        ] + alpha[:, None] * hand.q_mano[np.ix_(frames, dropped_columns)]
        T_delta = inverse_tf_mat(T_obj_base_a[None])[0] @ T_best[0]
        T_blend = np.broadcast_to(np.eye(4), (len(frames), 4, 4)).copy()
        T_blend[:, :3, :3] = axis_angle_to_matrix(alpha[:, None] * matrix_to_axis_angle(T_delta[:3, :3]))
        T_blend[:, :3, 3] = alpha[:, None] * T_delta[:3, 3]
        hand.T_obj_base[frames] = hand.T_obj_base[frames] @ T_blend

        held = np.flatnonzero(enforced[start:stop]) + start
        hand.contact_mask[held] = False
        hand.contact_mask[held[:, None], slots[None]] = True
        contacts_obj = _apply_se3(
            _pad_frames(robot, hand.q[held], hand.T_obj_base[held], pad_links, device), pad_points
        )
        _, _, closest = hand.object_scene.query_sdf(
            wp.from_numpy(contacts_obj.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=device),
            scene_offsets=wp.array([0, contacts_obj.size // 3], dtype=wp.int32, device=device),
        )
        hand.contacts_obj[held[:, None], slots[None]] = contacts_obj
        hand.targets_obj[held[:, None], slots[None]] = closest.numpy().reshape(contacts_obj.shape)
    hand.q = np.clip(hand.q, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1]).astype(
        np.float32
    )


def solve_combo_grasp(
    combo: ComboClip,
    preset: ComboPreset,
    device: str,
    *,
    config: GraspConfig,
    max_frames: int = 0,
    body_retargeter: Optional[ComboRetargeter] = None,
) -> ComboGrasp:
    """Retarget every engaged hand of ``combo``, synthesizing a canonical grasp wherever the hand
    carries its object, then optionally drive the whole body from the result."""
    num_frames = combo.num_frames
    if max_frames:
        num_frames = min(num_frames, max_frames)
    assert num_frames >= 3, "at least 3 frames are required"
    window = _odd_window(combo.fps)
    sides = [side for side in combo.sides if side in preset.hands]

    # Which hands get a canonical grasp: sustained contact AND an object that actually travels.
    held: Dict[str, np.ndarray] = {}
    grasping: Dict[str, bool] = {}
    refine_contacts: Dict[str, np.ndarray] = {}
    moving_objects: List[str] = []
    for side in sides:
        track = combo.hands[side]
        if track.object_name is None or track.object_mesh is None:
            print(f"{side}/free: MANO (no collision mesh)")
            refine_contacts[side] = np.zeros((num_frames, len(CONTACT_MANO_INDICES)), dtype=bool)
            continue
        contact = (
            track.contact[:num_frames] if config.object_contact else track.contact_mask[:num_frames].sum(axis=1) >= 2
        )
        held[side] = _majority(contact, window)
        object_T = pose7_to_tf_mat(track.object_pose_native[:num_frames])
        travel = np.linalg.norm(track.object_pose_native[:num_frames, :3] - track.object_pose_native[0, :3], axis=-1)
        turn = np.arccos(
            np.clip((np.einsum("ij,fij->f", object_T[0, :3, :3], object_T[:, :3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        )
        distance, angle = float(np.percentile(travel, 95)), float(np.percentile(turn, 95))
        grasping[side] = (distance > MOVE_DISTANCE or angle > MOVE_ANGLE) and bool(held[side].any())
        smoothed = np.asarray(
            uniform_filter1d(track.contact_mask[:num_frames].astype(np.float32), window, axis=0, mode="nearest"),
            dtype=np.float32,
        )
        slots = smoothed >= 0.5
        refine_contacts[side] = np.zeros_like(slots) if grasping[side] else slots & held[side][:, None]
        mode = "GraspKit" if grasping[side] else "contact" if held[side].any() else "collision"
        print(
            f"{side}/{track.object_name}: motion={distance * 100:.1f} cm, {np.rad2deg(angle):.1f} deg, "
            f"contact={held[side].mean() * 100:.0f}% -> MANO+collision+{mode}"
        )
        if grasping[side] and track.object_name not in moving_objects:
            moving_objects.append(track.object_name)

    start = time.time()
    hands: Dict[str, _Hand] = {}
    for side in sides:
        hand = _prepare_hand(
            side,
            combo,
            preset,
            config,
            num_frames=num_frames,
            refine_contacts=refine_contacts[side],
            refine=not grasping.get(side, False),
            device=device,
        )
        hands[side] = hand
        if not grasping.get(side):
            continue
        pads = _prepare_pads(hand, preset, side, config, device)

        object_delta = inverse_tf_mat(hand.T_world_obj[:-1]) @ hand.T_world_obj[1:]
        moving = np.r_[
            False,
            (np.linalg.norm(object_delta[:, :3, 3], axis=-1) > MOTION_STEP_DISTANCE)
            | (np.linalg.norm(matrix_to_axis_angle(object_delta[:, :3, :3]), axis=-1) > MOTION_STEP_ANGLE),
        ]
        stable = _majority(moving, window) & held[side]
        if config.single_grasp:
            frames = np.flatnonzero(stable)
            if not len(frames):
                frames = np.flatnonzero(moving & held[side])
            spans = [(int(frames[0]), int(frames[-1] + 1))] if len(frames) else []
        else:
            runs = [
                (int(run_start), int(run_stop))
                for run_start, run_stop in np.flatnonzero(np.diff(np.r_[False, stable, False])).reshape(-1, 2)
                if run_stop - run_start >= window
            ]
            spans = []
            for run_start, run_stop in runs:
                regrip = (
                    inverse_tf_mat(hand.T_obj_wrist[spans[-1][1] - 1]) @ hand.T_obj_wrist[run_start] if spans else None
                )
                if regrip is not None and (
                    np.linalg.norm(regrip[:3, 3]) <= 3.0 * RESIDUAL_TRANSLATION_LIMIT
                    and float(np.linalg.norm(matrix_to_axis_angle(regrip[:3, :3]))) <= 3.0 * RESIDUAL_ROTATION_LIMIT
                ):
                    spans[-1] = (spans[-1][0], run_stop)
                else:
                    spans.append((run_start, run_stop))
            if not runs:
                frames = np.flatnonzero(moving & held[side])
                spans = [(int(frames[0]), int(frames[-1] + 1))] if len(frames) else []

        grasps = []
        for start_frame, stop_frame in spans:
            key = _medoid_frame(hand.T_obj_wrist, start_frame, stop_frame)
            phase = _Phase(start_frame, stop_frame, key)
            q_init = np.median(
                hand.q[np.clip(np.arange(key - 1, key + 2), start_frame, stop_frame - 1)], axis=0
            ).astype(np.float32)
            grasp = _canonical_grasp(hand, pads, config, phase, q_init, device=device)
            if grasp is not None:
                grasps.append(grasp)
        if not grasps:
            keep = "MANO" if config.single_grasp else "MANO+collision"
            print(f"{hand.label}: no collision-safe canonical grasp; keep {keep}")
            continue
        hand.num_phases = len(grasps)
        _blend_grasps(hand, grasps, device)
        if config.prune is not None and hand.contact_mask.any():
            _prune_grasps(hand, pads, config, device)

    if not any(grasping.values()):
        print("no moving hand-object contact; solved common MANO/collision/contact only")
    print(
        f"solved {len(sides)} symmetric hand trajectories and {sum(hand.num_phases for hand in hands.values())} "
        f"canonical grasps in {time.time() - start:.1f}s on {device}"
    )

    # Everything above lives in the object frame; lift it into the scaled body world once.
    T_world_obj = {side: pose7_to_tf_mat(combo.hands[side].object_pose_world[:num_frames]) for side in hands}
    result = ComboGrasp(
        combo=combo,
        num_frames=num_frames,
        robots={side: hand.robot for side, hand in hands.items()},
        joints={side: hand.q for side, hand in hands.items()},
        bases_world={side: tf_mat_to_pose7(T_world_obj[side] @ hand.T_obj_base) for side, hand in hands.items()},
        targets_world={side: _apply_se3(T_world_obj[side][:, None], hand.targets_obj) for side, hand in hands.items()},
        contacts_world={
            side: _apply_se3(T_world_obj[side][:, None], hand.contacts_obj) for side, hand in hands.items()
        },
        contact_masks={side: hand.contact_mask for side, hand in hands.items()},
        moving_objects=tuple(moving_objects),
        qpos=None,
    )
    if body_retargeter is None:
        return result

    start = time.time()
    result.qpos = body_retargeter.solve(
        combo,
        max_frames=num_frames,
        hand_trajectories={
            side: HandTrajectory(q=result.joints[side], T_world_base=result.bases_world[side]) for side in result.joints
        },
    )
    body_robot = body_retargeter.body_robot
    ndof = body_robot.spec.num_actuated_joints
    body_links = (
        body_robot.forward_kinematics_via_matrix_torch(
            torch.from_numpy(result.qpos[:, 7 : 7 + ndof]).to(device),
            torch.from_numpy(pose7_to_tf_mat(result.qpos[:, :7]).astype(np.float32)).to(device),
        )
        .cpu()
        .numpy()
    )
    for side, robot in result.robots.items():
        binding = preset.hands[side]
        achieved = body_links[:, body_robot.spec.link_names.index(binding.wrist_link)]
        target = (
            robot.forward_kinematics_via_matrix_torch(
                torch.from_numpy(result.joints[side]).to(device),
                torch.from_numpy(pose7_to_tf_mat(result.bases_world[side]).astype(np.float32)).to(device),
            )
            .cpu()
            .numpy()[:, robot.spec.link_names.index(binding.hand_wrist_link)]
        )
        active = np.asarray(result.contact_masks[side].any(axis=1))
        if not active.any():
            active[:] = True
        position_error = np.linalg.norm(achieved[active, :3, 3] - target[active, :3, 3], axis=-1)
        rotation_error = np.arccos(
            np.clip((np.einsum("fij,fij->f", achieved[active, :3, :3], target[active, :3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        )
        print(
            f"{side} whole-body palm tracking: "
            f"median={np.median(position_error) * 1000:.1f} mm/{np.rad2deg(np.median(rotation_error)):.1f} deg, "
            f"p95={np.percentile(position_error, 95) * 1000:.1f} mm/"
            f"{np.rad2deg(np.percentile(rotation_error, 95)):.1f} deg, "
            f">10 mm frames={np.flatnonzero(active)[position_error > 0.01].tolist()}"
        )
    print(f"solved whole body in {time.time() - start:.1f}s")
    return result


__all__ = [
    "PARAHOME_GRASP",
    "SAGA_GRASP",
    "GraspConfig",
    "ComboGrasp",
    "PruneConfig",
    "solve_combo_grasp",
]
