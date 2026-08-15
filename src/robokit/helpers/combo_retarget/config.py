"""Robot bindings for combo retargeting, and the hand-asset repair they drive.

`ComboPreset` binds a whole-body preset to the standalone hand used per side; instances live in
`presets/`. `HandBinding.prepare_assets` repairs the standalone hand assets on first use:
copying the assembly's `*_tip` joint origins into the standalone urdf (`align_tips_to_assembly`)
and re-expressing the right hand's collision spheres into left link frames (`mirror_spheres`).
Repaired files are written once as `*_combo*` siblings of the source URDF and reused; delete
them to regenerate.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from robokit.helpers.combo_retarget.clip import FINGERS
from robokit.helpers.hand_retargeting.presets import TARGET_NAMES
from robokit.helpers.hand_retargeting.presets import inspire as inspire_preset
from robokit.helpers.humanoid_retarget.config import HumanoidRetargetingOnlineConfig
from robokit.robo import Robot
from robokit.utils.hand_coord_utils import HandCoordinateSpec
from robokit.xform.numpy import inverse_tf_mat


TIP_SPHERE_RADIUS = 0.005  # the shipped set stops at the intermediate links, leaving tip pads bare


def _base_frames_and_centroids(urdf: str, device: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-link ``T_handbase_link`` and mesh centroid (hand-base frame) for a standalone hand."""
    robot = Robot.load(urdf, load_meshes=True, mesh_dir=str(Path(urdf).parent))
    q0 = torch.zeros((1, robot.spec.num_actuated_joints), device=device)
    fk = robot.forward_kinematics_via_matrix_torch(q0, torch.eye(4, device=device)[None]).cpu().numpy()[0]
    base = inverse_tf_mat(fk[robot.spec.link_names.index("hand_base_link")][None])[0]
    frames, centroids = {}, {}
    for index, name in enumerate(robot.spec.link_names):
        local = (base @ fk[index]).astype(np.float32)
        frames[name] = local
        geom = robot.spec.link_visual_geometries.get(name)
        vertices = None if geom is None else np.asarray(geom.to_mesh().vertices, dtype=np.float32)
        if vertices is not None and vertices.shape[0]:
            centroids[name] = (vertices @ local[:3, :3].T + local[:3, 3]).mean(axis=0)
    return frames, centroids


def link_transfer_maps(source_urdf: str, target_urdf: str, device: str) -> Dict[str, np.ndarray]:
    """Per-link 4x4 taking a point in a SOURCE link frame to the mirrored TARGET link frame.

    Reflection across the hand-base z plane handles the frames; a per-link translation from the two
    mesh centroids absorbs the thumb's genuinely different modelling.
    """
    src_frames, src_centroids = _base_frames_and_centroids(source_urdf, device)
    dst_frames, dst_centroids = _base_frames_and_centroids(target_urdf, device)
    reflect = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
    maps = {}
    for name, src_local in src_frames.items():
        dst_local = dst_frames.get(name)
        if dst_local is None:
            continue
        shift = np.eye(4, dtype=np.float32)
        if name in src_centroids and name in dst_centroids:
            mirrored = reflect[:3, :3] @ src_centroids[name]
            shift[:3, 3] = dst_centroids[name] - mirrored
        maps[name] = (inverse_tf_mat(dst_local[None])[0] @ shift @ reflect @ src_local).astype(np.float32)
    return maps


@dataclass(frozen=True)
class HandBinding:
    """One side's standalone hand assets and how its joints and arm map into the assembly.

    Standalone joint `<name>` corresponds to assembly joint `<joint_prefix><name>`; `wrist_link`
    (assembly) and `hand_wrist_link` (standalone) denote the same physical frame. The arm snap
    pins `orientation_links` as extra anchors; they must be rigid in the wrist frame.
    """

    urdf: str
    collision_spheres: str
    wrist_link: str
    hand_wrist_link: str = "hand_base_link"
    joint_prefix: str = "R_"
    orientation_links: Tuple[str, ...] = ("R_thumb_proximal_base", "R_index_proximal", "R_pinky_proximal")
    elbow_link: str = "right_elbow_link"  # forearm direction, for palm-aware grasp seeds
    arm_prefix: str = "right_"  # selects this side's arm joints for the snap IK
    wrist_yaw_link: str = "right_wrist_yaw_link"  # body-model link the wrist target is injected on
    wrist_coord_spec: Optional[HandCoordinateSpec] = None
    mirror_spheres: bool = False  # spheres authored for the OTHER hand
    align_tips_to_assembly: bool = False  # standalone *_tip frames use another convention

    def prepare_assets(
        self, body_urdf: str, device: str, sphere_source: Optional["HandBinding"] = None
    ) -> Tuple[str, str, str]:
        """Return `(urdf_path, spheres_with_tips, spheres_plain)`, generating the repaired assets
        on the first call and reusing them thereafter.

        Two sphere sets: the contact refine needs fingertip spheres appended, while the hand pass
        keeps the shipped set its tuned weights were validated against.
        """
        src = Path(self.urdf)
        tag = self.joint_prefix.rstrip("_").lower() or "r"  # "l" / "r"
        # Repaired URDF stays beside the shared meshes/ dir so its relative mesh paths resolve.
        urdf_out = src.with_name(f"{src.stem}_combo.urdf") if self.align_tips_to_assembly else src
        plain_out = (
            src.with_name(f"combo_spheres_{tag}_plain.yaml") if self.mirror_spheres else Path(self.collision_spheres)
        )
        tips_out = src.with_name(f"combo_spheres_{tag}_tips.yaml")
        if urdf_out.exists() and plain_out.exists() and tips_out.exists():
            return str(urdf_out), str(tips_out), str(plain_out)

        # 1. tip-frame alignment: copy the assembly's authoritative *_tip origins into the standalone
        #    urdf. Match whole <joint> elements so the origin lookup cannot leak into a neighbour.
        if self.align_tips_to_assembly:
            assembly_text = Path(body_urdf).read_text()
            text = src.read_text()
            for finger in FINGERS:
                asm = re.search(rf'<joint name="{self.joint_prefix}{finger}_tip_joint".*?</joint>', assembly_text, re.S)
                assert asm is not None, f"{self.joint_prefix}{finger}_tip_joint missing from the assembly"
                xyz = re.search(r'<origin[^>]*xyz="([^"]*)"', asm.group(0))
                assert xyz is not None, f"{self.joint_prefix}{finger}_tip_joint has no origin"
                block = re.search(rf'<joint name="{finger}_tip_joint".*?</joint>', text, re.S)
                assert block is not None, f"{finger}_tip_joint missing from the standalone urdf"
                segment = re.sub(r'(<origin[^>]*xyz=")[^"]*(")', rf"\g<1>{xyz.group(1)}\g<2>", block.group(0), count=1)
                text = text[: block.start()] + segment + text[block.end() :]
            urdf_out.write_text(text)

        # 2. spheres: mirror onto this side (if authored for the other), then append tip pads.
        spheres = yaml.safe_load(Path(self.collision_spheres).read_text())
        if self.mirror_spheres:
            assert sphere_source is not None, "mirror_spheres needs the binding the spheres were authored for"
            maps = link_transfer_maps(sphere_source.urdf, str(urdf_out), device)
            spheres["collision_spheres"] = {
                link: [
                    {
                        "center": (maps[link][:3, :3] @ np.asarray(e["center"], dtype=np.float32) + maps[link][:3, 3])
                        .astype(float)
                        .tolist(),
                        "radius": e["radius"],
                    }
                    for e in entries
                ]
                if link in maps
                else entries
                for link, entries in spheres["collision_spheres"].items()
            }
            plain_out.write_text(yaml.safe_dump(spheres))

        for mano_tip in (4, 8, 12, 16, 20):
            tip_link = inspire_preset.spec.target_link_names[TARGET_NAMES[mano_tip]]
            spheres["collision_spheres"].setdefault(tip_link, []).append(
                {"center": [0.0, 0.0, 0.0], "radius": TIP_SPHERE_RADIUS}
            )
        tips_out.write_text(yaml.safe_dump(spheres))
        return str(urdf_out), str(tips_out), str(plain_out)


@dataclass(frozen=True)
class ComboPreset:
    """Per-robot binding: the whole-body preset plus the standalone hand for each side."""

    body: HumanoidRetargetingOnlineConfig
    hands: Dict[str, HandBinding]
