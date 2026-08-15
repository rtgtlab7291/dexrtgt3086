"""ComboClip - the per-clip input contract between `loaders/` and `ComboRetargeter`.

Each engaged hand gets a `HandTrack` naming its own object (two hands may work two objects).
Poses are translation-first, scalar-first `[x, y, z, qw, qx, qy, qz]`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from robokit.smplx import SMPLX_JOINT_NAMES, SMPLX_STATIC_LANDMARK_NAMES


# The 10 finger keypoints (intermediates + tips) that carry contact labels to robot links.
CONTACT_MANO_INDICES = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
CONTACT_DISTANCE = 0.025  # joint-center-to-labeled-surface threshold (native world)

HANDS = ("right", "left")  # canonical ordering wherever hands are enumerated
FINGERS = ("thumb", "index", "middle", "ring", "pinky")  # canonical finger order (MANO and inspire)


@dataclass
class SceneGeometry:
    """Furniture and untouched objects around the interaction.

    Mesh and pose are scaled by `ComboClip.scale` about the ground, because the room is contacted
    by the scaled body (grasped objects stay native instead - see `HandTrack`).
    """

    names: List[str]  # P part names, e.g. "diningtable_base"
    meshes: List[trimesh.Trimesh]  # P canonical meshes, SCALED to the body world
    poses: np.ndarray  # (P, T, 7) body-world pose per part per frame
    static: np.ndarray  # (P,) bool, part never moves over the clip (place it once)


@dataclass
class HandTrack:
    """One hand's captured motion, and the object it interacts with if it grasps one.

    Keypoints and contacts are native scale; `object_pose_world` is the same object in the scaled
    body world. A free hand (fingers captured, nothing grasped) leaves `object_name`/`object_mesh`
    `None` and uses its own wrist pose as the object anchor.
    """

    mano: np.ndarray  # (T, 21, 3) NATIVE keypoints, manopth order
    mano_overlay: np.ndarray  # (T, 21, 3) keypoints bridged into the scaled body world (viz)
    wrist_pos: np.ndarray  # (T, 3) native
    wrist_quat_wxyz: np.ndarray  # (T, 4)
    object_name: Optional[str]  # the grasped object (None = free hand); shared name = shared qpos slot
    object_mesh: Optional[trimesh.Trimesh]  # canonical mesh, NATIVE size (None = free hand)
    object_pose_native: np.ndarray  # (T, 7) grasped-object (or free-hand wrist) pose, native world
    object_pose_world: np.ndarray  # (T, 7) the same anchor in the scaled body world
    contact: np.ndarray  # (T,) bool, any object contact this frame
    contact_points: np.ndarray  # (T, 10, 3) NATIVE-world contact point per CONTACT_MANO_INDICES slot
    contact_normals: np.ndarray  # (T, 10, 3) outward surface normal at each contact point
    contact_mask: np.ndarray  # (T, 10) bool, slot has a contact within CONTACT_DISTANCE


@dataclass
class ComboClip:
    """One body + hands + object interaction segment, two-scale.

    Human motion is native and ground-normalized; `scale` (robot height / subject height) lifts it
    into the robot-sized body world. The hand-object subsystem stays native, because a robot hand
    is human-hand sized and must wrap the real object.
    """

    human_pose7: np.ndarray  # (T, 55, 7) NATIVE SMPL-X joint poses, ground-normalized (body-pass input)
    human_height: float  # subject T-pose height (the body pass rescales against it)
    scale: float  # global isotropic scale = robot_height / human_height
    fps: int
    hands: Dict[str, HandTrack]  # "right" and/or "left"
    scene: Optional[SceneGeometry] = None  # surrounding room, when the source captured one

    @property
    def num_frames(self) -> int:
        """Frames in the segment."""
        return int(self.human_pose7.shape[0])

    @property
    def sides(self) -> List[str]:
        """Engaged hands in canonical order."""
        return [s for s in HANDS if s in self.hands]

    @property
    def object_names(self) -> List[str]:
        """Distinct grasped objects in canonical hand order; the qpos tail carries one 7-pose each."""
        names: List[str] = []
        for side in self.sides:
            name = self.hands[side].object_name
            if name is not None and name not in names:
                names.append(name)
        return names


# manopth/OpenPose 21-keypoint order: wrist, then thumb/index/middle/ring/pinky as joint1..3 + tip.
def mano_index(side: str = "right") -> Tuple[List[int], List[int]]:
    """Returns (smplx joint index or -1 per slot, landmark index per tip slot) for one hand.

    Example:
        >>> joints, tips = mano_index("left")
        >>> len(joints), joints[0] >= 0, joints[4], tips[4] >= 0
        (21, True, -1, True)
    """
    joint_idx, tip_landmark_idx = [], []
    joint_idx.append(SMPLX_JOINT_NAMES.index(f"{side}_wrist"))
    tip_landmark_idx.append(-1)
    for finger in FINGERS:
        for k in (1, 2, 3):
            joint_idx.append(SMPLX_JOINT_NAMES.index(f"{side}_{finger}{k}"))
            tip_landmark_idx.append(-1)
        joint_idx.append(-1)
        tip_landmark_idx.append(SMPLX_STATIC_LANDMARK_NAMES.index(f"{side}_{finger}"))
    return joint_idx, tip_landmark_idx


__all__ = [
    "CONTACT_DISTANCE",
    "CONTACT_MANO_INDICES",
    "FINGERS",
    "HANDS",
    "ComboClip",
    "HandTrack",
    "SceneGeometry",
    "mano_index",
]
