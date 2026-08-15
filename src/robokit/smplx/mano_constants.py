"""MANO model constants: joint count, joint names, fingertip vertex IDs."""

from typing import List, Literal


Side = Literal["left", "right"]

MANO_NUM_VERTICES: int = 778
MANO_NUM_JOINTS: int = 16
MANO_NUM_POSE_DIRS: int = 135  # (MANO_NUM_JOINTS - 1) * 9
MANO_NUM_PCA_COMPONENTS: int = 45
MANO_NUM_SHAPE_COEFFS: int = 10

# wrist + 5 fingers × 3 phalanges
# Order matches the MPI MANO `kintree_table`.
MANO_JOINT_NAMES: List[str] = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]

# fingertip vertex IDs in [thumb, index, middle, ring, pinky] order
# Right and left differ in the middle finger by mesh asymmetry.
MANO_FINGERTIP_NAMES: List[str] = ["thumb", "index", "middle", "ring", "pinky"]
MANO_FINGERTIP_VERTEX_IDS_RIGHT: List[int] = [745, 317, 444, 556, 673]
MANO_FINGERTIP_VERTEX_IDS_LEFT: List[int] = [745, 317, 445, 556, 673]


def mano_fingertip_vertex_ids(side: Side) -> List[int]:
    """`[thumb, index, middle, ring, pinky]` mesh vertex IDs for `side`."""
    if side == "right":
        return MANO_FINGERTIP_VERTEX_IDS_RIGHT
    return MANO_FINGERTIP_VERTEX_IDS_LEFT


# MANO-native joint order (wrist, index, middle, pinky, ring, thumb) +
# 5 fingertips → OpenPose hand-keypoint order
# (wrist, thumb×3+tip, index×3+tip, middle×3+tip, ring×3+tip, pinky×3+tip).
MANOPTH_KEYPOINT_PERMUTATION: List[int] = [
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
]
