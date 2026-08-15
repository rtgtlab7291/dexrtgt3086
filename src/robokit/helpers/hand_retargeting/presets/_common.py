"""Shared semantic target topology for built-in hand presets."""

from typing import Dict, Mapping, Sequence, Tuple

from robokit.utils.hand_coord_utils import MANOPTH_HAND_COORD_SPEC


TARGET_COORD_SPEC = MANOPTH_HAND_COORD_SPEC
TARGET_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
TARGET_CHAINS = (
    ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"),
    ("index_mcp", "index_pip", "index_dip", "index_tip"),
    ("middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
    ("ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
    ("pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
)
FIVE_FINGERTIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
FOUR_FINGERTIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip")
THREE_FINGERTIPS = ("thumb_tip", "index_tip", "middle_tip")

FIVE_ROOT_TIP_PAIRS = tuple(("wrist", tip) for tip in FIVE_FINGERTIPS)
FOUR_ROOT_TIP_PAIRS = tuple(("wrist", tip) for tip in FOUR_FINGERTIPS)
THREE_ROOT_TIP_PAIRS = tuple(("wrist", tip) for tip in THREE_FINGERTIPS)
FIVE_TIP_PAIRS = tuple((origin, task) for i, origin in enumerate(FIVE_FINGERTIPS) for task in FIVE_FINGERTIPS[i + 1 :])
FOUR_TIP_PAIRS = tuple((origin, task) for i, origin in enumerate(FOUR_FINGERTIPS) for task in FOUR_FINGERTIPS[i + 1 :])
THREE_PINCH_PAIRS = (("thumb_tip", "index_tip"), ("thumb_tip", "middle_tip"))
FIVE_PINCH_PAIRS = tuple(("thumb_tip", tip) for tip in FIVE_FINGERTIPS[1:])
FIVE_ROOT_JOINT_PAIRS = (
    ("wrist", "thumb_mcp"),
    ("wrist", "index_pip"),
    ("wrist", "middle_pip"),
    ("wrist", "ring_pip"),
    ("wrist", "pinky_pip"),
)


def correspondences(
    target_link_names: Mapping[str, str], target_pairs: Sequence[Tuple[str, str]], weight: float = 1.0
) -> Dict[Tuple[str, str, str, str], float]:
    """Build weighted correspondences from semantic target pairs."""
    return {(target_link_names[origin], target_link_names[task], origin, task): weight for origin, task in target_pairs}


__all__ = []
