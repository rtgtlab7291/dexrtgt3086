"""Sharpa Wave Hand retargeting presets."""

from robokit.helpers.hand_retargeting.config import (
    HandRetargetingOfflineConfig,
    HandRetargetingOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    FIVE_PINCH_PAIRS,
    FIVE_ROOT_JOINT_PAIRS,
    FIVE_ROOT_TIP_PAIRS,
    FIVE_TIP_PAIRS,
    TARGET_CHAINS,
    TARGET_COORD_SPEC,
    TARGET_NAMES,
    correspondences,
)
from robokit.utils.hand_coord_utils import HandCoordinateSpec


_LINKS = {
    "wrist": "right_hand_C_MC",
    "thumb_cmc": "right_thumb_MC",
    "thumb_mcp": "right_thumb_PP",
    "thumb_ip": "right_thumb_DP",
    "thumb_tip": "right_thumb_fingertip",
    "index_mcp": "right_index_PP",
    "index_pip": "right_index_MP",
    "index_dip": "right_index_DP",
    "index_tip": "right_index_fingertip",
    "middle_mcp": "right_middle_PP",
    "middle_pip": "right_middle_MP",
    "middle_dip": "right_middle_DP",
    "middle_tip": "right_middle_fingertip",
    "ring_mcp": "right_ring_PP",
    "ring_pip": "right_ring_MP",
    "ring_dip": "right_ring_DP",
    "ring_tip": "right_ring_fingertip",
    "pinky_mcp": "right_pinky_PP",
    "pinky_pip": "right_pinky_MP",
    "pinky_dip": "right_pinky_DP",
    "pinky_tip": "right_pinky_fingertip",
}
_OFFLINE_LINKS = {
    name: _LINKS[name]
    for name in (
        "wrist",
        "thumb_mcp",
        "thumb_tip",
        "index_pip",
        "index_tip",
        "middle_pip",
        "middle_tip",
        "ring_pip",
        "ring_tip",
        "pinky_pip",
        "pinky_tip",
    )
}
_THUMB_PAIRS = (("thumb_mcp", "thumb_ip"), ("thumb_ip", "thumb_tip"))
_VECTOR_PAIRS = FIVE_ROOT_TIP_PAIRS + FIVE_ROOT_JOINT_PAIRS + FIVE_TIP_PAIRS + _THUMB_PAIRS
_DIRECTION_PAIRS = tuple((chain[i], chain[i + 1]) for chain in TARGET_CHAINS for i in range(len(chain) - 1))


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-z", four_fingers_up_axis="y"),
)
online = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS, FIVE_ROOT_TIP_PAIRS + FIVE_ROOT_JOINT_PAIRS + FIVE_TIP_PAIRS, 0.05),
        **correspondences(_LINKS, (_THUMB_PAIRS[0],), 0.2),
        **correspondences(_LINKS, (_THUMB_PAIRS[1],), 0.4),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.2,
    vector_huber_delta=0.01,
    vector_soft_gate_start_distance=0.024,
    vector_soft_gate_full_distance=0.006,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.02,
    self_collision_weight=10.0,
    self_collision_margin=0.002,
    self_collision_max_active_pairs=32,
)
offline = HandRetargetingOfflineConfig(target_scale=1.2, global_position_weight=26.0)


__all__ = ["offline", "online", "spec"]
