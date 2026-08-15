"""Inspire Hand retargeting presets."""

from robokit.helpers.hand_retargeting.config import (
    HandRetargetingOfflineConfig,
    HandRetargetingOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    FIVE_PINCH_PAIRS,
    FIVE_ROOT_TIP_PAIRS,
    FIVE_TIP_PAIRS,
    TARGET_CHAINS,
    TARGET_COORD_SPEC,
    TARGET_NAMES,
    correspondences,
)
from robokit.utils.hand_coord_utils import HandCoordinateSpec


_ONLINE_LINKS = {
    "wrist": "base",
    "thumb_mcp": "thumb_proximal",
    "thumb_tip": "thumb_tip",
    "index_mcp": "index_proximal",
    "index_tip": "index_tip",
    "middle_mcp": "middle_proximal",
    "middle_tip": "middle_tip",
    "ring_mcp": "ring_proximal",
    "ring_tip": "ring_tip",
    "pinky_mcp": "pinky_proximal",
    "pinky_tip": "pinky_tip",
}
_OFFLINE_LINKS = {
    "wrist": "hand_base_link",
    "thumb_mcp": "thumb_intermediate",
    "thumb_tip": "thumb_tip",
    "index_pip": "index_intermediate",
    "index_tip": "index_tip",
    "middle_pip": "middle_intermediate",
    "middle_tip": "middle_tip",
    "ring_pip": "ring_intermediate",
    "ring_tip": "ring_tip",
    "pinky_pip": "pinky_intermediate",
    "pinky_tip": "pinky_tip",
}
_VECTOR_PAIRS = FIVE_ROOT_TIP_PAIRS + FIVE_TIP_PAIRS
_DIRECTION_PAIRS = (
    ("thumb_mcp", "thumb_tip"),
    ("index_mcp", "index_tip"),
    ("middle_mcp", "middle_tip"),
    ("ring_mcp", "ring_tip"),
    ("pinky_mcp", "pinky_tip"),
)


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-x", four_fingers_up_axis="-y"),
)
online = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_ONLINE_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
        **correspondences(_ONLINE_LINKS, FIVE_TIP_PAIRS, 8.0 / 15.0),
    },
    direction_weights=correspondences(_ONLINE_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_ONLINE_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.15,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.023,
    vector_soft_gate_full_distance=0.00575,
    output_ema_alpha=0.45,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.03,
    self_collision_weight=10.0,
    self_collision_margin=0.002,
    self_collision_max_active_pairs=32,
)
offline = HandRetargetingOfflineConfig(target_scale=1.15, global_position_weight=26.0)


spec_left = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_ONLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-x", four_fingers_up_axis="-y"),
)
online_left = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_ONLINE_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
        **correspondences(_ONLINE_LINKS, FIVE_TIP_PAIRS, 0.2),
    },
    direction_weights=correspondences(_ONLINE_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_ONLINE_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.15,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.046,
    vector_soft_gate_full_distance=0.0115,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["offline", "online", "online_left", "spec", "spec_left"]
