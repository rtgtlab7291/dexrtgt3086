"""Ability Hand retargeting presets."""

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
    "wrist": "base_link",
    "thumb_cmc": "thumb_L1",
    "thumb_mcp": "thumb_L2",
    "thumb_tip": "thumb_tip",
    "index_mcp": "index_L1",
    "index_tip": "index_tip",
    "middle_mcp": "middle_L1",
    "middle_tip": "middle_tip",
    "ring_mcp": "ring_L1",
    "ring_tip": "ring_tip",
    "pinky_mcp": "pinky_L1",
    "pinky_tip": "pinky_tip",
}
_OFFLINE_LINKS = {
    "wrist": "base",
    "thumb_mcp": "thumb_L2",
    "thumb_tip": "thumb_tip",
    "index_pip": "index_L2",
    "index_tip": "index_tip",
    "middle_pip": "middle_L2",
    "middle_tip": "middle_tip",
    "ring_pip": "ring_L2",
    "ring_tip": "ring_tip",
    "pinky_pip": "pinky_L2",
    "pinky_tip": "pinky_tip",
}
_VECTOR_PAIRS = FIVE_ROOT_TIP_PAIRS + FIVE_TIP_PAIRS
_DIRECTION_PAIRS = (
    ("thumb_cmc", "thumb_mcp"),
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
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="z"),
)
online = HandRetargetingOnlineConfig(
    vector_weights=correspondences(_ONLINE_LINKS, _VECTOR_PAIRS, 1.0 / 15.0),
    direction_weights=correspondences(_ONLINE_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_ONLINE_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=0.99,
    vector_huber_delta=0.01,
    vector_soft_gate_start_distance=0.0198,
    vector_soft_gate_full_distance=0.00495,
    output_ema_alpha=0.7,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.01,
)
offline = HandRetargetingOfflineConfig(target_scale=0.99, global_position_weight=24.0)


spec_left = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_ONLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="z"),
)
online_left = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_ONLINE_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
        **correspondences(_ONLINE_LINKS, FIVE_TIP_PAIRS, 0.2),
    },
    direction_weights=correspondences(_ONLINE_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_ONLINE_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.1,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.044,
    vector_soft_gate_full_distance=0.011,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["offline", "online", "online_left", "spec", "spec_left"]
