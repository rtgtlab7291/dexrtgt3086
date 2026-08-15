"""Barrett Hand online retargeting preset."""

from robokit.helpers.hand_retargeting.config import HandRetargetingOnlineConfig, HandSpec
from robokit.helpers.hand_retargeting.presets._common import (
    TARGET_CHAINS,
    TARGET_COORD_SPEC,
    TARGET_NAMES,
    THREE_FINGERTIPS,
    THREE_PINCH_PAIRS,
    THREE_ROOT_TIP_PAIRS,
    correspondences,
)


_LINKS = {
    "wrist": "base_link",
    "thumb_tip": "finger_3_dist_link",
    "index_mcp": "finger_2_med_link",
    "index_tip": "finger_2_dist_link",
    "middle_mcp": "finger_1_med_liink",
    "middle_tip": "finger_1_dist_link",
}
_VECTOR_PAIRS = THREE_ROOT_TIP_PAIRS + THREE_PINCH_PAIRS
_DIRECTION_PAIRS = (("index_mcp", "index_tip"), ("middle_mcp", "middle_tip"))


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=THREE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=TARGET_COORD_SPEC,
)
online = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS, THREE_ROOT_TIP_PAIRS, 0.2),
        **correspondences(_LINKS, THREE_PINCH_PAIRS, 0.6),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, THREE_PINCH_PAIRS)),
    vector_target_scale=1.5,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.06,
    vector_soft_gate_full_distance=0.015,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["online", "spec"]
