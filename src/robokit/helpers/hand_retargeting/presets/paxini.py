"""Paxini DexH13 retargeting presets."""

from robokit.helpers.hand_retargeting.config import (
    HandRetargetingOfflineConfig,
    HandRetargetingOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FOUR_FINGERTIPS,
    FOUR_ROOT_TIP_PAIRS,
    FOUR_TIP_PAIRS,
    TARGET_CHAINS,
    TARGET_COORD_SPEC,
    TARGET_NAMES,
    correspondences,
)
from robokit.utils.hand_coord_utils import HandCoordinateSpec


_LINKS = {
    "wrist": "right_palm_link",
    "thumb_cmc": "right_thumb_link_1",
    "thumb_mcp": "right_thumb_link_2",
    "thumb_ip": "right_thumb_link_3",
    "thumb_tip": "thumb_tip",
    "index_mcp": "right_index_link_1",
    "index_pip": "right_index_link_2",
    "index_dip": "right_index_link_3",
    "index_tip": "index_tip",
    "middle_mcp": "right_middle_link_1",
    "middle_pip": "right_middle_link_2",
    "middle_dip": "right_middle_link_3",
    "middle_tip": "middle_tip",
    "ring_mcp": "right_ring_link_1",
    "ring_pip": "right_ring_link_2",
    "ring_dip": "right_ring_link_3",
    "ring_tip": "ring_tip",
}
_LINKS_LEFT = {name: link.replace("right_", "left_") for name, link in _LINKS.items()}
_OFFLINE_LINKS = {
    "wrist": "right_palm_link",
    "thumb_mcp": "right_thumb_link_1",
    "thumb_tip": "thumb_tip",
    "index_pip": "right_index_link_1",
    "index_tip": "index_tip",
    "middle_pip": "right_middle_link_1",
    "middle_tip": "middle_tip",
    "ring_pip": "right_ring_link_1",
    "ring_tip": "ring_tip",
}
_VECTOR_PAIRS = FOUR_ROOT_TIP_PAIRS + FOUR_TIP_PAIRS
_PINCH_PAIRS = tuple(("thumb_tip", tip) for tip in FOUR_FINGERTIPS[1:])
_DIRECTION_PAIRS = tuple(
    (chain[i], chain[i + 1])
    for chain in (
        ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"),
        ("index_mcp", "index_pip", "index_dip", "index_tip"),
        ("middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
        ("ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
    )
    for i in range(len(chain) - 1)
)


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FOUR_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="z", four_fingers_up_axis="x"),
)
online = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS, FOUR_ROOT_TIP_PAIRS, 0.1),
        **correspondences(_LINKS, FOUR_TIP_PAIRS, 0.2),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, _PINCH_PAIRS)),
    vector_target_scale=1.36,
    vector_huber_delta=0.01,
    vector_soft_gate_start_distance=0.04352,
    vector_soft_gate_full_distance=0.01088,
    output_ema_alpha=0.8,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.02,
)
offline = HandRetargetingOfflineConfig(target_scale=1.36, global_position_weight=24.0)


spec_left = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS_LEFT,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FOUR_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="z", four_fingers_up_axis="x"),
)
online_left = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS_LEFT, FOUR_ROOT_TIP_PAIRS, 0.1),
        **correspondences(_LINKS_LEFT, FOUR_TIP_PAIRS, 0.3),
    },
    direction_weights=correspondences(_LINKS_LEFT, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS_LEFT, _PINCH_PAIRS)),
    vector_target_scale=1.7,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.068,
    vector_soft_gate_full_distance=0.017,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["offline", "online", "online_left", "spec", "spec_left"]
