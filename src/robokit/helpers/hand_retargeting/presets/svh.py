"""Schunk SVH retargeting presets."""

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


_LINKS = {
    "wrist": "right_hand_base_link",
    "thumb_mcp": "right_hand_a",
    "thumb_tip": "thtip",
    "index_mcp": "right_hand_l",
    "index_pip": "right_hand_p",
    "index_tip": "fftip",
    "middle_mcp": "right_hand_k",
    "middle_pip": "right_hand_o",
    "middle_tip": "mftip",
    "ring_mcp": "right_hand_j",
    "ring_tip": "rftip",
    "pinky_mcp": "right_hand_i",
    "pinky_tip": "lftip",
}
_LINKS_LEFT = {name: link.replace("right_hand_", "left_hand_") for name, link in _LINKS.items()}
_OFFLINE_LINKS = {
    "wrist": "right_hand_base_link",
    "thumb_mcp": "right_hand_b",
    "thumb_tip": "right_hand_c",
    "index_pip": "right_hand_p",
    "index_tip": "right_hand_t",
    "middle_pip": "right_hand_o",
    "middle_tip": "right_hand_s",
    "ring_pip": "right_hand_n",
    "ring_tip": "right_hand_r",
    "pinky_pip": "right_hand_i",
    "pinky_tip": "right_hand_q",
}
_VECTOR_PAIRS = FIVE_ROOT_TIP_PAIRS + FIVE_TIP_PAIRS
_DIRECTION_PAIRS = (
    ("thumb_mcp", "thumb_tip"),
    ("index_mcp", "index_pip"),
    ("index_pip", "index_tip"),
    ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_tip"),
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
    vector_weights={
        **correspondences(_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
        **correspondences(_LINKS, FIVE_TIP_PAIRS, 2.0 / 15.0),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.2,
    vector_huber_delta=0.01,
    vector_soft_gate_start_distance=0.0384,
    vector_soft_gate_full_distance=0.0096,
    output_ema_alpha=0.8,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.01,
)
offline = HandRetargetingOfflineConfig(target_scale=1.2, global_position_weight=24.0)


spec_left = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS_LEFT,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="z"),
)
online_left = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS_LEFT, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
        **correspondences(_LINKS_LEFT, FIVE_TIP_PAIRS, 0.2),
    },
    direction_weights=correspondences(_LINKS_LEFT, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS_LEFT, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.2,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.048,
    vector_soft_gate_full_distance=0.012,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["offline", "online", "online_left", "spec", "spec_left"]
