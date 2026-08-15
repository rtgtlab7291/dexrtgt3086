"""XHand retargeting presets."""

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
    "wrist": "right_hand_link",
    "thumb_mcp": "right_hand_thumb_rota_link1",
    "thumb_ip": "right_hand_thumb_rota_link2",
    "thumb_tip": "right_hand_thumb_rota_tip",
    "index_mcp": "right_hand_index_rota_link1",
    "index_pip": "right_hand_index_rota_link2",
    "index_tip": "right_hand_index_rota_tip",
    "middle_mcp": "right_hand_mid_link1",
    "middle_pip": "right_hand_mid_link2",
    "middle_tip": "right_hand_mid_tip",
    "ring_mcp": "right_hand_ring_link1",
    "ring_pip": "right_hand_ring_link2",
    "ring_tip": "right_hand_ring_tip",
    "pinky_mcp": "right_hand_pinky_link1",
    "pinky_pip": "right_hand_pinky_link2",
    "pinky_tip": "right_hand_pinky_tip",
}
_LINKS_LEFT = {name: link.replace("right_hand_", "left_hand_") for name, link in _LINKS.items()}
_OFFLINE_LINKS = {
    "wrist": "right_hand_link",
    "thumb_mcp": "right_hand_thumb_rota_link1",
    "thumb_tip": "right_hand_thumb_rota_tip",
    "index_pip": "right_hand_index_rota_link1",
    "index_tip": "right_hand_index_rota_tip",
    "middle_pip": "right_hand_mid_link1",
    "middle_tip": "right_hand_mid_tip",
    "ring_pip": "right_hand_ring_link1",
    "ring_tip": "right_hand_ring_tip",
    "pinky_pip": "right_hand_pinky_link1",
    "pinky_tip": "right_hand_pinky_tip",
}
_VECTOR_LINKS = {**_LINKS, **_OFFLINE_LINKS}
_VECTOR_LINKS_LEFT = {name: link.replace("right_hand_", "left_hand_") for name, link in _VECTOR_LINKS.items()}
_VECTOR_PAIRS = FIVE_ROOT_TIP_PAIRS + FIVE_ROOT_JOINT_PAIRS + FIVE_TIP_PAIRS
_DIRECTION_PAIRS = (
    ("thumb_mcp", "thumb_ip"),
    ("thumb_ip", "thumb_tip"),
    ("index_mcp", "index_pip"),
    ("index_pip", "index_tip"),
    ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_tip"),
    ("ring_mcp", "ring_pip"),
    ("ring_pip", "ring_tip"),
    ("pinky_mcp", "pinky_pip"),
    ("pinky_pip", "pinky_tip"),
)


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="-z"),
)
online = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_VECTOR_LINKS, FIVE_ROOT_TIP_PAIRS + FIVE_ROOT_JOINT_PAIRS, 0.05),
        **correspondences(_VECTOR_LINKS, FIVE_TIP_PAIRS, 0.1),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_VECTOR_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.12,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.04032,
    vector_soft_gate_full_distance=0.01008,
    output_ema_alpha=0.4,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.02,
    regularization_weight=0.01,
    self_collision_weight=10.0,
    self_collision_margin=0.002,
    self_collision_max_active_pairs=32,
)
offline = HandRetargetingOfflineConfig(target_scale=1.12, global_position_weight=48.0)


spec_left = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS_LEFT,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="-z"),
)
online_left = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_VECTOR_LINKS_LEFT, FIVE_ROOT_TIP_PAIRS + FIVE_ROOT_JOINT_PAIRS, 0.05),
        **correspondences(_VECTOR_LINKS_LEFT, FIVE_TIP_PAIRS, 0.15),
    },
    direction_weights=correspondences(_LINKS_LEFT, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_VECTOR_LINKS_LEFT, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.4,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.056,
    vector_soft_gate_full_distance=0.014,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["offline", "online", "online_left", "spec", "spec_left"]
