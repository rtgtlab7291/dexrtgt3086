"""Shadow Hand retargeting presets."""

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
    "wrist": "palm",
    "thumb_cmc": "thproximal",
    "thumb_mcp": "thhub",
    "thumb_ip": "thmiddle",
    "thumb_tip": "thtip",
    "index_mcp": "ffproximal",
    "index_pip": "ffmiddle",
    "index_dip": "ffdistal",
    "index_tip": "fftip",
    "middle_mcp": "mfproximal",
    "middle_pip": "mfmiddle",
    "middle_dip": "mfdistal",
    "middle_tip": "mftip",
    "ring_mcp": "rfproximal",
    "ring_pip": "rfmiddle",
    "ring_dip": "rfdistal",
    "ring_tip": "rftip",
    "pinky_mcp": "lfproximal",
    "pinky_pip": "lfmiddle",
    "pinky_dip": "lfdistal",
    "pinky_tip": "lftip",
}
_OFFLINE_LINKS = {
    "wrist": "palm",
    "thumb_mcp": "thmiddle",
    "thumb_tip": "thtip",
    "index_pip": "ffmiddle",
    "index_tip": "fftip",
    "middle_pip": "mfmiddle",
    "middle_tip": "mftip",
    "ring_pip": "rfmiddle",
    "ring_tip": "rftip",
    "pinky_pip": "lfmiddle",
    "pinky_tip": "lftip",
}
_VECTOR_PAIRS = FIVE_ROOT_TIP_PAIRS + FIVE_TIP_PAIRS
_DIRECTION_PAIRS = (
    ("thumb_cmc", "thumb_mcp"),
    ("thumb_ip", "thumb_tip"),
    ("index_mcp", "index_pip"),
    ("index_pip", "index_dip"),
    ("index_dip", "index_tip"),
    ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_dip"),
    ("middle_dip", "middle_tip"),
    ("ring_mcp", "ring_pip"),
    ("ring_pip", "ring_dip"),
    ("ring_dip", "ring_tip"),
    ("pinky_mcp", "pinky_pip"),
    ("pinky_pip", "pinky_dip"),
    ("pinky_dip", "pinky_tip"),
)


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-y", four_fingers_up_axis="-z"),
)
online = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
        **correspondences(_LINKS, FIVE_TIP_PAIRS, 2.0 / 15.0),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.17,
    vector_huber_delta=0.01,
    vector_soft_gate_start_distance=0.1404,
    vector_soft_gate_full_distance=0.0351,
    output_ema_alpha=0.9,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.01,
    regularization_weight=0.0065,
)
offline = HandRetargetingOfflineConfig(target_scale=1.17, global_position_weight=24.0)


spec_left = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-y", four_fingers_up_axis="-z"),
)
online_left = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
        **correspondences(_LINKS, FIVE_TIP_PAIRS, 0.2),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, FIVE_PINCH_PAIRS)),
    vector_target_scale=1.2,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.048,
    vector_soft_gate_full_distance=0.012,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["offline", "online", "online_left", "spec", "spec_left"]
