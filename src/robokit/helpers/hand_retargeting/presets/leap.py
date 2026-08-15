"""LEAP Hand retargeting presets."""

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
    "wrist": "base",
    "thumb_cmc": "thumb_pip",
    "thumb_mcp": "thumb_dip",
    "thumb_ip": "thumb_fingertip",
    "thumb_tip": "thumb_tip_head",
    "index_mcp": "pip",
    "index_pip": "dip",
    "index_dip": "fingertip",
    "index_tip": "index_tip_head",
    "middle_mcp": "pip_2",
    "middle_pip": "dip_2",
    "middle_dip": "fingertip_2",
    "middle_tip": "middle_tip_head",
    "ring_mcp": "pip_3",
    "ring_pip": "dip_3",
    "ring_dip": "fingertip_3",
    "ring_tip": "ring_tip_head",
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
    )
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
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-z", four_fingers_up_axis="y"),
)
online = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS, FOUR_ROOT_TIP_PAIRS, 0.1),
        **correspondences(_LINKS, FOUR_TIP_PAIRS, 0.15),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, _PINCH_PAIRS)),
    vector_target_scale=1.28,
    vector_huber_delta=0.01,
    vector_soft_gate_start_distance=0.04096,
    vector_soft_gate_full_distance=0.01024,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.02,
)
offline = HandRetargetingOfflineConfig(target_scale=1.28, global_position_weight=26.0)


spec_left = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FOUR_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-z", four_fingers_up_axis="y"),
)
online_left = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS, FOUR_ROOT_TIP_PAIRS, 0.1),
        **correspondences(_LINKS, FOUR_TIP_PAIRS, 0.3),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, _PINCH_PAIRS)),
    vector_target_scale=1.6,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.064,
    vector_soft_gate_full_distance=0.016,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["offline", "online", "online_left", "spec", "spec_left"]
