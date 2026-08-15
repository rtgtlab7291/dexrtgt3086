"""Allegro Hand retargeting presets."""

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
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.utils.hand_coord_utils import HandCoordinateSpec


_LINKS = {
    "wrist": "wrist",
    "thumb_cmc": "link_13.0",
    "thumb_mcp": "link_14.0",
    "thumb_ip": "link_15.0",
    "thumb_tip": "link_15.0_tip",
    "index_mcp": "link_1.0",
    "index_pip": "link_2.0",
    "index_dip": "link_3.0",
    "index_tip": "link_3.0_tip",
    "middle_mcp": "link_5.0",
    "middle_pip": "link_6.0",
    "middle_dip": "link_7.0",
    "middle_tip": "link_7.0_tip",
    "ring_mcp": "link_9.0",
    "ring_pip": "link_10.0",
    "ring_dip": "link_11.0",
    "ring_tip": "link_11.0_tip",
}
_LINKS_LEFT = {
    **{name: link for name, link in _LINKS.items() if name == "wrist" or name.startswith("thumb")},
    "index_mcp": "link_9.0",
    "index_pip": "link_10.0",
    "index_dip": "link_11.0",
    "index_tip": "link_11.0_tip",
    "middle_mcp": "link_5.0",
    "middle_pip": "link_6.0",
    "middle_dip": "link_7.0",
    "middle_tip": "link_7.0_tip",
    "ring_mcp": "link_1.0",
    "ring_pip": "link_2.0",
    "ring_dip": "link_3.0",
    "ring_tip": "link_3.0_tip",
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
        **correspondences(_LINKS, FOUR_TIP_PAIRS, 0.4),
    },
    direction_weights=correspondences(_LINKS, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS, _PINCH_PAIRS)),
    vector_target_scale=1.6,
    vector_huber_delta=0.01,
    vector_soft_gate_start_distance=0.032,
    vector_soft_gate_full_distance=0.008,
    output_ema_alpha=0.7,
    sampling_distance=0.0,
    solver=MultiSeedSolverConfig(
        stages=[StageConfig(num_seeds=1, iters=10, lm_lambda=1e-2)],
        cuda_graph_mode="full",
    ),
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.02,
    regularization_weight=0.01,
)
offline = HandRetargetingOfflineConfig(target_scale=1.6, global_position_weight=26.0)


spec_left = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS_LEFT,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FOUR_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-z", four_fingers_up_axis="y"),
)
online_left = HandRetargetingOnlineConfig(
    vector_weights={
        **correspondences(_LINKS_LEFT, FOUR_ROOT_TIP_PAIRS, 0.1),
        **correspondences(_LINKS_LEFT, FOUR_TIP_PAIRS, 0.3),
    },
    direction_weights=correspondences(_LINKS_LEFT, _DIRECTION_PAIRS),
    pinch_correspondences=tuple(correspondences(_LINKS_LEFT, _PINCH_PAIRS)),
    vector_target_scale=1.6,
    vector_huber_delta=0.02,
    vector_soft_gate_start_distance=0.064,
    vector_soft_gate_full_distance=0.016,
    pinch_threshold=0.03,
    pinch_release_threshold=0.05,
    q_smoothness_weight=0.05,
)


__all__ = ["offline", "online", "online_left", "spec", "spec_left"]
