"""xArm7 with Ability Hand offline retargeting presets."""

from typing import Dict

import numpy as np

from robokit.helpers.hand_retargeting.config import HandRetargetingOfflineConfig, HandSpec
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    TARGET_CHAINS,
    TARGET_COORD_SPEC,
    TARGET_NAMES,
)
from robokit.utils.hand_coord_utils import HandCoordinateSpec


_LINKS = {
    "wrist": "base",
    "thumb_tip": "thumb_tip",
    "index_tip": "index_tip",
    "middle_tip": "middle_tip",
    "ring_tip": "ring_tip",
    "pinky_tip": "pinky_tip",
}
_ARM_Q: Dict[str, float] = dict(
    zip(
        ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
        np.deg2rad((-1.8, 7.7, 1.8, 13.7, 180.0, 83.9, -0.2)).tolist(),
    )
)
_ROOT_COORD_SPEC = HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="-z")


spec = HandSpec(
    floating_base=False,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=_ROOT_COORD_SPEC,
    init_q_by_name=_ARM_Q,
)
offline = HandRetargetingOfflineConfig(target_scale=0.99, global_position_weight=24.0)


spec_no_mimic = HandSpec(
    floating_base=False,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    target_coord_spec=TARGET_COORD_SPEC,
    root_link_coord_spec=_ROOT_COORD_SPEC,
    init_q_by_name=_ARM_Q,
)
offline_no_mimic = HandRetargetingOfflineConfig(target_scale=0.99, global_position_weight=26.0)


__all__ = ["offline", "offline_no_mimic", "spec", "spec_no_mimic"]
