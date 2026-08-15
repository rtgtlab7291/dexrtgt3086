"""Config builders for before/after calibration comparisons.

The genuinely *uncalibrated* mapping is per-joint scale 1.0 with zero
position_offset - i.e. SMPL-X joints mapped straight onto the robot with no
size or offset correction. `rotation_offset` is kept because it is the fixed
SMPL-X→robot frame alignment (Y-up→Z-up), set manually and NOT learned by
calibration; zeroing it would just rotate every target into the wrong frame.
"""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retarget import HumanoidRetargetingOnlineConfig
from robokit.helpers.humanoid_retarget.presets.g1 import g1


def uncalibrated(base: HumanoidRetargetingOnlineConfig = g1) -> HumanoidRetargetingOnlineConfig:
    """Return `base` reset to the non-calibrated mapping (scale 1.0, zero offset)."""
    link_mapping = {
        rl: dataclasses.replace(m, position_offset=np.zeros(3, dtype=np.float32)) for rl, m in base.link_mapping.items()
    }
    scale_table = {j: 1.0 for j in base.scale_table}
    return dataclasses.replace(base, scale_table=scale_table, link_mapping=link_mapping)
