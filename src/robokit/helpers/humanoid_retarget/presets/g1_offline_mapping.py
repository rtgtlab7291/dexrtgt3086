"""G1 offline mapping with lower pelvis weights, paired with the `g1_offline` optimizer preset."""

import dataclasses

from robokit.helpers.humanoid_retarget.presets.g1_cali import g1_cali


g1_offline_mapping = dataclasses.replace(
    g1_cali,
    link_mapping={
        **g1_cali.link_mapping,
        "pelvis": dataclasses.replace(g1_cali.link_mapping["pelvis"], position_weight=100, orientation_weight=20),
    },
)


__all__ = ["g1_offline_mapping"]
