"""G1 calibrated preset with collision-sphere self-collision avoidance."""

import dataclasses

from robokit.helpers.humanoid_retarget.presets.g1_cali import g1_cali


g1_collide = dataclasses.replace(
    g1_cali,
    self_collision_weight=100.0,
    self_collision_margin=0.01,
)


__all__ = ["g1_collide"]
