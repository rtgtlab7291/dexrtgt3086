"""G1 calibrated preset with a correspondence edge that transfers human wrist-to-wrist motion."""

import dataclasses

from robokit.helpers.humanoid_retarget.config import CorrespondenceEdge
from robokit.helpers.humanoid_retarget.presets.g1_cali import g1_cali


g1_claps = dataclasses.replace(
    g1_cali,
    correspondence_edges=[
        CorrespondenceEdge(
            origin_link="left_wrist_yaw_link",
            task_link="right_wrist_yaw_link",
            origin_joint="left_wrist",
            task_joint="right_wrist",
            weight=120.0,
        ),
    ],
)


__all__ = ["g1_claps"]
