"""GR3 calibrated preset with a correspondence edge that transfers human wrist-to-wrist motion."""

import dataclasses

from robokit.helpers.humanoid_retarget.config import CorrespondenceEdge
from robokit.helpers.humanoid_retarget.presets.gr3_cali import gr3_cali


gr3_claps = dataclasses.replace(
    gr3_cali,
    correspondence_edges=[
        CorrespondenceEdge(
            origin_link="left_end_effector_link",
            task_link="right_end_effector_link",
            origin_joint="left_wrist",
            task_joint="right_wrist",
            weight=120.0,
        ),
    ],
)
