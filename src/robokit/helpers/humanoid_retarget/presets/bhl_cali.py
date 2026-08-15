"""Berkeley Humanoid Lite calibrated retargeting preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retarget.presets.bhl import bhl


_OFFSETS = {
    "imu_2": np.array([0.0691, 0.002, 0.3337], dtype=np.float32),
    "leg_left_hip_roll": np.array([0.0261, -0.0294, 0.0315], dtype=np.float32),
    "leg_left_knee_pitch": np.array([-0.0057, 0.0948, 0.0008], dtype=np.float32),
    "leg_left_ankle_roll": np.array([-0.0642, -0.0428, -0.0405], dtype=np.float32),
    "leg_right_hip_roll": np.array([-0.0159, -0.0443, -0.0305], dtype=np.float32),
    "leg_right_knee_pitch": np.array([-0.002, -0.0953, -0.0041], dtype=np.float32),
    "leg_right_ankle_roll": np.array([0.0801, 0.0171, -0.0535], dtype=np.float32),
    "arm_left_elbow_pitch": np.array([-0.0504, -0.0564, -0.0341], dtype=np.float32),
    "arm_left_hand_link": np.array([0.0472, 0.0367, 0.1719], dtype=np.float32),
    "arm_right_elbow_pitch": np.array([0.0374, -0.0693, -0.0107], dtype=np.float32),
    "arm_right_hand_link": np.array([-0.04, 0.037, 0.1876], dtype=np.float32),
}


bhl_cali = dataclasses.replace(
    bhl,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.357,
        "left_hip": 0.5679,
        "right_hip": 0.5675,
        "left_knee": 0.4362,
        "right_knee": 0.4364,
        "left_foot": 0.3309,
        "right_foot": 0.2616,
        "left_elbow": 0.9389,
        "right_elbow": 0.9403,
        "left_wrist": 0.9302,
        "right_wrist": 0.9381,
    },
    link_mapping={
        link: dataclasses.replace(bhl.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
