"""Fourier GR-3 calibrated retargeting preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retarget.presets.gr3 import gr3


_OFFSETS = {
    "base_link": np.array([0.0004, 0.0001, 0.0122], dtype=np.float32),
    "left_thigh_roll_link": np.array([0.0168, 0.0544, 0.0048], dtype=np.float32),
    "left_shank_pitch_link": np.array([0.0133, 0.0199, 0.0032], dtype=np.float32),
    "left_foot_roll_link": np.array([0.0152, -0.0053, 0.0396], dtype=np.float32),
    "right_thigh_roll_link": np.array([-0.0004, -0.0533, 0.0206], dtype=np.float32),
    "right_shank_pitch_link": np.array([0.022, -0.025, 0.0018], dtype=np.float32),
    "right_foot_roll_link": np.array([0.0129, -0.0264, 0.0249], dtype=np.float32),
    "torso_link": np.array([0.0031, 0.0073, 0.0502], dtype=np.float32),
    "left_upper_arm_roll_link": np.array([-0.0044, -0.0005, -0.0042], dtype=np.float32),
    "left_lower_arm_pitch_link": np.array([0.0016, 0.0592, 0.0045], dtype=np.float32),
    "left_end_effector_link": np.array([-0.0056, -0.0015, -0.0197], dtype=np.float32),
    "right_upper_arm_roll_link": np.array([-0.0011, -0.001, 0.002], dtype=np.float32),
    "right_lower_arm_pitch_link": np.array([-0.0037, -0.0394, 0.0318], dtype=np.float32),
    "right_end_effector_link": np.array([0.0094, -0.0103, 0.015], dtype=np.float32),
}


gr3_cali = dataclasses.replace(
    gr3,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.976,
        "spine3": 1.1472,
        "left_hip": 0.8356,
        "right_hip": 0.8354,
        "left_knee": 0.9204,
        "right_knee": 0.9198,
        "left_foot": 0.9515,
        "right_foot": 0.9284,
        "left_shoulder": 1.0148,
        "right_shoulder": 1.0162,
        "left_elbow": 0.994,
        "right_elbow": 1.0049,
        "left_wrist": 0.9896,
        "right_wrist": 1.0189,
    },
    link_mapping={
        link: dataclasses.replace(gr3.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
