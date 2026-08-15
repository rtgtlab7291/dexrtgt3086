"""Unitree H1-2 calibrated retargeting preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retarget.presets.h1_2 import h1_2


_OFFSETS = {
    "pelvis": np.array([-0.0, 0.0, 0.086], dtype=np.float32),
    "left_hip_roll_link": np.array([0.0075, 0.0774, -0.0058], dtype=np.float32),
    "left_knee_link": np.array([0.0089, 0.0322, 0.0032], dtype=np.float32),
    "left_ankle_roll_link": np.array([-0.0073, 0.0133, 0.0835], dtype=np.float32),
    "right_hip_roll_link": np.array([0.005, -0.0763, 0.0082], dtype=np.float32),
    "right_knee_link": np.array([0.023, -0.0349, -0.0017], dtype=np.float32),
    "right_ankle_roll_link": np.array([-0.0049, -0.0257, 0.0488], dtype=np.float32),
    "torso_link": np.array([0.0028, 0.0024, -0.0942], dtype=np.float32),
    "left_shoulder_roll_link": np.array([0.0055, 0.0199, -0.0011], dtype=np.float32),
    "left_elbow_link": np.array([0.0285, 0.1077, 0.065], dtype=np.float32),
    "left_wrist_yaw_link": np.array([-0.0324, 0.0595, 0.064], dtype=np.float32),
    "right_shoulder_roll_link": np.array([0.0192, -0.0214, -0.0051], dtype=np.float32),
    "right_elbow_link": np.array([0.0131, -0.0815, 0.0704], dtype=np.float32),
    "right_wrist_yaw_link": np.array([-0.0493, -0.0604, 0.0896], dtype=np.float32),
}


h1_2_cali = dataclasses.replace(
    h1_2,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.9992,
        "spine3": 0.7227,
        "left_hip": 0.9873,
        "right_hip": 0.988,
        "left_knee": 1.0113,
        "right_knee": 1.011,
        "left_foot": 1.0453,
        "right_foot": 1.0048,
        "left_shoulder": 1.1523,
        "right_shoulder": 1.1532,
        "left_elbow": 1.2174,
        "right_elbow": 1.2195,
        "left_wrist": 1.1885,
        "right_wrist": 1.2007,
    },
    link_mapping={
        link: dataclasses.replace(h1_2.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
