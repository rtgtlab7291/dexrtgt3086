"""G1 calibrated retargeting preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retarget.presets.g1 import g1


_OFFSETS = {
    "pelvis": np.array([0.0024, 0.0006, 0.0365], dtype=np.float32),
    "left_hip_roll_link": np.array([0.0088, 0.0485, -0.0126], dtype=np.float32),
    "left_knee_link": np.array([0.0099, 0.0177, 0.0068], dtype=np.float32),
    "left_toe_link": np.array([0.0696, 0.0201, 0.0257], dtype=np.float32),
    "right_hip_roll_link": np.array([0.0053, -0.0464, 0.0051], dtype=np.float32),
    "right_knee_link": np.array([0.0111, -0.0189, 0.0028], dtype=np.float32),
    "right_toe_link": np.array([0.0587, -0.0099, 0.0202], dtype=np.float32),
    "torso_link": np.array([0.0009, 0.0018, -0.0564], dtype=np.float32),
    "left_shoulder_yaw_link": np.array([0.0099, -0.0023, -0.0698], dtype=np.float32),
    "left_elbow_link": np.array([0.0028, 0.0428, 0.0394], dtype=np.float32),
    "left_wrist_yaw_link": np.array([-0.0027, 0.0392, 0.0448], dtype=np.float32),
    "right_shoulder_yaw_link": np.array([0.0106, -0.0003, -0.0731], dtype=np.float32),
    "right_elbow_link": np.array([-0.0058, -0.0305, 0.0278], dtype=np.float32),
    "right_wrist_yaw_link": np.array([-0.0104, -0.0382, 0.0422], dtype=np.float32),
}


g1_cali = dataclasses.replace(
    g1,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.7978,
        "spine3": 0.8342,
        "left_hip": 0.9859,
        "right_hip": 0.9863,
        "left_knee": 0.8787,
        "right_knee": 0.8782,
        "left_foot": 0.8374,
        "right_foot": 0.8304,
        "left_shoulder": 0.7766,
        "right_shoulder": 0.7777,
        "left_elbow": 0.7814,
        "right_elbow": 0.7784,
        "left_wrist": 0.7675,
        "right_wrist": 0.768,
    },
    link_mapping={
        link: dataclasses.replace(g1.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
