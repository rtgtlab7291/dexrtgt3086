"""Unitree G1 + Inspire hand assembly assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/assembly/g1_inspire/**",
        "robots/robot_description/end_effectors/inspire_hand/**",  # assembly URDFs reference these meshes
        "robots/collision_spheres/g1_inspire/**",
        "robots/contact_points/g1_inspire/**",
        "robots/contact_avoid_points/g1_inspire/**",
        "robots/self_collision/g1_inspire/**",
        "robots/robot_description/humanoids/g1_inspire/**",
        "robots/robot_description/humanoids/g1/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/g1_inspire/g1_inspire.urdf"
URDF_UPPER_HEAD_PATH: Path = _DIR / "robots/robot_description/assembly/g1_inspire/g1_inspire_upper_head.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/g1_inspire/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/g1_inspire/contact_points.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/g1_inspire/contact_avoid_points.json"
SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/g1_inspire/ignore.capsule.yml"
URDF_NO_MIMIC_PATH: Path = _DIR / "robots/robot_description/assembly/g1_inspire/g1_inspire_no_mimic.urdf"
URDF_UPPER_HEAD_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/assembly/g1_inspire/g1_inspire_upper_head_no_mimic.urdf"
)
HUMANOID_29DOF_HEAD_URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/g1_inspire/urdf/g1_29dof_head.urdf"
HUMANOID_UPPER_HEAD_URDF_PATH: Path = (
    _DIR / "robots/robot_description/humanoids/g1_inspire/urdf/g1_inspire_upper_head.urdf"
)
HUMANOID_R_HAND_URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/g1_inspire/urdf/r_hand_inspire.urdf"
