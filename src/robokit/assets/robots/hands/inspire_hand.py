"""Inspire Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/inspire_hand/**",
        "robots/collision_spheres/inspire_hand_right/**",
        "robots/contact_points/inspire_hand_right/**",
        "robots/contact_avoid_points/inspire_hand_right/**",
        "robots/self_collision/inspire_hand_right/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/inspire_hand/inspire_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/inspire_hand/inspire_hand_left.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/inspire_hand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/inspire_hand_right/contact_points.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/inspire_hand_right/contact_avoid_points.json"
SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/inspire_hand_right/ignore.capsule.yml"
URDF_NO_MIMIC_PATH: Path = _DIR / "robots/robot_description/end_effectors/inspire_hand/inspire_hand_right_nomimic.urdf"
FREE_URDF_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/end_effectors/inspire_hand/inspire_hand_free_right_nomimic.urdf"
)
