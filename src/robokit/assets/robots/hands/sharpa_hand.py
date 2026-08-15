"""Sharpa Wave Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/sharpa_hand/**",
        "robots/collision_spheres/sharpa_hand_right/**",
        "robots/collision_spheres/sharpa_hand_left/**",
        "robots/contact_points/sharpa_hand_right/**",
        "robots/contact_points/sharpa_hand_left/**",
        "robots/self_collision/sharpa_hand_right/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/sharpa_hand/right_sharpa_wave.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/sharpa_hand/left_sharpa_wave.urdf"
COLLISION_SPHERE_RIGHT_PATH: Path = _DIR / "robots/collision_spheres/sharpa_hand_right/collision_spheres.yaml"
COLLISION_SPHERE_LEFT_PATH: Path = _DIR / "robots/collision_spheres/sharpa_hand_left/collision_spheres.yaml"
COLLISION_SPHERE_PATH = COLLISION_SPHERE_RIGHT_PATH  # alias, default side
CONTACT_POINTS_RIGHT_DIR: Path = _DIR / "robots/contact_points/sharpa_hand_right"
CONTACT_POINTS_LEFT_DIR: Path = _DIR / "robots/contact_points/sharpa_hand_left"
SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/sharpa_hand_right/ignore.sphere.yml"
FREE_URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/sharpa_hand/right_sharpa_wave_free.urdf"
