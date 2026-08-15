"""Shadow Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/shadow_hand/**",
        "robots/collision_spheres/shadow_hand_right/**",
        "robots/collision_spheres/shadow_hand_mjcf_right/**",
        "robots/robot_description/end_effectors/shadow_hand_mjcf/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand/shadow_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand/shadow_hand_left.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/shadow_hand_right/collision_spheres.yaml"
MJCF_PATH: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand_mjcf/shadow_hand_wrist_free.xml"
MJCF_CONTACT_POINTS_PATH: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand_mjcf/contact_points.json"
MJCF_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/shadow_hand_mjcf_right/collision_spheres.yaml"
MJCF_SAPIEN_PATH: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand_mjcf/shadow_hand_sapien.xml"
MJCF_FREE_SAPIEN_PATH: Path = (
    _DIR / "robots/robot_description/end_effectors/shadow_hand_mjcf/shadow_hand_free_sapien.xml"
)
FREE_URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand/shadow_hand_free_right.urdf"
