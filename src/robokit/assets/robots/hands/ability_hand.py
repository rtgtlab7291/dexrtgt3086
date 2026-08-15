"""PSYONIC Ability Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/ability_hand/**",
        "robots/collision_spheres/ability_hand_right/**",
        "robots/contact_avoid_points/ability_hand_right/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_left.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_right_glb.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/ability_hand_right/collision_spheres.yaml"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/ability_hand_right/contact_avoid_points.json"
URDF_GLB_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_right_glb_no_mimic.urdf"
)
URDF_GLB_FREE_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_right_glb_free_no_mimic.urdf"
)
URDF_GLB_LEFT_FREE_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_left_glb_free_no_mimic.urdf"
)
