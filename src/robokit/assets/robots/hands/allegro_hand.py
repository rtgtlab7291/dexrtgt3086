"""Allegro Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/allegro_hand/**",
        "robots/collision_spheres/allegro_hand_right/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/allegro_hand/allegro_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/allegro_hand/allegro_hand_left.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/end_effectors/allegro_hand/allegro_hand_right_glb.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/allegro_hand_right/collision_spheres.yaml"
URDF_GLB_FREE_PATH: Path = _DIR / "robots/robot_description/end_effectors/allegro_hand/allegro_hand_right_glb_free.urdf"
