"""MANO hand (end-effector URDF) assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/mano_hand/**",
        "robots/collision_spheres/mano_hand/**",
        "robots/contact_avoid_points/mano_hand/**",
        "robots/self_collision/mano_hand/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/mano_hand/mano_hand.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/mano_hand/collision_spheres.yaml"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/mano_hand/contact_avoid_points.json"
SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/mano_hand/ignore.capsule.yml"
FREE_URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/mano_hand/mano_hand_free.urdf"
