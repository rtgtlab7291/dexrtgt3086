"""xArm7 + MANO hand assembly assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm7_mano/**",
        "robots/robot_description/arms/xarm7/**",
        "robots/robot_description/end_effectors/mano_hand/**",
        "robots/collision_spheres/xarm_mano/**",
        "robots/contact_avoid_points/xarm_mano/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_mano/xarm7_mano_right_hand.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm_mano/collision_spheres.yaml"
COLLISION_SPHERE_1_2_PATH: Path = _DIR / "robots/collision_spheres/xarm_mano/collision_spheres_1.2.yaml"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/xarm_mano/contact_avoid_points.json"
