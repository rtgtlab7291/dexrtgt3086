"""xArm7 + XHand assembly assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm7_xhand/**",
        "robots/robot_description/arms/xarm7/**",
        "robots/collision_spheres/xarm_xhand/**",
        "robots/contact_points/xarm_xhand/**",
        "robots/contact_avoid_points/xarm_xhand/**",
        "robots/self_collision/xarm_xhand/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_xhand/xarm_xhand_righthand.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_xhand/xarm7_xhand_right_hand_glb.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm_xhand/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/xarm_xhand/contact_points.json"
CONTACT_POINTS_PINCH_V2_PATH: Path = _DIR / "robots/contact_points/xarm_xhand/contact_points_pinch_v2.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/xarm_xhand/contact_avoid_points.json"
SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_xhand/ignore.capsule.yml"
