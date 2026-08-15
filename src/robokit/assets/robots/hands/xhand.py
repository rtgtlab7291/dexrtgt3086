"""XHand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/xhand/**",
        "robots/collision_spheres/xhand/**",
        "robots/contact_points/xhand/**",
        "robots/contact_avoid_points/xhand/**",
        "robots/self_collision/xhand/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/xhand/xhand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/xhand/xhand_left.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/end_effectors/xhand/xhand_right_glb.urdf"
FREE_URDF_GLB_PATH: Path = _DIR / "robots/robot_description/end_effectors/xhand/xhand_right_glb_free.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xhand/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/xhand/contact_points.json"
CONTACT_POINTS_TIP_PATH: Path = _DIR / "robots/contact_points/xhand/contact_points_tip.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/xhand/contact_avoid_points.json"
SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xhand/ignore.capsule.yml"
