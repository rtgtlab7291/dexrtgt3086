"""Unitree G1 + MANO hand assembly assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/assembly/g1_mano/**",
        "robots/collision_spheres/g1_mano/**",
        "robots/contact_points/g1_mano/**",
        "robots/contact_avoid_points/g1_mano/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/g1_mano/g1_mano.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/g1_mano/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/g1_mano/contact_points.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/g1_mano/contact_avoid_points.json"
