"""Unitree G1 + XHand assembly assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/collision_spheres/g1_xhand/**",
        "robots/contact_points/g1_xhand/**",
    ]
)
COLLISION_SPHERE_FIXED_PATH: Path = _DIR / "robots/collision_spheres/g1_xhand/collision_spheres_fixed.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/g1_xhand/contact_points.json"
