"""Fourier GR3 humanoid assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["robots/robot_description/humanoids/fourier_gr3/**", "robots/collision_spheres/fourier_gr3/**"])
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/fourier_gr3/basic_urdf/gr3v2_1_1_dummy_hand.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/fourier_gr3/full_body.yaml"
