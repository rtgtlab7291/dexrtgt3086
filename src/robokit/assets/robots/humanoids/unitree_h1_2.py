"""Unitree H1-2 humanoid assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["robots/robot_description/humanoids/unitree_h1_2/**", "robots/collision_spheres/unitree_h1_2/**"])
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/unitree_h1_2/h1_2_handless.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/unitree_h1_2/full_body.yaml"
