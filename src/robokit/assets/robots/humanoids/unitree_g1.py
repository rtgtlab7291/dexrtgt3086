"""Unitree G1 humanoid assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["robots/robot_description/humanoids/unitree_g1/**", "robots/collision_spheres/g1/**"])
_MODEL = _DIR / "robots/robot_description/humanoids/unitree_g1"
_SPHERES = _DIR / "robots/collision_spheres/g1"

URDF_PATH: Path = _MODEL / "g1_custom_collision_29dof.urdf"
URDF_29DOF_PATH: Path = _MODEL / "g1_29dof.urdf"
URDF_SPHEREHAND_PATH: Path = _MODEL / "g1_29dof_spherehand.urdf"

COLLISION_SPHERE_PATH: Path = _SPHERES / "collision_spheres.yaml"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _SPHERES / "full_body.yaml"
COLLISION_SPHERE_FULL_BODY_SPHEREHAND_PATH: Path = _SPHERES / "full_body_spherehand.yaml"
