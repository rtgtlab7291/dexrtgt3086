"""Franka Emika Panda assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/arms/franka_panda/**",
        "robots/collision_spheres/franka_panda/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_v2.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/franka_panda/collision_spheres.yaml"
GRIPPER_URDF_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_v2_gripper.urdf"
GRIPPER_NO_MIMIC_URDF_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_v2_no_mimic.urdf"
URDF_NO_MIMIC_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_v2_no_mimic.urdf"
