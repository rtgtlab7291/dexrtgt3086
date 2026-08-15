"""xArm6 + Sharpa Wave hand assembly assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm6_sharpa/**",
        "robots/collision_spheres/xarm6_sharpa/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm6_sharpa/xarm6_sharpa_right.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm6_sharpa/collision_spheres.yaml"
