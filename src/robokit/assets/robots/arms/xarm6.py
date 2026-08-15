"""UFactory xArm6 assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["robots/robot_description/arms/xarm6/**"])
URDF_PATH: Path = _DIR / "robots/robot_description/arms/xarm6/xarm6.urdf"
