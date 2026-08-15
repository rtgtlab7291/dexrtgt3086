"""D'Claw Gripper assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["robots/robot_description/end_effectors/dclaw_gripper/**"])
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/dclaw_gripper/dclaw_gripper.urdf"
