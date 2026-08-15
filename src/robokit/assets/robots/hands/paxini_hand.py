"""Paxini DexH13 Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["robots/robot_description/end_effectors/paxini_hand/**"])
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/paxini_hand/dexh13_hand_right_description.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/paxini_hand/dexh13_hand_left_description.urdf"
