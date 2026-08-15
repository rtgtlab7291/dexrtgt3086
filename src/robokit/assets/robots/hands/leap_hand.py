"""LEAP Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["robots/robot_description/end_effectors/leap_hand/**"])
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/leap_hand/leap_hand_right_glb.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/leap_hand/leap_hand_left_glb.urdf"
FREE_URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/leap_hand/leap_hand_right_glb_free.urdf"
