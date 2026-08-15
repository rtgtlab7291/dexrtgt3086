"""Schunk SVH Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/schunk_hand/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/schunk_hand/schunk_svh_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/schunk_hand/schunk_svh_hand_left.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/end_effectors/schunk_hand/schunk_svh_hand_right_glb.urdf"
URDF_GLB_FREE_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/end_effectors/schunk_hand/schunk_svh_hand_right_glb_free_nomimic.urdf"
)
