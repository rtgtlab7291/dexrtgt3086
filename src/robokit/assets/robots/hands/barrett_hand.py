"""Barrett Hand assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/barrett_hand/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/barrett_hand/bhand_model.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/end_effectors/barrett_hand/bhand_model_glb_fixed_base.urdf"
URDF_GLB_FREE_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/end_effectors/barrett_hand/bhand_model_glb_fixed_base_free_nomimic.urdf"
)
