"""UFactory xArm7 assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/arms/xarm7/**",
        "robots/robot_description/arms/xarm7/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/xarm7/xarm7.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/arms/xarm7/xarm7_glb.urdf"
