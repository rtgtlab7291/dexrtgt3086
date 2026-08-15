"""Berkeley Humanoid Lite assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/humanoids/berkeley_humanoid_lite/**",
        "robots/collision_spheres/berkeley_humanoid_lite/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/berkeley_humanoid_lite/berkeley_humanoid_lite.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/berkeley_humanoid_lite/full_body.yaml"
