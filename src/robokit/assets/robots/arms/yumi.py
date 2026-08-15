"""ABB YuMi (collision spheres only) assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["robots/collision_spheres/yumi/**"])
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/yumi/collision_spheres.yaml"
