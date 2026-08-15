"""dex-retargeting motion clips assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["motions/dex-retargeting/**"])
DIR: Path = _DIR / "motions/dex-retargeting"
DEXYCB_MOTION_PATH: Path = _DIR / "motions/dex-retargeting/dexycb_motion.pkl"
