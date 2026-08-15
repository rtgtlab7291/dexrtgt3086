"""MANO hand body model assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["body_models/mano/**"])
DIR: Path = _DIR / "body_models/mano"
RIGHT_NPZ_PATH: Path = _DIR / "body_models/mano/MANO_RIGHT.npz"
LEFT_NPZ_PATH: Path = _DIR / "body_models/mano/MANO_LEFT.npz"
