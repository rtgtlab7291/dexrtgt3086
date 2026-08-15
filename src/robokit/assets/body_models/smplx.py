"""SMPL-X body model assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["body_models/smplx/**"])
DIR: Path = _DIR / "body_models/smplx"
NEUTRAL_NPZ_PATH: Path = _DIR / "body_models/smplx/SMPLX_NEUTRAL.npz"
MALE_NPZ_PATH: Path = _DIR / "body_models/smplx/SMPLX_MALE.npz"
FEMALE_NPZ_PATH: Path = _DIR / "body_models/smplx/SMPLX_FEMALE.npz"
