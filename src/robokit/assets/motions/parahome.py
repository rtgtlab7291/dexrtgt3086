"""ParaHome motion data used by the HOI loader."""

from pathlib import Path

from robokit.assets import fetch


def data_root() -> Path:
    """Return the directory containing ``seq`` and ``smplx_seq``."""
    return fetch(["motions/ParaHome/**"]) / "motions/ParaHome"
