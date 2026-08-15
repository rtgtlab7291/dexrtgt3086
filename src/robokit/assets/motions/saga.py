"""SAGA FullGraspPose motion subset."""

from pathlib import Path

from robokit.assets import fetch


def data_root() -> Path:
    """Fetch the SAGA motion subset and return its ``FullGraspPose``-style dir (``<split>/<subject>/*.npz``)."""
    return fetch(["motions/SAGA/**"]) / "motions/SAGA"
