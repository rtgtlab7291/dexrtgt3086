"""Watertight ParaHome object collision meshes."""

from pathlib import Path

from robokit.assets import fetch


def mesh_dir() -> Path:
    """Return the directory containing baked part meshes."""
    return fetch(["objects/ParaHome/**"]) / "objects/ParaHome"
