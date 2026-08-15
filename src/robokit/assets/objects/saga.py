"""Watertight SAGA grasp-object meshes."""

from pathlib import Path

from robokit.assets import fetch


def mesh_dir() -> Path:
    """Fetch every SAGA baked object mesh and return the directory holding the ``<object>.obj`` files."""
    return fetch(["objects/SAGA/**"]) / "objects/SAGA"
