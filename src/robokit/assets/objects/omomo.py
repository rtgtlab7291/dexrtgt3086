"""OMOMO interaction props (the 13 objects the OMOMO clips manipulate)."""

from pathlib import Path

from robokit.assets import fetch


NAMES = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "monitor",
    "plasticbox",
    "smallbox",
    "smalltable",
    "suitcase",
    "trashcan",
    "tripod",
    "whitechair",
    "woodchair",
)


def mesh_path(name: str) -> Path:
    """Path to one prop's `.obj`; fetches just that object."""
    assert name in NAMES, f"unknown OMOMO object {name!r}; expected one of {NAMES}"
    return fetch([f"objects/OMOMO/{name}/**"]) / f"objects/OMOMO/{name}/{name}.obj"
