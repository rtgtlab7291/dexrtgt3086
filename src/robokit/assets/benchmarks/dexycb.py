"""DexYCB benchmark assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["benchmarks/dexycb/**"])
DIR: Path = _DIR / "benchmarks/dexycb"
