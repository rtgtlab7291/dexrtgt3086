"""Module-level HEC preset constants."""

from dataclasses import replace

from robokit.helpers.hec.config import HECHelperConfig, _default_levels


_levels = _default_levels()

rgb = HECHelperConfig()
rgbd = HECHelperConfig(levels=_levels[:-2] + [replace(level, use_depth=True) for level in _levels[-2:]])
depth = HECHelperConfig(levels=[replace(level, use_depth=True) for level in _levels])

__all__ = ["depth", "rgb", "rgbd"]
