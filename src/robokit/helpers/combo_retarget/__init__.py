"""Combo retargeting: whole-body pass + dexterous fingers + object and scene contact.

One driver (``retarget.py``) over one clip contract (``clip.py``, built by ``loaders/``). The source
recording already has articulated fingers, so they retarget through the hand stack and a contact
refine firms up the grasp. Imports torch, so nothing is re-exported at the umbrella level.
"""

from robokit.helpers.combo_retarget.clip import ComboClip, HandTrack, SceneGeometry
from robokit.helpers.combo_retarget.config import ComboPreset, HandBinding
from robokit.helpers.combo_retarget.retarget import ComboRetargeter, HandTrajectory


# preset name -> the presets/ module that defines it (lazy: importing a preset module fetches assets).
_PRESET_MODULES = {
    "g1_inspire_right": "g1_inspire",
    "g1_inspire_bimanual": "g1_inspire",
}


class _LazyPresets(dict):
    """name -> ComboPreset, importing the robot's preset module on first lookup and caching it."""

    def __missing__(self, name: str) -> ComboPreset:
        import importlib

        module = importlib.import_module(f"robokit.helpers.combo_retarget.presets.{_PRESET_MODULES[name]}")
        preset = getattr(module, name)
        self[name] = preset
        return preset


PRESETS = _LazyPresets()


__all__ = [
    "ComboClip",
    "HandTrack",
    "HandTrajectory",
    "ComboPreset",
    "ComboRetargeter",
    "HandBinding",
    "PRESETS",
    "SceneGeometry",
]
