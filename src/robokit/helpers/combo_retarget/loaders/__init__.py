"""Dataset loaders: source recordings -> `ComboClip`, one module per source.

Each loader owns its dataset's conventions (frame, units, hand parameterization, contact labels)
and returns the shared clip contract, keeping the driver dataset-agnostic.
"""

from robokit.helpers.combo_retarget.loaders.parahome import (
    ParaHomeInterval,
    find_parahome_intervals,
    load_parahome_clip,
)
from robokit.helpers.combo_retarget.loaders.saga import load_saga_clip


__all__ = ["ParaHomeInterval", "find_parahome_intervals", "load_parahome_clip", "load_saga_clip"]
