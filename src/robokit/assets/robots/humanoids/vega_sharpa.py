"""Dexmate Vega-1 + bimanual Sharpa Wave hand assembly assets.

The combined URDF references meshes relatively across three subtrees, so fetch all three so the
`../../humanoids/vega_1/...` and `../../end_effectors/sharpa_hand/...` paths resolve.
"""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_description/assembly/vega_sharpa/**",
        "robots/robot_description/humanoids/vega_1/**",
        "robots/robot_description/end_effectors/sharpa_hand/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/vega_sharpa/vega_sharpa.urdf"
