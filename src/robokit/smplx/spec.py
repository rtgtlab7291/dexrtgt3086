"""Parametric body-model specification (SMPL / SMPL-H / SMPL-X / MANO / FLAME)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from jaxtyping import Float, Int


@dataclass(frozen=True)
class BodyModelSpec:
    """Parsed, device-agnostic body-model data for any LBS variant.

    Format-specific extras (`gender`, `side`, `hands_meanl`, ...) live
    in `metadata`; loaders are the only code that populates them.
    """

    name: str
    """Identifier such as `"smplx_neutral"` or `"mano_right"`."""

    v_template: Float[np.ndarray, "V 3"]
    shapedirs: Float[np.ndarray, "V 3 num_betas"]
    posedirs: Float[np.ndarray, "P Vx3"]
    """`(P, V*3)` layout consumed by `pose_feature @ posedirs`; `P = (J - 1) * 9`."""

    J_regressor: Float[np.ndarray, "J V"]
    parents: Int[np.ndarray, "J"]
    """`parents[0] == -1` marks the root."""

    lbs_weights: Float[np.ndarray, "V J"]
    faces: Int[np.ndarray, "F 3"]
    joint_names: List[str]
    num_betas: int

    static_landmark_names: Optional[List[str]] = None
    static_landmark_vertex_ids: Optional[Int[np.ndarray, "L"]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
