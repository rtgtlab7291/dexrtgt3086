"""Per-call pose + shape state for a parametric body model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np


if TYPE_CHECKING:
    import torch

    _ArrayLike = Union[np.ndarray, torch.Tensor]
else:
    _ArrayLike = np.ndarray


@dataclass(frozen=True)
class BodyModelState:
    """Batched pose + shape + root translation.

    The default runtime path (`body_lbs_warp` / `body_fk_warp`) consumes
    `np.ndarray` so the numpy+warp pipeline never requires `torch`.
    Differentiable callers (`mano_forward_pca_mm`, the torch oracle) may
    pass `torch.Tensor`; `body_lbs_torch` accepts either via
    `torch.as_tensor`.

    `full_pose_aa` is raw axis-angle in MANO/SMPL-X joint order; loaders
    that want the "relaxed hand" mean (or similar offsets) apply them before
    constructing the state.
    """

    betas: _ArrayLike
    """`[B, num_betas]` shape coefficients."""
    full_pose_aa: _ArrayLike
    """`[B, J, 3]` raw axis-angle pose per joint."""
    transl: _ArrayLike
    """`[B, 3]` root translation."""

    @property
    def batch_size(self) -> int:
        return int(self.full_pose_aa.shape[0])
