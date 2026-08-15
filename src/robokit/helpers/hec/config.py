from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence


@dataclass
class PyramidLevel:
    """One coarse-to-fine level: downscaled mask + seeds/iters for this stage."""

    downscale: int
    num_seeds: int
    max_iter: int
    blur_sigma: float
    patience: int
    early_stopping_interval: int = 1
    # use only the top-K largest-mask views at this level (None = all N)
    obs_subset_size: Optional[int] = None
    # enable the depth residual; fine levels only (meaningless during coarse basin search)
    use_depth: bool = False


def _default_levels() -> List[PyramidLevel]:
    # coarse-level obs_subset_size bounds per-iter mesh-transform work (~S*N*V); final polish uses all views
    return [
        PyramidLevel(
            downscale=16,
            num_seeds=32,
            max_iter=15,
            blur_sigma=2.5,
            patience=3,
            early_stopping_interval=2,
            obs_subset_size=5,
        ),
        PyramidLevel(
            downscale=8,
            num_seeds=3,
            max_iter=10,
            blur_sigma=1.8,
            patience=2,
            early_stopping_interval=2,
            obs_subset_size=5,
        ),
        PyramidLevel(
            downscale=4,
            num_seeds=2,
            max_iter=12,
            blur_sigma=1.2,
            patience=2,
            early_stopping_interval=2,
            obs_subset_size=8,
        ),
        PyramidLevel(
            downscale=2,
            num_seeds=1,
            max_iter=10,
            blur_sigma=0.6,
            patience=2,
            early_stopping_interval=1,
        ),
        PyramidLevel(
            downscale=1,
            num_seeds=1,
            max_iter=30,
            blur_sigma=0.0,
            patience=4,
            early_stopping_interval=1,
        ),
    ]


@dataclass
class HECHelperConfig:
    """`HECHelper` configuration: pyramid levels, LM parameters, seed sampling."""

    levels: Sequence[PyramidLevel] = field(default_factory=_default_levels)
    # depth residual weight, balances meters against [0, 1] mask units
    depth_weight: float = 100.0
    # Huber delta (meters) for the depth residual; tune to the depth sensor's noise floor
    depth_huber_delta: float = 0.02
    lm_lambda: float = 1e-3
    lambda_factor: float = 2.0
    rho_min: float = 1e-3
    cuda_graph_mode: Literal["none", "full", "iter"] = "iter"
    seed_noise_t: float = 0.35
    seed_noise_r: float = 0.7
    seed: int = 0

    def __post_init__(self):
        if not self.levels or self.levels[-1].num_seeds != 1:
            raise ValueError("levels must end with num_seeds=1")
