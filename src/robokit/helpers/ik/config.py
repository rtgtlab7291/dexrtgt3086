"""IKConfig - robot-free recipe for IK setup.

A pure dataclass holding terms, score_terms, solver, and scalar knobs.
Robot and target links are passed to `IK(config, robot, link)`; the config
itself is robot-free, so a single recipe (e.g. `presets.basic`) can be
reused across robots.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.task import ResidualTask


def _default_ik_solver_config() -> MultiSeedSolverConfig:
    return MultiSeedSolverConfig(
        stages=[
            StageConfig(num_seeds=64, iters=6, lm_lambda=10.0),
            StageConfig(num_seeds=4, iters=10, lm_lambda=1.0),
            StageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
        ],
        cuda_graph_mode="full",
    )


@dataclass
class IKConfig:
    terms: List[ResidualTask] = field(default_factory=list)
    score_terms: List[ResidualTask] = field(default_factory=list)
    solver: MultiSeedSolverConfig = field(default_factory=_default_ik_solver_config)

    seed: Optional[int] = None
    init_sample_range: float = 0.2
    base_init_sample_range: float = 0.1
    keep_init_seed: bool = True
    enable_T_world_base: bool = False
    base_sample_translation_mask: Sequence[float] = (1.0, 1.0, 0.0)
    base_lock_mask: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def __post_init__(self):
        if not self.solver.stages:
            raise ValueError("solver.stages must not be empty")
        if len(self.base_lock_mask) != 6:
            raise ValueError(f"base_lock_mask length {len(self.base_lock_mask)} != 6")
        if any(m != 1.0 for m in self.base_lock_mask) and not self.enable_T_world_base:
            raise ValueError("base_lock_mask requires enable_T_world_base=True")

    def add(self, term: ResidualTask, *, name: Optional[str] = None) -> "IKConfig":
        self._assign_name(term, name)
        self._validate(term)
        self.terms.append(term)
        return self

    def add_score(self, term: ResidualTask, *, name: Optional[str] = None) -> "IKConfig":
        self._assign_name(term, name)
        self._validate(term)
        self.score_terms.append(term)
        return self

    def _assign_name(self, term: ResidualTask, name: Optional[str]):
        if name is not None:
            term.name = name
            return
        base = term.cost_name
        existing = {t.name for t in self.terms} | {t.name for t in self.score_terms}
        idx = 0
        candidate = f"{base}_{idx}"
        while candidate in existing:
            idx += 1
            candidate = f"{base}_{idx}"
        term.name = candidate

    def _validate(self, term: ResidualTask):
        if isinstance(term, PositionLimit) and term.base_axis is not None and not self.enable_T_world_base:
            raise ValueError("PositionLimit base bounds require enable_T_world_base=True")


__all__ = ["IKConfig"]
