# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportInvalidTypeForm=false
from dataclasses import dataclass, field, replace
from typing import List, Literal, Optional

import warp as wp

from robokit.opt.gd_optimizer import GDOptimizer, GDOptimizerConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.population_solver import (
    PopulationSolver,
    PopulationSolverConfig,
    StageConfig,
)
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizer, SparseLMOptimizerConfig
from robokit.opt.var_values import VarValues
from robokit.terms.task import ResidualTask, SparseTask
from robokit.utils.warp_utils import wp_device_type


# Re-exported so existing importers of these names keep working.
__all__ = ["StageConfig", "MultiSeedSolverConfig", "MultiSeedSolver"]


@dataclass
class MultiSeedSolverConfig(PopulationSolverConfig):
    optimizer_type: Literal["dense", "sparse"] = "dense"
    gain_ratio_epsilon: float = 1e-8
    active_dof_mask: Optional[wp.array] = field(default=None, hash=False, compare=False)
    lambda_factor: float = 2.0
    rho_min: float = 1e-3
    cg_max_iter_override: Optional[int] = None
    # Setting either routes _build_stage_updater to that optimizer instead of LM/SparseLM; per-stage
    # max_iter and use_cuda_graph are overridden from stages/cuda_graph_mode.
    lbfgs: Optional[LBFGSOptimizerConfig] = None
    gd: Optional[GDOptimizerConfig] = None


class MultiSeedSolver(PopulationSolver):
    """Population solver whose per-stage update is a gradient (LM / Sparse / LBFGS / GD) step."""

    def __init__(
        self,
        terms: List[List[ResidualTask]],
        config: MultiSeedSolverConfig,
        device: wp_device_type,
        score_terms: Optional[List[Optional[ResidualTask]]] = None,
    ):
        super().__init__(terms=terms, config=config, device=device, score_terms=score_terms)

    def _build_stage_updater(self, stage_idx: int, stage_var: VarValues):
        stage = list(self.config.stages)[stage_idx]
        if self.config.lbfgs is not None:
            return LBFGSOptimizer(
                terms=list(self.terms[stage_idx]),
                device=self.device,
                config=replace(self.config.lbfgs, max_iter=stage.iters, use_cuda_graph=self._use_sub_graph),
                active_dof_mask=self.config.active_dof_mask,
            )
        if self.config.gd is not None:
            return GDOptimizer(
                terms=list(self.terms[stage_idx]),
                device=self.device,
                config=replace(
                    self.config.gd,
                    max_iter=stage.iters,
                    use_early_stopping=self._use_early_stopping and self.config.gd.use_early_stopping,
                    use_cuda_graph=self._use_sub_graph,
                ),
                active_dof_mask=self.config.active_dof_mask,
            )
        if self.config.optimizer_type == "dense":
            return LMOptimizer(
                terms=list(self.terms[stage_idx]),
                config=LMOptimizerConfig(
                    num_dofs=stage_var.tangent_dim,
                    max_iter=stage.iters,
                    lm_lambda=stage.lm_lambda,
                    lambda_factor=self.config.lambda_factor,
                    rho_min=self.config.rho_min,
                    use_early_stopping=self._use_early_stopping,
                    early_stopping_interval=stage.early_stopping_interval,
                    gain_ratio_epsilon=self.config.gain_ratio_epsilon,
                    use_cuda_graph=self._use_sub_graph,
                    active_dof_mask=self.config.active_dof_mask,
                ),
                device=self.device,
            )
        for task in self.terms[stage_idx]:
            if not isinstance(task, SparseTask):
                raise TypeError(f"SparseLMOptimizer requires SparseTask, got {type(task).__name__}")
        return SparseLMOptimizer(
            term=list(self.terms[stage_idx]),
            batch_size=stage_var.batch_size,
            total_tangent_dim=stage_var.tangent_dim,
            device=self.device,
            config=SparseLMOptimizerConfig(
                max_iter=stage.iters,
                lm_lambda=stage.lm_lambda,
                lambda_factor=self.config.lambda_factor,
                rho_min=self.config.rho_min,
                gain_ratio_epsilon=self.config.gain_ratio_epsilon,
                use_early_stopping=self._use_early_stopping,
                cg_max_iter_override=self.config.cg_max_iter_override,
                use_cuda_graph=self._use_sub_graph,
            ),
            active_dof_mask=self.config.active_dof_mask,
        )

    def _solve_stage(self, stage_idx: int, warmup: bool = False) -> wp.array:
        updater = self._updaters[stage_idx]
        if not warmup:
            _, costs = updater.solve(self._stage_vars[stage_idx])
            return costs
        # Cheap eager pass before "full"-mode graph capture (result discarded): cap max_iter so
        # this JIT/allocation-only run doesn't pay the full eager cost the capture is meant to
        # avoid (LMOptimizer/SparseLMOptimizer/LBFGSOptimizer/GDOptimizer each read config.max_iter
        # fresh on every solve(), so restoring it below is enough to make later solves un-capped).
        full_config = updater.config
        updater.config = replace(full_config, max_iter=min(2, full_config.max_iter))
        _, costs = updater.solve(self._stage_vars[stage_idx])
        updater.config = full_config
        return costs
