from typing import TYPE_CHECKING

from robokit.opt.var_values import Var, VarValues


if TYPE_CHECKING:
    from robokit.opt.sparse_lm_optimizer import SparseLMOptimizer, SparseLMOptimizerConfig


def __getattr__(name: str):
    if name == "SparseLMOptimizer":
        from robokit.opt.sparse_lm_optimizer import SparseLMOptimizer

        return SparseLMOptimizer
    if name == "SparseLMOptimizerConfig":
        from robokit.opt.sparse_lm_optimizer import SparseLMOptimizerConfig

        return SparseLMOptimizerConfig
    raise AttributeError(f"module 'robokit.opt' has no attribute {name!r}")


__all__ = ["SparseLMOptimizer", "SparseLMOptimizerConfig", "Var", "VarValues"]
