from typing import Optional, Sequence

import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.task import ResidualTask


class CompositeScoreTask(ResidualTask):
    """Concatenate residuals from multiple tasks."""

    def __init__(self, tasks: Sequence[ResidualTask]):
        self.tasks = list(tasks)
        self.SUPPORTS_CUDA_GRAPH = all(task.SUPPORTS_CUDA_GRAPH for task in tasks)

    @property
    def residual_dim(self) -> int:
        return sum(task.residual_dim for task in self.tasks)

    def precompute(self, var_values: VarValues, need_gradient: bool = False):
        """Prepare every child task."""
        for task in self.tasks:
            task.precompute(var_values, need_gradient=need_gradient)

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if out_residual is None:
            raise ValueError("out_residual is required.")
        for task in self.tasks:
            task.compute_weighted_residual(var_values, out_residual=out_residual, row_offset=row_offset)
            row_offset += task.residual_dim
        return out_residual
