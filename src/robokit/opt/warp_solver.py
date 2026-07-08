# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
import os
from dataclasses import dataclass, field
from typing import Any, Generic, List, Literal, Optional, Protocol, Sequence, Tuple, TypeVar

import numpy as np
import warp as wp

from robokit.opt.sparse_warp_optimizer import SparseWarpOptimizer, SparseWarpOptimizerConfig
from robokit.opt.variables import WarpVar
from robokit.opt.warp_optimizer import WarpLMOptimizer, WarpLMOptimizerConfig, aggregate_residuals_to_costs
from robokit.terms.terms import SparseWarpTask, WarpTask
from robokit.utils.warp_utils import wp_device_type


WarpVarT = TypeVar("WarpVarT", bound=WarpVar)


class WarpOptimizerProtocol(Protocol[WarpVarT]):
    """Protocol defining common interface for Warp optimizers."""

    costs: wp.array

    def solve(self, var: WarpVarT) -> WarpVarT:
        """Solve optimization problem."""
        ...


@dataclass
class WarpStageConfig:
    num_seeds: int
    iters: int
    lm_lambda: float = 10.0


def _default_single_stage() -> List[WarpStageConfig]:
    return [WarpStageConfig(num_seeds=1, iters=10, lm_lambda=1.0)]


@dataclass
class WarpSolverConfig:
    stages: Sequence[WarpStageConfig] = field(default_factory=_default_single_stage)
    use_cuda_graph: bool = False
    optimizer_type: Literal["dense", "sparse"] = "dense"
    acceptance_mode: Literal["strict_monotonic", "rho_only"] = "strict_monotonic"


class WarpSolver(Generic[WarpVarT]):
    def __init__(
        self,
        config: WarpSolverConfig,
        placeholder_var: WarpVarT,
        terms: List[List[WarpTask]],
        score_terms: Optional[List[Optional[WarpTask]]] = None,
    ):
        stages = list(config.stages)
        if len(stages) < 1:
            raise ValueError("At least one stage is required")
        if stages[-1].num_seeds != 1:
            raise ValueError("Final stage must have num_seeds=1")
        if len(terms) != len(stages):
            raise ValueError("terms must have same length as config.stages")

        self.batch_size = int(placeholder_var.batch_size)
        self.config = config
        self.num_stages = len(stages)

        self.device: wp_device_type = placeholder_var.device

        self.terms = terms
        self.score_terms: List[Optional[WarpTask]] = (
            score_terms if score_terms is not None else [None] * self.num_stages
        )

        # Compute total batch sizes per stage
        self.total_batches: List[int] = []
        for stage in stages:
            self.total_batches.append(self.batch_size * stage.num_seeds)

        # Auto-create per-stage vars via gather from placeholder_var
        self._stage_vars: List[WarpVarT] = []
        for stage_batch in self.total_batches:
            indices = wp.zeros((stage_batch,), dtype=wp.int32, device=self.device)
            self._stage_vars.append(placeholder_var.gather(indices))

        # Build beam selection kernels (only for multi-stage beam search)
        self.best_indices: List[wp.array] = []
        self._tile_threads: List[int] = []
        self._select_best_kernels: List = []

        for i in range(self.num_stages - 1):
            current_seeds = stages[i].num_seeds
            next_seeds = stages[i + 1].num_seeds
            self.best_indices.append(wp.empty((self.batch_size, next_seeds), dtype=wp.int32, device=self.device))
            self._tile_threads.append(max(32, current_seeds))
            self._select_best_kernels.append(self._build_select_best_k_kernel(current_seeds, next_seeds))

        self.best_costs = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)

        # Create per-stage optimizers
        use_cuda_graph = config.use_cuda_graph
        optimizer_type = config.optimizer_type

        self._optimizers: List[WarpOptimizerProtocol] = []
        for i in range(self.num_stages):
            stage = stages[i]

            if optimizer_type == "dense":
                optimizer = WarpLMOptimizer(
                    terms=list(terms[i]),
                    device=self.device,
                    config=WarpLMOptimizerConfig(
                        max_iter=stage.iters,
                        lm_lambda=stage.lm_lambda,
                        verbose=False,
                        use_early_stopping=not use_cuda_graph,
                        acceptance_mode=config.acceptance_mode,
                    ),
                    placeholder_var=self._stage_vars[i],
                    use_parallel_terms=True,
                )
            else:  # sparse
                for task in terms[i]:
                    if not isinstance(task, SparseWarpTask):
                        raise TypeError(f"SparseWarpOptimizer requires SparseWarpTask, got {type(task).__name__}")

                optimizer = SparseWarpOptimizer(
                    term=list(terms[i]),
                    batch_size=self._stage_vars[i].batch_size,
                    total_tangent_dim=self._stage_vars[i].tangent_dim,
                    device=self.device,
                    config=SparseWarpOptimizerConfig(
                        max_iter=stage.iters,
                        lm_lambda=stage.lm_lambda,
                        verbose=False,
                        use_early_stopping=not use_cuda_graph,
                    ),
                )

            self._optimizers.append(optimizer)

        # Allocate score buffers
        self._score_residual_buffers: List[Optional[wp.array]] = []
        for i in range(self.num_stages):
            score_term = self.score_terms[i]
            if score_term is not None:
                buf = wp.empty((self.total_batches[i], score_term.residual_dim), dtype=wp.float32, device=self.device)
                self._score_residual_buffers.append(buf)
            else:
                self._score_residual_buffers.append(None)

        self._score_costs: List[Optional[wp.array]] = []
        for i in range(self.num_stages):
            if self.score_terms[i] is not None:
                self._score_costs.append(wp.empty((self.total_batches[i],), dtype=wp.float32, device=self.device))
            else:
                self._score_costs.append(None)

        self._use_cuda_graph: bool = use_cuda_graph
        self._cuda_graphs: List[Optional[Any]] = [None] * self.num_stages
        self._graph_capture_complete: bool = False

        if self._use_cuda_graph and self.device.is_cuda:
            self._warmup_and_capture()

        self._handoff_validation_tol = 1e-6

    @property
    def initial_var(self) -> WarpVarT:
        """First-stage variable. Fill with initial seed values before calling solve()."""
        return self._stage_vars[0]

    def _warmup_and_capture(self) -> None:
        for i in range(self.num_stages):
            self._stage_vars[i].invalidate()
            self._optimizers[i].solve(self._stage_vars[i])

        wp.synchronize()
        for i in range(self.num_stages):
            self._stage_vars[i].invalidate()
            with wp.ScopedCapture(device=self.device) as capture:
                self._optimizers[i].solve(self._stage_vars[i])
            self._cuda_graphs[i] = capture.graph

        self._graph_capture_complete = True

    def _build_select_best_k_kernel(self, num_seeds: int, beam_width: int):
        NUM_SEEDS = int(num_seeds)
        BEAM_WIDTH = int(beam_width)

        def _template(
            costs: wp.array1d(dtype=wp.float32),
            best_indices: wp.array2d(dtype=wp.int32),
        ):
            batch_idx = wp.tid()
            start = batch_idx * NUM_SEEDS

            cost_tile = wp.tile_load(costs, shape=(NUM_SEEDS,), offset=(start,), storage="shared")
            index_tile = wp.tile_arange(0, NUM_SEEDS, dtype=wp.int32, storage="shared")

            wp.tile_sort(cost_tile, index_tile)

            for i in range(BEAM_WIDTH):
                best_indices[batch_idx, i] = start + wp.tile_extract(index_tile, i)

        _template.__name__ = f"_simple_select_best_k_{NUM_SEEDS}to{BEAM_WIDTH}"
        _template.__qualname__ = f"_simple_select_best_k_{NUM_SEEDS}to{BEAM_WIDTH}"
        return wp.kernel(enable_backward=False, module="unique")(_template)

    def _compute_optimizer_costs(self, stage_index: int, var: WarpVarT, out_costs: wp.array) -> None:
        optimizer = self._optimizers[stage_index]
        if isinstance(optimizer, WarpLMOptimizer):
            optimizer.compute_residuals(optimizer.terms, var, optimizer.residuals)
            wp.launch(
                kernel=aggregate_residuals_to_costs,
                dim=var.batch_size,
                inputs=[optimizer.residuals, out_costs],
                device=self.device,
            )
            return
        if isinstance(optimizer, SparseWarpOptimizer):
            optimizer._fill_residuals(var, optimizer.residuals)
            wp.launch(
                kernel=aggregate_residuals_to_costs,
                dim=var.batch_size,
                inputs=[optimizer.residuals, out_costs],
                device=self.device,
            )
            return
        raise TypeError(f"Unsupported optimizer type for handoff validation: {type(optimizer).__name__}")

    def _validate_stage_handoff_costs(self, stage_index: int, flat_indices: wp.array) -> None:
        parent_var = self._stage_vars[stage_index]
        child_var = self._stage_vars[stage_index + 1]

        parent_costs = wp.empty((parent_var.batch_size,), dtype=wp.float32, device=self.device)
        child_costs = wp.empty((child_var.batch_size,), dtype=wp.float32, device=self.device)

        self._compute_optimizer_costs(stage_index, parent_var, parent_costs)
        self._compute_optimizer_costs(stage_index + 1, child_var, child_costs)

        parent_costs_np = parent_costs.numpy()
        child_costs_np = child_costs.numpy()
        flat_indices_np = flat_indices.numpy()
        mapped_parent_costs_np = parent_costs_np[flat_indices_np]
        diffs = np.abs(mapped_parent_costs_np - child_costs_np)
        max_diff = float(np.max(diffs))
        if max_diff <= self._handoff_validation_tol:
            return

        worst_flat_index = int(np.argmax(diffs))
        selected_parent_index = int(flat_indices_np[worst_flat_index])
        raise RuntimeError(
            "WarpSolver stage handoff cost mismatch: "
            f"stage={stage_index}, worst_selected_idx={worst_flat_index}, parent_idx={selected_parent_index}, "
            f"parent_cost={mapped_parent_costs_np[worst_flat_index]:.8f}, "
            f"child_cost={child_costs_np[worst_flat_index]:.8f}, diff={max_diff:.8e}, "
            f"tol={self._handoff_validation_tol:.1e}"
        )

    def solve(self) -> Tuple[WarpVarT, wp.array]:
        """Run multi-stage beam search optimization.

        Caller must fill initial_var with seed values before calling.

        Returns:
            Tuple of (best_var, best_costs) where best_var is the variable from the final stage
            and best_costs is [batch_size] array of final costs.
        """
        debug_validate_handoff = os.environ.get("ROBOKIT_DEBUG_VALIDATE_STAGE_HANDOFF", "0") == "1"
        use_graphs = self._use_cuda_graph and self._graph_capture_complete
        final_scores: wp.array = self._optimizers[0].costs
        for i in range(self.num_stages):
            self._stage_vars[i].invalidate()
            if use_graphs and self._cuda_graphs[i] is not None:
                wp.capture_launch(self._cuda_graphs[i])
            else:
                self._optimizers[i].solve(self._stage_vars[i])

            score_term = self.score_terms[i]
            if score_term is None:
                scores = self._optimizers[i].costs
            else:
                residual_buf = self._score_residual_buffers[i]
                cost_buf = self._score_costs[i]
                score_term.compute_weighted_residual(self._stage_vars[i], residual_buffer=residual_buf, row_offset=0)
                wp.launch(
                    kernel=aggregate_residuals_to_costs,
                    dim=self.total_batches[i],
                    inputs=[residual_buf, cost_buf],
                    device=self.device,
                )
                scores = cost_buf  # type: ignore[assignment]

            if i < self.num_stages - 1:
                wp.launch_tiled(
                    self._select_best_kernels[i],
                    dim=[self.batch_size],
                    inputs=[scores, self.best_indices[i]],
                    block_dim=self._tile_threads[i],
                    device=self.device,
                )
                next_batch_total = self.total_batches[i + 1]
                flat_indices = self.best_indices[i].reshape((next_batch_total,))
                self._stage_vars[i].gather(flat_indices, self._stage_vars[i + 1])
                if debug_validate_handoff:
                    self._validate_stage_handoff_costs(i, flat_indices)

            final_scores = scores  # type: ignore[assignment]

        wp.copy(self.best_costs, final_scores)

        return self._stage_vars[-1], self.best_costs
