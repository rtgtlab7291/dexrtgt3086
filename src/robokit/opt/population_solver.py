# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportInvalidTypeForm=false
import abc
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import warp as wp

from robokit.opt.optimizer import aggregate_residuals_to_costs_kernel
from robokit.opt.var_values import VarValues
from robokit.terms.task import ResidualTask, SparseTask
from robokit.utils.warp_utils import wp_device_type


@wp.kernel
def _sum_costs_kernel(
    src: wp.array1d(dtype=wp.float32),
    dst: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    dst[batch_idx] = dst[batch_idx] + src[batch_idx]


@wp.kernel
def _gather_selected_values_kernel(
    values: wp.array1d(dtype=wp.float32),
    indices: wp.array2d(dtype=wp.int32),
    selected_values: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    selected_values[batch_idx] = values[indices[batch_idx, 0]]


@wp.kernel
def _plain_select_best_k_kernel(
    costs: wp.array1d(dtype=wp.float32),
    work: wp.array2d(dtype=wp.int32),
    best_indices: wp.array2d(dtype=wp.int32),
    num_seeds: int,
    beam_width: int,
):
    batch_idx = wp.tid()
    start = batch_idx * num_seeds
    for s in range(num_seeds):
        work[batch_idx, s] = s
    for k in range(beam_width):
        min_pos = k
        min_cost = costs[start + work[batch_idx, k]]
        for s in range(k + 1, num_seeds):
            c = costs[start + work[batch_idx, s]]
            if c < min_cost:
                min_cost = c
                min_pos = s
        tmp = work[batch_idx, k]
        work[batch_idx, k] = work[batch_idx, min_pos]
        work[batch_idx, min_pos] = tmp
        best_indices[batch_idx, k] = start + work[batch_idx, k]


@dataclass
class StageConfig:
    num_seeds: int  # candidates per problem in this stage
    iters: int  # update iterations run in this stage
    lm_lambda: float = 10.0  # initial LM damping for this stage
    early_stopping_interval: int = 1  # check the early-stop condition every N iterations (0 = never)


def _default_single_stage() -> List[StageConfig]:
    return [StageConfig(num_seeds=1, iters=10, lm_lambda=1.0)]


@dataclass
class PopulationSolverConfig:
    # beam-search schedule; solve() reduces the final stage to one result per instance
    stages: Sequence[StageConfig] = field(default_factory=_default_single_stage)
    # "full" captures the whole solve as one graph (disables early stopping),
    # "iter" captures per-stage updater graphs, "none" runs eagerly.
    cuda_graph_mode: Literal["none", "full", "iter"] = "iter"
    use_early_stopping: bool = True


class PopulationSolver(abc.ABC):
    """Staged beam-search solver over a population of candidates per problem.

    Owns the optimizer-agnostic machinery: instance-major buffer layout
    (`start = batch_idx * num_seeds`), stage variables via `gather`, best-cost
    selection, the score path, warm-start between stages, the solve loop, and CUDA-graph
    orchestration. Subclasses supply the per-stage solve rule via two seam methods  -
    `_build_stage_updater` (build the stage engine) and `_solve_stage` (run it and
    return per-candidate costs). `MultiSeedSolver` fills these with a gradient (LM/Sparse/LBFGS/GD)
    step; `ParticleSolver` fills them with an MPPI sampling step.
    """

    def __init__(
        self,
        terms: List[List[ResidualTask]],
        config: PopulationSolverConfig,
        device: wp_device_type,
        score_terms: Optional[List[Optional[ResidualTask]]] = None,
    ):
        stages = list(config.stages)
        if len(stages) < 1:
            raise ValueError("At least one stage is required")
        if len(terms) != len(stages):
            raise ValueError("terms must have same length as config.stages")

        self.config = config
        self.device: wp_device_type = wp.get_device(device) if isinstance(device, str) else device
        self.num_stages = len(stages)
        self.terms = terms
        self.score_terms: List[Optional[ResidualTask]] = (
            score_terms if score_terms is not None else [None] * self.num_stages
        )
        self._supports_cuda_graph = (
            self.device.is_cuda
            and all(term.SUPPORTS_CUDA_GRAPH for stage_terms in self.terms for term in stage_terms)
            and all(score_term is None or score_term.SUPPORTS_CUDA_GRAPH for score_term in self.score_terms)
        )

        # beam selection kernels (only depend on num_seeds from config)
        _warp_ver = tuple(int(x) for x in wp.__version__.split(".")[:2])
        self._use_tiled_select: bool = self.device.is_cuda or _warp_ver >= (1, 9)

        if self._use_tiled_select:
            self._tile_threads: List[int] = []
            self._select_best_kernels: List = []
            for i in range(self.num_stages - 1):
                self._tile_threads.append(max(32, stages[i].num_seeds))
                self._select_best_kernels.append(
                    self._build_select_best_k_kernel(stages[i].num_seeds, stages[i + 1].num_seeds)
                )
            self._final_tile_threads = max(32, stages[-1].num_seeds)
            self._final_select_best_kernel = self._build_select_best_k_kernel(stages[-1].num_seeds, 1)

        self._initialized = False

    # --- per-stage update rule ---
    # Subclasses supply this rule.
    @abc.abstractmethod
    def _build_stage_updater(self, stage_idx: int, stage_var: VarValues):
        """Build the per-stage update engine for `stage_idx` (stored in `self._updaters`)."""
        ...

    @abc.abstractmethod
    def _solve_stage(self, stage_idx: int, warmup: bool = False) -> wp.array:
        """Solve stage `stage_idx` in place on `self._stage_vars[stage_idx]`.

        Returns the `[total_batches[stage_idx]]` per-candidate cost array used for beam
        selection when this stage has no dedicated score term. `warmup=True` marks the
        single eager pass before "full"-mode graph capture (see `_warmup_and_capture`):
        the result is discarded, so subclasses may substitute a cheaper stand-in run to
        allocate buffers / JIT kernels without paying the full eager `max_iter` cost.
        """
        ...

    def setup(self, var: VarValues):
        stages = list(self.config.stages)
        mode = self.config.cuda_graph_mode
        self._use_sub_graph = mode == "iter"
        self._use_early_stopping = self.config.use_early_stopping and mode != "full"

        self.batch_size = int(var.batch_size) // stages[0].num_seeds
        self.total_batches: List[int] = [self.batch_size * stage.num_seeds for stage in stages]

        # stage vars: stage 0 is the caller's var, rest via gather
        self._stage_vars: List[VarValues] = [var]
        for stage_batch in self.total_batches[1:]:
            indices = wp.zeros((stage_batch,), dtype=wp.int32, device=self.device)
            self._stage_vars.append(var.gather(indices))

        # beam search index buffers
        self.best_indices: List[wp.array] = []
        for i in range(self.num_stages - 1):
            self.best_indices.append(
                wp.empty((self.batch_size, stages[i + 1].num_seeds), dtype=wp.int32, device=self.device)
            )
        self.best_costs = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.winner_indices = wp.from_numpy(
            np.arange(self.batch_size, dtype=np.int32).reshape((-1, 1)), dtype=wp.int32, device=self.device
        )
        self.final_var = self._stage_vars[-1]  # Full final population before selecting the best candidate.
        self._best_var = self.final_var
        if stages[-1].num_seeds > 1:
            self._best_var = self._stage_vars[-1].gather(self.winner_indices.reshape((self.batch_size,)))

        # plain select workspace (CPU + warp <= 1.8 only)
        if not self._use_tiled_select:
            max_seeds = max(stage.num_seeds for stage in stages)
            self._plain_select_work = wp.empty((self.batch_size, max_seeds), dtype=wp.int32, device=self.device)

        # per-stage update engines (seam)
        self._updaters: List = []
        for i in range(self.num_stages):
            self._updaters.append(self._build_stage_updater(i, self._stage_vars[i]))

        # score buffers (merged loop)
        self._score_residual_buffers: List[Optional[wp.array]] = []
        self._score_costs: List[Optional[wp.array]] = []
        for i in range(self.num_stages):
            score_term = self.score_terms[i]
            if score_term is not None:
                self._score_residual_buffers.append(
                    wp.empty((self.total_batches[i], score_term.residual_dim), dtype=wp.float32, device=self.device)
                )
                self._score_costs.append(wp.empty((self.total_batches[i],), dtype=wp.float32, device=self.device))
            else:
                self._score_residual_buffers.append(None)
                self._score_costs.append(None)

        self._full_graph = None
        if mode == "full" and self._supports_cuda_graph:
            self._warmup_and_capture()

        self._initialized = True

    def _run_solve_body(self, sync_each_stage: bool = False):
        # `sync_each_stage=True` is used during eager warmup before graph capture
        # to drain warp's launch queue between stages. Without this drain, queueing
        # the full body asynchronously intermittently corrupts the CUDA context
        # under multi-process load (>=4 concurrent CUDA contexts on the host),
        # surfacing as `wp_cuda_context_synchronize: error 700` at the next sync.
        # Must be False during captured replay - wp.synchronize_device raises
        # inside graph capture.
        final_scores: wp.array = self.best_costs
        for i in range(self.num_stages):
            if i == 0:
                self._stage_vars[i].invalidate()
            updater_costs = self._solve_stage(i, warmup=sync_each_stage)
            self.candidate_scores = updater_costs
            if sync_each_stage:
                wp.synchronize_device(self.device)

            score_term = self.score_terms[i]
            if score_term is None:
                scores = updater_costs
            else:
                residual_buf = self._score_residual_buffers[i]
                cost_buf = self._score_costs[i]
                score_term.precompute(self._stage_vars[i], need_gradient=False)
                score_term.compute_weighted_residual(self._stage_vars[i], out_residual=residual_buf, row_offset=0)
                wp.launch(
                    kernel=aggregate_residuals_to_costs_kernel,
                    dim=self.total_batches[i],
                    inputs=[residual_buf, cost_buf],
                    device=self.device,
                )
                scores = cost_buf  # type: ignore[assignment]

            if i < self.num_stages - 1:
                if self._use_tiled_select:
                    wp.launch_tiled(
                        self._select_best_kernels[i],
                        dim=[self.batch_size],
                        inputs=[scores, self.best_indices[i]],
                        block_dim=self._tile_threads[i],
                        device=self.device,
                    )
                else:
                    stages = list(self.config.stages)
                    wp.launch(
                        _plain_select_best_k_kernel,
                        dim=[self.batch_size],
                        inputs=[
                            scores,
                            self._plain_select_work,
                            self.best_indices[i],
                            stages[i].num_seeds,
                            stages[i + 1].num_seeds,
                        ],
                        device=self.device,
                    )
                next_batch_total = self.total_batches[i + 1]
                flat_indices = self.best_indices[i].reshape((next_batch_total,))
                self._stage_vars[i].gather(flat_indices, self._stage_vars[i + 1])

            final_scores = scores  # type: ignore[assignment]

        final_num_seeds = list(self.config.stages)[-1].num_seeds
        if final_num_seeds == 1:
            wp.copy(self.best_costs, final_scores)
            return
        if self._use_tiled_select:
            wp.launch_tiled(
                self._final_select_best_kernel,
                dim=[self.batch_size],
                inputs=[final_scores, self.winner_indices],
                block_dim=self._final_tile_threads,
                device=self.device,
            )
        else:
            wp.launch(
                _plain_select_best_k_kernel,
                dim=[self.batch_size],
                inputs=[
                    final_scores,
                    self._plain_select_work,
                    self.winner_indices,
                    final_num_seeds,
                    1,
                ],
                device=self.device,
            )
        flat_indices = self.winner_indices.reshape((self.batch_size,))
        self._stage_vars[-1].gather(flat_indices, out=self._best_var)
        wp.launch(
            _gather_selected_values_kernel,
            dim=self.batch_size,
            inputs=[final_scores, self.winner_indices],
            outputs=[self.best_costs],
            device=self.device,
        )

    def _warmup_and_capture(self):
        self._run_solve_body(sync_each_stage=True)
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self._run_solve_body(sync_each_stage=False)
        self._full_graph = capture.graph

    @staticmethod
    def _build_select_best_k_kernel(num_seeds: int, beam_width: int):
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

    def solve(self, var: VarValues) -> Tuple[VarValues, wp.array]:
        """Run multi-stage beam search optimization.

        Args:
            var: Stage-0 variable filled with initial seed values.

        Returns:
            Tuple of (best_var, best_costs) where best_var is the variable from the final stage
            and best_costs is [batch_size] array of final costs.
        """
        if not self._initialized:
            self.setup(var)

        if self._full_graph is not None:
            # Captured replay reads the buffers bound at setup; the caller writes fresh
            # seeds into the shared stage-0 leaf, so no re-point is needed here.
            wp.capture_launch(self._full_graph)
        else:
            self._stage_vars[0] = var
            self._run_solve_body()

        return self._best_var, self.best_costs

    def compute_costs(self, var: VarValues) -> Dict[str, wp.array]:
        stage_terms = self.terms[-1]
        batch_size = var.batch_size
        total_residual_dim = sum(term.residual_dim for term in stage_terms)

        out_residual = wp.zeros((batch_size, total_residual_dim), dtype=wp.float32, device=self.device)

        for term in stage_terms:
            term.precompute(var, need_gradient=False)
        offset = 0
        for term in stage_terms:
            # Sparse (trajectory) terms launch at their own `batch_size`, so point it at this var's batch.
            if isinstance(term, SparseTask):
                term.batch_size = batch_size
            term.compute_weighted_residual(var, out_residual=out_residual, row_offset=offset)
            offset += term.residual_dim

        costs: Dict[str, wp.array] = {}
        offset = 0
        for term in stage_terms:
            term_residual = out_residual[:, offset : offset + term.residual_dim]
            term_cost = wp.zeros((batch_size,), dtype=wp.float32, device=self.device)
            wp.launch(
                aggregate_residuals_to_costs_kernel,
                dim=batch_size,
                inputs=[term_residual, term_cost],
                device=self.device,
            )
            existing = costs.get(term.cost_name)
            if existing is None:
                costs[term.cost_name] = term_cost
            else:
                # Multiple terms share a cost_name (e.g. mobile config adds a base-lock
                # RestTask alongside a joint-rest RestTask). Sum so neither is dropped.
                wp.launch(
                    _sum_costs_kernel,
                    dim=batch_size,
                    inputs=[term_cost, existing],
                    device=self.device,
                )
            offset += term.residual_dim

        return costs
