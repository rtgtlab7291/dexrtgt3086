# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
import abc
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.task import ResidualTask, SparseTask, Task
from robokit.utils.warp_utils import wp_device_type


@dataclass(frozen=True)
class OptimizerConfig:
    # --- stopping criteria ---
    max_iter: int = 100
    use_early_stopping: bool = True
    early_stopping_interval: int = 1
    cost_tol: float = 1e-9

    # --- Levenberg-Marquardt parameters ---
    lm_lambda: float = 1e-3
    lambda_factor: float = 2.0
    lambda_min: float = 1e-5
    lambda_max: float = 1e10

    # --- execution ---
    use_parallel_terms: bool = True  # evaluate terms on parallel CUDA streams
    use_cuda_graph: bool = False  # capture + replay the iteration body (locks the batch size)


class Optimizer(abc.ABC):
    terms: Sequence[Task]
    config: OptimizerConfig

    def __init__(self, terms: Union[Task, Sequence[Task]], config: Optional[OptimizerConfig] = None):
        if not isinstance(terms, Sequence):
            terms = [terms]
        self.terms = terms
        if config is None:
            config = OptimizerConfig()
        self.config = config

    @abc.abstractmethod
    def solve(self, var: VarValues) -> Tuple[VarValues, wp.array]: ...


@wp.kernel
def aggregate_residuals_to_costs_kernel(
    residuals: wp.array2d(dtype=wp.float32),
    costs: wp.array(dtype=wp.float32),
):
    """Aggregate squared residuals to compute cost per batch element."""
    batch_idx = wp.tid()

    cost = wp.float32(0.0)
    for i in range(residuals.shape[1]):
        r = residuals[batch_idx, i]
        cost += r * r

    costs[batch_idx] = wp.float32(0.5) * cost


@wp.kernel
def apply_dof_mask_kernel(vec: wp.array2d(dtype=wp.float32), mask: wp.array1d(dtype=wp.float32)):
    """Zero masked tangent dims of a (B, D) vector (gradient or search direction)."""
    batch_idx, dof_idx = wp.tid()  # type: ignore[misc]
    vec[batch_idx, dof_idx] = vec[batch_idx, dof_idx] * mask[dof_idx]


@wp.kernel
def _sparse_jtr_split_kernel(
    residuals: wp.array2d(dtype=wp.float32),  # (B, R) shared residual buffer (all terms)
    jac_values: wp.array2d(dtype=wp.float32),  # (B, NNZ) shared sparse-Jacobian values (all terms)
    value_perm: wp.array1d(dtype=wp.int32),  # column-sorted nnz order
    row_perm: wp.array1d(dtype=wp.int32),  # residual row per sorted nnz
    split_cols: wp.array1d(dtype=wp.int32),  # gradient column per split
    split_offsets: wp.array1d(dtype=wp.int32),  # sorted-nnz range per split
    gradient: wp.array2d(dtype=wp.float32),  # (B, D) accumulated (caller zeroes)
):
    # One J^T r pass over the merged sparsity of ALL terms - replaces one CSC launch per term.
    # Column runs are cut into small splits (one atomic each) so long columns don't serialize.
    batch_idx, split_idx = wp.tid()  # type: ignore[misc]
    g = float(0.0)
    for i in range(split_offsets[split_idx], split_offsets[split_idx + 1]):
        g += jac_values[batch_idx, value_perm[i]] * residuals[batch_idx, row_perm[i]]
    wp.atomic_add(gradient, batch_idx, split_cols[split_idx], g)


_TILED_COST_KERNELS: Dict[int, object] = {}


def _tiled_cost_kernel(R: int):
    """Tiled per-row `0.5‖r‖²` ASSIGNED to costs - one launch for all terms' residuals."""
    kernel = _TILED_COST_KERNELS.get(R)
    if kernel is None:
        Rc = wp.constant(R)

        def cost(residuals: wp.array2d(dtype=wp.float32), costs: wp.array1d(dtype=wp.float32)):
            batch_idx = wp.tid()
            r = wp.tile_load(residuals[batch_idx], shape=(Rc,))
            costs[batch_idx] = wp.float32(0.5) * wp.tile_sum(wp.tile_map(wp.mul, r, r))[0]

        cost.__name__ = cost.__qualname__ = f"_tiled_cost_r{R}"
        kernel = wp.kernel(enable_backward=False, module="unique")(cost)
        _TILED_COST_KERNELS[R] = kernel
    return kernel


class OptimizerBase(Optimizer):
    """Shared infrastructure for Warp-based optimizers (LM, L-BFGS)."""

    # Tiled dense J^T r kernel + block size, provided by the (R, D)-specialized GD/LBFGS subclasses.
    _dense_jtr_kernel: ClassVar = None
    TILE_THREADS: ClassVar = None

    @property
    def is_all_sparse(self) -> bool:
        return all(isinstance(t, SparseTask) for t in self.terms)

    def _build_terms(
        self,
        terms: Union[Tuple[ResidualTask, ...], List[ResidualTask]],
        var: VarValues,
        device: Optional[wp_device_type],
        active_dof_mask: Optional[wp.array],
        residuals_require_grad: bool = False,
    ):
        self.batch_size = var.batch_size
        self.total_tangent_dim = var.tangent_dim
        self.residual_dims = [term.residual_dim for term in terms]
        self.total_residual_dim = sum(self.residual_dims)
        self.device = wp.get_device(device) if isinstance(device, str) else device

        # residual offsets
        offset = 0
        self.residual_offsets = []
        for term in terms:
            self.residual_offsets.append(offset)
            offset += term.residual_dim

        # common buffers
        self.residuals = wp.zeros(
            (self.batch_size, self.total_residual_dim),
            dtype=wp.float32,
            device=device,
            requires_grad=residuals_require_grad,
        )
        # Jacobian storage (dense buffer or sparse machinery) is allocated by `_build_jacobian`.
        self.jacobians: Optional[wp.array] = None
        self.costs = wp.zeros((self.batch_size,), dtype=wp.float32, device=device)

        # CUDA streams for parallel term evaluation
        if self.device is not None and self.device.is_cuda:
            self.term_streams = [wp.Stream(self.device) for _ in terms]
            self.sync_events = [wp.Event(self.device) for _ in terms]
        else:
            self.term_streams = [None] * len(terms)
            self.sync_events = [None] * len(terms)

        # Build residual and jacobian functions. The variable is always a VarValues;
        # each term routes its own leaf and Jacobian column block via `var_key`.
        self._residual_fns = [
            lambda offset, var, buf, _fn=t.compute_weighted_residual: _fn(var, out_residual=buf, row_offset=offset)
            for t in terms
        ]
        self._jacobian_fns = [
            lambda offset, var, buf, _fn=t.compute_weighted_jacobian: _fn(var, out_jacobian=buf, row_offset=offset)
            for t in terms
        ]

        # active DOF mask (wp.array, shape [total_tangent_dim], dtype float32)
        self._active_dof_mask: Optional[wp.array] = active_dof_mask
        # analytic_jacobian storage; _build_jacobian flips this to sparse when the terms allow it
        self._sparse_jac = False

    def _parallel_for_objectives(self, fns, *extra):
        """Run pre-bound fns across terms on parallel CUDA streams."""
        use_streams = self.config.use_parallel_terms and self.device is not None and self.device.is_cuda
        if use_streams:
            main = wp.get_stream(self.device)
            init_evt = main.record_event()
            for offset, fn, term_stream, sync_event in zip(
                self.residual_offsets, fns, self.term_streams, self.sync_events
            ):
                term_stream.wait_event(init_evt)
                with wp.ScopedStream(term_stream):
                    fn(offset, *extra)
                term_stream.record_event(sync_event)
            for sync_event in self.sync_events:
                main.wait_event(sync_event)
        else:
            for offset, fn in zip(self.residual_offsets, fns):
                fn(offset, *extra)

    def _precompute_terms(self, var: VarValues, need_gradient: bool):
        for term in self.terms:
            term.precompute(var, need_gradient=need_gradient)

    def _build_jacobian(self, var: VarValues, jac_rows: int) -> None:
        """Build the analytic_jacobian storage: merged-CSC sparse when every term is a SparseTask,
        else a dense (B,R,D) buffer.

        Sparse: per-term residual/Jacobian-value kernels write shared buffers at their row/nnz
        offsets, then ONE cost launch + ONE merged-CSC gradient launch replace the two per-term
        reductions. Patterns are structural (index-only kernels), built once from the first-solve var."""
        self._sparse_jac = self.is_all_sparse
        device = self.device
        if not self._sparse_jac:
            self.jacobians = wp.zeros(
                (self.batch_size, self.total_residual_dim, self.total_tangent_dim),
                dtype=wp.float32,
                device=device,
            )
            return
        rows_list: list = []
        cols_list: list = []
        nnz_offsets: list = []
        nnz_total = 0
        for t, row_off in zip(self.terms, self.residual_offsets):
            pattern = t.compute_sparse_jacobian_pattern(var, offset=row_off)
            rows_np = pattern.row_indices.numpy()
            cols_np = pattern.col_indices.numpy().astype(np.int64) + var.tangent_offset(t.var_key)
            nnz_offsets.append(nnz_total)
            nnz_total += rows_np.shape[0]
            rows_list.append(rows_np)
            cols_list.append(cols_np)
        merged_cols = np.concatenate(cols_list)
        merged_rows = np.concatenate(rows_list)
        # Column-sorted splits of at most `split_size` nonzeros (one atomic per split), mirroring
        # SparseLMOptimizer's J^T layout - long columns (collision) never serialize one thread.
        split_size = 8
        order = np.argsort(merged_cols, kind="stable")
        cols_sorted = merged_cols[order]
        col_run_starts = np.flatnonzero(np.r_[True, cols_sorted[1:] != cols_sorted[:-1]])
        split_starts = np.concatenate(
            [np.arange(s, e, split_size) for s, e in zip(col_run_starts, np.r_[col_run_starts[1:], nnz_total])]
        )
        self._sparse_value_perm = wp.from_numpy(order.astype(np.int32), dtype=wp.int32, device=device)
        self._sparse_row_perm = wp.from_numpy(merged_rows[order].astype(np.int32), dtype=wp.int32, device=device)
        self._sparse_split_cols = wp.from_numpy(
            cols_sorted[split_starts].astype(np.int32), dtype=wp.int32, device=device
        )
        self._sparse_split_offsets = wp.from_numpy(
            np.append(split_starts, nnz_total).astype(np.int32), dtype=wp.int32, device=device
        )
        self._sparse_num_splits = int(split_starts.shape[0])
        self._sparse_cost_kernel = _tiled_cost_kernel(self.total_residual_dim)
        self._sparse_jac_values = wp.zeros((jac_rows, nnz_total), dtype=wp.float32, device=device)
        self._sparse_fill_fns = [
            lambda offset, var, res, jac, _t=t, _n=nnz_off: _t.fill_residual_and_jacobian(
                var, res, offset, out_jacobian=jac, nnz_offset=_n
            )
            for t, nnz_off in zip(self.terms, nnz_offsets)
        ]

    def _fill_jacobian(self, var: VarValues, residuals: wp.array, need_jacobian: bool) -> None:
        """Fill the shared residual buffer and (optionally) the Jacobian storage, term by term."""
        if self._sparse_jac:
            self._parallel_for_objectives(
                self._sparse_fill_fns, var, residuals, self._sparse_jac_values if need_jacobian else None
            )
            return
        self._parallel_for_objectives(self._residual_fns, var, residuals)
        if need_jacobian:
            self.jacobians.zero_()
            self._parallel_for_objectives(self._jacobian_fns, var, self.jacobians)

    def _reduce_cost_gradient(self, residuals: wp.array, costs: wp.array, gradient: Optional[wp.array]) -> None:
        """Reduce the filled buffers ONCE across all terms to per-row cost and (optional) J^T r."""
        rows = int(residuals.shape[0])
        if not self._sparse_jac:
            wp.launch(
                kernel=aggregate_residuals_to_costs_kernel,
                dim=rows,
                inputs=[residuals, costs],
                device=self.device,
            )
            if gradient is not None:
                wp.launch_tiled(
                    self._dense_jtr_kernel,
                    dim=[rows],
                    inputs=[self.jacobians, residuals],
                    outputs=[gradient],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )
            return
        wp.launch_tiled(
            self._sparse_cost_kernel,
            dim=[rows],
            inputs=[residuals],
            outputs=[costs],
            block_dim=128,
            device=self.device,
        )
        if gradient is None:
            return
        gradient.zero_()
        wp.launch(
            kernel=_sparse_jtr_split_kernel,
            dim=(rows, self._sparse_num_splits),
            inputs=[
                residuals,
                self._sparse_jac_values,
                self._sparse_value_perm,
                self._sparse_row_perm,
                self._sparse_split_cols,
                self._sparse_split_offsets,
            ],
            outputs=[gradient],
            device=self.device,
        )

    def _mask_gradient(self, gradient: wp.array) -> None:
        # Zero locked tangent dims (reduced-DOF subset + lock_endpoints) so no term moves them.
        if self._active_dof_mask is None:
            return
        wp.launch(
            kernel=apply_dof_mask_kernel,
            dim=(int(gradient.shape[0]), self.total_tangent_dim),
            inputs=[gradient, self._active_dof_mask],
            device=self.device,
        )

    def _compute_cost_and_gradient_jacobian(self, var: VarValues) -> None:
        """analytic_jacobian mode: fill J (sparse or dense) -> reduce cost + J^T r -> mask."""
        self._precompute_terms(var, need_gradient=True)
        self._fill_jacobian(var, self.residuals, need_jacobian=True)
        self._reduce_cost_gradient(self.residuals, self.costs, self.gradient)
        self._mask_gradient(self.gradient)
