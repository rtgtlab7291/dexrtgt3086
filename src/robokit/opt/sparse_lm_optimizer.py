# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Sparse Gauss-Newton optimizer using structured CSR format and CG solver."""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.opt.optimizer import OptimizerConfig, aggregate_residuals_to_costs_kernel
from robokit.opt.var_values import VarValues
from robokit.terms.task import SparseTask
from robokit.utils.warp_utils import wp_device_type


# --- sparse Jacobian-transpose product -------------------------------------
# Computes `out += J^T w` in parallel.
# Gathers over the transpose (CSC) pattern in fixed-size per-column splits, so each
# thread issues a single atomic per split instead of one per nonzero. This keeps the
# atomic contention depth low and the kernel's runtime insensitive to where the output
# buffer happens to land in memory (deep same-address atomics are L2-placement-sensitive).
@wp.kernel
def sparse_jacobian_transpose_w_parallel_kernel(
    values: wp.array2d(dtype=wp.float32),
    csc_value_perm: wp.array1d(dtype=wp.int32),
    csc_row_indices: wp.array1d(dtype=wp.int32),
    split_cols: wp.array1d(dtype=wp.int32),
    split_offsets: wp.array1d(dtype=wp.int32),
    w: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, split_idx = wp.tid()  # type: ignore
    acc = wp.float32(0.0)
    for i in range(split_offsets[split_idx], split_offsets[split_idx + 1]):
        acc += values[batch_idx, csc_value_perm[i]] * w[batch_idx, csc_row_indices[i]]
    wp.atomic_add(out, batch_idx, split_cols[split_idx], acc)


# --- sparse Jacobian-vector product ----------------------------------------
# Computes the product in parallel.
@wp.kernel
def sparse_jacobian_mv_parallel_kernel(
    values: wp.array2d(dtype=wp.float32),
    col_indices: wp.array1d(dtype=wp.int32),
    residual_nnz_offsets: wp.array1d(dtype=wp.int32),
    vec: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, res_idx = wp.tid()  # type: ignore

    start = residual_nnz_offsets[res_idx]
    end = residual_nnz_offsets[res_idx + 1]

    acc = wp.float32(0.0)
    for nz_idx in range(start, end):
        col = col_indices[nz_idx]
        acc += values[batch_idx, nz_idx] * vec[batch_idx, col]

    out[batch_idx, res_idx] = acc


# --- conjugate-gradient vector operations ----------------------------------
# Runs the vector operations in parallel.
@wp.kernel
def batch_dot_parallel_kernel(
    vec_a: wp.array2d(dtype=wp.float32),
    vec_b: wp.array2d(dtype=wp.float32),
    out: wp.array1d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    wp.atomic_add(out, batch_idx, vec_a[batch_idx, dof_idx] * vec_b[batch_idx, dof_idx])


@wp.kernel
def init_cg_state_parallel_kernel(
    gradient: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    rTr: wp.array1d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    val = -gradient[batch_idx, dof_idx]
    r[batch_idx, dof_idx] = val
    p[batch_idx, dof_idx] = val
    wp.atomic_add(rTr, batch_idx, val * val)


@wp.kernel
def update_x_and_r_parallel_kernel(
    x: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    Ap: wp.array2d(dtype=wp.float32),
    alpha: wp.array1d(dtype=wp.float32),
    rTr_out: wp.array1d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    a = alpha[batch_idx]

    x_val = x[batch_idx, dof_idx] + a * p[batch_idx, dof_idx]
    r_val = r[batch_idx, dof_idx] - a * Ap[batch_idx, dof_idx]

    x[batch_idx, dof_idx] = x_val
    r[batch_idx, dof_idx] = r_val
    wp.atomic_add(rTr_out, batch_idx, r_val * r_val)


@wp.kernel
def update_p_parallel_kernel(
    p: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    beta: wp.array1d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    b = beta[batch_idx]
    p[batch_idx, dof_idx] = r[batch_idx, dof_idx] + b * p[batch_idx, dof_idx]


@wp.kernel
def compute_cg_alpha_kernel(
    rTr: wp.array1d(dtype=wp.float32),
    pAp: wp.array1d(dtype=wp.float32),
    eps: float,
    alpha: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    alpha[batch_idx] = rTr[batch_idx] / (pAp[batch_idx] + eps)


@wp.kernel
def compute_cg_beta_and_update_rTr_kernel(
    rTr: wp.array1d(dtype=wp.float32),
    new_rTr: wp.array1d(dtype=wp.float32),
    eps: float,
    beta: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    beta[batch_idx] = new_rTr[batch_idx] / (rTr[batch_idx] + eps)
    rTr[batch_idx] = new_rTr[batch_idx]


@wp.kernel
def add_lambda_parallel_kernel(
    vec: wp.array2d(dtype=wp.float32),
    lambda_values: wp.array1d(dtype=wp.float32),
    base: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    out[batch_idx, dof_idx] = base[batch_idx, dof_idx] + lambda_values[batch_idx] * vec[batch_idx, dof_idx]


@wp.kernel
def compute_predicted_reduction_fixed_kernel(
    delta: wp.array2d(dtype=wp.float32),
    gradient: wp.array2d(dtype=wp.float32),
    Av: wp.array2d(dtype=wp.float32),
    out: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    grad_dot = wp.float32(0.0)
    quad_dot = wp.float32(0.0)
    width = delta.shape[1]
    for i in range(width):
        d = delta[batch_idx, i]
        grad_dot += d * gradient[batch_idx, i]
        quad_dot += d * Av[batch_idx, i]
    out[batch_idx] = -grad_dot - wp.float32(0.5) * quad_dot


@wp.kernel
def accept_reject_kernel(
    cost_curr: wp.array1d(dtype=wp.float32),
    cost_prop: wp.array1d(dtype=wp.float32),
    pred_red: wp.array1d(dtype=wp.float32),
    rho_min: float,
    gain_ratio_epsilon: float,
    accept: wp.array1d(dtype=wp.int32),
):
    batch_idx = wp.tid()
    accepted = pred_red[batch_idx] > 0.0
    if accepted:
        rho = (cost_curr[batch_idx] - cost_prop[batch_idx]) / (pred_red[batch_idx] + gain_ratio_epsilon)
        accepted = rho >= rho_min
    accept[batch_idx] = wp.int32(1) if accepted else wp.int32(0)


@wp.kernel
def update_step_kernel(
    accept_flags: wp.array1d(dtype=wp.int32),
    costs_prop: wp.array(dtype=wp.float32),
    proposed_residuals: wp.array2d(dtype=wp.float32),
    lambda_factor: float,
    lambda_min: float,
    lambda_max: float,
    lambda_values: wp.array1d(dtype=wp.float32),
    costs_curr: wp.array(dtype=wp.float32),
    current_residuals: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    accepted = accept_flags[batch_idx] == 1

    if accepted:
        lambda_values[batch_idx] = wp.max(lambda_values[batch_idx] / lambda_factor, lambda_min)
    else:
        new_lambda = lambda_values[batch_idx] * lambda_factor
        lambda_values[batch_idx] = wp.clamp(new_lambda, lambda_min, lambda_max)

    if accepted:
        residual_dim = current_residuals.shape[1]
        for i in range(residual_dim):
            current_residuals[batch_idx, i] = proposed_residuals[batch_idx, i]
        costs_curr[batch_idx] = costs_prop[batch_idx]

    tangent_dim = delta.shape[1]
    mask = wp.float32(1.0) if accepted else wp.float32(0.0)
    for i in range(tangent_dim):
        delta[batch_idx, i] = delta[batch_idx, i] * mask


@dataclass(frozen=True)
class SparseLMOptimizerConfig(OptimizerConfig):
    rho_min: float = 1e-3
    gain_ratio_epsilon: float = 1e-8  # stabilizer in the rho denominator; 0 uses the exact ratio
    # CG convergence tolerance for early stopping
    cg_tol: float = 1e-6
    # check CG convergence every N iterations
    cg_check_interval: int = 4
    # use adaptive CG iteration count
    use_adaptive_cg: bool = True
    # Overrides the auto-derivation below; useful for a warm-started stage already near-converged.
    cg_max_iter_override: Optional[int] = None


@wp.kernel
def zero_masked_jacobian_values_kernel(
    values: wp.array2d(dtype=wp.float32),
    col_indices: wp.array1d(dtype=wp.int32),
    active_dof_mask: wp.array1d(dtype=wp.float32),
):
    batch_idx, nz_idx = wp.tid()  # type: ignore
    col = col_indices[nz_idx]
    values[batch_idx, nz_idx] = values[batch_idx, nz_idx] * active_dof_mask[col]


class SparseLMOptimizer:
    """Sparse Gauss-Newton optimizer using structured CSR format and CG solver."""

    def __init__(
        self,
        term: Union[SparseTask, Sequence[SparseTask]],
        batch_size: int,
        total_tangent_dim: int,
        device: Optional[wp_device_type] = None,
        config: Optional[SparseLMOptimizerConfig] = None,
        active_dof_mask: Optional[wp.array] = None,
    ):
        self.tasks = list(term) if isinstance(term, Sequence) else [term]
        if len(self.tasks) == 0:
            raise ValueError("SparseLMOptimizer requires at least one task.")
        for task in self.tasks:
            if not isinstance(task, SparseTask):
                raise TypeError("SparseLMOptimizer expects SparseTask instances.")

        self.term = self.tasks[0]
        self.batch_size = batch_size
        self.total_tangent_dim = total_tangent_dim
        self.device = wp.get_device(device) if isinstance(device, str) else device
        self.config = config if config is not None else SparseLMOptimizerConfig()
        if active_dof_mask is not None and active_dof_mask.shape != (total_tangent_dim,):
            raise ValueError("active_dof_mask must have shape (total_tangent_dim,).")

        self._use_cuda_graph: bool = self.config.use_cuda_graph and self.device is not None and self.device.is_cuda
        self._cuda_graph: Optional[Any] = None
        self._graph_capture_complete: bool = False

        self.task_residual_offsets = []
        residual_offset = 0
        for task in self.tasks:
            self.task_residual_offsets.append(residual_offset)
            residual_offset += task.residual_dim
        self.total_residual_dim = residual_offset

        self.residuals = wp.empty((self.batch_size, self.total_residual_dim), dtype=wp.float32, device=self.device)
        self.costs = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.proposed_residuals = wp.empty(
            (self.batch_size, self.total_residual_dim), dtype=wp.float32, device=self.device
        )
        self.proposed_costs = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.gradient = wp.empty((self.batch_size, self.total_tangent_dim), dtype=wp.float32, device=self.device)
        self.delta = wp.empty((self.batch_size, self.total_tangent_dim), dtype=wp.float32, device=self.device)
        self.lm = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.accept = wp.zeros((self.batch_size,), dtype=wp.int32, device=self.device)
        self.pred_reduction = wp.zeros((self.batch_size,), dtype=wp.float32, device=self.device)
        self._gain_ratio_epsilon = self.config.gain_ratio_epsilon

        # sparse pattern storage
        self.row_indices: Optional[wp.array] = None
        self.col_indices: Optional[wp.array] = None
        self.values: Optional[wp.array] = None
        self.task_nnz: Optional[List[int]] = None
        self.nnz: int = 0

        self.residual_nnz_offsets: Optional[wp.array] = None

        # transpose (CSC) pattern for J^T products, built alongside the CSR pattern
        self.csc_value_perm: Optional[wp.array] = None
        self.csc_row_indices: Optional[wp.array] = None
        self.jt_split_cols: Optional[wp.array] = None
        self.jt_split_offsets: Optional[wp.array] = None
        self.num_jt_splits: int = 0

        self.jx = wp.empty((self.batch_size, self.total_residual_dim), dtype=wp.float32, device=self.device)
        self.jt_jx = wp.empty((self.batch_size, self.total_tangent_dim), dtype=wp.float32, device=self.device)

        self.cg_r = wp.empty((self.batch_size, self.total_tangent_dim), dtype=wp.float32, device=self.device)
        self.cg_p = wp.empty((self.batch_size, self.total_tangent_dim), dtype=wp.float32, device=self.device)
        self.cg_Ap = wp.empty((self.batch_size, self.total_tangent_dim), dtype=wp.float32, device=self.device)

        self.cg_rTr = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.cg_new_rTr = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.cg_pAp = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)

        self.cg_alpha = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.cg_beta = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)

        if self.config.cg_max_iter_override is not None:
            self.cg_max_iter = self.config.cg_max_iter_override
        elif self.config.use_adaptive_cg:
            self.cg_max_iter = max(4, min(64, self.total_tangent_dim // 4))
        else:
            self.cg_max_iter = max(4, min(128, self.total_tangent_dim))
        self._cg_eps = 1e-8

        self._active_dof_mask = active_dof_mask
        self._proposed_var: Optional[VarValues] = None

    def _validate_residual_layout(self):
        residual_dims = [task.residual_dim for task in self.tasks]
        total_dim = int(sum(residual_dims))
        if total_dim == self.total_residual_dim:
            return
        term_desc = ", ".join(
            f"[{idx}] {type(task).__name__} residual_dim={dim}"
            for idx, (task, dim) in enumerate(zip(self.tasks, residual_dims))
        )
        raise ValueError(
            "SparseLMOptimizer residual layout mismatch: "
            f"expected total_residual_dim={self.total_residual_dim}, got {total_dim}. "
            f"TaskDims=[{term_desc}]"
        )

    def _build_sparse_pattern(self, var: VarValues):
        """Build CSR-like sparse pattern with per-residual offsets for parallel access."""
        task_nnz: List[int] = []
        total_nnz = 0

        for task, row_offset in zip(self.tasks, self.task_residual_offsets):
            pattern = task.compute_sparse_jacobian_pattern(var, offset=row_offset)
            nnz = pattern.col_indices.shape[0]
            task_nnz.append(nnz)
            total_nnz += nnz

        self.row_indices = wp.empty(total_nnz, dtype=wp.int32, device=self.device)
        self.col_indices = wp.empty(total_nnz, dtype=wp.int32, device=self.device)
        self.values = wp.empty((self.batch_size, total_nnz), dtype=wp.float32, device=self.device)

        nnz_offset = 0
        for task, row_offset, nnz in zip(self.tasks, self.task_residual_offsets, task_nnz):
            task.compute_sparse_jacobian_pattern(
                var,
                offset=row_offset,
                out_row_indices=self.row_indices,
                out_col_indices=self.col_indices,
                nnz_offset=nnz_offset,
            )
            nnz_offset += nnz

        self.task_nnz = task_nnz
        self.nnz = total_nnz

        col_np = self.col_indices.numpy()
        row_np = self.row_indices.numpy()

        # Row widths come from the pattern itself, so a row may touch any number of tangent dims (a
        # link deep in the chain has more ancestor DOFs than one near the root). The only requirement
        # is that each row's entries are contiguous, i.e. tasks emit their rows in order.
        if np.any(np.diff(row_np) < 0):
            raise ValueError(
                "SparseLMOptimizer requires each task to emit sparsity rows in non-decreasing order; "
                f"tasks=[{', '.join(type(t).__name__ for t in self.tasks)}]"
            )
        residual_offsets_np = np.zeros(self.total_residual_dim + 1, dtype=np.int32)
        residual_offsets_np[1:] = np.cumsum(np.bincount(row_np, minlength=self.total_residual_dim))
        self.residual_nnz_offsets = wp.from_numpy(residual_offsets_np, dtype=wp.int32, device=self.device)

        # Transpose (CSC) pattern: sort nnz by column, then cut each column's run into
        # splits of at most `split_size` nonzeros (one atomic per split in the kernel).
        split_size = 8
        order = np.argsort(col_np, kind="stable")
        cols_sorted = col_np[order]
        col_starts = np.flatnonzero(np.r_[True, cols_sorted[1:] != cols_sorted[:-1]])
        split_starts = np.concatenate(
            [np.arange(s, e, split_size) for s, e in zip(col_starts, np.r_[col_starts[1:], total_nnz])]
        )
        self.csc_value_perm = wp.from_numpy(order.astype(np.int32), dtype=wp.int32, device=self.device)
        self.csc_row_indices = wp.from_numpy(row_np[order], dtype=wp.int32, device=self.device)
        self.jt_split_cols = wp.from_numpy(cols_sorted[split_starts], dtype=wp.int32, device=self.device)
        self.jt_split_offsets = wp.from_numpy(
            np.append(split_starts, total_nnz).astype(np.int32), dtype=wp.int32, device=self.device
        )
        self.num_jt_splits = split_starts.shape[0]

    def _ensure_sparse_pattern(self, var: VarValues):
        """Ensure sparse pattern is built and values are up-to-date."""
        if (
            self.row_indices is None
            or self.col_indices is None
            or self.values is None
            or self.task_nnz is None
            or self.residual_nnz_offsets is None
        ):
            self._build_sparse_pattern(var)

        assert self.values is not None
        for task in self.tasks:
            task.precompute(var, need_gradient=True)
        nnz_offset = 0
        for task, row_offset, nnz in zip(self.tasks, self.task_residual_offsets, self.task_nnz):
            task.compute_weighted_sparse_jacobian_values(
                var, out_jacobian_values=self.values, offset=row_offset, nnz_offset=nnz_offset
            )
            nnz_offset += nnz

        if self._active_dof_mask is not None:
            assert self.col_indices is not None
            wp.launch(
                kernel=zero_masked_jacobian_values_kernel,
                dim=(self.batch_size, self.nnz),
                inputs=[self.values, self.col_indices, self._active_dof_mask],
                device=self.device,
            )

    def _fill_residuals(self, var: VarValues, buffer: wp.array):
        """Fill residual buffer from all tasks."""
        for task in self.tasks:
            task.precompute(var, need_gradient=False)
        for task, row_offset in zip(self.tasks, self.task_residual_offsets):
            task.compute_weighted_residual(var, out_residual=buffer, row_offset=row_offset)

    def _compute_gradient(self):
        """Compute gradient = J^T * r using parallelized kernel."""
        self.gradient.zero_()
        assert self.values is not None
        assert self.col_indices is not None
        assert self.residual_nnz_offsets is not None

        wp.launch(
            kernel=sparse_jacobian_transpose_w_parallel_kernel,
            dim=(self.batch_size, self.num_jt_splits),
            inputs=[
                self.values,
                self.csc_value_perm,
                self.csc_row_indices,
                self.jt_split_cols,
                self.jt_split_offsets,
                self.residuals,
            ],
            outputs=[self.gradient],
            device=self.device,
        )

    def _apply_linear_operator(self, vec: wp.array, out: wp.array):
        """Compute out = (J^T J + lambda I) * vec using parallelized kernels."""
        self.jx.zero_()
        self.jt_jx.zero_()

        assert self.values is not None
        assert self.col_indices is not None
        assert self.residual_nnz_offsets is not None

        wp.launch(
            kernel=sparse_jacobian_mv_parallel_kernel,
            dim=(self.batch_size, self.total_residual_dim),
            inputs=[self.values, self.col_indices, self.residual_nnz_offsets, vec],
            outputs=[self.jx],
            device=self.device,
        )

        wp.launch(
            kernel=sparse_jacobian_transpose_w_parallel_kernel,
            dim=(self.batch_size, self.num_jt_splits),
            inputs=[
                self.values,
                self.csc_value_perm,
                self.csc_row_indices,
                self.jt_split_cols,
                self.jt_split_offsets,
                self.jx,
            ],
            outputs=[self.jt_jx],
            device=self.device,
        )

        wp.launch(
            kernel=add_lambda_parallel_kernel,
            dim=(self.batch_size, self.total_tangent_dim),
            inputs=[vec, self.lm, self.jt_jx],
            outputs=[out],
            device=self.device,
        )

    def _compute_sparse_delta(self, skip_early_stop: bool = False):
        """Solve (J^T J + lambda I) delta = -g using parallelized CG."""
        self.delta.zero_()
        self.cg_rTr.zero_()

        wp.launch(
            kernel=init_cg_state_parallel_kernel,
            dim=(self.batch_size, self.total_tangent_dim),
            inputs=[self.gradient],
            outputs=[self.cg_r, self.cg_p, self.cg_rTr],
            device=self.device,
        )

        for cg_iter in range(self.cg_max_iter):
            self._apply_linear_operator(self.cg_p, self.cg_Ap)

            self.cg_pAp.zero_()
            wp.launch(
                kernel=batch_dot_parallel_kernel,
                dim=(self.batch_size, self.total_tangent_dim),
                inputs=[self.cg_p, self.cg_Ap],
                outputs=[self.cg_pAp],
                device=self.device,
            )

            wp.launch(
                kernel=compute_cg_alpha_kernel,
                dim=self.batch_size,
                inputs=[self.cg_rTr, self.cg_pAp, self._cg_eps],
                outputs=[self.cg_alpha],
                device=self.device,
            )

            self.cg_new_rTr.zero_()
            wp.launch(
                kernel=update_x_and_r_parallel_kernel,
                dim=(self.batch_size, self.total_tangent_dim),
                inputs=[self.delta, self.cg_p, self.cg_r, self.cg_Ap, self.cg_alpha],
                outputs=[self.cg_new_rTr],
                device=self.device,
            )

            if not skip_early_stop and cg_iter % self.config.cg_check_interval == 0 and cg_iter > 0:
                max_residual = self.cg_new_rTr.numpy().max()
                if max_residual < self.config.cg_tol:
                    break

            wp.launch(
                kernel=compute_cg_beta_and_update_rTr_kernel,
                dim=self.batch_size,
                inputs=[self.cg_rTr, self.cg_new_rTr, self._cg_eps],
                outputs=[self.cg_beta],
                device=self.device,
            )

            wp.launch(
                kernel=update_p_parallel_kernel,
                dim=(self.batch_size, self.total_tangent_dim),
                inputs=[self.cg_p, self.cg_r, self.cg_beta],
                device=self.device,
            )

        # pred is the reduction of the *undamped* Gauss-Newton model, -gᵀδ - ½δᵀJᵀJδ, matching
        # LMOptimizer. _apply_linear_operator leaves JᵀJδ in jt_jx; cg_Ap carries the extra λδ.
        self._apply_linear_operator(self.delta, self.cg_Ap)
        wp.launch(
            kernel=compute_predicted_reduction_fixed_kernel,
            dim=self.batch_size,
            inputs=[self.delta, self.gradient, self.jt_jx],
            outputs=[self.pred_reduction],
            device=self.device,
        )

    def _warmup_and_capture(self, var: VarValues):
        var.invalidate()
        self._solve_impl(var, skip_early_stop=True)

        wp.synchronize()

        var.invalidate()
        with wp.ScopedCapture(device=self.device) as capture:
            self._solve_impl(var, skip_early_stop=True)
        self._cuda_graph = capture.graph

        self._graph_capture_complete = True

    def _solve_impl(self, var: VarValues, skip_early_stop: bool = False) -> VarValues:
        """Internal solve implementation.

        Args:
            var: Initial variable values.

        Returns:
            Optimized variable values and their final aggregate costs.
        """
        if self._proposed_var is None:
            self._proposed_var = var.clone()
        proposed_var = self._proposed_var
        self.lm.fill_(self.config.lm_lambda)
        self._fill_residuals(var, self.residuals)
        wp.launch(
            kernel=aggregate_residuals_to_costs_kernel,
            dim=self.batch_size,
            inputs=[self.residuals, self.costs],
            device=self.device,
        )

        for _ in range(self.config.max_iter):
            self._ensure_sparse_pattern(var)
            self._compute_gradient()
            self._compute_sparse_delta(skip_early_stop=skip_early_stop or not self.config.use_early_stopping)
            proposed_var = var.integrate(self.delta, out=proposed_var, tangent_mask=self._active_dof_mask)
            self._fill_residuals(proposed_var, self.proposed_residuals)
            wp.launch(
                kernel=aggregate_residuals_to_costs_kernel,
                dim=self.batch_size,
                inputs=[self.proposed_residuals, self.proposed_costs],
                device=self.device,
            )
            wp.launch(
                kernel=accept_reject_kernel,
                dim=self.batch_size,
                inputs=[
                    self.costs,
                    self.proposed_costs,
                    self.pred_reduction,
                    self.config.rho_min,
                    self._gain_ratio_epsilon,
                ],
                outputs=[self.accept],
                device=self.device,
            )
            wp.launch(
                kernel=update_step_kernel,
                dim=self.batch_size,
                inputs=[
                    self.accept,
                    self.proposed_costs,
                    self.proposed_residuals,
                    self.config.lambda_factor,
                    self.config.lambda_min,
                    self.config.lambda_max,
                ],
                outputs=[self.lm, self.costs, self.residuals, self.delta],
                device=self.device,
            )
            var = var.integrate(self.delta, out=var, tangent_mask=self._active_dof_mask)

        return var

    def solve(self, var: VarValues) -> Tuple[VarValues, wp.array]:
        """Solve the optimization problem.

        Args:
            var: Initial variable values.

        Returns:
            Optimized variable values.
        """
        self._validate_residual_layout()
        if self._use_cuda_graph:
            if not self._graph_capture_complete:
                self._warmup_and_capture(var)
            var.invalidate()
            wp.capture_launch(self._cuda_graph)
            return var, self.costs
        self._solve_impl(var)
        return var, self.costs
