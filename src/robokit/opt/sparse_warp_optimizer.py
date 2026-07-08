# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Sparse Gauss-Newton optimizer using structured CSR format and CG solver."""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.opt.optimizer import OptimizerConfig
from robokit.opt.variables import WarpVar
from robokit.terms.terms import SparseWarpTask
from robokit.utils.warp_utils import wp_device_type


_ACCEPTANCE_MODE_STRICT_MONOTONIC = 0
_ACCEPTANCE_MODE_RHO_ONLY = 1


def _encode_acceptance_mode(mode: Literal["strict_monotonic", "rho_only"]) -> int:
    if mode == "strict_monotonic":
        return _ACCEPTANCE_MODE_STRICT_MONOTONIC
    if mode == "rho_only":
        return _ACCEPTANCE_MODE_RHO_ONLY
    raise ValueError(f"Unknown acceptance mode: {mode}")


@wp.kernel
def aggregate_residuals_to_costs(
    residuals: wp.array2d(dtype=wp.float32),
    costs: wp.array(dtype=wp.float32),
):
    batch_idx = wp.tid()
    cost = wp.float32(0.0)
    for i in range(residuals.shape[1]):
        r = residuals[batch_idx, i]
        cost += r * r
    costs[batch_idx] = wp.float32(0.5) * cost


# --- Parallelized gradient accumulation ---
@wp.kernel
def accumulate_gradient_parallel(
    residuals: wp.array2d(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    col_indices: wp.array1d(dtype=wp.int32),
    residual_nnz_offsets: wp.array1d(dtype=wp.int32),
    gradient: wp.array2d(dtype=wp.float32),
):
    batch_idx, res_idx = wp.tid()  # type: ignore
    r_val = residuals[batch_idx, res_idx]

    start = residual_nnz_offsets[res_idx]
    end = residual_nnz_offsets[res_idx + 1]

    for nz_idx in range(start, end):
        col = col_indices[nz_idx]
        val = values[batch_idx, nz_idx]
        wp.atomic_add(gradient, batch_idx, col, val * r_val)


# --- Parallelized sparse Jacobian-vector product ---
@wp.kernel
def sparse_jacobian_mv_parallel(
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


# --- Parallelized sparse Jacobian-transpose-vector product ---
@wp.kernel
def sparse_jacobian_transpose_mv_parallel(
    values: wp.array2d(dtype=wp.float32),
    col_indices: wp.array1d(dtype=wp.int32),
    residual_nnz_offsets: wp.array1d(dtype=wp.int32),
    vec: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, res_idx = wp.tid()  # type: ignore
    v_val = vec[batch_idx, res_idx]

    start = residual_nnz_offsets[res_idx]
    end = residual_nnz_offsets[res_idx + 1]

    for nz_idx in range(start, end):
        col = col_indices[nz_idx]
        val = values[batch_idx, nz_idx]
        wp.atomic_add(out, batch_idx, col, val * v_val)


# --- Parallelized CG vector operations ---
@wp.kernel
def batch_dot_parallel(
    vec_a: wp.array2d(dtype=wp.float32),
    vec_b: wp.array2d(dtype=wp.float32),
    out: wp.array1d(dtype=wp.float32),
):
    batch_idx, i = wp.tid()  # type: ignore
    wp.atomic_add(out, batch_idx, vec_a[batch_idx, i] * vec_b[batch_idx, i])


@wp.kernel
def init_cg_state_parallel(
    gradient: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    rTr: wp.array1d(dtype=wp.float32),
):
    batch_idx, i = wp.tid()  # type: ignore
    val = -gradient[batch_idx, i]
    r[batch_idx, i] = val
    p[batch_idx, i] = val
    wp.atomic_add(rTr, batch_idx, val * val)


@wp.kernel
def update_x_and_r_parallel(
    x: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    Ap: wp.array2d(dtype=wp.float32),
    alpha: wp.array1d(dtype=wp.float32),
    rTr_out: wp.array1d(dtype=wp.float32),
):
    batch_idx, i = wp.tid()  # type: ignore
    a = alpha[batch_idx]

    x_val = x[batch_idx, i] + a * p[batch_idx, i]
    r_val = r[batch_idx, i] - a * Ap[batch_idx, i]

    x[batch_idx, i] = x_val
    r[batch_idx, i] = r_val
    wp.atomic_add(rTr_out, batch_idx, r_val * r_val)


@wp.kernel
def update_p_parallel(
    p: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    beta: wp.array1d(dtype=wp.float32),
):
    batch_idx, i = wp.tid()  # type: ignore
    b = beta[batch_idx]
    p[batch_idx, i] = r[batch_idx, i] + b * p[batch_idx, i]


@wp.kernel
def compute_cg_alpha(
    rTr: wp.array1d(dtype=wp.float32),
    pAp: wp.array1d(dtype=wp.float32),
    eps: float,
    alpha: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    alpha[batch_idx] = rTr[batch_idx] / (pAp[batch_idx] + eps)


@wp.kernel
def compute_cg_beta_and_update_rTr(
    rTr: wp.array1d(dtype=wp.float32),
    new_rTr: wp.array1d(dtype=wp.float32),
    eps: float,
    beta: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    beta[batch_idx] = new_rTr[batch_idx] / (rTr[batch_idx] + eps)
    rTr[batch_idx] = new_rTr[batch_idx]


@wp.kernel
def add_lambda_parallel(
    vec: wp.array2d(dtype=wp.float32),
    lambda_values: wp.array1d(dtype=wp.float32),
    base: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, idx = wp.tid()  # type: ignore
    out[batch_idx, idx] = base[batch_idx, idx] + lambda_values[batch_idx] * vec[batch_idx, idx]


@wp.kernel
def compute_predicted_reduction_fixed(
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
def accept_reject(
    cost_curr: wp.array1d(dtype=wp.float32),
    cost_prop: wp.array1d(dtype=wp.float32),
    pred_red: wp.array1d(dtype=wp.float32),
    rho_min: float,
    acceptance_mode: int,
    accept: wp.array1d(dtype=wp.int32),
):
    problem_idx = wp.tid()
    rho = (cost_curr[problem_idx] - cost_prop[problem_idx]) / (pred_red[problem_idx] + 1e-8)
    rho_only_accept = rho >= rho_min
    strict_accept = rho_only_accept and pred_red[problem_idx] > 0.0 and cost_prop[problem_idx] <= cost_curr[problem_idx]
    accepted = strict_accept if acceptance_mode == _ACCEPTANCE_MODE_STRICT_MONOTONIC else rho_only_accept
    accept[problem_idx] = wp.int32(1) if accepted else wp.int32(0)


@wp.kernel
def update_step(
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
    problem_idx = wp.tid()
    accepted = accept_flags[problem_idx] == 1

    if accepted:
        lambda_values[problem_idx] = wp.max(lambda_values[problem_idx] / lambda_factor, lambda_min)
    else:
        new_lambda = lambda_values[problem_idx] * lambda_factor
        lambda_values[problem_idx] = wp.clamp(new_lambda, lambda_min, lambda_max)

    if accepted:
        residual_dim = current_residuals.shape[1]
        for i in range(residual_dim):
            current_residuals[problem_idx, i] = proposed_residuals[problem_idx, i]
        costs_curr[problem_idx] = costs_prop[problem_idx]

    tangent_dim = delta.shape[1]
    mask = wp.float32(1.0) if accepted else wp.float32(0.0)
    for i in range(tangent_dim):
        delta[problem_idx, i] = delta[problem_idx, i] * mask


@dataclass(frozen=True)
class SparseWarpOptimizerConfig(OptimizerConfig):
    rho_min: float = 1e-3
    acceptance_mode: Literal["strict_monotonic", "rho_only"] = "strict_monotonic"
    # CG convergence tolerance for early stopping
    cg_tol: float = 1e-6
    # Check CG convergence every N iterations
    cg_check_interval: int = 4
    # Use adaptive CG iteration count
    use_adaptive_cg: bool = True


class SparseWarpOptimizer:
    """Sparse Gauss-Newton optimizer using structured CSR format and CG solver."""

    def __init__(
        self,
        term: Union[SparseWarpTask, Sequence[SparseWarpTask]],
        batch_size: int,
        total_tangent_dim: int,
        device: Optional[wp_device_type] = None,
        config: Optional[SparseWarpOptimizerConfig] = None,
        use_cuda_graph: bool = False,
        placeholder_var: Optional[WarpVar] = None,
    ):
        self.tasks = list(term) if isinstance(term, Sequence) else [term]
        if len(self.tasks) == 0:
            raise ValueError("SparseWarpOptimizer requires at least one task.")
        for task in self.tasks:
            if not isinstance(task, SparseWarpTask):
                raise TypeError("SparseWarpOptimizer expects SparseWarpTask instances.")

        self.term = self.tasks[0]
        self.batch_size = batch_size
        self.total_tangent_dim = total_tangent_dim
        self.device = device
        self.config = config if config is not None else SparseWarpOptimizerConfig()

        self._use_cuda_graph: bool = use_cuda_graph
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
        self.accept_count = 0
        self.reject_count = 0
        self._acceptance_mode = _encode_acceptance_mode(self.config.acceptance_mode)

        # Sparse pattern storage
        self.row_indices: Optional[wp.array] = None
        self.col_indices: Optional[wp.array] = None
        self.values: Optional[wp.array] = None
        self.task_nnz: Optional[List[int]] = None
        self.task_nnz_per_residual: Optional[List[int]] = None
        self.nnz: int = 0

        self.residual_nnz_offsets: Optional[wp.array] = None

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

        if self.config.use_adaptive_cg:
            self.cg_max_iter = max(4, min(64, self.total_tangent_dim // 4))
        else:
            self.cg_max_iter = max(4, min(128, self.total_tangent_dim))
        self._cg_eps = 1e-8

        if self._use_cuda_graph and placeholder_var is not None and self.device is not None and self.device.is_cuda:
            self._warmup_and_capture(placeholder_var)

    def _validate_residual_layout(self) -> None:
        residual_dims = [task.residual_dim for task in self.tasks]
        total_dim = int(sum(residual_dims))
        if total_dim == self.total_residual_dim:
            return
        term_desc = ", ".join(
            f"[{idx}] {type(task).__name__} residual_dim={dim}"
            for idx, (task, dim) in enumerate(zip(self.tasks, residual_dims))
        )
        raise ValueError(
            "SparseWarpOptimizer residual layout mismatch: "
            f"expected total_residual_dim={self.total_residual_dim}, got {total_dim}. "
            f"TaskDims=[{term_desc}]"
        )

    def _build_sparse_pattern(self, var: WarpVar) -> None:
        """Build CSR-like sparse pattern with per-residual offsets for parallel access."""
        task_nnz: List[int] = []
        task_nnz_per_residual: List[int] = []
        total_nnz = 0

        for task, row_offset in zip(self.tasks, self.task_residual_offsets):
            pattern = task.compute_sparse_jacobian_pattern(var, offset=row_offset)
            nnz = pattern.col_indices.shape[0]
            if nnz % task.residual_dim != 0:
                raise ValueError("SparseWarpOptimizer requires constant nnz per residual for each task.")
            nnz_per_residual = nnz // task.residual_dim

            task_nnz.append(nnz)
            task_nnz_per_residual.append(nnz_per_residual)
            total_nnz += nnz

        self.row_indices = wp.empty(total_nnz, dtype=wp.int32, device=self.device)
        self.col_indices = wp.empty(total_nnz, dtype=wp.int32, device=self.device)
        self.values = wp.empty((self.batch_size, total_nnz), dtype=wp.float32, device=self.device)

        nnz_offset = 0
        for task, row_offset, nnz in zip(self.tasks, self.task_residual_offsets, task_nnz):
            task.compute_sparse_jacobian_pattern(
                var,
                offset=row_offset,
                row_indices_buffer=self.row_indices,
                col_indices_buffer=self.col_indices,
                nnz_offset=nnz_offset,
            )
            nnz_offset += nnz

        self.task_nnz = task_nnz
        self.task_nnz_per_residual = task_nnz_per_residual
        self.nnz = total_nnz

        residual_offsets_np = np.zeros(self.total_residual_dim + 1, dtype=np.int32)
        offset = 0
        res_idx = 0
        for task, nnz_per_res in zip(self.tasks, task_nnz_per_residual):
            for _ in range(task.residual_dim):
                residual_offsets_np[res_idx] = offset
                offset += nnz_per_res
                res_idx += 1
        residual_offsets_np[res_idx] = offset  # Final element is total nnz

        self.residual_nnz_offsets = wp.from_numpy(residual_offsets_np, dtype=wp.int32, device=self.device)

    def _ensure_sparse_pattern(self, var: WarpVar):
        """Ensure sparse pattern is built and values are up-to-date."""
        if (
            self.row_indices is None
            or self.col_indices is None
            or self.values is None
            or self.task_nnz is None
            or self.task_nnz_per_residual is None
            or self.residual_nnz_offsets is None
        ):
            self._build_sparse_pattern(var)

        assert self.values is not None
        nnz_offset = 0
        for task, row_offset, nnz in zip(self.tasks, self.task_residual_offsets, self.task_nnz):
            task.compute_weighted_sparse_jacobian_values(
                var, jacobian_values_buffer=self.values, offset=row_offset, nnz_offset=nnz_offset
            )
            nnz_offset += nnz

    def _fill_residuals(self, var: WarpVar, buffer: wp.array) -> None:
        """Fill residual buffer from all tasks."""
        for task, row_offset in zip(self.tasks, self.task_residual_offsets):
            task.compute_weighted_residual(var, residual_buffer=buffer, row_offset=row_offset)

    def _compute_gradient(self):
        """Compute gradient = J^T * r using parallelized kernel."""
        self.gradient.zero_()
        assert self.values is not None
        assert self.col_indices is not None
        assert self.residual_nnz_offsets is not None

        wp.launch(
            kernel=accumulate_gradient_parallel,
            dim=(self.batch_size, self.total_residual_dim),
            inputs=[self.residuals, self.values, self.col_indices, self.residual_nnz_offsets],
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
            kernel=sparse_jacobian_mv_parallel,
            dim=(self.batch_size, self.total_residual_dim),
            inputs=[self.values, self.col_indices, self.residual_nnz_offsets, vec],
            outputs=[self.jx],
            device=self.device,
        )

        wp.launch(
            kernel=sparse_jacobian_transpose_mv_parallel,
            dim=(self.batch_size, self.total_residual_dim),
            inputs=[self.values, self.col_indices, self.residual_nnz_offsets, self.jx],
            outputs=[self.jt_jx],
            device=self.device,
        )

        wp.launch(
            kernel=add_lambda_parallel,
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
            kernel=init_cg_state_parallel,
            dim=(self.batch_size, self.total_tangent_dim),
            inputs=[self.gradient],
            outputs=[self.cg_r, self.cg_p, self.cg_rTr],
            device=self.device,
        )

        self._cg_iters_used = 0  # Track actual iterations for diagnostics
        for cg_iter in range(self.cg_max_iter):
            self._cg_iters_used = cg_iter + 1
            self._apply_linear_operator(self.cg_p, self.cg_Ap)

            self.cg_pAp.zero_()
            wp.launch(
                kernel=batch_dot_parallel,
                dim=(self.batch_size, self.total_tangent_dim),
                inputs=[self.cg_p, self.cg_Ap],
                outputs=[self.cg_pAp],
                device=self.device,
            )

            wp.launch(
                kernel=compute_cg_alpha,
                dim=self.batch_size,
                inputs=[self.cg_rTr, self.cg_pAp, self._cg_eps],
                outputs=[self.cg_alpha],
                device=self.device,
            )

            self.cg_new_rTr.zero_()
            wp.launch(
                kernel=update_x_and_r_parallel,
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
                kernel=compute_cg_beta_and_update_rTr,
                dim=self.batch_size,
                inputs=[self.cg_rTr, self.cg_new_rTr, self._cg_eps],
                outputs=[self.cg_beta],
                device=self.device,
            )

            wp.launch(
                kernel=update_p_parallel,
                dim=(self.batch_size, self.total_tangent_dim),
                inputs=[self.cg_p, self.cg_r, self.cg_beta],
                device=self.device,
            )

        self._apply_linear_operator(self.delta, self.cg_Ap)
        wp.launch(
            kernel=compute_predicted_reduction_fixed,
            dim=self.batch_size,
            inputs=[self.delta, self.gradient, self.cg_Ap],
            outputs=[self.pred_reduction],
            device=self.device,
        )

    def _warmup_and_capture(self, var: WarpVar) -> None:
        var.invalidate()
        self._solve_impl(var, skip_early_stop=True)

        wp.synchronize()

        var.invalidate()
        with wp.ScopedCapture(device=self.device) as capture:
            self._solve_impl(var, skip_early_stop=True)
        self._cuda_graph = capture.graph

        self._graph_capture_complete = True

    def _solve_impl(self, var: WarpVar, skip_early_stop: bool = False) -> WarpVar:
        """Internal solve implementation.

        Args:
            var: Initial robot state

        Returns:
            Optimized robot state
        """
        self._validate_residual_layout()
        proposed_var = var.clone()
        self.lm.fill_(self.config.lm_lambda)
        self._fill_residuals(var, self.residuals)
        wp.launch(
            kernel=aggregate_residuals_to_costs,
            dim=self.batch_size,
            inputs=[self.residuals, self.costs],
            device=self.device,
        )

        iterations_run = 0
        for _ in range(self.config.max_iter):
            iterations_run += 1
            self._ensure_sparse_pattern(var)
            self._compute_gradient()
            self._compute_sparse_delta(skip_early_stop=skip_early_stop)
            proposed_var = var.integrate(self.delta, out=proposed_var)
            self._fill_residuals(proposed_var, self.proposed_residuals)
            wp.launch(
                kernel=aggregate_residuals_to_costs,
                dim=self.batch_size,
                inputs=[self.proposed_residuals, self.proposed_costs],
                device=self.device,
            )
            wp.launch(
                kernel=accept_reject,
                dim=self.batch_size,
                inputs=[
                    self.costs,
                    self.proposed_costs,
                    self.pred_reduction,
                    self.config.rho_min,
                    self._acceptance_mode,
                ],
                outputs=[self.accept],
                device=self.device,
            )
            wp.launch(
                kernel=update_step,
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
            var = var.integrate(self.delta, out=var)

        if not skip_early_stop:
            self._update_accept_counts(iterations_run)
        return var

    def _update_accept_counts(self, iterations_run: int) -> None:
        if iterations_run <= 0:
            self.accept_count = 0
            self.reject_count = 0
            return
        self.accept_count = int(np.sum(self.accept.numpy()))
        self.reject_count = self.batch_size - self.accept_count

    def solve(self, var: WarpVar) -> WarpVar:
        """Solve the optimization problem.

        Args:
            var: Initial robot state

        Returns:
            Optimized robot state
        """
        self._validate_residual_layout()
        use_graph = (
            self._use_cuda_graph and self.device is not None and self.device.is_cuda and self._graph_capture_complete
        )

        if use_graph and self._cuda_graph is not None:
            var.invalidate()
            wp.capture_launch(self._cuda_graph)
            iterations_run = 1 if self.config.max_iter > 0 else 0
            self._update_accept_counts(iterations_run)
            return var
        else:
            return self._solve_impl(var)
