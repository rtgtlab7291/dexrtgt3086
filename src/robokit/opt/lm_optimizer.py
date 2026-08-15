# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
"""Generalized Levenberg-Marquardt optimizer over normal equations.

Fast path (CUDA + all default Jacobian terms): default terms write their
residual and Jacobian to slices of a shared wide `J`/`r` buffer (in
parallel per-term streams when ≥2 terms). `_solve_kernel` turns wide
`J`/`r` + λ into δ + pred_reduction + costs in one launch (`JᵀJ`/`Jᵀr`
stay tile-local); `_accept_kernel` fuses cost-reduce with accept/reject.

Fallback path (streaming terms, CPU, or other): each term contributes
additively to global `JᵀJ`/`Jᵀr`/`costs` via `accumulate_normal_equations`,
then `_cholesky_kernel` computes δ; the accept side uses
`lm_accept_reject_cost_kernel`. Supports fused (one graph) and split (two graphs)
CUDA-graph capture modes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.opt.optimizer import Optimizer, OptimizerConfig
from robokit.opt.var_values import VarValues
from robokit.terms.task import EagerTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type


TILE_THREADS = 32
# Above _LM_FUSED_MAX_ROWS the single-tile solve overflows shared memory; stream J in _LM_SOLVE_CHUNK rows.
_LM_SOLVE_CHUNK = 64
_LM_FUSED_MAX_ROWS = 512
_KERNEL_CACHE: Dict[tuple, Any] = {}


def _cholesky_kernel(D: int):
    """Reads `JᵀJ` + `Jᵀr` + λ, computes `(JᵀJ+λI)δ = -Jᵀr` and pred_reduction."""
    key = ("chol", D)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Dc = wp.constant(D)

    def cholesky(
        JtJ: wp.array3d(dtype=wp.float32),
        Jtr: wp.array3d(dtype=wp.float32),
        lm_lambda: wp.array1d(dtype=wp.float32),
        delta_out: wp.array2d(dtype=wp.float32),
        pred_out: wp.array1d(dtype=wp.float32),
    ):
        batch_idx = wp.tid()
        lam = lm_lambda[batch_idx]
        A = wp.tile_load(JtJ[batch_idx], shape=(Dc, Dc))
        diag = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            diag[i] = lam
        A = wp.tile_diag_add(A, diag)
        g_col = wp.tile_load(Jtr[batch_idx], shape=(Dc, 1))
        g = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            g[i] = g_col[i, 0]
        delta = wp.tile_cholesky_solve(wp.tile_cholesky(A), wp.tile_map(wp.neg, g))
        wp.tile_store(delta_out[batch_idx], delta)
        ld = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            ld[i] = lam * delta[i]
        pred_out[batch_idx] = wp.float32(0.5) * wp.tile_sum(wp.tile_map(wp.mul, delta, wp.tile_map(wp.sub, ld, g)))[0]

    cholesky.__name__ = cholesky.__qualname__ = f"_lm_chol_{D}"
    k = wp.kernel(enable_backward=False, module="unique")(cholesky)
    _KERNEL_CACHE[key] = k
    return k


def _solve_kernel(R: int, D: int):
    """Fused fast-path kernel: wide `J`/`r` + λ → δ, pred_reduction, costs.

    Keeps `JᵀJ` / `Jᵀr` tile-local (no D×D round-trip). Only valid when
    there are no streaming terms.
    """
    key = ("solve", R, D)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Rc, Dc = wp.constant(R), wp.constant(D)

    def solve(
        jacobians: wp.array3d(dtype=wp.float32),
        residuals_col: wp.array3d(dtype=wp.float32),
        lm_lambda: wp.array1d(dtype=wp.float32),
        costs_out: wp.array1d(dtype=wp.float32),
        delta_out: wp.array2d(dtype=wp.float32),
        pred_out: wp.array1d(dtype=wp.float32),
    ):
        batch_idx = wp.tid()
        lam = lm_lambda[batch_idx]
        J = wp.tile_load(jacobians[batch_idx], shape=(Rc, Dc))
        r = wp.tile_load(residuals_col[batch_idx], shape=(Rc, 1))
        Jt = wp.tile_transpose(J)
        H = wp.tile_zeros(shape=(Dc, Dc), dtype=wp.float32)
        wp.tile_matmul(Jt, J, H)
        diag = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            diag[i] = lam
        A = wp.tile_diag_add(H, diag)
        g_col = wp.tile_zeros(shape=(Dc, 1), dtype=wp.float32)
        wp.tile_matmul(Jt, r, g_col)
        g = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            g[i] = g_col[i, 0]
        delta = wp.tile_cholesky_solve(wp.tile_cholesky(A), wp.tile_map(wp.neg, g))
        wp.tile_store(delta_out[batch_idx], delta)
        ld = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            ld[i] = lam * delta[i]
        pred_out[batch_idx] = wp.float32(0.5) * wp.tile_sum(wp.tile_map(wp.mul, delta, wp.tile_map(wp.sub, ld, g)))[0]
        rr = wp.tile_zeros(shape=(1, 1), dtype=wp.float32)
        wp.tile_matmul(wp.tile_transpose(r), r, rr)
        costs_out[batch_idx] = wp.float32(0.5) * rr[0, 0]

    solve.__name__ = solve.__qualname__ = f"_lm_solve_{R}x{D}"
    k = wp.kernel(enable_backward=False, module="unique")(solve)
    _KERNEL_CACHE[key] = k
    return k


def _accum_kernel_chunked(D: int, C: int):
    """Accumulate `JᵀJ` / `Jᵀr` / cost one C×D block at a time from `J`/`r` reshaped to
    `(B, num_chunks, C, D)`, so shared memory stays bounded; `_cholesky_kernel` then solves.

    `num_chunks` is a runtime arg: a compile-time bound lets Warp unroll the loop and allocate one
    tile per iteration, reintroducing the overflow for small chunk counts.
    """
    key = ("accum_chunked", D, C)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Dc, Cc = wp.constant(D), wp.constant(C)

    def accumulate(
        jacobians: wp.array4d(dtype=wp.float32),
        residuals_col: wp.array4d(dtype=wp.float32),
        num_chunks: wp.int32,
        JtJ: wp.array3d(dtype=wp.float32),
        Jtr: wp.array3d(dtype=wp.float32),
        costs: wp.array1d(dtype=wp.float32),
    ):
        batch_idx = wp.tid()
        H = wp.tile_zeros(shape=(Dc, Dc), dtype=wp.float32)
        g = wp.tile_zeros(shape=(Dc, 1), dtype=wp.float32)
        rr = wp.tile_zeros(shape=(1, 1), dtype=wp.float32)
        for c in range(num_chunks):
            Jc = wp.tile_load(jacobians[batch_idx, c], shape=(Cc, Dc))
            rc = wp.tile_load(residuals_col[batch_idx, c], shape=(Cc, 1))
            Jct = wp.tile_transpose(Jc)
            wp.tile_matmul(Jct, Jc, H)
            wp.tile_matmul(Jct, rc, g)
            wp.tile_matmul(wp.tile_transpose(rc), rc, rr)
        wp.tile_store(JtJ[batch_idx], H)
        wp.tile_store(Jtr[batch_idx], g)
        costs[batch_idx] = wp.float32(0.5) * rr[0, 0]

    accumulate.__name__ = accumulate.__qualname__ = f"_lm_accum_chunked_c{C}x{D}"
    k = wp.kernel(enable_backward=False, module="unique")(accumulate)
    _KERNEL_CACHE[key] = k
    return k


def _accum_kernel(R: int, D: int):
    """Additively contribute wide `J`/`r` → `JᵀJ`/`Jᵀr`/`costs`. Caller pre-zeros."""
    key = ("accum", R, D)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Rc, Dc = wp.constant(R), wp.constant(D)

    def accumulate(
        jacobians: wp.array3d(dtype=wp.float32),
        residuals_col: wp.array3d(dtype=wp.float32),
        JtJ: wp.array3d(dtype=wp.float32),
        Jtr: wp.array3d(dtype=wp.float32),
        costs: wp.array1d(dtype=wp.float32),
    ):
        batch_idx = wp.tid()
        J = wp.tile_load(jacobians[batch_idx], shape=(Rc, Dc))
        r = wp.tile_load(residuals_col[batch_idx], shape=(Rc, 1))
        Jt = wp.tile_transpose(J)
        H = wp.tile_load(JtJ[batch_idx], shape=(Dc, Dc))
        wp.tile_matmul(Jt, J, H)
        wp.tile_store(JtJ[batch_idx], H)
        g = wp.tile_load(Jtr[batch_idx], shape=(Dc, 1))
        wp.tile_matmul(Jt, r, g)
        wp.tile_store(Jtr[batch_idx], g)
        rr = wp.tile_zeros(shape=(1, 1), dtype=wp.float32)
        wp.tile_matmul(wp.tile_transpose(r), r, rr)
        costs[batch_idx] = costs[batch_idx] + wp.float32(0.5) * rr[0, 0]

    accumulate.__name__ = accumulate.__qualname__ = f"_lm_accum_{R}x{D}"
    k = wp.kernel(enable_backward=False, module="unique")(accumulate)
    _KERNEL_CACHE[key] = k
    return k


def _cost_kernel(R: int):
    """Additively contribute `0.5 ‖r‖²` → `costs`. Caller pre-zeros."""
    key = ("cost", R)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Rc = wp.constant(R)

    def cost(residuals: wp.array2d(dtype=wp.float32), costs: wp.array1d(dtype=wp.float32)):
        batch_idx = wp.tid()
        r = wp.tile_load(residuals[batch_idx], shape=(Rc,))
        costs[batch_idx] = costs[batch_idx] + wp.float32(0.5) * wp.tile_sum(wp.tile_map(wp.mul, r, r))[0]

    cost.__name__ = cost.__qualname__ = f"_lm_cost_r{R}"
    k = wp.kernel(enable_backward=False, module="unique")(cost)
    _KERNEL_CACHE[key] = k
    return k


def _accept_kernel(R: int):
    """Fused accept/reject reading proposed residuals directly (fast path).

    Inlines `0.5 ‖r_prop‖²`, rho check, λ update, and writes the new cost in
    place on accept - replaces cost-reduce + accept_reject_kernel with one launch.
    """
    key = ("accept", R)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Rc = wp.constant(R)

    def accept(
        costs: wp.array1d(dtype=wp.float32),
        r_prop: wp.array2d(dtype=wp.float32),
        pred: wp.array1d(dtype=wp.float32),
        rho_min: wp.float32,
        gain_ratio_epsilon: wp.float32,
        lam_f: wp.float32,
        lam_lo: wp.float32,
        lam_hi: wp.float32,
        lam: wp.array1d(dtype=wp.float32),
        acc: wp.array1d(dtype=wp.int32),
    ):
        batch_idx = wp.tid()
        s = wp.float32(0.0)
        for i in range(Rc):
            v = r_prop[batch_idx, i]
            s += v * v
        c = wp.float32(0.5) * s
        accepted = pred[batch_idx] > wp.float32(0.0)
        if accepted:
            rho = (costs[batch_idx] - c) / (pred[batch_idx] + gain_ratio_epsilon)
            accepted = rho >= rho_min
        if accepted:
            acc[batch_idx] = 1
            lam[batch_idx] = wp.max(lam[batch_idx] / lam_f, lam_lo)
            costs[batch_idx] = c
        else:
            acc[batch_idx] = 0
            lam[batch_idx] = wp.clamp(lam[batch_idx] * lam_f, lam_lo, lam_hi)

    accept.__name__ = accept.__qualname__ = f"_lm_accept_r{R}"
    k = wp.kernel(enable_backward=False, module="unique")(accept)
    _KERNEL_CACHE[key] = k
    return k


@wp.kernel(enable_backward=False)
def lm_accept_reject_cost_kernel(
    costs_curr: wp.array1d(dtype=wp.float32),
    costs_prop: wp.array1d(dtype=wp.float32),
    pred_red: wp.array1d(dtype=wp.float32),
    rho_min: wp.float32,
    gain_ratio_epsilon: wp.float32,
    lam_f: wp.float32,
    lam_lo: wp.float32,
    lam_hi: wp.float32,
    lam: wp.array1d(dtype=wp.float32),
    acc: wp.array1d(dtype=wp.int32),
):
    """Streaming-path accept/reject (reads precomputed proposed cost scalar)."""
    batch_idx = wp.tid()
    accepted = pred_red[batch_idx] > wp.float32(0.0)
    if accepted:
        rho = (costs_curr[batch_idx] - costs_prop[batch_idx]) / (pred_red[batch_idx] + gain_ratio_epsilon)
        accepted = rho >= rho_min
    if accepted:
        acc[batch_idx] = 1
        lam[batch_idx] = wp.max(lam[batch_idx] / lam_f, lam_lo)
        costs_curr[batch_idx] = costs_prop[batch_idx]
    else:
        acc[batch_idx] = 0
        lam[batch_idx] = wp.clamp(lam[batch_idx] * lam_f, lam_lo, lam_hi)


@wp.kernel(enable_backward=False)
def _mask_jacobian_cols_kernel(J: wp.array3d(dtype=wp.float32), mask: wp.array1d(dtype=wp.float32)):
    """Zero Jacobian columns for inactive DOFs (`mask[d] == 0`)."""
    batch_idx, res_idx, dof_idx = wp.tid()
    J[batch_idx, res_idx, dof_idx] = J[batch_idx, res_idx, dof_idx] * mask[dof_idx]


@wp.kernel(enable_backward=False)
def _mask_hessian_kernel(H: wp.array3d(dtype=wp.float32), mask: wp.array1d(dtype=wp.float32)):
    """Zero `JᵀJ` rows and columns for inactive DOFs (symmetric)."""
    batch_idx, i, j = wp.tid()
    H[batch_idx, i, j] = H[batch_idx, i, j] * (mask[i] * mask[j])


@wp.kernel(enable_backward=False)
def _mask_gradient_kernel(g: wp.array3d(dtype=wp.float32), mask: wp.array1d(dtype=wp.float32)):
    """Zero `Jᵀr` entries for inactive DOFs. `g` has shape `(B, D, 1)`."""
    batch_idx, dof_idx = wp.tid()
    g[batch_idx, dof_idx, 0] = g[batch_idx, dof_idx, 0] * mask[dof_idx]


@dataclass(frozen=True)
class LMOptimizerConfig(OptimizerConfig):
    num_dofs: int = 0  # total tangent dim D (required, must be > 0)
    rho_min: float = 1e-3  # minimum gain ratio rho to accept a step
    gain_ratio_epsilon: float = 1e-8  # stabilizer in the rho denominator; 0 uses the exact ratio
    # float32 [num_dofs]; 0 freezes a dof, 1 keeps it active
    active_dof_mask: Optional[wp.array] = field(default=None, hash=False, compare=False)

    def __post_init__(self):
        if self.num_dofs <= 0:
            raise ValueError(f"num_dofs must be positive, got {self.num_dofs}")


class LMOptimizer(Optimizer):
    """Generalized normal-equation LM. Fast fused path when all terms are default Jacobian."""

    def __init__(
        self,
        terms: Union[ResidualTask, Sequence[ResidualTask]],
        config: LMOptimizerConfig,
        device: Optional[wp_device_type] = None,
    ):
        Optimizer.__init__(self, terms=terms, config=config)
        self.device = wp.get_device(device) if device is None or isinstance(device, str) else device
        self.total_tangent_dim = config.num_dofs
        self._initialized = False

    def setup(self, var: VarValues):
        B, D, dev = var.batch_size, self.total_tangent_dim, self.device
        zf = lambda s, dt=wp.float32: wp.zeros(s, dtype=dt, device=dev)  # noqa: E731
        self.batch_size = B
        self._proposed_var = var.clone()
        self.hessian, self.gradient, self.delta = zf((B, D, D)), zf((B, D, 1)), zf((B, D))
        self.lm = wp.empty((B,), dtype=wp.float32, device=dev)
        self.costs, self.proposed_costs, self.pred_reduction = zf((B,)), zf((B,)), zf((B,))
        self.accept_mask = zf((B,), wp.int32)
        self._cholesky_kernel = _cholesky_kernel(D)
        c = self.config
        self._accept_args = (c.rho_min, c.gain_ratio_epsilon, c.lambda_factor, c.lambda_min, c.lambda_max)
        self._active_dof_mask = c.active_dof_mask

        # one-pass term partition: record default-term row offsets in wide buffer
        self._default_accum: List[ResidualTask] = []
        self._default_cost: List[ResidualTask] = []
        self._accum_offsets: List[int] = []
        self._cost_offsets: List[int] = []
        ao = co = 0
        has_stream = False
        for t in self.terms:
            is_default_acc = type(t).accumulate_normal_equations is ResidualTask.accumulate_normal_equations
            is_default_cost = is_default_acc
            if is_default_acc:
                self._default_accum.append(t)
                self._accum_offsets.append(ao)
                ao += t.residual_dim
            if is_default_cost:
                self._default_cost.append(t)
                self._cost_offsets.append(co)
                co += t.residual_dim
            has_stream = has_stream or not (is_default_acc and is_default_cost)
        self._R_accum = ao
        # Streaming terms get eager lifecycle hooks (prepare/on_step) run outside any
        # captured graph; their presence also forces the two-graph split-capture mode.
        self._stream_terms: List[EagerTask] = [t for t in self.terms if isinstance(t, EagerTask)]
        self._fused_mode = len(self._stream_terms) == 0
        self._task_stop = False

        # Fast path activates only when CUDA + every term is default (no streaming override).
        is_cuda = dev is not None and dev.is_cuda
        fast = is_cuda and not has_stream
        self._use_streams = fast and len(self.terms) > 1 and (len(self.terms) < 6 or B >= 128)
        self._term_streams = [wp.Stream(dev) for _ in self.terms] if self._use_streams else []
        self._term_events = [wp.Event(dev) for _ in self.terms] if self._use_streams else []
        self._wide_J = self._wide_r = self._wide_pr = None
        self._fused_solve = self._fused_accept = self._chunked_accum = None
        # _R_accum_buf is the padded row count the solve kernel sees (chunked path); _R_accum stays
        # the true count for term writes and masking.
        self._R_accum_buf = ao
        self._chunk_nc = 0
        if fast and ao > 0:
            if ao > _LM_FUSED_MAX_ROWS:
                self._R_accum_buf = -(-ao // _LM_SOLVE_CHUNK) * _LM_SOLVE_CHUNK
                self._chunk_nc = self._R_accum_buf // _LM_SOLVE_CHUNK
                self._chunked_accum = _accum_kernel_chunked(D, _LM_SOLVE_CHUNK)
            else:
                self._fused_solve = _solve_kernel(ao, D)
            self._wide_r = wp.zeros((B, self._R_accum_buf), dtype=wp.float32, device=dev)
            self._wide_J = wp.zeros((B, self._R_accum_buf, D), dtype=wp.float32, device=dev)
        if fast and co > 0:
            self._wide_pr = wp.zeros((B, co), dtype=wp.float32, device=dev)
            self._fused_accept = _accept_kernel(co)

        # optional CUDA-graph capture: 2-pass (eager warmup, then record)
        self._graph = self._propose_graph = self._accept_graph = None
        if self.config.use_cuda_graph and is_cuda:
            self._reset(var)
            self._run_iteration(var)
            wp.synchronize()
            self._reset(var)
            if self._fused_mode:
                with wp.ScopedCapture(device=dev) as cap:
                    self._propose_body(var)
                    self._accept_body(var)
                self._graph = cap.graph
            else:
                for t in self._stream_terms:
                    t.prepare(var, False, 0)
                with wp.ScopedCapture(device=dev) as cap:
                    self._propose_body(var)
                self._propose_graph = cap.graph
                for t in self._stream_terms:
                    t.prepare(self._proposed_var, True, 0)
                with wp.ScopedCapture(device=dev) as cap:
                    self._accept_body(var)
                self._accept_graph = cap.graph
        self._initialized = True

    def _reset(self, var: VarValues):
        var.invalidate()
        self.lm.fill_(self.config.lm_lambda)
        self._iter = 0

    def _fanout(self, terms: List[ResidualTask], offsets: List[int], fn):
        """Run `fn(term, offset)` per term, using per-term streams when available."""
        if not self._term_streams:
            for term, off in zip(terms, offsets):
                fn(term, off)
            return
        main = wp.get_stream(self.device)
        init = main.record_event()
        for term, off, stream, evt in zip(terms, offsets, self._term_streams, self._term_events):
            stream.wait_event(init)
            with wp.ScopedStream(stream):
                fn(term, off)
            stream.record_event(evt)
        for evt in self._term_events[: len(terms)]:
            main.wait_event(evt)

    def _propose_body(self, var: VarValues):
        B = self.batch_size
        if self._fused_solve is not None or self._chunked_accum is not None:
            for term in self.terms:
                term.precompute(var, need_gradient=True)
            # Analytic Jacobian kernels write only their own row-slice; zero wide_J
            # once so untouched rows stay 0. Residual kernels overwrite their slice.
            self._wide_J.zero_()

            def _fn(t, o):
                t.compute_weighted_residual(var, out_residual=self._wide_r, row_offset=o)
                t.compute_weighted_jacobian(var, out_jacobian=self._wide_J, row_offset=o)

            self._fanout(self._default_accum, self._accum_offsets, _fn)
            # Mask Jacobian columns for frozen DOFs BEFORE forming JᵀJ/Jᵀr so
            # frozen columns don't leak into the active-DOF step.
            if self._active_dof_mask is not None:
                wp.launch(
                    _mask_jacobian_cols_kernel,
                    dim=(B, self._R_accum, self.total_tangent_dim),
                    inputs=[self._wide_J, self._active_dof_mask],
                    device=self.device,
                )
            if self._chunked_accum is not None:
                # many residuals: accumulate JᵀJ in row-blocks, then solve the small D×D system
                Dd = self.total_tangent_dim
                J4 = self._wide_J.reshape((B, self._chunk_nc, _LM_SOLVE_CHUNK, Dd))
                r4 = self._wide_r.reshape((B, self._chunk_nc, _LM_SOLVE_CHUNK, 1))
                wp.launch_tiled(
                    self._chunked_accum,
                    dim=[B],
                    inputs=[J4, r4, self._chunk_nc],
                    outputs=[self.hessian, self.gradient, self.costs],
                    block_dim=TILE_THREADS,
                    device=self.device,
                )
                wp.launch_tiled(
                    self._cholesky_kernel,
                    dim=[B],
                    inputs=[self.hessian, self.gradient, self.lm],
                    outputs=[self.delta, self.pred_reduction],
                    block_dim=TILE_THREADS,
                    device=self.device,
                )
            else:
                r_col = self._wide_r.reshape((B, self._R_accum_buf, 1))
                wp.launch_tiled(
                    self._fused_solve,
                    dim=[B],
                    inputs=[self._wide_J, r_col, self.lm],
                    outputs=[self.costs, self.delta, self.pred_reduction],
                    block_dim=TILE_THREADS,
                    device=self.device,
                )
        else:
            self.hessian.zero_()
            self.gradient.zero_()
            self.costs.zero_()
            for t in self.terms:
                t.accumulate_normal_equations(var, costs=self.costs, JtJ=self.hessian, Jtr=self.gradient)
            # mask JᵀJ rows/cols and Jᵀr entries for frozen DOFs
            if self._active_dof_mask is not None:
                D = self.total_tangent_dim
                wp.launch(
                    _mask_hessian_kernel,
                    dim=(B, D, D),
                    inputs=[self.hessian, self._active_dof_mask],
                    device=self.device,
                )
                wp.launch(
                    _mask_gradient_kernel,
                    dim=(B, D),
                    inputs=[self.gradient, self._active_dof_mask],
                    device=self.device,
                )
            wp.launch_tiled(
                self._cholesky_kernel,
                dim=[B],
                inputs=[self.hessian, self.gradient, self.lm],
                outputs=[self.delta, self.pred_reduction],
                block_dim=TILE_THREADS,
                device=self.device,
            )
        var.integrate(self.delta, out=self._proposed_var, tangent_mask=self._active_dof_mask)

    def _accept_body(self, var: VarValues):
        B, dev = self.batch_size, self.device
        out = [self.lm, self.accept_mask]
        if self._fused_accept is not None:
            for term in self.terms:
                term.precompute(self._proposed_var, need_gradient=False)
            pv = self._proposed_var

            def _fn(t, o):
                t.compute_weighted_residual(pv, out_residual=self._wide_pr, row_offset=o)

            self._fanout(self._default_cost, self._cost_offsets, _fn)
            args = [self.costs, self._wide_pr, self.pred_reduction, *self._accept_args]
            wp.launch(self._fused_accept, dim=B, inputs=args, outputs=out, device=dev)
        else:
            self.proposed_costs.zero_()
            for t in self.terms:
                t.accumulate_normal_equations(self._proposed_var, costs=self.proposed_costs)
            args = [self.costs, self.proposed_costs, self.pred_reduction, *self._accept_args]
            wp.launch(lm_accept_reject_cost_kernel, dim=B, inputs=args, outputs=out, device=dev)
        var.accept(self.accept_mask, self._proposed_var)

    def _run_iteration(self, var: VarValues):
        if self._fused_mode:
            if self._graph is not None:
                wp.capture_launch(self._graph)
            else:
                self._propose_body(var)
                self._accept_body(var)
            self._iter += 1
            return
        for t in self._stream_terms:
            t.prepare(var, False, self._iter)
        if self._propose_graph is not None:
            wp.capture_launch(self._propose_graph)
        else:
            self._propose_body(var)
        for t in self._stream_terms:
            t.prepare(self._proposed_var, True, self._iter)
        if self._accept_graph is not None:
            wp.capture_launch(self._accept_graph)
        else:
            self._accept_body(var)
        stop = False
        for t in self._stream_terms:
            stop = t.on_step(self.accept_mask, self._iter, self.costs) or stop
        self._task_stop = stop
        self._iter += 1

    def _compute_current_costs(self, var: VarValues):
        """Populate `self.costs` with `0.5·‖r(var)‖²` at the current var.

        Used when the caller runs 0 iterations (`max_iter=0`) so scoring
        consumers (e.g. `MultiSeedSolver`) read a valid cost, not the
        stale zeros left by `_reset`.
        """
        self.costs.zero_()
        for t in self.terms:
            t.accumulate_normal_equations(var, costs=self.costs)

    def solve(self, var: VarValues) -> Tuple[VarValues, wp.array]:
        wp.init()
        if not self._initialized:
            self.setup(var)
        elif var.batch_size != self.batch_size:
            if self.config.use_cuda_graph:
                raise ValueError(
                    f"batch size changed from {self.batch_size} to {var.batch_size}, "
                    "but use_cuda_graph=True locks the batch size"
                )
            self.setup(var)
        self._reset(var)
        cfg = self.config
        if cfg.max_iter == 0:
            self._compute_current_costs(var)
            return var, self.costs
        stop_interval = cfg.early_stopping_interval if cfg.use_early_stopping else 0
        for iter_idx in range(cfg.max_iter):
            self._run_iteration(var)
            if stop_interval > 0 and (iter_idx + 1) % stop_interval == 0:
                if self._task_stop:
                    break
                if bool(np.all(self.costs.numpy() <= cfg.cost_tol)):
                    break
        return var, self.costs
