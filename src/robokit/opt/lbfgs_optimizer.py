# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import warp as wp

from robokit.opt.optimizer import (
    OptimizerBase,
    OptimizerConfig,
    aggregate_residuals_to_costs_kernel,
    apply_dof_mask_kernel,
)
from robokit.opt.var_values import VarValues
from robokit.terms.task import GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type


@wp.kernel
def _gather_winner_cost_grad_kernel(
    winner_idx: wp.array1d(dtype=wp.int32),
    expanded_costs: wp.array1d(dtype=wp.float32),
    expanded_gradient: wp.array2d(dtype=wp.float32),
    costs_out: wp.array1d(dtype=wp.float32),
    gradient_out: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()
    w = winner_idx[batch_idx]
    if w >= 0:
        if dof_idx == 0:
            costs_out[batch_idx] = expanded_costs[w]
        gradient_out[batch_idx, dof_idx] = expanded_gradient[w, dof_idx]


@wp.kernel
def _generate_candidate_velocities_kernel(
    search_direction: wp.array2d(dtype=wp.float32),  # (P, D)
    alphas: wp.array1d(dtype=wp.float32),  # (S,)
    n_line_search: int,
    out: wp.array2d(dtype=wp.float32),  # (P*S, D)
):
    batch_idx, step_idx, dof_idx = wp.tid()
    out[batch_idx * n_line_search + step_idx, dof_idx] = alphas[step_idx] * search_direction[batch_idx, dof_idx]


@dataclass(frozen=True)
class LBFGSOptimizerConfig(OptimizerConfig):
    history_len: int = 10
    h0_scale: float = 1.0
    line_search_alphas: tuple = (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
    wolfe_c1: float = 1e-4
    normalize_direction: bool = True
    # Barzilai-Borwein initial-Hessian scaling (gamma = sy/yy). Off by default = fixed h0_scale;
    # turn on for the small-h0_scale collision setups where a self-scaled H0 matters.
    bb_h0: bool = False
    # analytic_jacobian: terms fill a shared J (sparse when every term is a SparseTask, else dense),
    # one J^T r reduce. analytic_gradient: each term's kernel atomic-adds cost+gradient directly.
    gradient_mode: Literal["analytic_jacobian", "autodiff", "analytic_gradient"] = "analytic_jacobian"
    # Fold the gradient into the line-search cost pass and gather the winner's gradient, skipping the
    # separate per-iteration recompute (FK + cost + grad). ~1.14x on the multi-seed trajectory solve.
    # Needs an expanded-batch gradient, so it only applies to analytic_gradient and sparse
    # analytic_jacobian; keep off for solves that read the returned iterate's gradient path
    # differently (IK / retargeting). Requires all terms to be correct at the expanded batch.
    fold_line_search_gradient: bool = False


class LBFGSOptimizer(OptimizerBase):
    TILE_N_RESIDUALS = None
    TILE_N_DOFS = None
    TILE_HISTORY_LEN = None
    TILE_N_LINE_STEPS = None
    _cache: ClassVar[Dict[Tuple[int, int, int, int], type]] = {}

    @classmethod
    def _build_specialized(cls, key):
        RES, DOF, M_HIST, N_LINE_SEARCH = key

        def _make_kernel(fn, name):
            fn.__name__ = fn.__qualname__ = name
            return wp.kernel(enable_backward=False, module="unique")(fn)

        def _compute_gradient_jtr_template(
            jacobians: wp.array3d(dtype=wp.float32),  # (B, R, D)
            residuals: wp.array2d(dtype=wp.float32),  # (B, R)
            gradient: wp.array2d(dtype=wp.float32),  # (B, D)
        ):
            batch_idx = wp.tid()
            R = _Specialized.TILE_N_RESIDUALS
            D = _Specialized.TILE_N_DOFS
            J = wp.tile_load(jacobians[batch_idx], shape=(R, D))
            r = wp.tile_load(residuals[batch_idx], shape=(R,))
            Jt = wp.tile_transpose(J)
            r_2d = wp.tile_reshape(r, shape=(R, 1))
            g_2d = wp.tile_matmul(Jt, r_2d)
            g = wp.tile_reshape(g_2d, shape=(D,))
            wp.tile_store(gradient[batch_idx], g)

        def _compute_search_direction_template(
            gradient: wp.array2d(dtype=wp.float32),  # (B, D)
            s_history: wp.array3d(dtype=wp.float32),  # (B, M, D)
            y_history: wp.array3d(dtype=wp.float32),  # (B, M, D)
            rho_history: wp.array2d(dtype=wp.float32),  # (B, M)
            alpha_history: wp.array2d(dtype=wp.float32),  # (B, M)
            history_count: wp.array1d(dtype=wp.int32),  # (B,)
            history_start: wp.array1d(dtype=wp.int32),  # (B,)
            h0_scale: float,
            normalize_dir: int,
            use_bb: int,
            search_direction: wp.array2d(dtype=wp.float32),  # (B, D)
            slope_out: wp.array1d(dtype=wp.float32),  # (B,)
        ):
            batch_idx = wp.tid()
            D = _Specialized.TILE_N_DOFS
            M = _Specialized.TILE_HISTORY_LEN

            q = wp.tile_load(gradient[batch_idx], shape=(D,))
            count = history_count[batch_idx]
            start = history_start[batch_idx]

            # first loop: backward through history
            for i in range(count):
                idx = (start + count - 1 - i) % M
                s_i = wp.tile_load(s_history[batch_idx, idx], shape=(D,), storage="shared")
                rho_i = rho_history[batch_idx, idx]
                s_dot_q = wp.tile_sum(wp.tile_map(wp.mul, s_i, q))
                alpha_i = rho_i * s_dot_q[0]
                alpha_history[batch_idx, idx] = alpha_i
                y_i = wp.tile_load(y_history[batch_idx, idx], shape=(D,), storage="shared")
                q = q - alpha_i * y_i

            # Initial Hessian: Barzilai-Borwein scaling gamma = (s·y)/(y·y) from the most recent
            # history pair - the standard L-BFGS H0, far better conditioned than a fixed scalar. Opt-in
            # via config.bb_h0 (default off = fixed h0_scale, the long-standing behavior the robometrics
            # benchmark is tuned to): BB helps the tiny-h0_scale graspkit collision setup but slightly
            # perturbs the well-tuned fixed-h0 path. Falls back to h0_scale before any history exists.
            gamma = h0_scale
            if count > 0 and use_bb != 0:
                last = (start + count - 1) % M
                s_last = wp.tile_load(s_history[batch_idx, last], shape=(D,), storage="shared")
                y_last = wp.tile_load(y_history[batch_idx, last], shape=(D,), storage="shared")
                sy = wp.tile_sum(wp.tile_map(wp.mul, s_last, y_last))[0]
                yy = wp.tile_sum(wp.tile_map(wp.mul, y_last, y_last))[0]
                if yy > 1.0e-12:
                    gamma = sy / yy
            q = gamma * q

            # second loop: forward through history
            for i in range(count):
                idx = (start + i) % M
                y_i = wp.tile_load(y_history[batch_idx, idx], shape=(D,), storage="shared")
                s_i = wp.tile_load(s_history[batch_idx, idx], shape=(D,), storage="shared")
                rho_i = rho_history[batch_idx, idx]
                alpha_i = alpha_history[batch_idx, idx]
                y_dot_q = wp.tile_sum(wp.tile_map(wp.mul, y_i, q))
                beta = rho_i * y_dot_q[0]
                diff = alpha_i - beta
                q = q + diff * s_i

            q = -q
            if normalize_dir != 0:
                norm_sq = wp.tile_sum(wp.tile_map(wp.mul, q, q))
                n = wp.sqrt(norm_sq[0])
                if n > wp.float32(1e-12):
                    q = (wp.float32(1.0) / n) * q
            wp.tile_store(search_direction[batch_idx], q)

            # compute slope = g^T * direction (for Armijo line search)
            g = wp.tile_load(gradient[batch_idx], shape=(D,))
            slope_out[batch_idx] = wp.tile_sum(wp.tile_map(wp.mul, g, q))[0]

        def _update_history_template(
            last_step: wp.array2d(dtype=wp.float32),  # (B, D)
            gradient: wp.array2d(dtype=wp.float32),  # (B, D)
            gradient_prev: wp.array2d(dtype=wp.float32),  # (B, D)
            history_len: int,
            s_history: wp.array3d(dtype=wp.float32),
            y_history: wp.array3d(dtype=wp.float32),
            rho_history: wp.array2d(dtype=wp.float32),
            history_count: wp.array1d(dtype=wp.int32),
            history_start: wp.array1d(dtype=wp.int32),
        ):
            batch_idx = wp.tid()
            D = _Specialized.TILE_N_DOFS

            s_k = wp.tile_load(last_step[batch_idx], shape=(D,))
            g_curr = wp.tile_load(gradient[batch_idx], shape=(D,))
            g_prev = wp.tile_load(gradient_prev[batch_idx], shape=(D,))
            y_k = wp.tile_map(wp.sub, g_curr, g_prev)

            y_dot_s = wp.tile_sum(wp.tile_map(wp.mul, y_k, s_k))[0]

            if y_dot_s > 1e-8:
                rho_k = wp.float32(1.0) / y_dot_s
                count = history_count[batch_idx]
                start = history_start[batch_idx]
                write_idx = (start + count) % history_len
                if count < history_len:
                    history_count[batch_idx] = count + 1
                else:
                    history_start[batch_idx] = (start + 1) % history_len
                wp.tile_store(s_history[batch_idx, write_idx], s_k)
                wp.tile_store(y_history[batch_idx, write_idx], y_k)
                rho_history[batch_idx, write_idx] = rho_k

        def _select_best_step_template(
            costs_flat: wp.array1d(dtype=wp.float32),  # (P*S,)
            dq_flat: wp.array2d(dtype=wp.float32),  # (P*S, D)
            n_line_search: int,
            cost_curr: wp.array1d(dtype=wp.float32),  # (B,)
            slope_initial: wp.array1d(dtype=wp.float32),  # (B,)
            alphas: wp.array1d(dtype=wp.float32),  # (S,)
            wolfe_c1: float,
            last_step_out: wp.array2d(dtype=wp.float32),  # (B, D)
            winner_idx_out: wp.array1d(dtype=wp.int32),  # (B,) expanded-batch row of the winner
        ):
            batch_idx = wp.tid()
            S = _Specialized.TILE_N_LINE_STEPS
            D = _Specialized.TILE_N_DOFS
            base = batch_idx * n_line_search

            cost_k = cost_curr[batch_idx]
            slope_k = slope_initial[batch_idx]

            # scan large->small alpha, accept first satisfying Armijo
            accept_idx = int(-1)
            for i in range(S - 1, -1, -1):
                cost_new = costs_flat[base + i]
                alpha = alphas[i]
                if slope_k < 0.0 and cost_new <= cost_k + wolfe_c1 * alpha * slope_k:
                    accept_idx = i
                    break

            winner_idx_out[batch_idx] = base + accept_idx if accept_idx >= 0 else -1
            best_dq = wp.tile_zeros(shape=(D,), dtype=wp.float32)
            if accept_idx >= 0:
                best_dq = wp.tile_load(dq_flat[base + accept_idx], shape=(D,), storage="shared")
            wp.tile_store(last_step_out[batch_idx], best_dq)

        _compute_gradient_jtr_tiled = _make_kernel(_compute_gradient_jtr_template, f"_lbfgs_grad_jtr_{RES}x{DOF}")
        _compute_search_direction_tiled = _make_kernel(
            _compute_search_direction_template, f"_lbfgs_search_dir_{DOF}x{M_HIST}"
        )
        _update_history_tiled = _make_kernel(_update_history_template, f"_lbfgs_update_hist_{DOF}x{M_HIST}")
        _select_best_step_tiled = _make_kernel(_select_best_step_template, f"_lbfgs_best_step_{DOF}x{N_LINE_SEARCH}")

        class _Specialized(LBFGSOptimizer):
            TILE_N_RESIDUALS = wp.constant(RES)
            TILE_N_DOFS = wp.constant(DOF)
            TILE_HISTORY_LEN = wp.constant(M_HIST)
            TILE_N_LINE_STEPS = wp.constant(N_LINE_SEARCH)
            TILE_THREADS = wp.constant(32)
            _dense_jtr_kernel = _compute_gradient_jtr_tiled
            _search_direction_kernel = _compute_search_direction_tiled
            _update_history_kernel = _update_history_tiled
            _select_best_step_kernel = _select_best_step_tiled

        _Specialized.__name__ = f"LBFGS_{RES}x{DOF}x{M_HIST}x{N_LINE_SEARCH}"
        return _Specialized

    def _tiled_search_direction(self):
        # 128 threads (vs TILE_THREADS=32): the two-loop recursion is a serial chain of ~2*history_len
        # tile loads/dots over D dims, and the wider block cuts each tile op ~4x. Measured: 88 -> 31 us
        # per iteration on the 4-seed trajectory solve; identical math (fp reduction order only).
        wp.launch_tiled(
            self._search_direction_kernel,
            dim=[self.batch_size],
            inputs=[
                self.gradient,
                self.s_history,
                self.y_history,
                self.rho_history,
                self.alpha_history,
                self.history_count,
                self.history_start,
                self.config.h0_scale,
                int(self.config.normalize_direction),
                int(self.config.bb_h0),
            ],
            outputs=[self.search_direction, self.initial_slope],
            block_dim=128,
            device=self.device,
        )

    def _tiled_update_history(self):
        wp.launch_tiled(
            self._update_history_kernel,
            dim=[self.batch_size],
            inputs=[self.last_step_dq, self.gradient, self.gradient_prev, self.config.history_len],
            outputs=[self.s_history, self.y_history, self.rho_history, self.history_count, self.history_start],
            block_dim=self.TILE_THREADS,
            device=self.device,
        )

    def _tiled_select_best_step(self):
        wp.launch_tiled(
            self._select_best_step_kernel,
            dim=[self.batch_size],
            inputs=[
                self._expanded_costs,
                self._candidate_velocities,
                self.n_line_search,
                self.costs,
                self.initial_slope,
                self.line_search_alphas,
                self.config.wolfe_c1,
            ],
            outputs=[self.last_step_dq, self._winner_expanded_idx],
            block_dim=self.TILE_THREADS,
            device=self.device,
        )

    def __init__(
        self,
        terms: Union[Tuple[ResidualTask, ...], List[ResidualTask]],
        device: Optional[wp_device_type] = None,
        config: Optional[LBFGSOptimizerConfig] = None,
        active_dof_mask: Optional[wp.array] = None,
    ):
        if config is None:
            config = LBFGSOptimizerConfig()
        super().__init__(terms=terms, config=config)
        # Buffers, kernel specialization, and graph capture happen lazily on the first solve(),
        # which supplies batch_size/tangent_dim - no placeholder var at construction.
        self._device = device
        self._active_dof_mask = active_dof_mask
        self._use_cuda_graph = config.use_cuda_graph and device is not None and device.is_cuda
        self._cuda_graph = None
        self._graph_capture_complete = False
        self._built = False

    def _build(self, var: VarValues):
        config = self.config
        terms = self.terms
        # Swap in the subclass specialized to (residual_dim, tangent_dim, history_len, line_steps),
        # carrying the tiled two-loop/line-search kernels.
        key = (sum(t.residual_dim for t in terms), var.tangent_dim, config.history_len, len(config.line_search_alphas))
        spec_cls = LBFGSOptimizer._cache.get(key)
        if spec_cls is None:
            spec_cls = LBFGSOptimizer._build_specialized(key)
            LBFGSOptimizer._cache[key] = spec_cls
        self.__class__ = spec_cls
        device = self._device
        jac_mode = config.gradient_mode == "analytic_jacobian"
        all_sparse = self.is_all_sparse
        self._build_terms(
            terms,
            var,
            device,
            self._active_dof_mask,
            residuals_require_grad=(config.gradient_mode == "autodiff"),
        )

        batch_size = self.batch_size
        total_tangent_dim = self.total_tangent_dim
        history_len = config.history_len
        n_line_search = len(config.line_search_alphas)
        self.n_line_search = n_line_search
        self.gradient = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self.gradient_prev = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self.search_direction = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self.last_step_dq = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self.initial_slope = wp.zeros((batch_size,), dtype=wp.float32, device=device)

        # L-BFGS history
        self.s_history = wp.zeros((batch_size, history_len, total_tangent_dim), dtype=wp.float32, device=device)
        self.y_history = wp.zeros((batch_size, history_len, total_tangent_dim), dtype=wp.float32, device=device)
        self.rho_history = wp.zeros((batch_size, history_len), dtype=wp.float32, device=device)
        self.alpha_history = wp.zeros((batch_size, history_len), dtype=wp.float32, device=device)
        self.history_count = wp.zeros((batch_size,), dtype=wp.int32, device=device)
        self.history_start = wp.zeros((batch_size,), dtype=wp.int32, device=device)

        # line search buffers (parallel expanded-batch evaluation)
        self.line_search_alphas = wp.from_numpy(
            np.array(config.line_search_alphas, dtype=np.float32), dtype=wp.float32, device=device
        )
        expanded_batch = batch_size * n_line_search
        # fold needs an expanded-batch gradient: analytic_gradient or sparse analytic_jacobian
        self._fold = config.fold_line_search_gradient and (
            config.gradient_mode == "analytic_gradient" or (jac_mode and all_sparse)
        )
        self._expanded_indices = wp.from_numpy(
            np.repeat(np.arange(batch_size), n_line_search).astype(np.int32), dtype=wp.int32, device=device
        )
        self._expanded_var = var.gather(self._expanded_indices)
        self._expanded_proposed_var = self._expanded_var.clone()
        self._expanded_residuals = wp.zeros((expanded_batch, self.total_residual_dim), dtype=wp.float32, device=device)
        self._expanded_costs = wp.zeros((expanded_batch,), dtype=wp.float32, device=device)
        self._candidate_velocities = wp.zeros((expanded_batch, total_tangent_dim), dtype=wp.float32, device=device)
        # select_best_step always records the winner row; the expanded gradient it indexes is fold-only
        # (the large buffer stays unallocated for every non-fold optimizer).
        self._winner_expanded_idx = wp.zeros((batch_size,), dtype=wp.int32, device=device)
        self._expanded_gradient = (
            wp.zeros((expanded_batch, total_tangent_dim), dtype=wp.float32, device=device) if self._fold else None
        )
        if config.gradient_mode == "autodiff":
            self._autodiff_tape = wp.Tape()
            self._autodiff_velocity = wp.zeros(
                (batch_size, total_tangent_dim), dtype=wp.float32, device=device, requires_grad=True
            )
            self._autodiff_proposed_var = var.integrate(self._autodiff_velocity, tangent_mask=self._active_dof_mask)
            self._compute_cost_and_gradient = self._compute_cost_and_gradient_autodiff
        elif config.gradient_mode == "analytic_jacobian":
            self._build_jacobian(var, expanded_batch if self._fold else batch_size)
            self._compute_cost_and_gradient = self._compute_cost_and_gradient_jacobian
        else:  # analytic_gradient
            self._direct_cost_grad_fns = []
            self._direct_cost_only_fns = []
            self._direct_expanded_cost_grad_fns = []
            for t in terms:
                assert isinstance(t, GradientTask)
                self._direct_cost_grad_fns.append(
                    lambda offset, var, _fn=t.compute_weighted_cost_and_gradient: _fn(
                        var, out_cost=self.costs, out_gradient=self.gradient
                    )
                )
                self._direct_cost_only_fns.append(
                    # out_gradient omitted → cost only
                    lambda offset, var, _fn=t.compute_weighted_cost_and_gradient: _fn(
                        var, out_cost=self._expanded_costs
                    )
                )
                self._direct_expanded_cost_grad_fns.append(
                    lambda offset, var, _fn=t.compute_weighted_cost_and_gradient: _fn(
                        var, out_cost=self._expanded_costs, out_gradient=self._expanded_gradient
                    )
                )
            self._compute_cost_and_gradient = self._compute_cost_and_gradient_direct
            self._evaluate_line_search_costs = self._evaluate_line_search_costs_direct

        self._built = True

    def _evaluate_line_search_costs(self, expanded_batch: int):
        """Costs (and, when folding, gradient) of the expanded line-search batch — jacobian modes."""
        self._precompute_terms(self._expanded_proposed_var, need_gradient=self._fold)
        self._fill_jacobian(self._expanded_proposed_var, self._expanded_residuals, need_jacobian=self._fold)
        self._reduce_cost_gradient(
            self._expanded_residuals, self._expanded_costs, self._expanded_gradient if self._fold else None
        )

    def _evaluate_line_search_costs_direct(self, expanded_batch: int):
        self._precompute_terms(self._expanded_proposed_var, need_gradient=self._fold)
        if not self._fold:
            self._expanded_costs.zero_()
            self._parallel_for_objectives(self._direct_cost_only_fns, self._expanded_proposed_var)
            return
        # fold: also compute the gradient here so the winner's can be gathered (skips the recompute pass)
        self._expanded_costs.zero_()
        self._expanded_gradient.zero_()
        self._parallel_for_objectives(self._direct_expanded_cost_grad_fns, self._expanded_proposed_var)

    def _compute_cost_and_gradient_direct(self, var: VarValues):
        """analytic_gradient mode: each term's kernel atomic-adds cost+gradient into shared buffers."""
        self._precompute_terms(var, need_gradient=True)
        self.gradient.zero_()
        self.costs.zero_()
        self._parallel_for_objectives(self._direct_cost_grad_fns, var)
        # Mask the gradient so the two-loop curvature (s, y, rho) lives purely in the active
        # subspace - proper reduced-DOF L-BFGS. No-op when no mask.
        self._mask_gradient(self.gradient)

    def _compute_cost_and_gradient_autodiff(self, var: VarValues):
        """Compute residuals, costs, and gradient via autodiff (no Jacobian materialization)."""
        self._autodiff_tape.reset()
        self._autodiff_tape.gradients = {}
        self._autodiff_velocity.zero_()

        with self._autodiff_tape:
            proposed = var.integrate(
                self._autodiff_velocity, out=self._autodiff_proposed_var, tangent_mask=self._active_dof_mask
            )
            self._precompute_terms(proposed, need_gradient=False)
            for offset, fn in zip(self.residual_offsets, self._residual_fns):
                fn(offset, proposed, self.residuals)

        wp.launch(
            kernel=aggregate_residuals_to_costs_kernel,
            dim=self.batch_size,
            inputs=[self.residuals, self.costs],
            device=self.device,
        )
        self._autodiff_tape.backward(grads={self.residuals: self.residuals})
        wp.copy(self.gradient, self._autodiff_velocity.grad)
        self._autodiff_tape.zero()

    def _reset(self, var: VarValues):
        self._compute_cost_and_gradient(var)
        self.history_count.zero_()
        self.history_start.zero_()

    def _solve_iteration(self, var: VarValues, iteration: int):
        # update L-BFGS history with previous step (skip on first iteration - no previous step)
        if iteration > 0:
            self._tiled_update_history()

        # compute search direction + slope (two-loop recursion; steepest descent when history is empty)
        self._tiled_search_direction()

        # Lock masked tangent dims (reduced-DOF subset + lock_endpoints) so those dofs/frames
        # cannot move regardless of how the gradient was produced.
        if self._active_dof_mask is not None:
            wp.launch(
                kernel=apply_dof_mask_kernel,
                dim=(self.batch_size, self.total_tangent_dim),
                inputs=[self.search_direction, self._active_dof_mask],
                device=self.device,
            )

        # save gradient for next iteration's history update
        wp.copy(self.gradient_prev, self.gradient)

        expanded_batch = self.batch_size * self.n_line_search
        wp.launch(
            kernel=_generate_candidate_velocities_kernel,
            dim=[self.batch_size, self.n_line_search, self.total_tangent_dim],
            inputs=[self.search_direction, self.line_search_alphas, self.n_line_search],
            outputs=[self._candidate_velocities],
            device=self.device,
        )
        var.gather(self._expanded_indices, out=self._expanded_var)
        self._expanded_var.integrate(
            self._candidate_velocities, out=self._expanded_proposed_var, tangent_mask=self._active_dof_mask
        )
        self._evaluate_line_search_costs(expanded_batch)
        self._tiled_select_best_step()

        # Apply the selected step in-place. Fold: gather the winner's already-computed cost+gradient
        # from the line-search pass; else recompute (FK + cost + gradient).
        var.integrate(self.last_step_dq, out=var, tangent_mask=self._active_dof_mask)
        if self._fold:
            wp.launch(
                kernel=_gather_winner_cost_grad_kernel,
                dim=(self.batch_size, self.total_tangent_dim),
                inputs=[self._winner_expanded_idx, self._expanded_costs, self._expanded_gradient],
                outputs=[self.costs, self.gradient],
                device=self.device,
            )
            self._mask_gradient(self.gradient)
        else:
            self._compute_cost_and_gradient(var)

    def _solve_impl(self, var: VarValues, max_iter: Optional[int] = None) -> VarValues:
        n_iter = self.config.max_iter if max_iter is None else max_iter
        self._reset(var)
        for iter_idx in range(n_iter):
            self._solve_iteration(var, iter_idx)
        return var

    def _warmup_and_capture(self, var: VarValues):
        # Warmup runs only a couple iterations - just enough to allocate every lazy buffer (history,
        # line-search batch shapes) and JIT all kernels. Running the FULL max_iter
        # here (eager) was the dominant single-plan cost; capture itself only RECORDS launches (no
        # execute), so a cheap warmup + captured replay is far faster.
        var.invalidate()
        self._solve_impl(var, max_iter=min(2, self.config.max_iter))
        wp.synchronize()
        var.invalidate()
        with wp.ScopedCapture(device=self.device) as capture:
            self._solve_impl(var)
        self._cuda_graph = capture.graph
        self._graph_capture_complete = True

    def solve(self, var: VarValues) -> Tuple[VarValues, wp.array]:
        wp.init()
        if not self._built:
            self._build(var)
        if self._use_cuda_graph:
            if not self._graph_capture_complete:
                self._warmup_and_capture(var)
            var.invalidate()
            wp.capture_launch(self._cuda_graph)
            return var, self.costs
        self._solve_impl(var)
        return var, self.costs
