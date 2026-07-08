# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
import logging
from dataclasses import dataclass
from typing import Any, Generic, List, Literal, Optional, Sequence, Tuple, TypeVar, Union, overload

import numpy as np
import warp as wp

from robokit.opt.optimizer import Optimizer, OptimizerConfig
from robokit.opt.var_values import WarpVarValues
from robokit.opt.variables import WarpVar
from robokit.terms import Term
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type


WarpVarT = TypeVar("WarpVarT", bound=WarpVar)
logger = logging.getLogger("robokit")

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
    """Aggregate squared residuals to compute cost per batch element."""
    batch_idx = wp.tid()

    cost = wp.float32(0.0)
    for i in range(residuals.shape[1]):
        r = residuals[batch_idx, i]
        cost += r * r

    costs[batch_idx] = wp.float32(0.5) * cost


@wp.kernel
def update_step(
    costs_curr: wp.array1d(dtype=wp.float32),
    proposed_residuals: wp.array2d(dtype=wp.float32),
    pred_red: wp.array1d(dtype=wp.float32),
    rho_min: float,
    acceptance_mode: int,
    lambda_factor: float,
    lambda_min: float,
    lambda_max: float,
    # outputs
    lambda_values: wp.array1d(dtype=wp.float32),
    current_residuals: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
):
    """Fused accept/reject and state update kernel."""
    problem_idx = wp.tid()

    # Compute proposed cost from proposed_residuals (inlined from aggregate_residuals_to_costs)
    cost_prop = wp.float32(0.0)
    residual_dim = proposed_residuals.shape[1]
    for i in range(residual_dim):
        r = proposed_residuals[problem_idx, i]
        cost_prop += r * r
    cost_prop = wp.float32(0.5) * cost_prop

    # Compute rho and accept decision (inlined from accept_reject)
    rho = (costs_curr[problem_idx] - cost_prop) / (pred_red[problem_idx] + 1e-8)
    rho_only_accept = rho >= rho_min
    strict_accept = rho_only_accept and pred_red[problem_idx] > 0.0 and cost_prop <= costs_curr[problem_idx]
    accepted = strict_accept if acceptance_mode == _ACCEPTANCE_MODE_STRICT_MONOTONIC else rho_only_accept

    # Update lambda
    if accepted:
        lambda_values[problem_idx] = wp.max(lambda_values[problem_idx] / lambda_factor, lambda_min)
    else:
        new_lambda = lambda_values[problem_idx] * lambda_factor
        lambda_values[problem_idx] = wp.clamp(new_lambda, lambda_min, lambda_max)

    # Update residuals and costs if accepted
    if accepted:
        for i in range(residual_dim):
            current_residuals[problem_idx, i] = proposed_residuals[problem_idx, i]
        costs_curr[problem_idx] = cost_prop

    # Mask delta based on accept flag
    tangent_dim = delta.shape[1]
    mask = wp.float32(1.0) if accepted else wp.float32(0.0)
    for i in range(tangent_dim):
        delta[problem_idx, i] = delta[problem_idx, i] * mask


@wp.kernel
def check_status_kernel(costs: wp.array1d(dtype=wp.float32), tol: float, status: wp.array1d(dtype=wp.int32)):
    idx = wp.tid()
    status[idx] = wp.int32(1) if costs[idx] <= tol else wp.int32(0)


@wp.kernel
def apply_jacobian_mask(
    jacobians: wp.array3d(dtype=wp.float32),
    mask: wp.array1d(dtype=wp.float32),
):
    batch_idx, res_idx, dof_idx = wp.tid()  # type: ignore[misc]
    jacobians[batch_idx, res_idx, dof_idx] = jacobians[batch_idx, res_idx, dof_idx] * mask[dof_idx]


@dataclass(frozen=True)
class WarpLMOptimizerConfig(OptimizerConfig):
    rho_min: float = 1e-3
    acceptance_mode: Literal["strict_monotonic", "rho_only"] = "strict_monotonic"


class WarpLMOptimizer(Optimizer, Generic[WarpVarT]):
    def __init__(
        self,
        terms: Union[Tuple[WarpTask, ...], List[WarpTask]],
        placeholder_var: WarpVarT,
        device: Optional[wp_device_type] = None,
        config: Optional[WarpLMOptimizerConfig] = None,
        use_parallel_terms: bool = True,
        active_dof_mask: Optional[np.ndarray] = None,
        use_cuda_graph: bool = False,
    ):
        if config is None:
            config = WarpLMOptimizerConfig()
        super().__init__(terms=terms, config=config)
        self._use_parallel_terms = use_parallel_terms
        batch_size = placeholder_var.batch_size
        total_tangent_dim = placeholder_var.tangent_dim
        self.residual_dims = [term.residual_dim for term in terms]
        self.total_residual_dim = sum(self.residual_dims)
        self.total_tangent_dim = total_tangent_dim
        self.batch_size = batch_size
        self.device = device
        self._proposed_var = placeholder_var.clone()

        self.residuals = wp.empty((self.batch_size, self.total_residual_dim), dtype=wp.float32, device=self.device)
        self.costs = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.proposed_residuals = wp.empty(
            (self.batch_size, self.total_residual_dim), dtype=wp.float32, device=self.device
        )
        self.jacobians = wp.empty(
            (self.batch_size, self.total_residual_dim, self.total_tangent_dim), dtype=wp.float32, device=self.device
        )
        self.hessian = wp.empty(
            (self.batch_size, self.total_tangent_dim, self.total_tangent_dim), dtype=wp.float32, device=self.device
        )
        self.gradient = wp.empty((self.batch_size, self.total_tangent_dim), dtype=wp.float32, device=self.device)
        self.delta = wp.empty((self.batch_size, self.total_tangent_dim), dtype=wp.float32, device=self.device)
        self.lm = wp.empty((self.batch_size,), dtype=wp.float32, device=self.device)
        self.pred_reduction = wp.zeros((self.batch_size,), dtype=wp.float32, device=self.device)
        self._acceptance_mode = _encode_acceptance_mode(self.config.acceptance_mode)

        self._tile_threads = 32
        self._compute_delta_tiled = self._build_fused_solver(self.total_residual_dim, self.total_tangent_dim)

        offset = 0
        self.residual_offsets = []
        for term in self.terms:
            self.residual_offsets.append(offset)
            offset += term.residual_dim

        self._init_cuda_streams()
        self._active_dof_mask = None
        self._use_cuda_graph = use_cuda_graph
        self._cuda_graph: Optional[Any] = None
        self._graph_capture_complete = False
        if active_dof_mask is not None:
            mask_np = np.asarray(active_dof_mask, dtype=np.float32)
            if mask_np.shape != (self.total_tangent_dim,):
                raise ValueError(f"Expected active_dof_mask shape {(self.total_tangent_dim,)}, got {mask_np.shape}")
            self._active_dof_mask = wp.from_numpy(mask_np, dtype=wp.float32, device=self.device)
        if self._use_cuda_graph and self.device is not None and self.device.is_cuda:
            self._warmup_and_capture(placeholder_var)

    def _build_fused_solver(self, n_residuals: int, n_dofs: int):
        """Build a fused LM solver kernel specialized for problem sizes."""

        RES = int(n_residuals)
        DOF = int(n_dofs)

        def _template(
            jacobians: wp.array3d(dtype=wp.float32),
            residuals: wp.array2d(dtype=wp.float32),
            lambda_values: wp.array1d(dtype=wp.float32),
            # outputs
            hessian_out: wp.array3d(dtype=wp.float32),
            gradient_out: wp.array2d(dtype=wp.float32),
            delta_out: wp.array2d(dtype=wp.float32),
            pred_reduction_out: wp.array1d(dtype=wp.float32),
        ):
            problem_idx = wp.tid()

            J = wp.tile_load(jacobians[problem_idx], shape=(RES, DOF))

            r_tile = wp.tile_zeros(shape=(RES, 1), dtype=wp.float32)
            for i in range(RES):
                r_tile[i, 0] = residuals[problem_idx, i]

            lam = lambda_values[problem_idx]

            Jt = wp.tile_transpose(J)
            H = wp.tile_zeros(shape=(DOF, DOF), dtype=wp.float32)
            wp.tile_matmul(Jt, J, H)

            diag = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                diag[i] = lam
            A = wp.tile_diag_add(H, diag)

            tmp2d = wp.tile_zeros(shape=(DOF, 1), dtype=wp.float32)
            wp.tile_matmul(Jt, r_tile, tmp2d)

            g = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                g[i] = tmp2d[i, 0]

            rhs = wp.tile_map(wp.neg, g)
            L = wp.tile_cholesky(A)
            delta = wp.tile_cholesky_solve(L, rhs)

            wp.tile_store(hessian_out[problem_idx], H)
            wp.tile_store(gradient_out[problem_idx], g)
            wp.tile_store(delta_out[problem_idx], delta)

            # compute predicted reduction: pred_red = 0.5 * delta^T * (lambda * delta - g)
            lambda_delta = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                lambda_delta[i] = lam * delta[i]

            diff = wp.tile_map(wp.sub, lambda_delta, g)
            prod = wp.tile_map(wp.mul, delta, diff)
            red = wp.tile_sum(prod)[0]
            pred_reduction_out[problem_idx] = wp.float32(0.5) * red

        _template.__name__ = f"_fused_lm_{RES}x{DOF}"
        _template.__qualname__ = f"_fused_lm_{RES}x{DOF}"
        return wp.kernel(enable_backward=False, module="unique")(_template)

    def _init_cuda_streams(self):
        """Allocate per-term Warp streams and sync events."""
        self.term_streams = []
        self.sync_events = []
        self.init_event = None

        if self.device is not None and self.device.is_cuda:
            self.init_event = wp.Event(self.device)
            for _ in range(len(self.terms)):
                stream = wp.Stream(self.device)
                event = wp.Event(self.device)
                self.term_streams.append(stream)
                self.sync_events.append(event)
        else:
            self.term_streams = [None] * len(self.terms)
            self.sync_events = [None] * len(self.terms)

    def _validate_residual_layout(self) -> None:
        residual_dims = [term.residual_dim for term in self.terms]
        total_dim = int(sum(residual_dims))
        if total_dim == self.total_residual_dim:
            return
        term_desc = ", ".join(
            f"[{idx}] {type(term).__name__} residual_dim={dim}"
            for idx, (term, dim) in enumerate(zip(self.terms, residual_dims))
        )
        raise ValueError(
            "WarpLMOptimizer residual layout mismatch: "
            f"expected total_residual_dim={self.total_residual_dim}, got {total_dim}. "
            f"TermDims=[{term_desc}]"
        )

    def _parallel_for_objectives(self, fn, *extra):
        """Run <fn(term, offset, *extra)> across terms on parallel CUDA streams."""
        if self._use_parallel_terms and self.device is not None and self.device.is_cuda:
            main = wp.get_stream(self.device)
            main.record_event(self.init_event)
            for term, offset, term_stream, sync_event in zip(
                self.terms, self.residual_offsets, self.term_streams, self.sync_events
            ):
                term_stream.wait_event(self.init_event)
                with wp.ScopedStream(term_stream):
                    fn(term, offset, *extra)
                term_stream.record_event(sync_event)
            for sync_event in self.sync_events:
                main.wait_event(sync_event)
        else:
            for term, offset in zip(self.terms, self.residual_offsets):
                fn(term, offset, *extra)

    def _precompute_shared_state(self, var: WarpVar) -> None:
        """Pre-compute shared state (FK, motion subspace) on the main stream before parallel dispatch."""
        for term in self.terms:
            term.precompute(var)

    def compute_jacobians(self, terms: Sequence[Term], var: WarpVar, jacobian_buffer: wp.array):
        jacobian_buffer.zero_()
        self._precompute_shared_state(var)

        def _compute_jacobian(term, offset, var, jacobian_buffer):
            if isinstance(var, WarpVarValues):
                sub_var = var.get(term.var_key)
                term.compute_weighted_jacobian(
                    sub_var,
                    jacobian_buffer=jacobian_buffer,
                    row_offset=offset,
                    col_offset=var.tangent_offset(term.var_key),
                )  # type: ignore[arg-type]
            else:
                term.compute_weighted_jacobian(var, jacobian_buffer=jacobian_buffer, row_offset=offset)  # type: ignore[arg-type]

        self._parallel_for_objectives(_compute_jacobian, var, jacobian_buffer)
        if self._active_dof_mask is not None:
            wp.launch(
                kernel=apply_jacobian_mask,
                dim=(self.batch_size, self.total_residual_dim, self.total_tangent_dim),
                inputs=[jacobian_buffer, self._active_dof_mask],
                device=self.device,
            )

    def compute_residuals(self, terms: Sequence[Term], var: WarpVar, residual_buffer: wp.array):
        residual_buffer.zero_()
        self._precompute_shared_state(var)

        def _compute_residual(term, offset, var, residual_buffer):
            sub_var = var.get(term.var_key) if isinstance(var, WarpVarValues) else var
            term.compute_weighted_residual(sub_var, residual_buffer=residual_buffer, row_offset=offset)  # type: ignore[arg-type]

        self._parallel_for_objectives(_compute_residual, var, residual_buffer)

    def _warmup_and_capture(self, var: WarpVarT) -> None:
        var.invalidate()
        self._solve_init(var)
        self._solve_iteration(var)
        wp.synchronize()

        var.invalidate()
        self._solve_init(var)
        with wp.ScopedCapture(device=self.device) as capture:
            self._solve_iteration(var)
        self._cuda_graph = capture.graph
        self._graph_capture_complete = True

    def _solve_init(self, var: WarpVarT) -> None:
        self._validate_residual_layout()
        self.lm.fill_(self.config.lm_lambda)
        self.compute_residuals(self.terms, var, self.residuals)
        self.compute_jacobians(self.terms, var, self.jacobians)
        wp.launch(
            kernel=aggregate_residuals_to_costs,
            dim=self.batch_size,
            inputs=[self.residuals, self.costs],
            device=self.device,
        )

    def _solve_iteration(self, var: WarpVarT) -> WarpVarT:
        proposed_var = self._proposed_var if self._proposed_var is not None else var.clone()

        wp.launch_tiled(
            self._compute_delta_tiled,
            dim=[self.batch_size],
            inputs=[self.jacobians, self.residuals, self.lm],
            outputs=[self.hessian, self.gradient, self.delta, self.pred_reduction],
            block_dim=self._tile_threads,
            device=self.device,
        )

        proposed_var = var.integrate(self.delta, out=proposed_var)
        self.compute_residuals(self.terms, proposed_var, self.proposed_residuals)

        wp.launch(
            kernel=update_step,
            dim=self.batch_size,
            inputs=[
                self.costs,
                self.proposed_residuals,
                self.pred_reduction,
                self.config.rho_min,
                self._acceptance_mode,
                self.config.lambda_factor,
                self.config.lambda_min,
                self.config.lambda_max,
            ],
            outputs=[self.lm, self.residuals, self.delta],
            device=self.device,
        )

        var = var.integrate(self.delta, out=var)  # type: ignore[call-arg]
        self.compute_jacobians(self.terms, var, self.jacobians)
        return var

    def _solve_impl(
        self,
        var: WarpVarT,
        return_status: bool = False,
        skip_early_stop: bool = False,
    ) -> Union[WarpVarT, Tuple[WarpVarT, wp.array]]:
        status = None
        if self.config.use_early_stopping and not skip_early_stop:
            status = wp.empty((self.batch_size,), dtype=wp.int32, device=self.device)

        self._solve_init(var)

        for iter_idx in range(self.config.max_iter):
            var = self._solve_iteration(var)

            if (
                self.config.use_early_stopping
                and not skip_early_stop
                and self.config.early_stopping_interval > 0
                and (iter_idx + 1) % self.config.early_stopping_interval == 0
            ):
                assert status is not None
                wp.launch(
                    kernel=check_status_kernel,
                    dim=self.batch_size,
                    inputs=[self.costs, self.config.cost_tol, status],
                    device=self.device,
                )
                if bool(np.all(status.numpy() == 1)):
                    break

        if return_status:
            if status is None:
                status = wp.empty((self.batch_size,), dtype=wp.int32, device=self.device)
                wp.launch(
                    kernel=check_status_kernel,
                    dim=self.batch_size,
                    inputs=[self.costs, self.config.cost_tol, status],
                    device=self.device,
                )
            return var, status

        return var

    @overload
    def solve(self, var: WarpVarT, return_status: Literal[False] = False) -> WarpVarT: ...
    @overload
    def solve(self, var: WarpVarT, return_status: Literal[True]) -> Tuple[WarpVarT, wp.array]: ...
    def solve(self, var: WarpVarT, return_status: bool = False) -> Union[WarpVarT, Tuple[WarpVarT, wp.array]]:
        wp.init()
        use_graph = (
            self._use_cuda_graph and self.device is not None and self.device.is_cuda and self._graph_capture_complete
        )
        if use_graph and self._cuda_graph is not None:
            var.invalidate()
            status = None
            if self.config.use_early_stopping and self.config.early_stopping_interval > 0:
                status = wp.empty((self.batch_size,), dtype=wp.int32, device=self.device)

            self._solve_init(var)
            for iter_idx in range(self.config.max_iter):
                wp.capture_launch(self._cuda_graph)
                if (
                    self.config.use_early_stopping
                    and self.config.early_stopping_interval > 0
                    and (iter_idx + 1) % self.config.early_stopping_interval == 0
                ):
                    assert status is not None
                    wp.launch(
                        kernel=check_status_kernel,
                        dim=self.batch_size,
                        inputs=[self.costs, self.config.cost_tol, status],
                        device=self.device,
                    )
                    if bool(np.all(status.numpy() == 1)):
                        break

            if return_status:
                if status is None:
                    status = wp.empty((self.batch_size,), dtype=wp.int32, device=self.device)
                    wp.launch(
                        kernel=check_status_kernel,
                        dim=self.batch_size,
                        inputs=[self.costs, self.config.cost_tol, status],
                        device=self.device,
                    )
                return var, status

            return var

        result = self._solve_impl(var, return_status=return_status, skip_early_stop=False)
        return result
