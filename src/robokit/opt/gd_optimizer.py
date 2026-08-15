# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import warp as wp

from robokit.opt.optimizer import OptimizerBase, OptimizerConfig, aggregate_residuals_to_costs_kernel
from robokit.opt.var_values import VarValues
from robokit.terms.task import GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type


@wp.kernel
def _gd_update_kernel(
    gradient: wp.array2d(dtype=wp.float32),
    learning_rate: float,
    velocity: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()
    velocity[batch_idx, dof_idx] = -learning_rate * gradient[batch_idx, dof_idx]


@wp.kernel
def _adam_update_kernel(
    gradient: wp.array2d(dtype=wp.float32),
    m: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    bias_correction1: float,
    bias_correction2: float,
):
    batch_idx, dof_idx = wp.tid()
    g = gradient[batch_idx, dof_idx]
    m_new = beta1 * m[batch_idx, dof_idx] + (1.0 - beta1) * g
    v_new = beta2 * v[batch_idx, dof_idx] + (1.0 - beta2) * g * g
    m[batch_idx, dof_idx] = m_new
    v[batch_idx, dof_idx] = v_new
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    velocity[batch_idx, dof_idx] = -lr * m_hat / (wp.sqrt(v_hat) + epsilon)


@dataclass(frozen=True)
class GDOptimizerConfig(OptimizerConfig):
    learning_rate: float = 0.01
    lr_schedule: Optional[Callable[[int, float], float]] = None
    # analytic_jacobian: terms fill a shared J (sparse when every term is a SparseTask, else dense),
    # one J^T r reduce. analytic_gradient: each term's kernel atomic-adds cost+gradient directly.
    gradient_mode: Literal["analytic_jacobian", "autodiff", "analytic_gradient"] = "analytic_jacobian"
    optimizer_type: Literal["gd", "adam", "adamw"] = "gd"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0


class GDOptimizer(OptimizerBase):
    """First-order optimizer supporting GD, Adam, and AdamW."""

    TILE_N_RESIDUALS = None
    TILE_N_DOFS = None
    _cache: ClassVar[Dict[Tuple[int, int], type]] = {}

    @classmethod
    def _build_specialized(cls, key):
        RES, DOF = key

        def _compute_gradient_jtr_template(
            jacobians: wp.array3d(dtype=wp.float32),
            residuals: wp.array2d(dtype=wp.float32),
            gradient: wp.array2d(dtype=wp.float32),
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

        _compute_gradient_jtr_template.__name__ = f"_fo_grad_jtr_{RES}x{DOF}"
        _compute_gradient_jtr_template.__qualname__ = f"_fo_grad_jtr_{RES}x{DOF}"
        _compute_gradient_jtr_tiled = wp.kernel(enable_backward=False, module="unique")(_compute_gradient_jtr_template)

        class _Specialized(GDOptimizer):
            TILE_N_RESIDUALS = wp.constant(RES)
            TILE_N_DOFS = wp.constant(DOF)
            TILE_THREADS = wp.constant(32)
            _dense_jtr_kernel = _compute_gradient_jtr_tiled

        _Specialized.__name__ = f"FO_{RES}x{DOF}"
        return _Specialized

    def __init__(
        self,
        terms: Union[Tuple[ResidualTask, ...], List[ResidualTask]],
        device: Optional[wp_device_type] = None,
        config: Optional[GDOptimizerConfig] = None,
        active_dof_mask: Optional[wp.array] = None,
    ):
        if config is None:
            config = GDOptimizerConfig()
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
        jac_mode = config.gradient_mode == "analytic_jacobian"
        dense_jac = jac_mode and not self.is_all_sparse
        if dense_jac:
            # Swap in the (residual_dim, tangent_dim)-specialized subclass carrying the tiled
            # J^T r kernel - only the dense jacobian reduce uses it.
            key = (sum(t.residual_dim for t in terms), var.tangent_dim)
            spec_cls = GDOptimizer._cache.get(key)
            if spec_cls is None:
                spec_cls = GDOptimizer._build_specialized(key)
                GDOptimizer._cache[key] = spec_cls
            self.__class__ = spec_cls

        device = self._device
        self._build_terms(
            terms,
            var,
            device,
            self._active_dof_mask,
            residuals_require_grad=(config.gradient_mode == "autodiff"),
        )

        batch_size = self.batch_size
        total_tangent_dim = self.total_tangent_dim
        self.gradient = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self.velocity = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self._optimizer_type = config.optimizer_type

        if config.optimizer_type in ("adam", "adamw"):
            self.m = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
            self.v = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
            self._step_count = 0
        self._lr_iter = 0

        if config.gradient_mode == "autodiff":
            self._autodiff_tape = wp.Tape()
            self._autodiff_velocity = wp.zeros(
                (batch_size, total_tangent_dim), dtype=wp.float32, device=device, requires_grad=True
            )
            self._autodiff_proposed_var = var.integrate(self._autodiff_velocity)
            self._compute_cost_and_gradient = self._compute_cost_and_gradient_autodiff
        elif config.gradient_mode == "analytic_jacobian":
            # GD has no expanded line-search batch, so jac rows = batch_size
            self._build_jacobian(var, self.batch_size)
            self._compute_cost_and_gradient = self._compute_cost_and_gradient_jacobian
        else:  # analytic_gradient
            self._cost_grad_fns: list = []
            for t in terms:
                assert isinstance(t, GradientTask)
                self._cost_grad_fns.append(
                    lambda offset, var, _fn=t.compute_weighted_cost_and_gradient: _fn(
                        var, out_cost=self.costs, out_gradient=self.gradient
                    )
                )
            self._compute_cost_and_gradient = self._compute_cost_and_gradient_direct

        self._built = True

    def _compute_cost_and_gradient_direct(self, var: VarValues):
        """analytic_gradient mode: each term's kernel atomic-adds cost+gradient into shared buffers."""
        self._precompute_terms(var, need_gradient=True)
        self.gradient.zero_()
        self.costs.zero_()
        self._parallel_for_objectives(self._cost_grad_fns, var)
        # Mask the gradient (the Jacobian never sees the mask in this mode). With m/v starting at
        # zero this keeps Adam/AdamW momentum zero on locked dims too.
        self._mask_gradient(self.gradient)

    def _compute_cost_and_gradient_autodiff(self, var: VarValues):
        """Autodiff mode: tape forward -> backward to get gradient directly (no Jacobian)."""
        self._autodiff_tape.reset()
        self._autodiff_tape.gradients = {}
        self._autodiff_velocity.zero_()

        with self._autodiff_tape:
            proposed = var.integrate(self._autodiff_velocity, out=self._autodiff_proposed_var)
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

    def _compute_velocity(self):
        cfg = self.config
        dim = (self.batch_size, self.total_tangent_dim)
        lr = cfg.lr_schedule(self._lr_iter, cfg.learning_rate) if cfg.lr_schedule is not None else cfg.learning_rate
        self._lr_iter += 1

        if self._optimizer_type == "gd":
            wp.launch(
                kernel=_gd_update_kernel,
                dim=dim,
                inputs=[self.gradient, lr],
                outputs=[self.velocity],
                device=self.device,
            )
        else:
            self._step_count += 1
            bc1 = 1.0 - cfg.beta1**self._step_count
            bc2 = 1.0 - cfg.beta2**self._step_count
            wp.launch(
                kernel=_adam_update_kernel,
                dim=dim,
                inputs=[
                    self.gradient,
                    self.m,
                    self.v,
                    self.velocity,
                    lr,
                    cfg.beta1,
                    cfg.beta2,
                    cfg.epsilon,
                    bc1,
                    bc2,
                ],
                device=self.device,
            )
        self._weight_decay = lr * cfg.weight_decay if self._optimizer_type == "adamw" else 0.0

    def _warmup_and_capture(self, var: VarValues):
        var.invalidate()
        self._solve_impl(var)
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

    def _solve_impl(self, var: VarValues) -> VarValues:
        # Restart the schedule and Adam state per solve. The counters are host-side and baked into
        # the captured graph as launch params, so without this reset a capture recorded after a
        # warmup run would bake the post-warmup lr/bias-correction tail instead of the schedule
        # start; m/v are buffers, so the zeroing is captured and replays reset them per problem.
        self._lr_iter = 0
        if self._optimizer_type in ("adam", "adamw"):
            self._step_count = 0
            self.m.zero_()
            self.v.zero_()
        self._compute_cost_and_gradient(var)

        stop_interval = self.config.early_stopping_interval if self.config.use_early_stopping else 0
        for iter_idx in range(self.config.max_iter):
            self._compute_velocity()
            # in-place: every integrate kernel is elementwise on the same index, so out=var aliases safely
            var.integrate(
                self.velocity,
                out=var,
                tangent_mask=self._active_dof_mask,
                weight_decay=self._weight_decay,
            )
            var.invalidate()
            self._compute_cost_and_gradient(var)

            if stop_interval > 0 and (iter_idx + 1) % stop_interval == 0:
                if bool(np.all(self.costs.numpy() <= self.config.cost_tol)):
                    break

        return var
