import logging
from dataclasses import dataclass
from typing import Sequence

import torch

from robokit.opt.optimizer import Optimizer, OptimizerConfig, VarT
from robokit.opt.variables import Var
from robokit.terms import Term


logger = logging.getLogger("robokit")


@dataclass(frozen=True)
class TorchOptimizerConfig(OptimizerConfig):
    early_stopping_interval: int = 5


class TorchOptimizer(Optimizer):
    def compute_jacobians(self, terms: Sequence[Term], var: Var) -> torch.Tensor:
        return torch.cat([term.compute_weighted_jacobian(var) for term in terms], dim=-2)

    def compute_residuals(self, terms: Sequence[Term], var: Var) -> torch.Tensor:
        return torch.cat([term.compute_weighted_residual(var) for term in terms], dim=-1)

    def solve(self, var: VarT) -> VarT:
        residuals = self.compute_residuals(self.terms, var)
        prev_cost = 0.5 * (residuals * residuals).sum(dim=-1)

        J = self.compute_jacobians(self.terms, var)
        JT = J.transpose(-2, -1)
        H = JT @ J
        c = (JT @ residuals.unsqueeze(-1)).squeeze(-1)

        batch_shape = residuals.shape[:-1]
        d = H.shape[-1]
        device = H.device
        dtype = H.dtype

        lm = torch.full(batch_shape, float(self.config.lm_lambda), device=device, dtype=dtype)

        for iteration in range(self.config.max_iter):
            eye = torch.eye(d, device=device, dtype=dtype)
            damped_H = H + lm[..., None, None] * eye

            delta = torch.linalg.solve(damped_H, (-c).unsqueeze(-1)).squeeze(-1)

            proposed_var = var.integrate(delta)
            proposed_residuals = self.compute_residuals(self.terms, proposed_var)
            proposed_cost = 0.5 * (proposed_residuals * proposed_residuals).sum(dim=-1)

            Hd = (H @ delta.unsqueeze(-1)).squeeze(-1)
            pred_red = -(c * delta).sum(dim=-1) - 0.5 * (delta * Hd).sum(dim=-1)
            act_red = prev_cost - proposed_cost

            rho = torch.full_like(pred_red, float("-inf"))
            pos = pred_red > 0
            rho[pos] = act_red[pos] / pred_red[pos]

            accept = torch.isfinite(rho) & (rho >= 1e-3)

            # Apply accepted updates via masked delta
            masked_delta = delta * accept.to(delta.dtype)[..., None]
            var = var.integrate(masked_delta)

            # Recompute residuals and cost consistently with updated var
            residuals = self.compute_residuals(self.terms, var)
            prev_cost = 0.5 * (residuals * residuals).sum(dim=-1)

            lm_down = torch.clamp(lm / self.config.lambda_factor, min=float(self.config.lambda_min))
            lm_up = torch.clamp(lm * self.config.lambda_factor, max=float(self.config.lambda_max))
            lm = torch.where(accept, lm_down, lm_up)

            J = self.compute_jacobians(self.terms, var)
            JT = J.transpose(-2, -1)
            H = JT @ J
            c = (JT @ residuals.unsqueeze(-1)).squeeze(-1)

            should_check_convergence = self.config.use_early_stopping and (
                iteration % self.config.early_stopping_interval == 0
            )

            if should_check_convergence:
                denom = torch.maximum(prev_cost.abs(), torch.tensor(1.0, device=device, dtype=dtype))
                cost_cond = (act_red.abs() / denom) < self.config.cost_tol
                velocity_cond = torch.linalg.vector_norm(delta, dim=-1) < self.config.velocity_tol
                gradient_cond = c.abs().amax(dim=-1) < self.config.gradient_tol
                converged = cost_cond | velocity_cond | gradient_cond
                done_ratio = converged.float().mean()
                should_terminate = not torch.any(~converged)
            else:
                done_ratio = 0.0
                should_terminate = False

            if self.config.verbose and iteration % 10 == 0:
                logger.info(
                    f"Iter {iteration}: mean_cost={float(prev_cost.mean()):.6g}, "
                    f"done={float(done_ratio):.2%}, "
                    f"lambda~{float(lm.mean()):.3g}"
                )

            if should_terminate:
                if self.config.verbose:
                    logger.info(
                        f"Converged (Iter {iteration}): mean_cost={float(prev_cost.mean()):.6g}, "
                        f"done={float(done_ratio):.2%}"
                    )
                break

        return var
