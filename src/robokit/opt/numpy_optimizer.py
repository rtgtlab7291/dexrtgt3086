# pyright: reportOperatorIssue=false
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import qpsolvers

from robokit.opt.optimizer import Optimizer, OptimizerConfig, VarT
from robokit.opt.variables import Var
from robokit.terms import Term
from robokit.terms.terms import QPLimit, Task


logger = logging.getLogger("robokit")


@dataclass(frozen=True)
class NumpyOptimizerConfig(OptimizerConfig):
    use_qpsolver: bool = False


class NumpyOptimizer(Optimizer):
    def compute_jacobians(self, terms: Sequence[Term], var: Var) -> np.ndarray:
        J = []
        for term in terms:
            if isinstance(term, Task):
                J_term = term.compute_weighted_jacobian(var)
                J.append(J_term)
        return np.vstack(J)

    def compute_residuals(self, terms: Sequence[Term], var: Var) -> np.ndarray:
        r = []
        for term in terms:
            if isinstance(term, Task):
                r_term = term.compute_weighted_residual(var)
                r.append(r_term)
        return np.hstack(r)

    def compute_qp_inequalities(
        self, terms: Sequence[Term], var: Var
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        G, h = [], []
        for term in terms:
            if isinstance(term, QPLimit):
                G_item, h_item = term.compute_qp_inequalities(var)
                G.append(G_item)
                h.append(h_item)
        if not G:
            return None, None
        return np.vstack(G), np.hstack(h)

    def solve(self, var: VarT) -> VarT:
        residuals = self.compute_residuals(self.terms, var)
        prev_cost = 0.5 * (residuals @ residuals)

        J = self.compute_jacobians(self.terms, var)
        H = J.T @ J
        c = J.T @ residuals

        if self.config.use_qpsolver:
            G, h = self.compute_qp_inequalities(self.terms, var)
        else:
            G, h = None, None

        lm_lambda = self.config.lm_lambda
        for iteration in range(self.config.max_iter):
            damped_H = H.copy()
            diag_idx = np.diag_indices_from(damped_H)
            damped_H[diag_idx] += lm_lambda

            if self.config.use_qpsolver:
                delta = qpsolvers.solve_qp(P=damped_H, q=c, G=G, h=h, solver="daqp")  # type: ignore
                if delta is None:
                    raise RuntimeError("QP solver failed to find a solution.")
            else:
                delta = np.linalg.solve(damped_H, -c)

            proposed_var = var.integrate(delta)
            proposed_residuals = self.compute_residuals(self.terms, proposed_var)
            proposed_cost = 0.5 * (proposed_residuals @ proposed_residuals)

            pred_red = -(c @ delta) - 0.5 * (delta @ (H @ delta))
            act_red = prev_cost - proposed_cost
            rho = -np.inf if pred_red <= 0 else act_red / pred_red

            accept = rho >= 1e-3
            should_check_convergence = (
                self.config.use_early_stopping and iteration % self.config.early_stopping_interval == 0
            )

            if accept:
                var = proposed_var
                residuals = proposed_residuals
                prev_cost = proposed_cost
                lm_lambda = max(lm_lambda / self.config.lambda_factor, self.config.lambda_min)

                J = self.compute_jacobians(self.terms, var)
                H = J.T @ J
                c = J.T @ residuals
                if self.config.use_qpsolver:
                    G, h = self.compute_qp_inequalities(self.terms, var)

                if should_check_convergence:
                    cost_cond = abs(act_red) / max(1.0, abs(float(prev_cost))) < self.config.cost_tol
                    velocity_cond = np.linalg.norm(delta) < self.config.velocity_tol
                    gradient_cond = np.linalg.norm(c, ord=np.inf) < self.config.gradient_tol
                else:
                    cost_cond, velocity_cond, gradient_cond = False, False, False
            else:
                lm_lambda = min(lm_lambda * self.config.lambda_factor, self.config.lambda_max)

                if should_check_convergence:
                    gradient_cond = np.linalg.norm(c, ord=np.inf) < self.config.gradient_tol
                else:
                    gradient_cond = False

                cost_cond, velocity_cond = False, False

            if self.config.verbose and iteration % 10 == 0:
                logger.info(
                    f"Iter {iteration}: cost={float(prev_cost):.6g}, "
                    f"rho={float(rho) if np.isfinite(rho) else rho}, "
                    f"lambda={lm_lambda:.3g}"
                )

            if cost_cond or velocity_cond or gradient_cond:
                if self.config.verbose:
                    logger.info(
                        f"Converged (Iter {iteration}): cost={float(prev_cost):.6g}, "
                        f"cond(cost/grad/vel)=[{cost_cond}/{gradient_cond}/{velocity_cond}]"
                    )
                break
        return var
