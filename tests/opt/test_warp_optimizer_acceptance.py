from typing import Tuple

import numpy as np
import warp as wp

from robokit.opt.sparse_warp_optimizer import SparseWarpOptimizerConfig, accept_reject
from robokit.opt.warp_optimizer import WarpLMOptimizerConfig, update_step


class TestWarpOptimizerAcceptance:
    @staticmethod
    def _device():
        wp.init()
        return wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

    def _run_dense_update_step(
        self,
        cost_curr: float,
        cost_prop: float,
        pred_red: float,
        acceptance_mode: int,
        rho_min: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        device = self._device()
        proposed_residual = np.sqrt(2.0 * cost_prop).astype(np.float32)

        costs_curr = wp.from_numpy(np.array([cost_curr], dtype=np.float32), dtype=wp.float32, device=device)
        proposed_residuals = wp.from_numpy(
            np.array([[proposed_residual]], dtype=np.float32), dtype=wp.float32, device=device
        )
        pred_reduction = wp.from_numpy(np.array([pred_red], dtype=np.float32), dtype=wp.float32, device=device)

        lambda_values = wp.from_numpy(np.array([1.0], dtype=np.float32), dtype=wp.float32, device=device)
        current_residuals = wp.from_numpy(np.array([[0.0]], dtype=np.float32), dtype=wp.float32, device=device)
        delta = wp.from_numpy(np.array([[0.3]], dtype=np.float32), dtype=wp.float32, device=device)

        wp.launch(
            kernel=update_step,
            dim=1,
            inputs=[costs_curr, proposed_residuals, pred_reduction, rho_min, acceptance_mode, 2.0, 1e-6, 1e6],
            outputs=[lambda_values, current_residuals, delta],
            device=device,
        )
        wp.synchronize_device(device)
        cost_out = costs_curr.numpy()
        accept = np.array([int(not np.isclose(cost_out[0], cost_curr, atol=1e-7))], dtype=np.int32)
        return cost_out, delta.numpy(), accept

    def _run_sparse_accept_reject(
        self,
        cost_curr: float,
        cost_prop: float,
        pred_red: float,
        acceptance_mode: int,
        rho_min: float = 1e-3,
    ) -> np.ndarray:
        device = self._device()
        cost_curr_arr = wp.from_numpy(np.array([cost_curr], dtype=np.float32), dtype=wp.float32, device=device)
        cost_prop_arr = wp.from_numpy(np.array([cost_prop], dtype=np.float32), dtype=wp.float32, device=device)
        pred_red_arr = wp.from_numpy(np.array([pred_red], dtype=np.float32), dtype=wp.float32, device=device)
        accept = wp.zeros((1,), dtype=wp.int32, device=device)

        wp.launch(
            kernel=accept_reject,
            dim=1,
            inputs=[cost_curr_arr, cost_prop_arr, pred_red_arr, rho_min, acceptance_mode],
            outputs=[accept],
            device=device,
        )
        wp.synchronize_device(device)
        return accept.numpy()

    def test_dense_strict_monotonic_rejects_uphill_when_rho_only_would_accept(self) -> None:
        cost, delta, accept = self._run_dense_update_step(
            cost_curr=1.0,
            cost_prop=1.1,
            pred_red=-0.05,
            acceptance_mode=0,
        )
        assert accept[0] == 0
        assert np.isclose(cost[0], 1.0, atol=1e-6)
        assert np.isclose(delta[0, 0], 0.0, atol=1e-6)

    def test_dense_rho_only_accepts_same_case(self) -> None:
        cost, delta, accept = self._run_dense_update_step(
            cost_curr=1.0,
            cost_prop=1.1,
            pred_red=-0.05,
            acceptance_mode=1,
        )
        assert accept[0] == 1
        assert np.isclose(cost[0], 1.1, atol=1e-5)
        assert np.isclose(delta[0, 0], 0.3, atol=1e-6)

    def test_dense_strict_monotonic_accepts_valid_downhill_step(self) -> None:
        cost, delta, accept = self._run_dense_update_step(
            cost_curr=1.0,
            cost_prop=0.8,
            pred_red=0.2,
            acceptance_mode=0,
        )
        assert accept[0] == 1
        assert np.isclose(cost[0], 0.8, atol=1e-5)
        assert np.isclose(delta[0, 0], 0.3, atol=1e-6)

    def test_sparse_strict_monotonic_rejects_uphill_when_rho_only_would_accept(self) -> None:
        accept = self._run_sparse_accept_reject(
            cost_curr=1.0,
            cost_prop=1.1,
            pred_red=-0.05,
            acceptance_mode=0,
        )
        assert accept[0] == 0

    def test_sparse_rho_only_accepts_same_case(self) -> None:
        accept = self._run_sparse_accept_reject(
            cost_curr=1.0,
            cost_prop=1.1,
            pred_red=-0.05,
            acceptance_mode=1,
        )
        assert accept[0] == 1

    def test_sparse_strict_monotonic_accepts_valid_downhill_step(self) -> None:
        accept = self._run_sparse_accept_reject(
            cost_curr=1.0,
            cost_prop=0.8,
            pred_red=0.2,
            acceptance_mode=0,
        )
        assert accept[0] == 1

    def test_acceptance_mode_config_defaults(self) -> None:
        assert WarpLMOptimizerConfig().acceptance_mode == "strict_monotonic"
        assert WarpLMOptimizerConfig(acceptance_mode="rho_only").acceptance_mode == "rho_only"
        assert SparseWarpOptimizerConfig().acceptance_mode == "strict_monotonic"
        assert SparseWarpOptimizerConfig(acceptance_mode="rho_only").acceptance_mode == "rho_only"
