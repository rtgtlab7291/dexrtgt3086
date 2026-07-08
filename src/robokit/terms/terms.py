import abc
from functools import lru_cache
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import warp as wp

from robokit.opt.variables import Var
from robokit.types import ArrayLike


@wp.kernel
def fill_seed_kernel(seed_array: wp.array2d(dtype=wp.float32), col_idx: int, value: float):
    i = wp.tid()
    seed_array[i, col_idx] = value


@wp.kernel
def fill_jacobian_row_kernel(
    grad: wp.array2d(dtype=wp.float32),
    jacobian: wp.array3d(dtype=wp.float32),
    row_idx: int,
    row_offset: int,
    col_offset: int,
):
    i, j = wp.tid()  # type: ignore[misc]
    jacobian[i, row_offset + row_idx, col_offset + j] = grad[i, j]


class Term(abc.ABC):
    """
    Base optimization term.
    """


class Task(Term):
    """
    Task term for quadratic programming optimization.

    All tasks accept a single `var` as the optimization variable.
    Additional args/kwargs are auxiliary arguments (not optimization variables).
    """

    residual_weight: Optional[Union[float, Sequence[float], np.ndarray]]

    @abc.abstractmethod
    def compute_weighted_residual(self, var: Var, *args: Any, **kwargs: Any) -> ArrayLike: ...

    @abc.abstractmethod
    def compute_weighted_jacobian(self, var: Var, *args: Any, **kwargs: Any) -> ArrayLike: ...


class NumpyTask(Task):
    @abc.abstractmethod
    def compute_residual(self, var: Var, *args: Any, **kwargs: Any) -> np.ndarray: ...

    @abc.abstractmethod
    def compute_jacobian(self, var: Var, *args: Any, **kwargs: Any) -> np.ndarray: ...

    @lru_cache()
    def get_residual_weight_matrix(self, num_dim: int) -> np.ndarray:
        if self.residual_weight is None:
            return np.eye(num_dim)
        else:
            return np.diag(
                [self.residual_weight] * num_dim if isinstance(self.residual_weight, float) else self.residual_weight
            )

    def compute_weighted_residual(self, var: Var, *args: Any, **kwargs: Any) -> np.ndarray:
        error = self.compute_residual(var, *args, **kwargs)
        if error.ndim == 1:
            weight = self.get_residual_weight_matrix(error.shape[0])
        else:
            weight = self.get_residual_weight_matrix(error.shape[-2])
        return weight @ error

    def compute_weighted_jacobian(self, var: Var, *args: Any, **kwargs: Any) -> np.ndarray:
        jacobian = self.compute_jacobian(var, *args, **kwargs)
        weight = self.get_residual_weight_matrix(jacobian.shape[-2])
        return weight @ jacobian


class TorchTask(Task):
    JACOBIAN_MODE: Literal["autodiff", "analytic"] = "analytic"

    @abc.abstractmethod
    def compute_residual(self, var: Var, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def compute_jacobian_analytic(self, var: Var, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Analytic Jacobian not implemented for this task.")

    def compute_jacobian_autodiff(self, var: Var, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute Jacobian using torch autodiff. Returns shape [..., residual_dim, tangent_dim]."""

        def residual_fn(velocity: torch.Tensor) -> torch.Tensor:
            new_var = var.integrate(velocity)
            return self.compute_residual(new_var, *args, **kwargs)

        residual = self.compute_residual(var, *args, **kwargs)
        tangent_dim = var.tangent_dim
        device, dtype = residual.device, residual.dtype

        if hasattr(torch, "func"):
            jacrev, vmap = torch.func.jacrev, torch.func.vmap  # type: ignore[attr-defined]
        else:
            import functorch

            jacrev, vmap = functorch.jacrev, functorch.vmap

        if residual.ndim == 1:
            zero_velocity = torch.zeros(tangent_dim, device=device, dtype=dtype)
            return jacrev(residual_fn)(zero_velocity)  # type: ignore[return-value]

        batch_shape = residual.shape[:-1]
        batch_size = residual.numel() // residual.shape[-1]
        zero_velocity = torch.zeros(batch_size, tangent_dim, device=device, dtype=dtype)
        jacobian: torch.Tensor = vmap(jacrev(residual_fn))(zero_velocity)  # type: ignore[assignment]
        return jacobian.view(*batch_shape, -1, tangent_dim)

    def compute_jacobian(self, var: Var, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.JACOBIAN_MODE == "analytic":
            return self.compute_jacobian_analytic(var, *args, **kwargs)
        else:
            return self.compute_jacobian_autodiff(var, *args, **kwargs)

    @lru_cache()
    def get_residual_weight_tensor(
        self, num_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Get the diagonal weights as a vector for faster element-wise multiplication."""
        if self.residual_weight is None:
            return torch.ones(num_dim, device=device, dtype=dtype)
        else:
            return torch.as_tensor(
                [self.residual_weight] * num_dim if isinstance(self.residual_weight, float) else self.residual_weight,
                device=device,
                dtype=dtype,
            )

    def compute_weighted_residual(self, var: Var, *args: Any, **kwargs: Any) -> torch.Tensor:
        error = self.compute_residual(var, *args, **kwargs)
        weight_vector = self.get_residual_weight_tensor(error.shape[-1], device=error.device, dtype=error.dtype)
        return weight_vector * error

    def compute_weighted_jacobian(self, var: Var, *args: Any, **kwargs: Any) -> torch.Tensor:
        jacobian = self.compute_jacobian(var, *args, **kwargs)
        weight_vector = self.get_residual_weight_tensor(
            jacobian.shape[-2], device=jacobian.device, dtype=jacobian.dtype
        )
        return weight_vector.unsqueeze(-1) * jacobian


class WarpTask(Task):
    JACOBIAN_MODE: Literal["autodiff", "analytic"] = "analytic"
    var_key: str = "robot"

    _autodiff_tape: Optional[wp.Tape] = None
    _autodiff_velocity: Optional[wp.array] = None
    _autodiff_grad_seed: Optional[wp.array] = None
    _autodiff_proposed_var: Optional[Var] = None

    @property
    @abc.abstractmethod
    def residual_dim(self) -> int:
        """Dimension of the residual vector."""
        ...

    @abc.abstractmethod
    def compute_weighted_residual(
        self,
        var: Var,
        *args: Any,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array: ...

    def compute_weighted_jacobian_analytic(
        self,
        var: Var,
        *args: Any,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
        col_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        raise NotImplementedError("Analytic Jacobian not implemented for this task.")

    def init_autodiff_buffers(self, var: Var):
        """Pre-allocate buffers for autodiff Jacobian computation."""
        batch_size = var.batch_size
        tangent_dim = var.tangent_dim
        device = var.device if hasattr(var, "device") else None

        self._autodiff_tape = wp.Tape()
        self._autodiff_velocity = wp.zeros(
            (batch_size, tangent_dim), dtype=wp.float32, device=device, requires_grad=True
        )
        self._autodiff_grad_seed = wp.zeros((batch_size, self.residual_dim), dtype=wp.float32, device=device)

    def compute_weighted_jacobian_autodiff(
        self,
        var: Var,
        *args: Any,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
        col_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        batch_size = var.batch_size
        tangent_dim = var.tangent_dim
        device = var.device if hasattr(var, "device") else None

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros((batch_size, self.residual_dim, tangent_dim), dtype=wp.float32, device=device)

        if self._autodiff_tape is None:
            self.init_autodiff_buffers(var)

        assert self._autodiff_tape is not None
        assert self._autodiff_velocity is not None
        assert self._autodiff_grad_seed is not None

        self._autodiff_tape.reset()
        self._autodiff_velocity.zero_()
        self._autodiff_grad_seed.zero_()

        with self._autodiff_tape:
            new_var = var.integrate(self._autodiff_velocity)
            residual = self.compute_weighted_residual(new_var, *args, residual_buffer=None, **kwargs)

        for k in range(self.residual_dim):
            wp.launch(kernel=fill_seed_kernel, dim=batch_size, inputs=[self._autodiff_grad_seed, k, 1.0], device=device)
            self._autodiff_tape.backward(grads={residual: self._autodiff_grad_seed})
            grad = self._autodiff_tape.gradients[self._autodiff_velocity]
            wp.launch(
                kernel=fill_jacobian_row_kernel,
                dim=(batch_size, tangent_dim),
                inputs=[grad, jacobian_buffer, k, row_offset, col_offset],
                device=device,
            )
            self._autodiff_tape.zero()
            wp.launch(kernel=fill_seed_kernel, dim=batch_size, inputs=[self._autodiff_grad_seed, k, 0.0], device=device)

        return jacobian_buffer

    def precompute(self, var: Var) -> None:
        """Pre-compute shared state on the main stream before parallel dispatch.

        When terms run on parallel CUDA streams, each stream only waits for an
        init_event recorded on the main stream. Shared state (FK, motion subspace)
        must be computed on the main stream before that event so all streams see it.

        The default calls FK and motion subspace via self.robot if available.
        Override for tasks with different precomputation needs.
        """
        robot = getattr(self, "robot", None)
        if robot is None:
            return
        if hasattr(var, "is_fk_computed") and not var.is_fk_computed:
            robot.forward_kinematics(var)
        if hasattr(var, "is_motion_subspace_computed") and not var.is_motion_subspace_computed:
            robot.compute_motion_subspace(var)

    def set_target(self, target) -> None:
        raise NotImplementedError

    def compute_weighted_jacobian(self, var: Var, *args: Any, **kwargs: Any) -> wp.array:
        if self.JACOBIAN_MODE == "analytic":
            return self.compute_weighted_jacobian_analytic(var, *args, **kwargs)
        else:
            return self.compute_weighted_jacobian_autodiff(var, *args, **kwargs)


@wp.struct
class WarpSparsityPattern:
    row_indices: wp.array(dtype=wp.int32)
    col_indices: wp.array(dtype=wp.int32)


class SparseWarpTask(WarpTask):
    JACOBIAN_MODE: Literal["analytic"] = "analytic"  # pyright: ignore[reportIncompatibleVariableOverride]

    @abc.abstractmethod
    def compute_sparse_jacobian_pattern(
        self, var: Var, *args: Any, offset: int = 0, **kwargs: Any
    ) -> WarpSparsityPattern:  # pyright: ignore[reportGeneralTypeIssues]
        """Sparsity pattern of the Jacobian matrix."""
        ...

    @abc.abstractmethod
    def compute_weighted_sparse_jacobian_values(
        self,
        var: Var,
        *args: Any,
        jacobian_values_buffer: Optional[wp.array] = None,
        offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        """Compute the non-zero values of the weighted Jacobian matrix according to the sparsity pattern."""
        ...


class QPLimit(Term):
    """
    Limit term for quadratic programming optimization.

    All limits accept a single `var` as the optimization variable.
    Additional args/kwargs are auxiliary arguments (not optimization variables).
    """

    @abc.abstractmethod
    def compute_qp_inequalities(self, var: Var, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]: ...
