import abc
from typing import Any, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues


# --- task protocols ---------------------------------------------------------
class Task(abc.ABC):
    """Base of every cost term: names its variable (`var_key`) and cost row (`cost_name`)."""

    SUPPORTS_CUDA_GRAPH: bool = True

    var_key: str = "robot"

    @property
    def cost_name(self) -> str:
        """Return the snake-case task name used in cost reports."""
        name = type(self).__name__
        return "".join("_" + c.lower() if c.isupper() else c for c in name).lstrip("_")

    def set_target(self, target: Any):
        """Update the task target."""
        raise NotImplementedError

    def set_weight(self, weight: Any):
        """Update the task weight."""
        raise NotImplementedError

    def precompute(self, var_values: VarValues, need_gradient: bool = False):
        """Prepare shared state before this term is evaluated."""
        pass


class GradientTask(Task):
    """Term consumed as a per-batch scalar cost and gradient.

    Used by `GDOptimizer`/`LBFGSOptimizer` in `gradient_mode="analytic_gradient"`.
    """

    @abc.abstractmethod
    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        *args: Any,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
        **kwargs: Any,
    ):
        """Accumulate this term's weighted cost and optional direct gradient.

        Args:
            var_values: Optimization variables.
            *args: Additional term inputs.
            out_cost: Shared per-batch cost buffer.
            out_gradient: Optional shared gradient buffer. If omitted, only cost is accumulated.
            **kwargs: Additional term inputs.
        """
        ...


class ResidualTask(Task):
    """Term consumed as a weighted residual and optional Jacobian."""

    residual_weight: Optional[Union[float, Sequence[float], np.ndarray]]

    @property
    @abc.abstractmethod
    def residual_dim(self) -> int:
        """Dimension of the residual vector."""
        ...

    @abc.abstractmethod
    def compute_weighted_residual(
        self,
        var_values: VarValues,
        *args: Any,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array: ...

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        *args: Any,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        raise NotImplementedError("Analytic Jacobian not implemented for this task.")

    def compute_weighted_jacobian(self, var_values: VarValues, *args: Any, **kwargs: Any) -> wp.array:
        return self.compute_weighted_jacobian_analytic(var_values, *args, **kwargs)

    # --- normal equations ---
    # the default path materializes the residual and Jacobian; streaming tasks override it
    _residual_buf: Optional[wp.array] = None
    _jacobian_buf: Optional[wp.array] = None
    _cost_residual_buf: Optional[wp.array] = None

    def accumulate_normal_equations(
        self, var_values: VarValues, *, costs: wp.array, JtJ: Optional[wp.array] = None, Jtr: Optional[wp.array] = None
    ):
        """Accumulate this term into shared normal-equation buffers.

        Lifecycle:
            1. Prepare term state and reusable buffers.
            2. Evaluate residuals and, for a build step, the Jacobian.
            3. Reduce into cost, `JᵀJ`, and `Jᵀr`.

        Args:
            var_values: Optimization variables.
            costs: Shared per-batch cost buffer.
            JtJ: Optional shared normal matrix. Omit for a cost-only step.
            Jtr: Optional shared normal-vector buffer. Omit for a cost-only step.
        """
        from robokit.opt.lm_optimizer import TILE_THREADS, _accum_kernel, _cost_kernel

        # --- prepare buffers ---
        self.precompute(var_values, need_gradient=JtJ is not None)
        batch_size, R = var_values.batch_size, self.residual_dim
        device = var_values.device

        # --- cost-only step ---
        if JtJ is None:
            if self._cost_residual_buf is None or self._cost_residual_buf.shape[0] != batch_size:
                self._cost_residual_buf = wp.zeros((batch_size, R), dtype=wp.float32, device=device)
            self.compute_weighted_residual(var_values, out_residual=self._cost_residual_buf, row_offset=0)
            wp.launch_tiled(
                _cost_kernel(R),
                dim=[batch_size],
                inputs=[self._cost_residual_buf],
                outputs=[costs],
                block_dim=TILE_THREADS,
                device=device,
            )
            return

        # --- normal-equation step ---
        D = JtJ.shape[1]
        if self._residual_buf is None or self._residual_buf.shape[0] != batch_size:
            self._residual_buf = wp.zeros((batch_size, R), dtype=wp.float32, device=device)
        if self._jacobian_buf is None or self._jacobian_buf.shape[0] != batch_size:
            self._jacobian_buf = wp.zeros((batch_size, R, D), dtype=wp.float32, device=device)

        self._jacobian_buf.zero_()
        self.compute_weighted_residual(var_values, out_residual=self._residual_buf, row_offset=0)
        self.compute_weighted_jacobian(var_values, out_jacobian=self._jacobian_buf, row_offset=0)
        wp.launch_tiled(
            _accum_kernel(R, D),
            dim=[batch_size],
            inputs=[self._jacobian_buf, self._residual_buf.reshape((batch_size, R, 1))],
            outputs=[JtJ, Jtr, costs],
            block_dim=TILE_THREADS,
            device=JtJ.device,
        )


class EagerTask(ResidualTask):
    """Task with per-iteration work outside a captured CUDA graph."""

    def prepare(self, var_values: VarValues, proposed: bool, iter_idx: int):
        """Prepare eager state before a Warp reduction.

        Args:
            var_values: Current or proposed optimization variables.
            proposed: Whether this is a line-search proposal.
            iter_idx: Solver iteration.
        """
        pass

    def on_step(self, accept_mask: wp.array, iter_idx: int, costs: wp.array) -> bool:
        """Update eager state after accepting or rejecting a step.

        Args:
            accept_mask: Per-batch accepted-step mask.
            iter_idx: Solver iteration.
            costs: Per-batch costs.

        Returns:
            Whether the solver should stop early.
        """
        return False


# --- sparse kernels ---------------------------------------------------------
@wp.struct
class SparsityPattern:
    row_indices: wp.array(dtype=wp.int32)
    col_indices: wp.array(dtype=wp.int32)


@wp.kernel
def _sparse_cost_deterministic_kernel(
    residuals: wp.array2d(dtype=wp.float32),
    out_cost: wp.array1d(dtype=wp.float32),
    residual_dim: int,
):
    batch_idx = wp.tid()
    c = float(0.0)
    for r in range(residual_dim):
        rv = residuals[batch_idx, r]
        c += 0.5 * rv * rv
    wp.atomic_add(out_cost, batch_idx, c)


@wp.kernel
def _sparse_gradient_csc_kernel(
    residuals: wp.array2d(dtype=wp.float32),
    jac_values: wp.array2d(dtype=wp.float32),
    col_perm: wp.array1d(dtype=wp.int32),
    col_nnz_starts: wp.array1d(dtype=wp.int32),
    row_indices: wp.array1d(dtype=wp.int32),
    col_offset: int,
    out_gradient: wp.array2d(dtype=wp.float32),
):
    batch_idx, col_idx = wp.tid()  # pyright: ignore[reportGeneralTypeIssues]
    start = col_nnz_starts[col_idx]
    end = col_nnz_starts[col_idx + 1]
    g = float(0.0)
    for i in range(start, end):
        nz_idx = col_perm[i]
        res_idx = row_indices[nz_idx]
        g += jac_values[batch_idx, nz_idx] * residuals[batch_idx, res_idx]
    wp.atomic_add(out_gradient, batch_idx, col_offset + col_idx, g)


@wp.kernel
def _set_frame_mask_kernel(
    valid_lengths: wp.array1d(dtype=wp.int32),
    frame_mask: wp.array2d(dtype=wp.uint8),
):
    batch_index, frame_index = wp.tid()  # pyright: ignore[reportGeneralTypeIssues]
    # the mask may be grown past the construction batch (expanded solves); rows repeat instance-major
    length_index = batch_index // (frame_mask.shape[0] // valid_lengths.shape[0])
    frame_mask[batch_index, frame_index] = wp.uint8(frame_index < valid_lengths[length_index])


# --- sparse task ------------------------------------------------------------
class SparseTask(ResidualTask):
    """Residual task with an explicitly stored sparse Jacobian."""

    _sparse_pattern: Optional[SparsityPattern] = None  # pyright: ignore[reportGeneralTypeIssues]
    _sparse_residual_buf: Optional[wp.array] = None
    # masking is disabled until `set_valid_lengths` supplies real-versus-padded frames
    use_mask: bool = False
    frame_mask: Optional[wp.array] = None
    _valid_lengths: Optional[wp.array] = None
    _dummy_mask: Optional[wp.array] = None
    _sparse_jac_buf: Optional[wp.array] = None
    _sparse_nnz_offsets: Optional[wp.array] = None
    _sparse_col_perm: Optional[wp.array] = None
    _sparse_col_nnz_starts: Optional[wp.array] = None
    _sparse_num_cols: int = 0

    @abc.abstractmethod
    def compute_sparse_jacobian_pattern(
        self, var_values: VarValues, *args: Any, offset: int = 0, **kwargs: Any
    ) -> SparsityPattern:  # pyright: ignore[reportGeneralTypeIssues]
        """Sparsity pattern of the Jacobian matrix."""
        ...

    @abc.abstractmethod
    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: VarValues,
        *args: Any,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        """Compute the non-zero values of the weighted Jacobian matrix according to the sparsity pattern."""
        ...

    def _build_sparse_gradient_buffers(self, var_values: VarValues):
        batch_size = var_values.batch_size
        if self._sparse_pattern is not None and self._sparse_residual_buf.shape[0] >= batch_size:
            return
        device = var_values.device
        if self._sparse_pattern is None:
            self._sparse_pattern = self.compute_sparse_jacobian_pattern(var_values, offset=0)
            row_np = self._sparse_pattern.row_indices.numpy()
            col_np = self._sparse_pattern.col_indices.numpy()
            offsets = np.searchsorted(row_np, np.arange(self.residual_dim + 1))
            self._sparse_nnz_offsets = wp.from_numpy(offsets.astype(np.int32), dtype=wp.int32, device=device)
            # build CSC ordering for deterministic gradient accumulation
            num_cols = int(col_np.max()) + 1 if col_np.size > 0 else 0
            self._sparse_num_cols = num_cols
            col_perm = np.argsort(col_np, kind="stable").astype(np.int32)
            col_nnz_starts = np.searchsorted(col_np[col_perm], np.arange(num_cols + 1)).astype(np.int32)
            self._sparse_col_perm = wp.from_numpy(col_perm, dtype=wp.int32, device=device)
            self._sparse_col_nnz_starts = wp.from_numpy(col_nnz_starts, dtype=wp.int32, device=device)
        nnz = self._sparse_pattern.row_indices.shape[0]
        self._sparse_residual_buf = wp.zeros((batch_size, self.residual_dim), dtype=wp.float32, device=device)
        self._sparse_jac_buf = wp.zeros((batch_size, nnz), dtype=wp.float32, device=device)

    def fill_residual_and_jacobian(
        self,
        var_values: VarValues,
        out_residual: wp.array,
        row_offset: int,
        out_jacobian: Optional[wp.array] = None,
        nnz_offset: int = 0,
    ):
        """Write residuals and optional sparse Jacobian values into shared buffers.

        Args:
            var_values: Optimization variables.
            out_residual: Shared residual buffer.
            row_offset: First residual row for this task.
            out_jacobian: Optional shared sparse-value buffer.
            nnz_offset: First sparse entry for this task.
        """
        if self.use_mask:
            self._build_frame_mask(var_values.device, var_values.batch_size)
        self.compute_weighted_residual(var_values, out_residual=out_residual, row_offset=row_offset)
        if out_jacobian is not None:
            self.compute_weighted_sparse_jacobian_values(
                var_values, out_jacobian_values=out_jacobian, nnz_offset=nnz_offset
            )

    def _build_frame_mask(self, device: Any, batch_size: int):
        """Grow the cached validity mask to cover the requested batch."""
        rows = batch_size
        if self.frame_mask is None or self.frame_mask.shape[0] < rows:
            self.frame_mask = wp.ones((rows, self.num_frames), dtype=wp.uint8, device=device)
            if self._valid_lengths is not None:
                # regrowing must not erase a set_valid_lengths mask; rows repeat instance-major
                wp.launch(
                    _set_frame_mask_kernel,
                    dim=(rows, self.num_frames),
                    inputs=[self._valid_lengths, self.frame_mask],
                    device=device,
                )

    def _get_frame_mask(self) -> wp.array:
        """Get the validity mask or a dummy buffer when masking is disabled."""
        if self.frame_mask is not None:
            return self.frame_mask
        if self._dummy_mask is None:
            self._dummy_mask = wp.zeros((1, 1), dtype=wp.uint8, device=self.device)
        return self._dummy_mask

    def set_valid_lengths(self, lengths: Union[np.ndarray, wp.array]):
        """Set the valid trajectory length for each batch element.

        Args:
            lengths: Per-batch valid frame counts.
        """
        self.use_mask = True
        if isinstance(lengths, np.ndarray):
            lengths = wp.from_numpy(lengths.astype(np.int32, copy=False), dtype=wp.int32, device=self.device)
        if lengths.ndim != 1:
            raise ValueError("valid_lengths must be one-dimensional")
        self._build_frame_mask(self.device, lengths.shape[0])
        self._valid_lengths = lengths
        wp.launch(
            _set_frame_mask_kernel,
            dim=(self.frame_mask.shape[0], self.num_frames),
            inputs=[lengths, self.frame_mask],
            device=self.device,
        )


def sparse_cost_and_gradient(
    self,
    var_values: VarValues,
    *args: Any,
    out_cost: wp.array,
    out_gradient: Optional[wp.array] = None,
    **kwargs: Any,
):
    """Compute sparse residual cost and optional `Jᵀr` gradient.

    Lifecycle:
        1. Build reusable residual, Jacobian, and CSC buffers.
        2. Fill residuals and optional sparse Jacobian values.
        3. Reduce cost and deterministic CSC gradient.

    Args:
        self: Sparse task instance.
        var_values: Optimization variables.
        *args: Additional task inputs.
        out_cost: Shared per-batch cost buffer.
        out_gradient: Optional shared gradient buffer.
        **kwargs: Additional task inputs.
    """
    # --- build buffers ---
    self._build_sparse_gradient_buffers(var_values)
    if self.use_mask:
        self._build_frame_mask(var_values.device, var_values.batch_size)
    self.compute_weighted_residual(var_values, out_residual=self._sparse_residual_buf, row_offset=0)
    if out_gradient is not None:
        self.compute_weighted_sparse_jacobian_values(var_values, out_jacobian_values=self._sparse_jac_buf, nnz_offset=0)
    # --- reduce cost ---
    wp.launch(
        kernel=_sparse_cost_deterministic_kernel,
        dim=var_values.batch_size,
        inputs=[self._sparse_residual_buf, out_cost, self.residual_dim],
        device=var_values.device,
    )
    if out_gradient is None:
        return  # cost-only line-search step
    # --- reduce gradient ---
    col_offset = var_values.tangent_offset(self.var_key)
    wp.launch(
        kernel=_sparse_gradient_csc_kernel,
        dim=(var_values.batch_size, self._sparse_num_cols),
        inputs=[
            self._sparse_residual_buf,
            self._sparse_jac_buf,
            self._sparse_col_perm,
            self._sparse_col_nnz_starts,
            self._sparse_pattern.row_indices,
            col_offset,
        ],
        outputs=[out_gradient],
        device=var_values.device,
    )
