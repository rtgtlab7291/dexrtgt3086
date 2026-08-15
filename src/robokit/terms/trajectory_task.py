# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
"""Replay a per-pose dense task at every frame of a trajectory."""

from typing import Any, Optional, Sequence

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import (
    GradientTask,
    ResidualTask,
    SparseTask,
    SparsityPattern,
    sparse_cost_and_gradient,
)
from robokit.utils.warp_utils import wp_device_type


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _pattern_kernel(
    frames: wp.array1d(dtype=wp.int32),
    residual_dim: int,
    tangent_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    slot, r, d = wp.tid()  # type: ignore
    row = slot * residual_dim + r
    row_indices[nnz_offset + row * tangent_dim + d] = row_offset + row
    col_indices[nnz_offset + row * tangent_dim + d] = frames[slot] * tangent_dim + d


@wp.kernel
def _select_kernel(
    src: wp.array2d(dtype=wp.float32),
    frames: wp.array1d(dtype=wp.int32),
    block: int,
    col_offset: int,
    frame_mask: wp.array2d(dtype=wp.uint8),
    use_mask: wp.bool,
    dst: wp.array2d(dtype=wp.float32),
):
    batch_idx, slot, i = wp.tid()  # type: ignore
    frame_idx = frames[slot]
    value = src[batch_idx, frame_idx * block + i]
    if use_mask and frame_mask[batch_idx, frame_idx] == wp.uint8(0):
        value = 0.0
    dst[batch_idx, col_offset + slot * block + i] = value


# --- task -------------------------------------------------------------------
class TrajectoryTask(RobotTask, SparseTask, GradientTask):
    """Apply a per-pose residual task independently to selected trajectory frames.

    The sparse Jacobian contains one dense tangent block per selected frame. Tasks that
    couple neighboring frames require a dedicated trajectory implementation.

    Args:
        dense_task: Per-pose residual task.
        num_frames: Number of trajectory frames.
        frame_indices: Frames to include. Defaults to all frames.
    """

    compute_weighted_cost_and_gradient = sparse_cost_and_gradient

    def __init__(
        self,
        dense_task: ResidualTask,
        num_frames: int,
        frame_indices: Optional[Sequence[int]] = None,
    ):
        self.dense_task = dense_task
        self.num_frames = num_frames
        self.frames = list(range(num_frames)) if frame_indices is None else [f % num_frames for f in frame_indices]
        self.var_key = dense_task.var_key
        self.robot = dense_task.robot
        self.precompute_collision_geometry = getattr(dense_task, "precompute_collision_geometry", "")
        self.device: Optional[wp_device_type] = None
        self._frames_wp: Optional[wp.array] = None
        self._residual_buf: Optional[wp.array] = None
        self._jacobian_buf: Optional[wp.array] = None

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self._frames_wp = wp.from_numpy(np.array(self.frames, dtype=np.int32), dtype=wp.int32, device=device)
        if hasattr(self.dense_task, "init_buffers"):
            self.dense_task.init_buffers(device)

    @property
    def residual_dim(self) -> int:
        return len(self.frames) * self.dense_task.residual_dim

    def _build_flat_var_values(self, state: Any) -> VarValues:
        if self.device is None:
            self.init_buffers(state.q.device)
        return VarValues.from_var(state.flatten(), self.var_key)

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        *args: Any,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        state = var_values.get(self.var_key)
        flat = self._build_flat_var_values(state)
        batch_size, rows = state.batch_size, self.dense_task.residual_dim
        device = state.q.device

        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=device)
            row_offset = 0
        if self._residual_buf is None or self._residual_buf.shape[0] != flat.batch_size:
            self._residual_buf = wp.zeros(
                (flat.batch_size, rows), dtype=wp.float32, device=device, requires_grad=state.q.requires_grad
            )

        self.dense_task.compute_weighted_residual(flat, out_residual=self._residual_buf)
        wp.launch(
            _select_kernel,
            dim=(batch_size, len(self.frames), rows),
            inputs=[
                self._residual_buf.reshape((batch_size, self.num_frames * rows)),
                self._frames_wp,
                rows,
                row_offset,
                self._get_frame_mask(),
                self.use_mask,
            ],
            outputs=[out_residual],
            device=device,
        )
        return out_residual

    def compute_sparse_jacobian_pattern(
        self,
        var_values: VarValues,
        *args: Any,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
        **kwargs: Any,
    ) -> SparsityPattern:
        state = var_values.get(self.var_key)
        if self.device is None:
            self.init_buffers(state.q.device)
        device = state.q.device
        rows = self.dense_task.residual_dim
        tangent_dim = (6 if state.has_floating_base else 0) + self.robot.spec.num_actuated_joints
        nnz = len(self.frames) * rows * tangent_dim

        pattern = SparsityPattern()
        pattern.row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=device) if out_row_indices is None else out_row_indices
        )
        pattern.col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=device) if out_col_indices is None else out_col_indices
        )
        wp.launch(
            _pattern_kernel,
            dim=(len(self.frames), rows, tangent_dim),
            inputs=[self._frames_wp, rows, tangent_dim, offset, nnz_offset],
            outputs=[pattern.row_indices, pattern.col_indices],
            device=device,
        )
        return pattern

    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: VarValues,
        *args: Any,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        state = var_values.get(self.var_key)
        flat = self._build_flat_var_values(state)
        batch_size, rows = state.batch_size, self.dense_task.residual_dim
        tangent_dim = flat.tangent_dim
        block = rows * tangent_dim
        device = state.q.device

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((batch_size, len(self.frames) * block), dtype=wp.float32, device=device)
            nnz_offset = 0
        if self._jacobian_buf is None or self._jacobian_buf.shape[0] != flat.batch_size:
            self._jacobian_buf = wp.zeros((flat.batch_size, rows, tangent_dim), dtype=wp.float32, device=device)

        # dense kernels only write their nonzeros, so the block starts clean every call
        self._jacobian_buf.zero_()
        self.dense_task.compute_weighted_jacobian_analytic(flat, out_jacobian=self._jacobian_buf)
        wp.launch(
            _select_kernel,
            dim=(batch_size, len(self.frames), block),
            inputs=[
                self._jacobian_buf.reshape((batch_size, self.num_frames * block)),
                self._frames_wp,
                block,
                nnz_offset,
                self._get_frame_mask(),
                self.use_mask,
            ],
            outputs=[out_jacobian_values],
            device=device,
        )
        return out_jacobian_values


__all__ = ["TrajectoryTask"]
