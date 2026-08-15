# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
from typing import Any, Optional, Sequence

import warp as wp
from warp.types import matrix, vector


# wp.Device is public in newer Warp; older versions (e.g. 1.10) only expose wp.context.Device.
wp_device_type = wp.Device if hasattr(wp, "Device") else wp.context.Device


class wp_vec6f(vector(length=6, dtype=wp.float32)):
    pass


class wp_vec7f(vector(length=7, dtype=wp.float32)):
    pass


class wp_mat66f(matrix(shape=(6, 6), dtype=wp.float32)):
    pass


wp_vec6 = wp_vec6f
wp_vec7 = wp_vec7f
wp_mat66 = wp_mat66f


# --- kernels ---------------------------------------------------------------
@wp.kernel
def _gather_2d_kernel(
    source: wp.array2d(dtype=Any),
    indices: wp.array1d(dtype=wp.int32),
    out: wp.array2d(dtype=Any),
):
    tid, j = wp.tid()
    out[tid, j] = source[indices[tid], j]


@wp.kernel
def _gather_3d_kernel(
    source: wp.array3d(dtype=Any),
    indices: wp.array1d(dtype=wp.int32),
    out: wp.array3d(dtype=Any),
):
    tid, i, j = wp.tid()
    out[tid, i, j] = source[indices[tid], i, j]


@wp.kernel
def _gather_4d_kernel(
    source: wp.array4d(dtype=Any),
    indices: wp.array1d(dtype=wp.int32),
    out: wp.array4d(dtype=Any),
):
    tid, i, j, k = wp.tid()
    out[tid, i, j, k] = source[indices[tid], i, j, k]


@wp.kernel
def _repeat_first_dim_1d_kernel(
    source: wp.array1d(dtype=Any),
    repeats: wp.int32,
    out: wp.array1d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid // int(repeats)
    out[tid] = source[src_idx]


@wp.kernel
def _repeat_first_dim_2d_kernel(
    source: wp.array2d(dtype=Any),
    repeats: wp.int32,
    out: wp.array2d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid // int(repeats)
    num_cols = out.shape[1]
    for j in range(num_cols):
        out[tid, j] = source[src_idx, j]


@wp.kernel
def _repeat_first_dim_3d_kernel(
    source: wp.array3d(dtype=Any),
    repeats: wp.int32,
    out: wp.array3d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid // int(repeats)
    dim1 = out.shape[1]
    dim2 = out.shape[2]
    for j in range(dim1):
        for k in range(dim2):
            out[tid, j, k] = source[src_idx, j, k]


@wp.kernel
def _repeat_first_dim_4d_kernel(
    source: wp.array4d(dtype=Any),
    repeats: wp.int32,
    out: wp.array4d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid // int(repeats)
    dim1 = out.shape[1]
    dim2 = out.shape[2]
    dim3 = out.shape[3]
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                out[tid, i, j, k] = source[src_idx, i, j, k]


@wp.kernel
def _tile_1d_kernel(
    source: wp.array1d(dtype=Any),
    repeats: wp.int32,
    out: wp.array1d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % int(source.shape[0])
    out[tid] = source[src_idx]


@wp.kernel
def _tile_2d_kernel(
    source: wp.array2d(dtype=Any),
    repeats: wp.int32,
    out: wp.array2d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % int(source.shape[0])
    num_cols = out.shape[1]
    for j in range(num_cols):
        out[tid, j] = source[src_idx, j]


@wp.kernel
def _tile_3d_kernel(
    source: wp.array3d(dtype=Any),
    repeats: wp.int32,
    out: wp.array3d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % int(source.shape[0])
    dim1 = out.shape[1]
    dim2 = out.shape[2]
    for j in range(dim1):
        for k in range(dim2):
            out[tid, j, k] = source[src_idx, j, k]


@wp.kernel
def _tile_4d_kernel(
    source: wp.array4d(dtype=Any),
    repeats: wp.int32,
    out: wp.array4d(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % int(source.shape[0])
    dim1 = out.shape[1]
    dim2 = out.shape[2]
    dim3 = out.shape[3]
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                out[tid, i, j, k] = source[src_idx, i, j, k]


@wp.kernel
def _stack_1d_kernel(
    src: wp.array1d(dtype=Any),
    out: wp.array2d(dtype=Any),
    stack_idx: int,
    axis: int,
):
    i = wp.tid()
    if axis == 0:
        out[stack_idx, i] = src[i]
    else:
        out[i, stack_idx] = src[i]


@wp.kernel
def _stack_2d_kernel(
    src: wp.array2d(dtype=Any),
    out: wp.array3d(dtype=Any),
    stack_idx: int,
    axis: int,
):
    i, j = wp.tid()  # pyright: ignore
    if axis == 0:
        out[stack_idx, i, j] = src[i, j]
    elif axis == 1:
        out[i, stack_idx, j] = src[i, j]
    else:
        out[i, j, stack_idx] = src[i, j]


@wp.kernel
def _stack_3d_kernel(
    src: wp.array3d(dtype=Any),
    out: wp.array4d(dtype=Any),
    stack_idx: int,
    axis: int,
):
    i, j, k = wp.tid()  # pyright: ignore
    if axis == 0:
        out[stack_idx, i, j, k] = src[i, j, k]
    elif axis == 1:
        out[i, stack_idx, j, k] = src[i, j, k]
    elif axis == 2:
        out[i, j, stack_idx, k] = src[i, j, k]
    else:
        out[i, j, k, stack_idx] = src[i, j, k]


@wp.kernel
def _masked_copy_2d_kernel(
    mask: wp.array1d(dtype=wp.int32),
    source: wp.array2d(dtype=Any),
    out: wp.array2d(dtype=Any),
):
    tid, j = wp.tid()
    if mask[tid] == 1:
        out[tid, j] = source[tid, j]


@wp.kernel
def _masked_copy_3d_kernel(
    mask: wp.array1d(dtype=wp.int32),
    source: wp.array3d(dtype=Any),
    out: wp.array3d(dtype=Any),
):
    tid, i, j = wp.tid()
    if mask[tid] == 1:
        out[tid, i, j] = source[tid, i, j]


@wp.kernel
def _masked_copy_kernel(
    mask: wp.array1d(dtype=wp.int32),
    source: wp.array4d(dtype=Any),
    out: wp.array4d(dtype=Any),
):
    tid, i, j, k = wp.tid()
    if mask[tid] == 1:
        out[tid, i, j, k] = source[tid, i, j, k]


# --- public functions ------------------------------------------------------
def masked_copy(mask: wp.array, src: wp.array, out: wp.array) -> wp.array:
    """
    Masked copy: `out[i] = src[i]` where `mask[i] == 1`, unchanged otherwise.

    Args:
        mask: 1D int32 array of 0/1 flags, shape [N].
        src: Array to copy from, shape [N, ...].
        out: Array to copy into (modified in-place), shape [N, ...].

    Returns:
        out array (modified in-place).
    """
    batch_size = mask.shape[0]
    device = src.device

    if src.ndim == 1:
        src_2d = src.reshape((src.shape[0], 1))
        dst_2d = out.reshape((out.shape[0], 1))
        wp.launch(_masked_copy_2d_kernel, dim=(batch_size, 1), inputs=[mask, src_2d, dst_2d], device=device)
    elif src.ndim == 2:
        wp.launch(_masked_copy_2d_kernel, dim=(batch_size, src.shape[1]), inputs=[mask, src, out], device=device)
    elif src.ndim == 3:
        wp.launch(
            _masked_copy_3d_kernel, dim=(batch_size, src.shape[1], src.shape[2]), inputs=[mask, src, out], device=device
        )
    elif src.ndim == 4:
        wp.launch(
            _masked_copy_kernel,
            dim=(batch_size, src.shape[1], src.shape[2], src.shape[3]),
            inputs=[mask, src, out],
            device=device,
        )
    else:
        raise ValueError(f"masked_copy() only supports 1D, 2D, 3D, or 4D arrays, got ndim={src.ndim}")

    return out


def gather(source: wp.array, indices: wp.array, out: Optional[wp.array] = None) -> wp.array:
    """
    Gather rows from an array using indices along the first dimension.

    This is equivalent to np.take(source, indices, axis=0) or source[indices].

    Args:
        source: Source array with shape [N, ...] to gather from.
        indices: 1D int32 array of row indices to gather, shape [M].
        out: Optional pre-allocated destination array with shape [M, ...].

    Returns:
        Gathered array with shape [M, ...] where M = len(indices).

    Example:
        [[1,2], [3,4], [5,6]] with indices=[2, 0] -> [[5,6], [1,2]]
    """
    num_gather = indices.shape[0]
    device = source.device

    if out is None:
        out_shape = (num_gather,) + source.shape[1:]
        out = wp.empty(out_shape, dtype=source.dtype, device=device, requires_grad=source.requires_grad)

    if source.ndim == 1:
        source_2d = source.reshape((source.shape[0], 1))
        dest_2d = out.reshape((num_gather, 1))
        wp.launch(_gather_2d_kernel, dim=(num_gather, 1), inputs=[source_2d, indices, dest_2d], device=device)
    elif source.ndim == 2:
        wp.launch(_gather_2d_kernel, dim=(num_gather, source.shape[1]), inputs=[source, indices, out], device=device)
    elif source.ndim == 3:
        wp.launch(
            _gather_3d_kernel,
            dim=(num_gather, source.shape[1], source.shape[2]),
            inputs=[source, indices, out],
            device=device,
        )
    elif source.ndim == 4:
        wp.launch(
            _gather_4d_kernel,
            dim=(num_gather, source.shape[1], source.shape[2], source.shape[3]),
            inputs=[source, indices, out],
            device=device,
        )
    else:
        raise ValueError(f"gather() only supports 1D, 2D, 3D, or 4D arrays, got ndim={source.ndim}")

    return out


def repeat(array: wp.array, repeats: int, out: Optional[wp.array] = None) -> wp.array:
    """
    Repeat elements along the first dimension, keeping elements contiguous.

    Each element in the first dimension is repeated `repeats` times consecutively.
    This is equivalent to np.repeat(array, repeats, axis=0).

    Args:
        array: Source array with shape [N, ...].
        repeats: Number of times to repeat each element.
        out: Optional pre-allocated destination array with shape [N*repeats, ...].

    Example:
        [a, b] with repeats=3 -> [a, a, a, b, b, b]
        [[1,2], [3,4]] with repeats=2 -> [[1,2], [1,2], [3,4], [3,4]]
    """
    if repeats < 1:
        raise ValueError("repeat requires repeats >= 1")

    device = array.device
    out_shape = (array.shape[0] * repeats,) + array.shape[1:]
    if out is None:
        out = wp.empty(out_shape, dtype=array.dtype, device=device, requires_grad=array.requires_grad)

    if array.ndim == 1:
        wp.launch(
            _repeat_first_dim_1d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 2:
        wp.launch(
            _repeat_first_dim_2d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 3:
        wp.launch(
            _repeat_first_dim_3d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 4:
        wp.launch(
            _repeat_first_dim_4d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    else:
        raise ValueError(f"repeat() only supports 1D, 2D, 3D, or 4D arrays, got ndim={array.ndim}")

    return out


def tile(array: wp.array, repeats: int) -> wp.array:
    """
    Tile an array by repeating the entire first dimension.

    The whole array along the first dimension is repeated `repeats` times.
    This is equivalent to np.tile(array, (repeats, 1, ...)).

    Example:
        [a, b] with repeats=3 -> [a, b, a, b, a, b]
        [[1,2], [3,4]] with repeats=2 -> [[1,2], [3,4], [1,2], [3,4]]
    """
    if repeats < 1:
        raise ValueError("tile requires repeats >= 1")

    device = array.device
    out_shape = (array.shape[0] * repeats,) + array.shape[1:]
    out = wp.empty(out_shape, dtype=array.dtype, device=device, requires_grad=array.requires_grad)

    if array.ndim == 1:
        wp.launch(
            _tile_1d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 2:
        wp.launch(
            _tile_2d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 3:
        wp.launch(
            _tile_3d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    elif array.ndim == 4:
        wp.launch(
            _tile_4d_kernel,
            dim=out_shape[0],
            inputs=[array, repeats, out],
            device=device,
        )
    else:
        raise ValueError(f"tile() only supports 1D, 2D, 3D, or 4D arrays, got ndim={array.ndim}")

    return out


def stack(arrays: Sequence[wp.array], axis: int = 0, out: Optional[wp.array] = None) -> wp.array:
    """
    Stack arrays along a new axis, like np.stack.

    All input arrays must have the same shape and dtype. A new dimension of size
    len(arrays) is inserted at position `axis` in the output.

    Supports 1D→2D, 2D→3D, and 3D→4D (warp maximum is 4D).

    Args:
        arrays: Sequence of arrays with identical shape and dtype.
        axis: Position of the new dimension in the output (default 0).
        out: Optional pre-allocated destination array.

    Example:
        [arr(N,)] * M with axis=0 -> (M, N)
        [arr(N,)] * M with axis=1 -> (N, M)
        [arr(N,K)] * M with axis=1 -> (N, M, K)
    """
    num_arrays = len(arrays)
    first = arrays[0]
    ndim = first.ndim
    shape = first.shape
    device = first.device
    dtype = first.dtype

    out_shape = list(shape)
    out_shape.insert(axis, num_arrays)
    out_shape = tuple(out_shape)

    if out is None:
        out = wp.empty(out_shape, dtype=dtype, device=device)

    if ndim == 1:
        for idx, arr in enumerate(arrays):
            wp.launch(_stack_1d_kernel, dim=shape[0], inputs=[arr, out, idx, axis], device=device)
    elif ndim == 2:
        for idx, arr in enumerate(arrays):
            wp.launch(_stack_2d_kernel, dim=(shape[0], shape[1]), inputs=[arr, out, idx, axis], device=device)
    elif ndim == 3:
        for idx, arr in enumerate(arrays):
            wp.launch(
                _stack_3d_kernel,
                dim=(shape[0], shape[1], shape[2]),
                inputs=[arr, out, idx, axis],
                device=device,
            )
    else:
        raise ValueError(f"stack() supports 1D, 2D, or 3D input arrays, got ndim={ndim}")

    return out


__all__ = [
    "wp_vec6f",
    "wp_vec7f",
    "wp_vec6",
    "wp_vec7",
    "wp_mat66f",
    "wp_mat66",
    "masked_copy",
    "gather",
    "repeat",
    "tile",
    "stack",
    "wp_device_type",
]
