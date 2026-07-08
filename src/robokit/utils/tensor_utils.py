from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, TypeVar, Union, overload

import numpy as np
import torch
import warp as wp
from einops import repeat
from numpy.typing import ArrayLike


@overload
def to_numpy(x: torch.Tensor, preserve_sequence: bool = ...) -> np.ndarray: ...
@overload
def to_numpy(x: np.ndarray, preserve_sequence: bool = ...) -> np.ndarray: ...
@overload
def to_numpy(x: wp.array, preserve_sequence: bool = ...) -> np.ndarray: ...
@overload
def to_numpy(x: ArrayLike, preserve_sequence: bool = ...) -> np.ndarray: ...
@overload
def to_numpy(x: None, preserve_sequence: bool = ...) -> None: ...
@overload
def to_numpy(x: Dict[Any, Any], preserve_sequence: bool = ...) -> Dict[Any, np.ndarray]: ...
@overload
def to_numpy(x: Sequence[Any], preserve_sequence: Literal[True]) -> Sequence[np.ndarray]: ...
@overload
def to_numpy(x: Sequence[Any], preserve_sequence: Literal[False] = ...) -> np.ndarray: ...
def to_numpy(
    x: Any, preserve_sequence: bool = True
) -> Optional[Union[np.ndarray, Dict[Any, np.ndarray], Sequence[np.ndarray]]]:
    """Convert input to numpy array.

    Args:
        x (Any): Input to be converted.
        preserve_sequence (bool, optional): Whether to preserve sequence or convert to numpy array. Defaults to True.

    Example:
        >>> to_numpy(torch.tensor([1, 2, 3]))
        array([1, 2, 3])
        >>> to_numpy([torch.tensor([1]), torch.tensor([2])], preserve_sequence=True)
        [array([1]), array([2])]
        >>> to_numpy(np.array([1, 2, 3]))
        array([1, 2, 3])

    Note:
        These helpers are handy, but because they try to handle many input shapes and types
        they can introduce ambiguity in complex "if" logic. If you know the input is a
        torch.Tensor, prefer calling `torch.numpy(...)` directly.
        Reserve this helper for simple or interactive code to reduce subtle bugs.
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.numpy(force=True)
    elif isinstance(x, wp.array):
        return x.numpy()
    elif x is None:
        return None
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif preserve_sequence and isinstance(x, Sequence) and not isinstance(x, (str, bytes, np.ndarray, torch.Tensor)):
        return type(x)(to_numpy(elem, preserve_sequence=preserve_sequence) for elem in x)  # pyright: ignore[reportCallIssue]
    try:
        return np.asarray(x)
    except Exception as _:
        return x  # pyright: ignore[reportReturnType]


@overload
def to_torch(
    x: np.ndarray,
    preserve_sequence: bool = ...,
    *,
    dtype: Optional[torch.dtype] = ...,
    device: Optional[Union[str, torch.device]] = ...,
) -> torch.Tensor: ...
@overload
def to_torch(
    x: torch.Tensor,
    preserve_sequence: bool = ...,
    *,
    dtype: Optional[torch.dtype] = ...,
    device: Optional[Union[str, torch.device]] = ...,
) -> torch.Tensor: ...
@overload
def to_torch(
    x: None,
    preserve_sequence: bool = ...,
    *,
    dtype: Optional[torch.dtype] = ...,
    device: Optional[Union[str, torch.device]] = ...,
) -> None: ...
@overload
def to_torch(
    x: Dict[Any, Any],
    preserve_sequence: bool = ...,
    *,
    dtype: Optional[torch.dtype] = ...,
    device: Optional[Union[str, torch.device]] = ...,
) -> Dict[Any, torch.Tensor]: ...
@overload
def to_torch(
    x: Sequence[Any],
    preserve_sequence: Literal[True],
    *,
    dtype: Optional[torch.dtype] = ...,
    device: Optional[Union[str, torch.device]] = ...,
) -> Sequence[torch.Tensor]: ...
@overload
def to_torch(
    x: Sequence[Any],
    preserve_sequence: Literal[False] = ...,
    *,
    dtype: Optional[torch.dtype] = ...,
    device: Optional[Union[str, torch.device]] = ...,
) -> torch.Tensor: ...
def to_torch(
    x: Any,
    preserve_sequence: bool = True,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[Union[torch.Tensor, Dict[Any, torch.Tensor], Sequence[torch.Tensor]]]:
    """Convert input to torch tensor.

    Args:
        x (Any): Input to be converted.
        preserve_sequence (bool, optional): Whether to preserve sequence or convert to torch tensor. Defaults to True.
        dtype (torch.dtype | None, optional): Desired dtype of the output tensor. If None, infer/keep.
        device (str | torch.device | None, optional): Desired device of the output tensor. If None, keep current/default.

    Example:
        >>> to_torch(np.array([1, 2, 3]))
        tensor([1, 2, 3])
        >>> to_torch([np.array([1]), np.array([2])], preserve_sequence=True)
        [tensor([1]), tensor([2])]
        >>> to_torch(torch.tensor([1, 2, 3]))
        tensor([1, 2, 3])

    Note:
        These helpers are handy, but because they try to handle many input shapes and types
        they can introduce ambiguity in complex "if" logic. If you know the input is a
        Numpy array, prefer calling `torch.from_numpy(...)` or `torch.as_tensor(...)`
        directly. Reserve this helper for simple or interactive code to reduce subtle bugs.
    """
    if isinstance(x, torch.Tensor):
        # Move/cast only if requested to avoid unnecessary copies
        if dtype is not None or device is not None:
            return x.to(dtype=dtype, device=device)
        return x
    elif isinstance(x, np.ndarray):
        # Use as_tensor to honor dtype/device while remaining zero-copy on CPU when possible
        return torch.as_tensor(x, dtype=dtype, device=device)
    elif isinstance(x, dict):
        return {k: to_torch(v, preserve_sequence=preserve_sequence, dtype=dtype, device=device) for k, v in x.items()}
    elif preserve_sequence and isinstance(x, Sequence) and not isinstance(x, (str, bytes, np.ndarray, torch.Tensor)):
        return type(x)(to_torch(elem, preserve_sequence=preserve_sequence, dtype=dtype, device=device) for elem in x)  # pyright: ignore[reportCallIssue]
    elif x is None:
        return None
    try:
        return torch.as_tensor(x, dtype=dtype, device=device)
    except Exception as _:
        return x  # pyright: ignore[reportReturnType]


T = TypeVar("T")


@overload
def batchify_tensor(x: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]: ...
@overload
def batchify_tensor(x: np.ndarray, batch_size: int) -> Iterator[np.ndarray]: ...
@overload
def batchify_tensor(x: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]: ...
def batchify_tensor(
    x: Union[torch.Tensor, np.ndarray, Sequence[T]], batch_size: int
) -> Iterator[Union[torch.Tensor, np.ndarray, Sequence[T]]]:
    """Split data into sequential batches of specified size.

    Args:
        x: Input data to split into batches. Can be:
            - torch.Tensor: Batched along first dimension
            - np.ndarray: Batched along first dimension
            - Sequence: Any homogeneous sequence
        batch_size: Size of each batch. Must be positive.

    Yields:
        Sequential batches of the input with same type as input.
        Each batch has size <= batch_size.

    Raises:
        ValueError: If batch_size <= 0

    Examples:
        >>> list(batchify_tensor(torch.tensor([1, 2, 3, 4, 5]), 2))[0].tolist()
        [1, 2]
        >>> list(batchify_tensor(np.array([10, 20, 30, 40]), 3))[1].tolist()
        [40]
        >>> list(batchify_tensor(['a', 'b', 'c', 'd'], 2))[0]
        ['a', 'b']
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    num_items = len(x)
    for start_idx in range(0, num_items, batch_size):
        end_idx = min(start_idx + batch_size, num_items)
        yield x[start_idx:end_idx]


@overload
def padding_tensor(x: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
@overload
def padding_tensor(x: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def padding_tensor(x: List[T], target_size: int) -> Tuple[List[T], List[bool]]: ...
def padding_tensor(
    x: Union[torch.Tensor, np.ndarray, List[T]], target_size: int
) -> Tuple[Union[torch.Tensor, np.ndarray, List[T]], Union[torch.Tensor, np.ndarray, List[bool]]]:
    """Pad data to target size by repeating last element.

    Args:
        x: Input data to pad (torch.Tensor, np.ndarray, or List)
        target_size: Desired size of first dimension after padding

    Returns:
        Tuple of (padded_data, mask) where:
            padded_data: Input padded to target_size by repeating last element
            mask: Boolean mask indicating original (True) vs padded (False) elements

    Examples:
        >>> padded, mask = padding_tensor(torch.tensor([1, 2, 3]), 5)
        >>> padded.tolist()
        [1, 2, 3, 3, 3]
        >>> mask.tolist()
        [True, True, True, False, False]
        >>> padded, mask = padding_tensor(np.array([10, 20]), 4)
        >>> padded.tolist()
        [10, 20, 20, 20]
        >>> padding_tensor(['a', 'b'], 4)[0]
        ['a', 'b', 'b', 'b']
    """
    current_size = len(x) if isinstance(x, list) else x.shape[0]
    if current_size >= target_size:
        if isinstance(x, torch.Tensor):
            return x, torch.ones(current_size, device=x.device, dtype=torch.bool)
        elif isinstance(x, np.ndarray):
            return x, np.ones(current_size, dtype=bool)
        return x, [True] * current_size

    padding_size = target_size - current_size
    last_element = [x[-1]] if isinstance(x, list) else x[-1:]

    if isinstance(x, torch.Tensor):
        padded = torch.cat([x, repeat(last_element, "1 ... -> n ...", n=padding_size)], dim=0)  # type: ignore[arg-type]
        mask = torch.ones(target_size, device=x.device, dtype=torch.bool)
        mask[current_size:] = False
    elif isinstance(x, np.ndarray):
        padded = np.concatenate([x, repeat(last_element, "1 ... -> n ...", n=padding_size)], axis=0)  # type: ignore[arg-type]
        mask = np.ones(target_size, dtype=bool)
        mask[current_size:] = False
    else:
        padded = x + [x[-1]] * padding_size
        mask = [True] * current_size + [False] * padding_size

    return padded, mask
