import abc
from typing import TYPE_CHECKING, Literal, Optional, Union, overload

import numpy as np
import pinocchio as pin

from robokit.types import ArrayLike


try:
    import torch  # torch is optional

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


try:
    import warp as wp  # warp is optional

    _WARP_AVAILABLE = True
except ImportError:
    _WARP_AVAILABLE = False

if TYPE_CHECKING:
    from robokit.lie.pinocchio_so3 import PinocchioSO3
    from robokit.lie.torch_so3 import TorchSO3
    from robokit.lie.warp_so3 import WarpSO3


class SO3(abc.ABC):
    """SO(3) representation using quaternions.

    Internal parameterization: [qw, qx, qy, qz].
    Tangent parameterization: [omega_x, omega_y, omega_z].
    """

    @staticmethod
    def _infer_backend(
        param: Union[pin.Quaternion, ArrayLike],
        backend: Optional[Literal["numpy", "torch", "warp"]] = None,
    ) -> Literal["numpy", "torch", "warp"]:
        if backend is not None:
            return backend
        if isinstance(param, (np.ndarray, pin.Quaternion)):
            return "numpy"
        elif _TORCH_AVAILABLE and isinstance(param, torch.Tensor):
            return "torch"
        elif _WARP_AVAILABLE and isinstance(param, wp.array):
            return "warp"
        else:
            raise ValueError(
                f"Cannot infer backend from type: {type(param)}. Expected pin.Quaternion, numpy.ndarray, torch.Tensor, or wp.array."
            )

    @staticmethod
    def _get_so3_class(
        backend: Literal["numpy", "torch", "warp"],
    ) -> Union["type[PinocchioSO3]", "type[TorchSO3]", "type[WarpSO3]"]:
        # fmt: off
        if backend == "numpy":
            from robokit.lie.pinocchio_so3 import PinocchioSO3
            return PinocchioSO3
        elif backend == "torch":
            from robokit.lie.torch_so3 import TorchSO3
            return TorchSO3
        elif backend == "warp":
            from robokit.lie.warp_so3 import WarpSO3
            return WarpSO3
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        # fmt: on

    @overload
    def __new__(cls, so3_like: "wp.array", backend: Optional[Literal["warp"]] = ...) -> "WarpSO3": ...
    @overload
    def __new__(cls, so3_like: "torch.Tensor", backend: Optional[Literal["torch"]] = ...) -> "TorchSO3": ...
    @overload
    def __new__(
        cls, so3_like: Union[pin.Quaternion, np.ndarray], backend: Optional[Literal["numpy"]] = ...
    ) -> "PinocchioSO3": ...
    def __new__(
        cls,
        so3_like: Union[pin.Quaternion, ArrayLike],
        backend: Optional[Literal["numpy", "torch", "warp"]] = None,
    ) -> "SO3":
        # If so3_like is already a SO3 instance, return it directly
        if isinstance(so3_like, SO3):
            return so3_like

        # If called directly on SO3, determine the backend and return appropriate subclass
        if cls is SO3:
            backend = cls._infer_backend(so3_like, backend)
            so3_cls = cls._get_so3_class(backend)
            return so3_cls.__new__(so3_cls, so3_like, backend)  # type: ignore
        else:
            # Called on subclass, use normal instantiation
            return super().__new__(cls)

    @property
    @abc.abstractmethod
    def wxyz(self) -> Union[np.ndarray, "torch.Tensor"]:
        """Quaternion part of SO(3) as [qw, qx, qy, qz]"""
        ...

    @staticmethod
    def from_matrix(
        matrix: Union[np.ndarray, "torch.Tensor"],
        backend: Optional[Literal["numpy", "torch", "warp"]] = None,
    ) -> "SO3":
        backend = SO3._infer_backend(matrix, backend)
        so3_cls = SO3._get_so3_class(backend)
        return so3_cls.from_matrix(matrix, backend)  # pyright: ignore[reportArgumentType]

    def as_matrix(self):
        raise NotImplementedError

    @staticmethod
    def exp(
        log_rot: Union[np.ndarray, "torch.Tensor"],
        backend: Optional[Literal["numpy", "torch", "warp"]] = None,
    ) -> "SO3":
        backend = SO3._infer_backend(log_rot, backend)
        so3_cls = SO3._get_so3_class(backend)
        return so3_cls.exp(log_rot, backend)  # pyright: ignore[reportArgumentType]

    def log(self) -> Union[np.ndarray, "torch.Tensor"]:
        """Return tangent vector [omega_x, omega_y, omega_z]."""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()

    def __mul__(self, other: "SO3") -> "SO3":
        raise NotImplementedError

    def __rmul__(self, other: "SO3") -> "SO3":
        raise NotImplementedError

    def __matmul__(self, other: "SO3") -> "SO3":
        return self.__mul__(other)

    def __rmatmul__(self, other: "SO3") -> "SO3":
        return self.__rmul__(other)
