import abc
from typing import TYPE_CHECKING, Literal, Optional, Union, overload

import numpy as np
import pinocchio as pin
import torch
import warp as wp
from typing_extensions import Self

from robokit.types import ArrayLike


if TYPE_CHECKING:
    from robokit.lie.pinocchio_se3 import PinocchioSE3
    from robokit.lie.torch_se3 import TorchSE3
    from robokit.lie.warp_se3 import WarpSE3


class SE3(abc.ABC):
    """SE(3) representation.

    Internal parameterization: [x, y, z, qw, qx, qy, qz].
    Tangent parameterization: [vx, vy, vz, omega_x, omega_y, omega_z].
    """

    @staticmethod
    def _infer_backend(
        param: Union[pin.SE3, ArrayLike],
        backend: Optional[Literal["numpy", "torch", "warp"]] = None,
    ) -> Literal["numpy", "torch", "warp"]:
        if backend is not None:
            return backend
        if isinstance(param, (np.ndarray, pin.SE3)):
            return "numpy"
        elif isinstance(param, torch.Tensor):
            return "torch"
        elif isinstance(param, wp.array):
            return "warp"
        else:
            raise ValueError(
                f"Cannot infer backend from type: {type(param)}. Expected pin.SE3, numpy.ndarray, torch.Tensor, or wp.array."
            )

    @staticmethod
    def _get_se3_class(
        backend: Literal["numpy", "torch", "warp"],
    ) -> Union["type[PinocchioSE3]", "type[TorchSE3]", "type[WarpSE3]"]:
        if backend == "numpy":
            from robokit.lie.pinocchio_se3 import PinocchioSE3

            return PinocchioSE3
        elif backend == "torch":
            from robokit.lie.torch_se3 import TorchSE3

            return TorchSE3
        elif backend == "warp":
            from robokit.lie.warp_se3 import WarpSE3

            return WarpSE3
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @overload
    def __new__(cls, se3_like: "wp.array", backend: Optional[Literal["warp"]] = ...) -> "WarpSE3": ...
    @overload
    def __new__(cls, se3_like: "torch.Tensor", backend: Optional[Literal["torch"]] = ...) -> "TorchSE3": ...
    @overload
    def __new__(
        cls, se3_like: Union[pin.SE3, np.ndarray], backend: Optional[Literal["numpy"]] = ...
    ) -> "PinocchioSE3": ...
    def __new__(
        cls,
        se3_like: Union[pin.SE3, ArrayLike],
        backend: Optional[Literal["numpy", "torch", "warp"]] = None,
    ) -> "SE3":
        # If se3_like is already a SE3 instance, return it directly
        if isinstance(se3_like, SE3):
            return se3_like

        # If called directly on SE3, determine the backend and return appropriate subclass
        if cls is SE3:
            backend = cls._infer_backend(se3_like, backend)
            se3_cls = cls._get_se3_class(backend)
            return se3_cls.__new__(se3_cls, se3_like, backend)  # type: ignore
        else:
            # Called on subclass, use normal instantiation
            return super().__new__(cls)

    @property
    @abc.abstractmethod
    def xyz_wxyz(self) -> Union[np.ndarray, "torch.Tensor"]:
        """Internal parameterization of SE(3) as [x, y, z, qw, qx, qy, qz]"""
        ...

    @property
    @abc.abstractmethod
    def xyz(self) -> Union[np.ndarray, "torch.Tensor"]:
        """Translation part of SE(3) as [x, y, z]"""
        ...

    @property
    @abc.abstractmethod
    def quat_wxyz(self) -> Union[np.ndarray, "torch.Tensor"]:
        """Quaternion part of SE(3) as [qw, qx, qy, qz]"""
        ...

    @abc.abstractmethod
    def as_matrix(self) -> Union[np.ndarray, "torch.Tensor"]: ...

    @staticmethod
    def exp(
        log_transform: Union[np.ndarray, "torch.Tensor"],
        backend: Optional[Literal["numpy", "torch", "warp"]] = None,
    ) -> "SE3":
        backend = SE3._infer_backend(log_transform, backend)
        se3_cls = SE3._get_se3_class(backend)
        return se3_cls.exp(log_transform, backend)  # type: ignore

    @abc.abstractmethod
    def inverse(self) -> Self: ...

    @abc.abstractmethod
    def log(self) -> Union[np.ndarray, "torch.Tensor"]:
        """The log representation in [v, omega] format, linear first, then angular."""
        ...

    @abc.abstractmethod
    def adjoint(self) -> Union[np.ndarray, "torch.Tensor"]:
        """The 6x6 adjoint matrix Ad_T = [[R, skew(t) @ R], [0, R]] mapping twists as v' = Ad_T @ v with [v, omega] ordering (linear first, then angular)."""
        ...

    # NOTE jlog for TorchSE3/WarpSE3 is not implemented yet
    # @abc.abstractmethod
    def jlog(self) -> Union[np.ndarray, "torch.Tensor"]: ...

    def __getitem__(self, key) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()

    @abc.abstractmethod
    def __mul__(self, other: "SE3") -> Self: ...

    @abc.abstractmethod
    def __rmul__(self, other: "SE3") -> Self: ...

    def __matmul__(self, other: "SE3") -> Self:
        return self.__mul__(other)

    def __rmatmul__(self, other: "SE3") -> Self:
        return self.__rmul__(other)
