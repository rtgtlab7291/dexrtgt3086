import abc
from typing import TYPE_CHECKING, Optional

from typing_extensions import Self

from robokit.types import ArrayLike


if TYPE_CHECKING:
    from robokit.utils.warp_utils import wp_device_type


class Var(abc.ABC):
    @property
    @abc.abstractmethod
    def tangent_dim(self) -> int: ...

    @abc.abstractmethod
    def clone(self) -> Self: ...

    @abc.abstractmethod
    def integrate(self, velocity: ArrayLike, out: Optional[Self] = None) -> Self: ...

    def invalidate(self):
        pass


class WarpVar(Var):
    @property
    @abc.abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def device(self) -> "wp_device_type": ...

    @abc.abstractmethod
    def gather(self, indices: ArrayLike, dest: Optional[Self] = None) -> Self: ...


__all__ = ["Var", "WarpVar"]
