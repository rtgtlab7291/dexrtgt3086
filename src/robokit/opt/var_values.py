# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
import abc
from typing import Dict, Optional

import warp as wp
from typing_extensions import Self

from robokit.types import ArrayLike
from robokit.utils.warp_utils import wp_device_type


class Var(abc.ABC):
    @property
    @abc.abstractmethod
    def tangent_dim(self) -> int: ...

    @abc.abstractmethod
    def clone(self) -> Self: ...

    @abc.abstractmethod
    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional[Self] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
    ) -> Self: ...

    def invalidate(self):
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def device(self) -> wp_device_type: ...

    @abc.abstractmethod
    def gather(self, indices: ArrayLike, out: Optional[Self] = None) -> Self: ...

    @abc.abstractmethod
    def accept(self, accept_mask: ArrayLike, proposed: Self) -> Self:
        """Accept proposed values where accept_mask[i] == 1, keep self unchanged otherwise.

        Modifies self in-place. Copies all state including derived quantities.
        """
        ...


class VarValues(Var):
    """Container mapping named Var instances to a combined optimization variable.

    Implements Var so the optimizer treats it as a single variable. Each sub-variable
    owns a contiguous block of columns in the unified tangent space.
    """

    def __init__(self, **named_vars: Var):
        if not named_vars:
            raise ValueError("VarValues requires at least one variable")

        self._vars: Dict[str, Var] = dict(named_vars)
        self._offsets: Dict[str, int] = {}
        offset = 0
        for name, var in self._vars.items():
            self._offsets[name] = offset
            offset += var.tangent_dim
        self._tangent_dim = offset

        first_var = next(iter(self._vars.values()))
        self._batch_size = first_var.batch_size
        self._device = first_var.device

    @classmethod
    def from_var(cls, var: Var, name: str) -> "VarValues":
        return cls(**{name: var})

    def get(self, name: str) -> Var:
        return self._vars[name]

    def values(self):
        return self._vars.values()

    def tangent_offset(self, name: str) -> int:
        return self._offsets[name]

    @property
    def tangent_dim(self) -> int:
        return self._tangent_dim

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> wp_device_type:
        return self._device

    def clone(self) -> "VarValues":
        cloned_vars = {name: var.clone() for name, var in self._vars.items()}
        return VarValues(**cloned_vars)

    def _get_velocity(self, name: str, velocity: ArrayLike):
        offset = self._offsets[name]
        dim = self._vars[name].tangent_dim
        if offset == 0 and dim == velocity.shape[1]:
            return velocity
        return velocity[:, offset : offset + dim]

    def _get_tangent_mask(self, name: str, tangent_mask: Optional[wp.array]):
        if tangent_mask is None:
            return None
        offset = self._offsets[name]
        dim = self._vars[name].tangent_dim
        return tangent_mask[offset : offset + dim]

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["VarValues"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
    ) -> "VarValues":
        if out is not None:
            for name, var in self._vars.items():
                var.integrate(
                    self._get_velocity(name, velocity),
                    out=out.get(name),
                    tangent_mask=self._get_tangent_mask(name, tangent_mask),
                    weight_decay=weight_decay,
                )
            return out
        integrated_vars = {}
        for name, var in self._vars.items():
            integrated_vars[name] = var.integrate(
                self._get_velocity(name, velocity),
                tangent_mask=self._get_tangent_mask(name, tangent_mask),
                weight_decay=weight_decay,
            )
        return VarValues(**integrated_vars)

    def gather(self, indices: ArrayLike, out: Optional["VarValues"] = None) -> "VarValues":
        if out is None:
            return VarValues(**{name: var.gather(indices) for name, var in self._vars.items()})
        for name, var in self._vars.items():
            var.gather(indices, out=out.get(name))
        return out

    def accept(self, accept_mask: ArrayLike, proposed: "VarValues") -> "VarValues":
        for name, var in self._vars.items():
            var.accept(accept_mask, proposed.get(name))
        return self

    def invalidate(self):
        for var in self._vars.values():
            var.invalidate()
