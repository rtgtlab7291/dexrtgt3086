# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
from typing import Dict, Optional

import warp as wp

from robokit.opt.variables import WarpVar
from robokit.types import ArrayLike
from robokit.utils.warp_utils import wp_device_type


@wp.kernel
def _slice_velocity_kernel(
    velocity: wp.array2d(dtype=wp.float32),
    col_start: int,
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, col_idx = wp.tid()
    out[batch_idx, col_idx] = velocity[batch_idx, col_start + col_idx]


class WarpVarValues(WarpVar):
    """Container mapping named WarpVar instances to a combined optimization variable.

    Implements WarpVar so the optimizer treats it as a single variable. Each sub-variable
    owns a contiguous block of columns in the unified tangent space.
    """

    def __init__(self, **named_vars: WarpVar):
        if not named_vars:
            raise ValueError("WarpVarValues requires at least one variable")

        self._vars: Dict[str, WarpVar] = dict(named_vars)
        self._offsets: Dict[str, int] = {}
        offset = 0
        for name, var in self._vars.items():
            self._offsets[name] = offset
            offset += var.tangent_dim
        self._tangent_dim = offset

        first_var = next(iter(self._vars.values()))
        self._batch_size = first_var.batch_size
        self._device = first_var.device

        self._velocity_buffers: Dict[str, wp.array] = {}
        for name, var in self._vars.items():
            self._velocity_buffers[name] = wp.zeros(
                (self._batch_size, var.tangent_dim),
                dtype=wp.float32,
                device=self._device,
                requires_grad=True,
            )

    @classmethod
    def from_var(cls, var: WarpVar, name: str = "robot") -> "WarpVarValues":
        return cls(**{name: var})

    def get(self, name: str) -> WarpVar:
        return self._vars[name]

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

    def clone(self) -> "WarpVarValues":
        cloned_vars = {name: var.clone() for name, var in self._vars.items()}
        return WarpVarValues(**cloned_vars)

    def integrate(self, velocity: ArrayLike, out: Optional["WarpVarValues"] = None) -> "WarpVarValues":
        if out is not None:
            for name, var in self._vars.items():
                offset = self._offsets[name]
                dim = var.tangent_dim
                vel_buf = self._velocity_buffers[name]
                wp.launch(
                    kernel=_slice_velocity_kernel,
                    dim=(self._batch_size, dim),
                    inputs=[velocity, offset],
                    outputs=[vel_buf],
                    device=self._device,
                )
                var.integrate(vel_buf, out=out.get(name))
            return out
        integrated_vars = {}
        for name, var in self._vars.items():
            offset = self._offsets[name]
            dim = var.tangent_dim
            vel_buf = self._velocity_buffers[name]
            wp.launch(
                kernel=_slice_velocity_kernel,
                dim=(self._batch_size, dim),
                inputs=[velocity, offset],
                outputs=[vel_buf],
                device=self._device,
            )
            integrated_vars[name] = var.integrate(vel_buf)
        return WarpVarValues(**integrated_vars)

    def gather(self, indices: ArrayLike, dest: Optional["WarpVarValues"] = None) -> "WarpVarValues":
        if dest is None:
            gathered = {name: var.gather(indices) for name, var in self._vars.items()}
            return WarpVarValues(**gathered)
        for name, var in self._vars.items():
            var.gather(indices, dest=dest.get(name))
        return dest

    def invalidate(self):
        for var in self._vars.values():
            var.invalidate()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        var_dict = self.__dict__.get("_vars")
        if var_dict is not None and len(var_dict) == 1:
            sole_var = next(iter(var_dict.values()))
            return getattr(sole_var, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
