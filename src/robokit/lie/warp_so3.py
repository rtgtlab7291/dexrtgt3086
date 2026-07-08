# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
from typing import Literal, Optional, Union

import numpy as np
import warp as wp

from robokit.lie.so3 import SO3
from robokit.lie.warp_so3_kernels import so3_jlog_kernel
from robokit.opt.variables import WarpVar
from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather, wp_device_type
from robokit.xform.warp.rotation_conversions import (
    axis_angle_to_quaternion_kernel,
    matrix_to_quaternion_kernel,
    quaternion_apply_kernel,
    quaternion_invert_kernel,
    quaternion_multiply_kernel,
    quaternion_to_axis_angle_kernel,
    quaternion_to_matrix_kernel,
    standardize_quaternion_kernel,
)


class WarpSO3(SO3, WarpVar):
    _wxyz: wp.array(dtype=wp.vec4)

    def __init__(self, wxyz: wp.array(dtype=wp.vec4), backend: Literal["warp"] = "warp"):
        self._wxyz = wxyz.contiguous()

    @property
    def wxyz(self) -> wp.array(dtype=wp.vec4):
        return self._wxyz

    @wxyz.setter
    def wxyz(self, value: wp.array(dtype=wp.vec4)):
        self._wxyz = value.contiguous()

    def __repr__(self):
        return f"WarpSO3(wxyz={np.round(self.wxyz.numpy(), decimals=4)})"

    @property
    def batch_size(self) -> int:
        if self._wxyz.ndim == 0:
            return 1
        return int(self._wxyz.shape[0])

    @property
    def device(self) -> wp_device_type:
        return self._wxyz.device

    @property
    def tangent_dim(self) -> int:
        return 3

    @staticmethod
    def from_matrix(
        matrix: wp.array(dtype=wp.mat33),
        backend: Literal["warp"] = "warp",
    ) -> "WarpSO3":
        """
        Construct SO(3) from a 3x3 rotation matrix.

        Example:
            >>> mat = wp.from_numpy(np.array([[0.0, -1.0, 0.0],
            ...                               [1.0,  0.0, 0.0],
            ...                               [0.0,  0.0, 1.0]]), dtype=wp.mat33)
            >>> so3 = WarpSO3.from_matrix(mat)
            >>> expected_quat = np.array([0.7071, 0.0, 0.0, 0.7071])
            >>> np.allclose(so3.standardize().wxyz.numpy(), expected_quat, atol=1e-4)
            True
        """
        wp.init()
        quat = wp.empty(matrix.shape, dtype=wp.vec4, requires_grad=matrix.requires_grad)  # type: ignore
        wp.launch(
            kernel=matrix_to_quaternion_kernel,
            dim=matrix.size,
            inputs=[matrix.flatten()],
            outputs=[quat.flatten()],
            device=matrix.device,
        )
        return WarpSO3(quat, backend)

    def as_matrix(self) -> wp.array(dtype=wp.mat33):
        """
        Convert SO(3) into 3x3 rotation matrix.

        Example:
            >>> quat = wp.from_numpy(np.array([0.7071, 0.0, 0.0, 0.7071]), dtype=wp.vec4)
            >>> so3 = WarpSO3(quat)
            >>> matrix = so3.as_matrix()
            >>> expected = np.array([[0.0, -1.0, 0.0],
            ...                      [1.0,  0.0, 0.0],
            ...                      [0.0,  0.0, 1.0]])
            >>> np.allclose(matrix.numpy(), expected, atol=1e-4)
            True
        """
        wp.init()
        matrix = wp.empty(self.wxyz.shape, dtype=wp.mat33, requires_grad=self.wxyz.requires_grad)
        wp.launch(
            kernel=quaternion_to_matrix_kernel,
            dim=self.wxyz.size,
            inputs=[self.wxyz.flatten()],
            outputs=[matrix.flatten()],
            device=self.wxyz.device,
        )
        return matrix

    def standardize(self) -> "WarpSO3":
        """
        Return an equivalent quaternion with non-negative scalar part and unit norm.

        Example:
            >>> quat = wp.from_numpy(np.array([[-0.7071, 0.0, 0.0, -0.7071]], dtype=np.float32), dtype=wp.vec4)
            >>> so3 = WarpSO3(quat)
            >>> so3_std = so3.standardize()
            >>> expected = np.array([0.7071, 0.0, 0.0, 0.7071])
            >>> np.allclose(so3_std.wxyz.numpy()[0], expected, atol=1e-4)
            True
        """
        wp.init()
        result = wp.empty(self.wxyz.shape, dtype=wp.vec4, requires_grad=self.wxyz.requires_grad)
        wp.launch(
            kernel=standardize_quaternion_kernel,
            dim=self.wxyz.size,
            inputs=[self.wxyz.flatten()],
            outputs=[result.flatten()],
            device=self.wxyz.device,
        )
        return WarpSO3(result, "warp")

    def apply(self, target: wp.array(dtype=wp.vec3)) -> wp.array(dtype=wp.vec3):
        """
        Rotate 3D vector(s) by this rotation.

        Example:
            >>> quat = wp.from_numpy(np.array([0.7071068, 0.0, 0.0, 0.7071068]), dtype=wp.vec4)
            >>> so3 = WarpSO3(quat)
            >>> v = wp.from_numpy(np.array([1.0, 0.0, 0.0]), dtype=wp.vec3)
            >>> result = so3.apply(v)
            >>> expected = np.array([0.0, 1.0, 0.0])
            >>> np.allclose(result.numpy(), expected, atol=1e-6)
            True
        """
        wp.init()
        result = wp.empty(target.shape, dtype=wp.vec3, requires_grad=target.requires_grad or self.wxyz.requires_grad)
        wp.launch(
            kernel=quaternion_apply_kernel,
            dim=target.size,
            inputs=[self.wxyz.flatten(), target.flatten()],
            outputs=[result.flatten()],
            device=target.device,
        )
        return result

    def multiply(self, other: "WarpSO3") -> "WarpSO3":
        """
        Compose rotations via quaternion (Hamilton) product.

        Example:
            >>> quat = wp.from_numpy(np.array([0.9238795, 0.0, 0.0, 0.3826834]), dtype=wp.vec4)
            >>> a = WarpSO3(quat)
            >>> b = WarpSO3(quat)
            >>> c = a.multiply(b)
            >>> expected = np.array([0.7071068, 0.0, 0.0, 0.7071068])
            >>> np.allclose(c.standardize().wxyz.numpy(), expected, atol=1e-6)
            True
        """
        wp.init()
        result = wp.empty(
            self.wxyz.shape, dtype=wp.vec4, requires_grad=self.wxyz.requires_grad or other.wxyz.requires_grad
        )
        wp.launch(
            kernel=quaternion_multiply_kernel,
            dim=self.wxyz.size,
            inputs=[self.wxyz.flatten(), other.wxyz.flatten()],
            outputs=[result.flatten()],
            device=self.wxyz.device,
        )
        return WarpSO3(result, "warp")

    def clone(self) -> "WarpSO3":
        return WarpSO3(self.wxyz.clone(), "warp")

    def integrate(self, velocity: ArrayLike, out: Optional["WarpSO3"] = None) -> "WarpSO3":
        delta = WarpSO3.exp(velocity)
        result = delta.multiply(self)
        if out is None:
            return result
        out.wxyz = result.wxyz
        return out

    def inverse(self) -> "WarpSO3":
        """
        Return the inverse rotation (quaternion conjugate for unit quaternions).

        Example:
            >>> quat = wp.from_numpy(np.array([0.7071068, 0.0, 0.0, 0.7071068]), dtype=wp.vec4)
            >>> so3 = WarpSO3(quat)
            >>> inv = so3.inverse()
            >>> identity = (so3 * inv).standardize().wxyz
            >>> expected = np.array([1.0, 0.0, 0.0, 0.0])
            >>> np.allclose(identity.numpy(), expected, atol=1e-6)
            True
        """
        wp.init()
        result = wp.empty(self.wxyz.shape, dtype=wp.vec4, requires_grad=self.wxyz.requires_grad)
        wp.launch(
            kernel=quaternion_invert_kernel,
            dim=self.wxyz.size,
            inputs=[self.wxyz.flatten()],
            outputs=[result.flatten()],
            device=self.wxyz.device,
        )
        return WarpSO3(result, "warp")

    @staticmethod
    def exp(
        log_rot: wp.array(dtype=wp.vec3),
        backend: Literal["warp"] = "warp",
    ) -> "WarpSO3":
        """
        Exponential map from so(3) (axis-angle/log rotation) to SO(3).

        Example:
            >>> log_rot = wp.from_numpy(np.array([0.0, 0.0, 1.5707963]), dtype=wp.vec3)
            >>> so3 = WarpSO3.exp(log_rot)
            >>> expected_wxyz = np.array([0.7071068, 0.0, 0.0, 0.7071068])
            >>> np.allclose(so3.standardize().wxyz.numpy(), expected_wxyz, atol=1e-6)
            True
        """
        wp.init()
        quat = wp.empty(log_rot.shape, dtype=wp.vec4, requires_grad=log_rot.requires_grad)
        wp.launch(
            kernel=axis_angle_to_quaternion_kernel,
            dim=log_rot.size,
            inputs=[log_rot.flatten()],
            outputs=[quat.flatten()],
            device=log_rot.device,
        )
        return WarpSO3(quat, backend)

    def log(self) -> wp.array(dtype=wp.vec3):
        """
        Logarithmic map from SO(3) to so(3), returning the axis-angle/log rotation.

        Example:
            >>> quat = wp.from_numpy(np.array([0.7071068, 0.0, 0.0, 0.7071068]), dtype=wp.vec4)
            >>> so3 = WarpSO3(quat)
            >>> log_rot = so3.log()
            >>> expected = np.array([0.0, 0.0, 1.5707963])
            >>> np.allclose(log_rot.numpy(), expected, atol=1e-6)
            True
        """
        wp.init()
        axis_angle = wp.empty(self.wxyz.shape, dtype=wp.vec3, requires_grad=self.wxyz.requires_grad)
        wp.launch(
            kernel=quaternion_to_axis_angle_kernel,
            dim=self.wxyz.size,
            inputs=[self.wxyz.flatten()],
            outputs=[axis_angle.flatten()],
            device=self.wxyz.device,
        )
        return axis_angle

    def jlog(self) -> wp.array(dtype=wp.mat33):
        """
        Compute the Jacobian of the logarithm map using Warp backend.

        Reference:
        Equations (144, 147, 174) from Micro-Lie theory:
        https://arxiv.org/pdf/1812.01537

        Example:
            >>> quat = wp.from_numpy(np.array([1.0, 0.0, 0.0, 0.0]), dtype=wp.vec4)
            >>> so3 = WarpSO3(quat)
            >>> J = so3.jlog()
            >>> expected = np.eye(3)
            >>> np.allclose(J.numpy(), expected, atol=1e-6)
            True
        """
        wp.init()
        theta = self.log()
        eps = 1e-4
        result = wp.empty(theta.shape, dtype=wp.mat33, requires_grad=theta.requires_grad)
        wp.launch(
            kernel=so3_jlog_kernel,
            dim=theta.size,
            inputs=[theta.flatten(), eps],
            outputs=[result.flatten()],
            device=theta.device,
        )
        return result

    def gather(self, indices: ArrayLike, dest: Optional["WarpSO3"] = None) -> "WarpSO3":
        dest_arr = dest._wxyz if dest is not None else None
        gathered = gather(self._wxyz, indices, dest_arr)
        return WarpSO3(gathered, "warp")

    def __mul__(self, other: Union[SO3, wp.array]) -> "WarpSO3":
        """
        Compose rotations via quaternion (Hamilton) product.

        Example:
            >>> quat = wp.from_numpy(np.array([0.9238795, 0.0, 0.0, 0.3826834]), dtype=wp.vec4)
            >>> a = WarpSO3(quat)
            >>> b = WarpSO3(quat)
            >>> c = a * b
            >>> expected = np.array([0.7071068, 0.0, 0.0, 0.7071068])
            >>> np.allclose(c.standardize().wxyz.numpy(), expected, atol=1e-6)
            True
        """
        if not isinstance(other, WarpSO3):
            if isinstance(other, wp.array):
                other = WarpSO3(other, "warp")
            else:
                return NotImplemented
        return self.multiply(other)

    def __rmul__(self, other: Union[SO3, wp.array]) -> "WarpSO3":
        if not isinstance(other, WarpSO3):
            if isinstance(other, wp.array):
                other = WarpSO3(other, "warp")
            else:
                return NotImplemented
        return other * self

    def __getitem__(self, key) -> "WarpSO3":
        return WarpSO3(self.wxyz.__getitem__(key), "warp")


__all__ = ["WarpSO3"]
