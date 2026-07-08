# pyright: reportArgumentType=false
# pyright: reportOptionalOperand=false
import logging
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.lie.se3 import SE3
from robokit.lie.warp_se3_kernels import (
    se3_adjoint_kernel,
    se3_apply_kernel,
    se3_exp_map_kernel,
    se3_exp_map_to_matrix_kernel,
    se3_from_matrix_kernel,
    se3_inverse_kernel,
    se3_jlog_kernel,
    se3_log_map_kernel,
    se3_multiply_kernel,
    se3_to_matrix_kernel,
)
from robokit.opt.variables import WarpVar
from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather, stack, wp_device_type, wp_mat66, wp_vec6, wp_vec7


logger = logging.getLogger("robokit")


@wp.kernel
def _float32_to_vec6_kernel(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array(dtype=wp_vec6),
):
    i = wp.tid()
    dst[i] = wp_vec6(src[i, 0], src[i, 1], src[i, 2], src[i, 3], src[i, 4], src[i, 5])


class WarpSE3(SE3, WarpVar):
    """
    Special Euclidean group SE(3) representation with warp backend.

    The internal parameterization is [x, y, z, qw, qx, qy, qz]: translation followed by a quaternion rotation.
    The tangent parameterization is [vx, vy, vz, omega_x, omega_y, omega_z]: linear and angular velocities.
    """

    _xyz_wxyz: wp.array

    def __new__(cls, se3_like: wp.array, backend: Optional[Literal["warp"]] = None) -> "WarpSE3":
        return SE3.__new__(cls, se3_like, backend="warp")

    def __init__(self, xyz_wxyz: wp.array, backend: Literal["warp"] = "warp"):
        self._xyz_wxyz = xyz_wxyz.contiguous()

    def __repr__(self):
        return f"WarpSE3(xyz={np.round(self._xyz_wxyz.numpy()[:, :3], decimals=4)}, wxyz={np.round(self._xyz_wxyz.numpy()[:, 3:], decimals=4)})"

    @property
    def xyz_wxyz(self) -> wp.array:
        return self._xyz_wxyz

    @xyz_wxyz.setter
    def xyz_wxyz(self, value: wp.array):
        self._xyz_wxyz = value.contiguous()

    @property
    def batch_size(self) -> int:
        if self._xyz_wxyz.ndim == 0:
            return 1
        return int(self._xyz_wxyz.shape[0])

    @property
    def device(self) -> wp_device_type:
        return self._xyz_wxyz.device

    @property
    def tangent_dim(self) -> int:
        return 6

    def _component_view(self, dtype: type, byte_offset: int = 0) -> wp.array:
        storage = self._xyz_wxyz
        if storage.requires_grad and storage.grad is None:
            storage.grad = wp.zeros_like(storage)

        grad_view = None
        if storage.grad is not None:
            grad_view = wp.array(
                ptr=storage.grad.ptr + byte_offset,
                shape=storage.shape,
                strides=storage.strides,
                dtype=dtype,
                device=storage.device,
                copy=False,
            )

        return wp.array(
            ptr=storage.ptr + byte_offset,
            shape=storage.shape,
            strides=storage.strides,
            dtype=dtype,
            device=storage.device,
            copy=False,
            requires_grad=storage.requires_grad,
            grad=grad_view,
        )

    @property
    def xyz(self) -> wp.array:
        return self._component_view(wp.vec3)  # aliased non-contiguous view into xyz of each vec7 element

    @property
    def quat_wxyz(self) -> wp.array:
        return self._component_view(wp.vec4, byte_offset=3 * 4)

    @staticmethod
    def identity(
        shape: Union[int, Tuple[int, ...]],
        device: Optional[wp_device_type] = None,
        backend: Literal["warp"] = "warp",
        requires_grad: bool = False,
    ) -> "WarpSE3":
        if isinstance(shape, int):
            shape = (shape,)
        identity_vec = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        identity_np = np.broadcast_to(identity_vec, (*shape, 7)).copy()
        identity_wp = wp.from_numpy(identity_np, dtype=wp_vec7, device=device, requires_grad=requires_grad)
        return WarpSE3(identity_wp, backend)

    @staticmethod
    def stack(se3_list: Sequence["WarpSE3"], axis: int = 0, dest: Optional[wp.array] = None) -> wp.array:
        """
        Stack a sequence of WarpSE3 instances along a new axis, like np.stack.

        Example:
            >>> a_np = np.array([[1, 2, 3, 1, 0, 0, 0]], dtype=np.float32)
            >>> b_np = np.array([[4, 5, 6, 1, 0, 0, 0]], dtype=np.float32)
            >>> a = WarpSE3(wp.from_numpy(a_np, dtype=wp_vec7))
            >>> b = WarpSE3(wp.from_numpy(b_np, dtype=wp_vec7))
            >>> stacked = WarpSE3.stack([a, b], axis=1)
            >>> stacked.shape
            (1, 2)
            >>> np.allclose(stacked.numpy()[0, 0], [1, 2, 3, 1, 0, 0, 0])
            True
            >>> np.allclose(stacked.numpy()[0, 1], [4, 5, 6, 1, 0, 0, 0])
            True
        """
        return stack([s.xyz_wxyz for s in se3_list], axis=axis, dest=dest)

    @staticmethod
    def from_matrix(
        matrix: wp.array,
        backend: Literal["warp"] = "warp",
    ) -> "WarpSE3":
        """
        Construct an SE(3) transform from a 4x4 homogeneous matrix.

        Example:
            >>> mat = np.array([[0.0, -1.0, 0.0, 1.0],
            ...                 [1.0,  0.0, 0.0, 2.0],
            ...                 [0.0,  0.0, 1.0, 3.0],
            ...                 [0.0,  0.0, 0.0, 1.0]], dtype=np.float32)
            >>> mat_wp = wp.from_numpy(mat.reshape(1, 4, 4), dtype=wp.mat44)
            >>> se3 = WarpSE3.from_matrix(mat_wp)
            >>> expected_xyz = np.array([1.0, 2.0, 3.0])
            >>> expected_wxyz = np.array([0.7071, 0.0, 0.0, 0.7071])
            >>> np.allclose(se3.xyz.numpy()[0], expected_xyz, atol=1e-4)
            True
            >>> np.allclose(se3.quat_wxyz.numpy()[0], expected_wxyz, atol=1e-4)
            True
        """
        wp.init()
        xyz_wxyz = wp.empty(matrix.shape, dtype=wp_vec7, requires_grad=matrix.requires_grad)

        wp.launch(
            kernel=se3_from_matrix_kernel,
            dim=matrix.size,
            inputs=[matrix.flatten()],
            outputs=[xyz_wxyz.flatten()],
            device=matrix.device,
        )

        return WarpSE3(xyz_wxyz, backend)

    def as_matrix(self) -> wp.array:
        """
        Convert this SE(3) pose to a 4x4 homogeneous transformation matrix.

        Example:
            >>> xyz_wxyz = np.array([1.0, 2.0, 3.0, 0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32)
            >>> se3 = WarpSE3(wp.from_numpy(xyz_wxyz.reshape(1, 7), dtype=wp_vec7))
            >>> mat = se3.as_matrix()
            >>> expected = np.array([
            ...     [0.0, -1.0, 0.0, 1.0],
            ...     [1.0,  0.0, 0.0, 2.0],
            ...     [0.0,  0.0, 1.0, 3.0],
            ...     [0.0,  0.0, 0.0, 1.0],
            ... ], dtype=np.float32)
            >>> np.allclose(mat.numpy()[0], expected, atol=1e-6)
            True
        """
        wp.init()
        matrix = wp.empty(self._xyz_wxyz.shape, dtype=wp.mat44, requires_grad=self._xyz_wxyz.requires_grad)

        wp.launch(
            kernel=se3_to_matrix_kernel,
            dim=self._xyz_wxyz.size,
            inputs=[self._xyz_wxyz.flatten()],
            outputs=[matrix.flatten()],
            device=self._xyz_wxyz.device,
        )

        return matrix

    def apply(self, points: wp.array) -> wp.array:
        """
        Apply this SE(3) transform to 3D point(s).

        Example:
            >>> xyz_wxyz = np.array([1.0, 2.0, 3.0, 0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32)
            >>> se3 = WarpSE3(wp.from_numpy(xyz_wxyz.reshape(1, 7), dtype=wp_vec7))
            >>> p = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            >>> result = se3.apply(wp.from_numpy(p.reshape(1, 3), dtype=wp.vec3))
            >>> np.allclose(result.numpy()[0], np.array([1.0, 3.0, 3.0]), atol=1e-6)
            True
        """
        wp.init()
        result = wp.empty(
            points.shape, dtype=wp.vec3, requires_grad=points.requires_grad or self._xyz_wxyz.requires_grad
        )

        wp.launch(
            kernel=se3_apply_kernel,
            dim=points.size,
            inputs=[self._xyz_wxyz.flatten(), points.flatten()],
            outputs=[result.flatten()],
            device=points.device,
        )

        return result

    def clone(self) -> "WarpSE3":
        return WarpSE3(wp.clone(self._xyz_wxyz), "warp")

    def integrate(self, velocity: ArrayLike, out: Optional["WarpSE3"] = None) -> "WarpSE3":
        if velocity.dtype == wp.float32 and velocity.ndim == 2:
            vec6_vel = wp.empty(
                velocity.shape[0], dtype=wp_vec6, device=velocity.device, requires_grad=velocity.requires_grad
            )
            wp.launch(
                kernel=_float32_to_vec6_kernel,
                dim=velocity.shape[0],
                inputs=[velocity],
                outputs=[vec6_vel],
                device=velocity.device,
            )
            velocity = vec6_vel
        delta = WarpSE3.exp(velocity)
        return delta.multiply(self, out=out)

    def inverse(self) -> "WarpSE3":
        """
        Invert an SE(3) pose.

        Example:
            >>> xyz_wxyz = np.array([1.0, 2.0, 3.0, 0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)
            >>> se3 = WarpSE3(wp.from_numpy(xyz_wxyz.reshape(1, 7), dtype=wp_vec7))
            >>> se3_inv = se3.inverse()
            >>> expected = np.array([-1.0, -3.0, 2.0, 0.70710678, -0.70710678, 0.0, 0.0], dtype=np.float32)
            >>> np.allclose(se3_inv.xyz_wxyz.numpy()[0], expected, atol=1e-6)
            True
        """
        wp.init()
        result = wp.empty(self._xyz_wxyz.shape, dtype=wp_vec7, requires_grad=self._xyz_wxyz.requires_grad)
        wp.launch(
            kernel=se3_inverse_kernel,
            dim=self._xyz_wxyz.size,
            inputs=[self._xyz_wxyz.flatten()],
            outputs=[result.flatten()],
            device=self._xyz_wxyz.device,
        )
        return WarpSE3(result, "warp")

    @staticmethod
    def exp(
        log_transform: wp.array,
        backend: Literal["warp"] = "warp",
        eps: float = 1e-4,
    ) -> "WarpSE3":
        """
        Compute the exponential map of the logarithm of the SE(3) transformation.
        The input is assumed to be in [log_translation, log_rotation] format.

        Example:
            >>> log_se3_np = np.array([1.0, 3.92699, 0.785398, 1.570796, 0.0, 0.0], dtype=np.float32)
            >>> log_se3_wp = wp.from_numpy(log_se3_np.reshape(1, 6), dtype=wp_vec6)
            >>> se3 = WarpSE3.exp(log_se3_wp)
            >>> expected = np.array([1.0, 2.0, 3.0, 0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)
            >>> np.allclose(se3.xyz_wxyz.numpy()[0], expected, atol=1e-6)
            True
        """
        wp.init()
        xyz_wxyz = wp.empty(log_transform.shape, dtype=wp_vec7, requires_grad=log_transform.requires_grad)

        wp.launch(
            kernel=se3_exp_map_kernel,
            dim=log_transform.size,
            inputs=[log_transform.flatten(), eps],
            outputs=[xyz_wxyz.flatten()],
            device=log_transform.device,
        )

        return WarpSE3(xyz_wxyz, backend)

    @staticmethod
    def exp_to_matrix(
        log_transform: wp.array,
        eps: float = 1e-4,
    ) -> wp.array:
        """
        Compute the exponential map of the logarithm of the SE(3) transformation
        and return the result as a 4x4 homogeneous matrix.

        Example:
            >>> log_se3_np = np.array([1.0, 3.92699, 0.785398, 1.570796, 0.0, 0.0], dtype=np.float32)
            >>> log_se3_wp = wp.from_numpy(log_se3_np.reshape(1, 6), dtype=wp_vec6)
            >>> mat = WarpSE3.exp_to_matrix(log_se3_wp)
            >>> expected = np.array([
            ...     [1.0, 0.0, 0.0, 1.0],
            ...     [0.0, 0.0, -1.0, 2.0],
            ...     [0.0, 1.0, 0.0, 3.0],
            ...     [0.0, 0.0, 0.0, 1.0]
            ... ], dtype=np.float32)
            >>> np.allclose(mat.numpy()[0], expected, atol=1e-4)
            True
        """
        wp.init()
        matrix = wp.empty(log_transform.shape, dtype=wp.mat44, requires_grad=log_transform.requires_grad)

        wp.launch(
            kernel=se3_exp_map_to_matrix_kernel,
            dim=log_transform.size,
            inputs=[log_transform.flatten(), eps],
            outputs=[matrix.flatten()],
            device=log_transform.device,
        )

        return matrix

    def log(self, eps: float = 1e-4) -> wp.array:
        """
        Compute the log map of an SE(3) pose.
        The output is in [log_translation, log_rotation] format.

        Example:
            >>> xyz_wxyz = np.array([1.0, 2.0, 3.0, 0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)
            >>> se3 = WarpSE3(wp.from_numpy(xyz_wxyz.reshape(1, 7), dtype=wp_vec7))
            >>> log_se3 = se3.log()
            >>> expected = np.array([1.0, 3.92699, 0.785398, 1.570796, 0.0, 0.0], dtype=np.float32)
            >>> np.allclose(log_se3.numpy()[0], expected, atol=1e-5)
            True
        """
        wp.init()

        log_transform = wp.empty(self._xyz_wxyz.shape, dtype=wp_vec6, requires_grad=self._xyz_wxyz.requires_grad)

        wp.launch(
            kernel=se3_log_map_kernel,
            dim=self._xyz_wxyz.size,
            inputs=[self._xyz_wxyz.flatten(), eps],
            outputs=[log_transform.flatten()],
            device=self._xyz_wxyz.device,
        )

        return log_transform

    def adjoint(self) -> wp.array:
        """
        Compute the 6x6 adjoint representation Ad_T of this SE(3) transform.
        This matrix maps twists between frames as: v' = Ad_T @ v.

        Convention: twist/log vectors use [v, omega] ordering (linear first, then angular).

        Returns:
            6x6 adjoint matrix:
            [[R, skew(t) @ R],
             [0, R]]

        Example:
            >>> xyz_wxyz = np.array([1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)
            >>> se3 = WarpSE3(wp.from_numpy(xyz_wxyz.reshape(1, 7), dtype=wp_vec7))
            >>> Ad = se3.adjoint()
            >>> Ad.shape
            (1,)
            >>> se3_inv = se3.inverse()
            >>> Ad_inv = se3_inv.adjoint()
            >>> np.allclose(Ad_inv.numpy()[0], np.linalg.inv(Ad.numpy()[0]), atol=1e-4)
            True
        """
        wp.init()
        result = wp.empty(self._xyz_wxyz.shape, dtype=wp_mat66, requires_grad=self._xyz_wxyz.requires_grad)

        wp.launch(
            kernel=se3_adjoint_kernel,
            dim=self._xyz_wxyz.size,
            inputs=[self._xyz_wxyz.flatten(), result.flatten()],
            device=self._xyz_wxyz.device,
        )

        return result

    def jlog(self) -> wp.array:
        """
        Compute the Jacobian of the logarithm map.

        The Jacobian J_log relates the variation in the SE(3) group to the variation
        in the se(3) algebra: d(log(T)) = J_log @ d(T)

        Returns:
            6x6 Jacobian matrix of the logarithm map.

        Example:
            >>> xyz_wxyz = np.array([0.1, 0.2, 0.3, 0.9987, 0.0314, 0.0314, 0.0314], dtype=np.float32)
            >>> se3 = WarpSE3(wp.from_numpy(xyz_wxyz.reshape(1, 7), dtype=wp_vec7))
            >>> J_log = se3.jlog()
            >>> bool(np.linalg.norm(J_log.numpy()[0] - np.eye(6)) < 0.5)
            True
        """
        wp.init()
        eps = 1e-4
        result = wp.empty(self._xyz_wxyz.shape, dtype=wp_mat66, requires_grad=self._xyz_wxyz.requires_grad)

        wp.launch(
            kernel=se3_jlog_kernel,
            dim=self._xyz_wxyz.size,
            inputs=[self._xyz_wxyz.flatten(), eps, result.flatten()],
            device=self._xyz_wxyz.device,
        )

        return result

    def multiply(self, other: "WarpSE3", out: Optional["WarpSE3"] = None) -> "WarpSE3":
        """
        Compose two SE(3) transforms: self * other.

        The resulting rotation is the quaternion product, and the translation is
        `t_self + R_self @ t_other`.

        Args:
            other: The SE(3) transform to compose with.
            out: Optional output WarpSE3 to store the result in-place. If provided,
                no new array is allocated.

        Example:
            >>> a_np = np.array([1.0, 0.0, 0.0, 0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32)
            >>> b_np = np.array([0.0, 1.0, 0.0, 0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32)
            >>> a = WarpSE3(wp.from_numpy(a_np.reshape(1, 7), dtype=wp_vec7))
            >>> b = WarpSE3(wp.from_numpy(b_np.reshape(1, 7), dtype=wp_vec7))
            >>> c = a.multiply(b)
            >>> result = c.xyz_wxyz.numpy()[0]
            >>> np.allclose(result[:3], [0.0, 0.0, 0.0], atol=1e-6)  # translation
            True
            >>> np.allclose(np.abs(result[3:]), [0.0, 0.0, 0.0, 1.0], atol=1e-6)  # 180° rotation about Z
            True
        """
        wp.init()
        if out is None:
            out = WarpSE3(
                wp.empty(
                    self._xyz_wxyz.shape,
                    dtype=wp_vec7,
                    requires_grad=self._xyz_wxyz.requires_grad or other._xyz_wxyz.requires_grad,
                )
            )

        wp.launch(
            kernel=se3_multiply_kernel,
            dim=self._xyz_wxyz.size,
            inputs=[self._xyz_wxyz.flatten(), other._xyz_wxyz.flatten()],
            outputs=[out._xyz_wxyz.flatten()],
            device=self._xyz_wxyz.device,
        )

        return out

    def gather(self, indices: ArrayLike, dest: "WarpSE3" = None) -> "WarpSE3":
        """
        Gather SE3 elements from this instance using indices. Pure Warp implementation.

        Args:
            indices: 1D int32 array of indices to gather, shape [M].
            dest: Optional pre-allocated destination WarpSE3. If None, a new one is allocated.

        Returns:
            Gathered WarpSE3 with first dimension = len(indices).

        Example:
            >>> xyz_wxyz = np.array([[1, 2, 3, 1, 0, 0, 0], [4, 5, 6, 1, 0, 0, 0], [7, 8, 9, 1, 0, 0, 0]], dtype=np.float32)
            >>> se3 = WarpSE3(wp.from_numpy(xyz_wxyz, dtype=wp_vec7))
            >>> idx = wp.from_numpy(np.array([2, 0], dtype=np.int32))
            >>> gathered = se3.gather(idx)
            >>> gathered.xyz_wxyz.shape
            (2,)
            >>> np.allclose(gathered.xyz.numpy()[0], [7, 8, 9])
            True
        """
        dest_arr = dest._xyz_wxyz if dest is not None else None
        gathered = gather(self._xyz_wxyz, indices, dest_arr)
        return WarpSE3(gathered, "warp")

    def __mul__(self, other: Union[SE3, wp.array]) -> "WarpSE3":
        """
        Compose two SE(3) transforms: self * other.
        """
        if not isinstance(other, WarpSE3):
            if isinstance(other, wp.array):
                other = WarpSE3(other, "warp")
            else:
                return NotImplemented
        return self.multiply(other)

    def __rmul__(self, other: Union[SE3, wp.array]) -> "WarpSE3":
        if not isinstance(other, WarpSE3):
            if isinstance(other, wp.array):
                other = WarpSE3(other, "warp")
            else:
                return NotImplemented
        return other * self

    def __getitem__(self, key) -> "WarpSE3":
        return WarpSE3(self._xyz_wxyz.__getitem__(key), "warp")
