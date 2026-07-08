from typing import Literal, Tuple, Union

import torch
from jaxtyping import Float

from robokit.lie.so3 import SO3
from robokit.xform.torch.rotation_conversions import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    standardize_quaternion,
)


class TorchSO3(SO3):
    """
    Special Orthogonal group SO(3) representation with torch backend.

    Internal parameterization: [qw, qx, qy, qz].
    Tangent parameterization: [omega_x, omega_y, omega_z].
    """

    _wxyz: Float[torch.Tensor, "... 4"]
    """Internal quaternion [qw, qx, qy, qz]"""

    def __init__(self, wxyz: Float[torch.Tensor, "... 4"], backend: Literal["torch"] = "torch"):
        self._wxyz = wxyz

    def __repr__(self):
        return f"TorchSO3(wxyz={torch.round(self.wxyz, decimals=4)})"

    @property
    def wxyz(self) -> Float[torch.Tensor, "... 4"]:
        return self._wxyz

    @wxyz.setter
    def wxyz(self, value: Float[torch.Tensor, "... 4"]):
        self._wxyz = value

    @staticmethod
    def from_matrix(
        matrix: Float[torch.Tensor, "... 3 3"],
        backend: Literal["torch"] = "torch",
    ) -> "TorchSO3":
        """
        Construct SO(3) from a 3x3 rotation matrix.

        Example:
            >>> so3 = TorchSO3.from_matrix(torch.tensor([[0.0, -1.0, 0.0],
                                                     [1.0, 0.0, 0.0],
                                                     [0.0, 0.0, 1.0]]))
            >>> expected_quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
            >>> torch.allclose(so3.standardize().wxyz, expected_quat, atol=1e-4)
            True
        """
        quat = matrix_to_quaternion(matrix)
        return TorchSO3(quat, backend)

    def as_matrix(self) -> Float[torch.Tensor, "... 3 3"]:
        """
        Convert SO(3) into 3x3 rotation matrix.

        Example:
            >>> so3 = TorchSO3(torch.tensor([0.7071, 0.0, 0.0, 0.7071]))
            >>> expected_matrix = torch.tensor([[0.0, -1.0, 0.0],
            ...                                 [1.0,  0.0, 0.0],
            ...                                 [0.0,  0.0, 1.0]])
            >>> torch.allclose(so3.as_matrix(), expected_matrix, atol=1e-4)
            True
        """
        return quaternion_to_matrix(self.wxyz)

    def standardize(self) -> "TorchSO3":
        """
        Return an equivalent quaternion with non-negative scalar part and unit norm.

        Example:
            >>> so3 = TorchSO3(torch.tensor([-0.7071, 0.0, 0.0, -0.7071]))
            >>> so3_std = so3.standardize()
            >>> torch.allclose(so3_std.wxyz, torch.tensor([0.7071, 0.0, 0.0, 0.7071]), atol=1e-4)
            True
        """
        return TorchSO3(standardize_quaternion(self.wxyz), "torch")

    def apply(self, v: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3"]:
        """
        Rotate 3D vector(s) v by this rotation.

        Example:
            >>> so3 = TorchSO3(torch.tensor([0.7071068, 0.0, 0.0, 0.7071068]))
            >>> v = torch.tensor([1.0, 0.0, 0.0])
            >>> torch.allclose(so3.apply(v), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
            True
        """
        return quaternion_apply(self.wxyz, v)

    @staticmethod
    def exp(
        log_rot: Float[torch.Tensor, "... 3"],
        backend: Literal["torch"] = "torch",
    ) -> "TorchSO3":
        """
        Exponential map from so(3) (axis-angle/log rotation) to SO(3).

        Example:
            >>> log_rot = torch.tensor([0.0, 0.0, 1.5707963])
            >>> so3 = TorchSO3.exp(log_rot)
            >>> expected_wxyz = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
            >>> torch.allclose(so3.standardize().wxyz, expected_wxyz, atol=1e-6)
            True
        """
        quat = axis_angle_to_quaternion(log_rot)
        return TorchSO3(quat, backend)

    def log(self) -> Float[torch.Tensor, "... 3"]:
        """
        Logarithmic map from SO(3) to so(3), returning the axis-angle/log rotation.

        Example:
            >>> so3 = TorchSO3(torch.tensor([0.7071068, 0.0, 0.0, 0.7071068]))
            >>> log_rot = so3.log()
            >>> expected = torch.tensor([0.0, 0.0, 1.5707963])
            >>> torch.allclose(log_rot, expected, atol=1e-6)
            True
        """
        return quaternion_to_axis_angle(self.wxyz)

    def inverse(self) -> "TorchSO3":
        """
        Return the inverse rotation (quaternion conjugate for unit quaternions).

        Example:
            >>> so3 = TorchSO3(torch.tensor([0.7071068, 0.0, 0.0, 0.7071068]))
            >>> inv = so3.inverse()
            >>> identity = (so3 * inv).standardize().wxyz
            >>> torch.allclose(identity, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
            True
        """
        quat = quaternion_invert(self.wxyz)
        return TorchSO3(quat, "torch")

    def jlog(self) -> Float[torch.Tensor, "... 3 3"]:
        """
        Compute the Jacobian of the logarithm map.

        Returns the right Jacobian Jr^(-1) which relates perturbations in the
        manifold to perturbations in the tangent space.

        Reference:
        Equations (144, 147, 174) from Micro-Lie theory:
        https://arxiv.org/pdf/1812.01537

        Example:
            >>> so3 = TorchSO3(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # Identity
            >>> J = so3.jlog()
            >>> torch.allclose(J, torch.eye(3), atol=1e-6)
            True
        """
        V_inv = _so3_jac_left_inv(self.log())
        return V_inv.transpose(-1, -2)

    def clone(self) -> "TorchSO3":
        """
        Return a copy of the TorchSO3 object.
        """
        return TorchSO3(self.wxyz.clone(), "torch")

    def __mul__(self, other: Union[SO3, Float[torch.Tensor, "... 4"]]) -> "TorchSO3":
        """
        Compose rotations via quaternion (Hamilton) product.

        Example:
            >>> a = TorchSO3(torch.tensor([0.9238795, 0.0, 0.0, 0.3826834]))
            >>> b = TorchSO3(torch.tensor([0.9238795, 0.0, 0.0, 0.3826834]))
            >>> c = a * b
            >>> expected = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
            >>> torch.allclose(c.standardize().wxyz, expected, atol=1e-6)
            True
        """
        if not isinstance(other, TorchSO3):
            if isinstance(other, torch.Tensor):
                other = TorchSO3(other, "torch")
            else:
                return NotImplemented
        quat = quaternion_multiply(self.wxyz, other.wxyz)
        return TorchSO3(quat, "torch")

    def __rmul__(self, other: Union[SO3, Float[torch.Tensor, "... 4"]]) -> "TorchSO3":
        if not isinstance(other, TorchSO3):
            if isinstance(other, torch.Tensor):
                other = TorchSO3(other, "torch")
            else:
                return NotImplemented
        return other * self

    def __getitem__(self, key) -> Union["TorchSO3", torch.Tensor]:
        out = self.wxyz.__getitem__(key)
        if isinstance(out, torch.Tensor) and out.shape[-1] == 4:
            return TorchSO3(out, "torch")
        return out


def hat(v: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(..., 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(..., 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """
    x, y, z = v.unbind(dim=-1)
    zeros = torch.zeros_like(x)

    row0 = torch.stack([zeros, -z, y], dim=-1)
    row1 = torch.stack([z, zeros, -x], dim=-1)
    row2 = torch.stack([-y, x, zeros], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def _so3_jac_left_inv(theta: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
    """
    Compute the inverse of the left jacobian for the given theta.

    Args:
        theta: The input angle(s) in axis-angle representation.

    Returns:
        A 3x3 matrix (or batch of 3x3 matrices) representing the inverse left jacobian.
    """
    theta_squared = torch.sum(torch.square(theta), dim=-1)
    eps = _get_epsilon(theta_squared.dtype)
    use_taylor = theta_squared < eps

    # Shim to avoid NaNs in torch.where branches, which cause failures for
    # reverse-mode AD.
    theta_squared_safe = torch.where(
        use_taylor,
        torch.ones_like(theta_squared),  # Any non-zero value should do here.
        theta_squared,
    )
    theta_safe = torch.sqrt(theta_squared_safe)
    half_theta_safe = theta_safe / 2.0

    skew_omega = hat(theta)
    skew_omega_squared = torch.matmul(skew_omega, skew_omega)

    # Identity matrix with appropriate batch dimensions
    batch_shape = theta.shape[:-1]
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device)
    eye = eye.expand(batch_shape + (3, 3))

    # Taylor series expansion for small angles
    taylor_result = eye - 0.5 * skew_omega + skew_omega_squared / 12.0

    # Full computation for larger angles
    cos_half_theta = torch.cos(half_theta_safe)
    sin_half_theta = torch.sin(half_theta_safe)

    # Coefficient for the skew_omega_squared term
    coeff = (1.0 - theta_safe * cos_half_theta / (2.0 * sin_half_theta)) / theta_squared_safe

    full_result = eye - 0.5 * skew_omega + coeff[..., None, None] * skew_omega_squared

    jac_left_inv = torch.where(
        use_taylor[..., None, None],
        taylor_result,
        full_result,
    )

    return jac_left_inv


def _so3_exp_map_quaternion(
    log_rot: Float[torch.Tensor, "... 3"], eps: float = 0.0001
) -> Tuple[
    Float[torch.Tensor, "... 4"],
    Float[torch.Tensor, "..."],
    Float[torch.Tensor, "... 3 3"],
    Float[torch.Tensor, "... 3 3"],
]:
    nrms = (log_rot * log_rot).sum(-1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    skews = hat(log_rot)
    skews_square = torch.matmul(skews, skews)

    q = axis_angle_to_quaternion(log_rot)

    return q, rot_angles, skews, skews_square


def _get_epsilon(dtype: torch.dtype) -> float:
    """Get appropriate epsilon value for the given dtype."""
    if dtype == torch.float32:
        return 1e-4
    elif dtype == torch.float64:
        return 1e-8
    else:
        return 1e-4  # Default fallback


__all__ = ["TorchSO3"]
