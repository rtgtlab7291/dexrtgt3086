import logging
from typing import Literal, Optional, Union

import torch
from jaxtyping import Float

from robokit.lie.se3 import SE3
from robokit.lie.torch_so3 import TorchSO3, _so3_exp_map_quaternion, hat
from robokit.xform.torch import (
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rot_tl_to_tf_mat,
)


logger = logging.getLogger("robokit")


class TorchSE3(SE3):
    """
    Special Euclidean group SE(3) representation with torch backend.

    The internal parameterization is [x, y, z, qw, qx, qy, qz]: translation followed by a quaternion rotation.
    The tangent parameterization is [vx, vy, vz, omega_x, omega_y, omega_z]: linear and angular velocities.
    """

    _xyz_wxyz: Float[torch.Tensor, "... 7"]
    """Internal parameterization of SE(3) as [x, y, z, qw, qx, qy, qz]"""

    def __new__(cls, se3_like: "torch.Tensor", backend: Optional[Literal["torch"]] = None) -> "TorchSE3":
        return super().__new__(cls, se3_like)

    def __init__(self, xyz_wxyz: Float[torch.Tensor, "... 7"], backend: Literal["torch"] = "torch"):
        self._xyz_wxyz = xyz_wxyz

    def __repr__(self):
        return f"TorchSE3(xyz={torch.round(self.xyz_wxyz[..., :3], decimals=4)}, wxyz={torch.round(self.xyz_wxyz[..., 3:], decimals=4)})"

    @property
    def xyz_wxyz(self) -> Float[torch.Tensor, "... 7"]:
        return self._xyz_wxyz

    @xyz_wxyz.setter
    def xyz_wxyz(self, value: Float[torch.Tensor, "... 7"]):
        self._xyz_wxyz = value

    @property
    def quat_wxyz(self) -> Float[torch.Tensor, "... 4"]:
        return self.xyz_wxyz[..., 3:]

    @property
    def xyz(self) -> Float[torch.Tensor, "... 3"]:
        return self.xyz_wxyz[..., :3]

    @staticmethod
    def identity(
        dim: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        backend: Literal["torch"] = "torch",
    ) -> "TorchSE3":
        identity_vec = torch.zeros((dim, 7), device=device, dtype=torch.float32)
        identity_vec[..., 3] = 1.0
        return TorchSE3(identity_vec, backend)

    @staticmethod
    def from_matrix(
        matrix: Float[torch.Tensor, "... 4 4"],
        backend: Literal["torch"] = "torch",
    ) -> "TorchSE3":
        """
        Construct an SE(3) transform from a 4x4 homogeneous matrix.

        Example:
            >>> wxyz = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])  # 90° about Z
            >>> t = torch.tensor([1.0, 2.0, 3.0])
            >>> T = rot_tl_to_tf_mat(quaternion_to_matrix(wxyz), t)
            >>> se3 = TorchSE3.from_matrix(T)
            >>> torch.allclose(se3.as_matrix(), T, atol=1e-6)
            True
        """
        xyz = matrix[..., :3, 3]
        wxyz = matrix_to_quaternion(matrix[..., :3, :3])
        return TorchSE3(torch.cat([xyz, wxyz], dim=-1), backend)

    def as_matrix(self) -> Float[torch.Tensor, "... 4 4"]:
        """
        Convert this SE(3) pose to a 4x4 homogeneous transformation matrix.

        Example:
            >>> se3 = TorchSE3(torch.tensor([1.0, 2.0, 3.0, 0.7071068, 0.0, 0.0, 0.7071068]))
            >>> expected = torch.tensor([
            ...     [0.0, -1.0, 0.0, 1.0],
            ...     [1.0,  0.0, 0.0, 2.0],
            ...     [0.0,  0.0, 1.0, 3.0],
            ...     [0.0,  0.0, 0.0, 1.0],
            ... ])
            >>> torch.allclose(se3.as_matrix(), expected, atol=1e-6)
            True
        """
        rot_mat = quaternion_to_matrix(self.quat_wxyz)
        return rot_tl_to_tf_mat(rot_mat=rot_mat, tl=self.xyz)

    @staticmethod
    def exp(
        log_transform: Float[torch.Tensor, "... 6"],
        backend: Literal["torch"] = "torch",
        eps: float = 1e-4,
    ) -> "TorchSE3":
        """
        Compute the exponential map of the logarithm of the SE(3) transformation.
        The input is assumed to be in [log_translation, log_rotation] format.

        Example:
            >>> log_se3 = torch.tensor([1.0, 3.92699, 0.785398, 1.570796, 0.0, 0.0])
            >>> se3 = TorchSE3.exp(log_se3)
            >>> torch.allclose(se3.xyz_wxyz, torch.tensor([1.0, 2.0, 3.0, 0.70710678, 0.70710678, 0.0, 0.0]), atol=1e-6)
            True
        """
        log_translation = log_transform[..., :3]
        log_rotation = log_transform[..., 3:]

        # rotation is an exponential map of log_rotation
        (
            q,
            rotation_angles,
            log_rotation_hat,
            log_rotation_hat_square,
        ) = _so3_exp_map_quaternion(log_rotation, eps=eps)

        # translation is V @ T
        V = _se3_V_matrix(
            log_rotation,
            log_rotation_hat,
            log_rotation_hat_square,
            rotation_angles,
            eps=eps,
        )
        xyz = torch.matmul(V, log_translation.unsqueeze(-1)).squeeze(-1)

        return TorchSE3(torch.cat([xyz, q], dim=-1), backend)

    def log(self) -> Float[torch.Tensor, "... 6"]:
        """
        Compute the log map of an SE(3) pose.
        The output is in [log_translation, log_rotation] format.

        Example:
            >>> xyzquat = torch.tensor([1.0, 2.0, 3.0, 0.70710678, 0.70710678, 0.0, 0.0])
            >>> se3 = TorchSE3(xyzquat)
            >>> log_se3 = se3.log()
            >>> torch.allclose(log_se3, torch.tensor([1.0, 3.92699, 0.785398, 1.570796, 0.0, 0.0]), atol=1e-6)
            True
        """
        log_rotation = quaternion_to_axis_angle(self.quat_wxyz)

        log_rotation_inputs = _get_se3_V_input(log_rotation)
        V = _se3_V_matrix(*log_rotation_inputs)

        log_translation = torch.linalg.solve(V, self.xyz[..., None]).squeeze(-1)

        return torch.cat([log_translation, log_rotation], dim=-1)

    def clone(self) -> "TorchSE3":
        return TorchSE3(self._xyz_wxyz.clone(), "torch")

    def inverse(self) -> "TorchSE3":
        """
        Invert an SE(3) pose in [x, y, z, qw, qx, qy, qz] format.

        Example:
            >>> se3 = TorchSE3(torch.tensor([1.0, 2.0, 3.0, 0.70710678, 0.70710678, 0.0, 0.0]))
            >>> se3_inv = se3.inverse()
            >>> torch.allclose(se3_inv.xyz_wxyz, torch.tensor([-1.0, -3.0, 2.0, 0.70710678, -0.70710678, 0.0, 0.0]))
            True
        """
        q_inv = quaternion_invert(self.quat_wxyz)
        t_inv = -quaternion_apply(q_inv, self.xyz)
        return TorchSE3(torch.cat([t_inv, q_inv], dim=-1), "torch")

    def apply(self, points: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3"]:
        """
        Apply this SE(3) transform to 3D point(s).

        Example:
            >>> se3 = TorchSE3(torch.tensor([1.0, 2.0, 3.0, 0.7071068, 0.0, 0.0, 0.7071068]))
            >>> p = torch.tensor([1.0, 0.0, 0.0])
            >>> torch.allclose(se3.apply(p), torch.tensor([1.0, 3.0, 3.0]), atol=1e-6)
            True
        """
        return quaternion_apply(self.quat_wxyz, points) + self.xyz

    def adjoint(self) -> Float[torch.Tensor, "... 6 6"]:
        """
        Compute the 6x6 adjoint representation Ad_T of this SE(3) transform.
        This matrix maps twists between frames as: v' = Ad_T @ v.

        Convention: twist/log vectors use [v, omega] ordering (linear first, then angular).

        Returns:
            6x6 adjoint matrix:
            [[R, skew(t) @ R],
             [0, R]]

        Example:
            >>> xyz_wxyz = torch.tensor([1.0, 2.0, 3.0, 0.7071, 0.7071, 0.0, 0.0])
            >>> se3 = TorchSE3(xyz_wxyz)
            >>> Ad = se3.adjoint()
            >>> Ad.shape
            torch.Size([6, 6])
            >>> se3_inv = se3.inverse()
            >>> Ad_inv = se3_inv.adjoint()
            >>> torch.allclose(Ad_inv, torch.linalg.inv(Ad), atol=1e-4)
            True
        """
        return _se3_adjoint(self.xyz, self.quat_wxyz)

    def jlog(self) -> Float[torch.Tensor, "... 6 6"]:
        """
        Compute the Jacobian of the logarithm map.

        The Jacobian J_log relates the variation in the SE(3) group to the variation
        in the se(3) algebra: d(log(T)) = J_log @ d(T)

        Returns:
            6x6 Jacobian matrix of the logarithm map.

        Example:
            >>> xyz_wxyz = torch.tensor([0.1, 0.2, 0.3, 0.9987, 0.0314, 0.0314, 0.0314])
            >>> se3 = TorchSE3(xyz_wxyz)
            >>> J_log = se3.jlog()
            >>> J_log.shape
            torch.Size([6, 6])
            >>> bool(torch.linalg.norm(J_log - torch.eye(6)) < 0.5)
            True
        """
        return _se3_jlog(self.xyz, self.quat_wxyz)

    def __mul__(self, other: Union["TorchSE3", Float[torch.Tensor, "... 7"]]) -> "TorchSE3":
        """
        Compose two SE(3) transforms: self * other.

        The resulting rotation is the quaternion product, and the translation is
        `t_self + R_self @ t_other`.

        Example:
            >>> a = TorchSE3(torch.tensor([1.0, 0.0, 0.0, 0.7071068, 0.0, 0.0, 0.7071068]))
            >>> b = TorchSE3(torch.tensor([0.0, 1.0, 0.0, 0.7071068, 0.0, 0.0, 0.7071068]))
            >>> c = a * b
            >>> expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # 180° about Z, zero translation
            >>> torch.allclose(c.xyz_wxyz, expected, atol=1e-6)
            True
        """
        if not isinstance(other, TorchSE3):
            if isinstance(other, torch.Tensor):
                other = TorchSE3(other, "torch")
            else:
                return NotImplemented
        xyz_wxyz_new = _se3_multiply(self.xyz_wxyz, other.xyz_wxyz)
        return TorchSE3(xyz_wxyz_new, "torch")

    def __rmul__(self, other: Union[SE3, Float[torch.Tensor, "... 7"]]) -> "TorchSE3":
        if not isinstance(other, TorchSE3):
            if isinstance(other, torch.Tensor):
                other = TorchSE3(other, "torch")
            else:
                return NotImplemented
        return other * self

    def __getitem__(self, key) -> "TorchSE3":
        out = self._xyz_wxyz.__getitem__(key)
        if isinstance(out, torch.Tensor) and out.shape[-1] == 7:
            return TorchSE3(out, "torch")
        else:
            logger.warning(
                "Indexing TorchSE3 returned a tensor that is not of shape (..., 7). Returning a tensor instead of TorchSE3."
            )
            return out  # pyright: ignore[reportReturnType]

    @property
    def device(self):
        return self._xyz_wxyz.device


def _get_se3_V_input(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    A helper function that computes the input variables to the `_se3_V_matrix`
    function.
    """
    nrms = (log_rotation**2).sum(-1)
    rotation_angles = torch.clamp(nrms, eps).sqrt()
    log_rotation_hat = hat(log_rotation)
    log_rotation_hat_square = torch.matmul(log_rotation_hat, log_rotation_hat)
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)
        + log_rotation_hat * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[..., None, None]
        + (
            log_rotation_hat_square
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[..., None, None]
        )
    )

    return V


def _se3_multiply(xyz_wxyz1: torch.Tensor, xyz_wxyz2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two SE(3) transforms represented as [x, y, z, qw, qx, qy, qz].

    The resulting rotation is the quaternion product, and the translation is
    `t1 + R1 @ t2`.
    """
    wxyz_new = quaternion_multiply(xyz_wxyz1[..., 3:], xyz_wxyz2[..., 3:])
    xyz_new = xyz_wxyz1[..., :3] + quaternion_apply(xyz_wxyz1[..., 3:], xyz_wxyz2[..., :3])
    return torch.cat([xyz_new, wxyz_new], dim=-1)


def _se3_adjoint(translation: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Compute SE(3) adjoint matrix."""
    rotation = quaternion_to_matrix(quat)
    # Create skew-symmetric matrix of translation and multiply by rotation
    t_skew_rotation = torch.matmul(hat(translation), rotation)
    batch_shape = rotation.shape[:-2]
    zeros = torch.zeros(batch_shape + (3, 3), dtype=rotation.dtype, device=rotation.device)

    # Assemble the adjoint matrix matching Pinocchio's convention
    top_row = torch.cat([rotation, t_skew_rotation], dim=-1)
    bottom_row = torch.cat([zeros, rotation], dim=-1)

    return torch.cat([top_row, bottom_row], dim=-2)


def _se3_jlog(translation: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """Jacobian of SE(3) logarithm map."""
    rotation = TorchSO3(quaternion, "torch")
    jlog_so3 = rotation.jlog()

    # Get rotation axis-angle representation
    w = rotation.log()
    theta = torch.norm(w, dim=-1, keepdim=True)
    theta_squared = torch.sum(w * w, dim=-1, keepdim=True)
    eps = 1e-6
    use_taylor = theta_squared.squeeze(-1) < eps

    # Compute trigonometric functions
    theta_safe = torch.where(use_taylor.unsqueeze(-1), torch.ones_like(theta), theta)
    theta_inv = 1.0 / theta_safe
    theta_squared_inv = theta_inv * theta_inv
    st = torch.sin(theta)
    ct = torch.cos(theta)

    # Compute intermediate terms
    inv_2_2ct = torch.where(use_taylor.unsqueeze(-1), 0.5 * torch.ones_like(theta), 1.0 / (2.0 * (1.0 - ct)))
    beta = theta_squared_inv - st * theta_inv * inv_2_2ct
    beta_dot_over_theta = (
        -2.0 * theta_squared_inv * theta_squared_inv + (1.0 + st * theta_inv) * theta_squared_inv * inv_2_2ct
    )

    # Cross term w^T * p
    wTp = torch.sum(w * translation, dim=-1, keepdim=True)

    # Compute v3_tmp term
    v3_tmp = beta_dot_over_theta * wTp * w - (theta_squared * beta_dot_over_theta + 2.0 * beta) * translation

    # Compute C matrix components
    C = (
        torch.einsum("...i,...j->...ij", v3_tmp, w)
        + beta.unsqueeze(-1) * torch.einsum("...i,...j->...ij", w, translation)
        + (wTp * beta).unsqueeze(-1) * torch.eye(3, dtype=w.dtype, device=w.device)
    )
    # Add skew-symmetric part
    C = C + 0.5 * hat(translation)
    # Compute B = C @ jlog_so3
    B = torch.matmul(C, jlog_so3)
    # For small angles, use Taylor expansion
    B_taylor = 0.5 * hat(translation)
    B_final = torch.where(use_taylor.unsqueeze(-1).unsqueeze(-1), B_taylor, B)

    # Assemble the full Jacobian matrix
    batch_shape = theta.shape[:-1]
    jlog_matrix = torch.zeros(batch_shape + (6, 6), dtype=w.dtype, device=w.device)
    jlog_matrix[..., :3, :3] = jlog_so3
    jlog_matrix[..., 3:, 3:] = jlog_so3
    jlog_matrix[..., :3, 3:] = B_final

    return jlog_matrix


__all__ = ["TorchSE3"]
