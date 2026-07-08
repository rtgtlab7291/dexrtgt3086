from typing import Literal, Union

import numpy as np
import pinocchio as pin
from jaxtyping import Float

from robokit.lie.se3 import SE3


class PinocchioSE3(SE3):
    """
    Pinocchio SE3 backend.

    NOTE this backend does not support batching.
    """

    pin_se3: pin.SE3

    def __new__(
        cls,
        se3_like: Union[pin.SE3, Float[np.ndarray, "7"]],
        backend: Literal["numpy"] = "numpy",
    ) -> "PinocchioSE3":
        return super().__new__(cls, se3_like)

    def __init__(
        self,
        se3_like: Union[pin.SE3, Float[np.ndarray, "7"]],
        backend: Literal["numpy"] = "numpy",
    ):
        if isinstance(se3_like, pin.SE3):
            self.pin_se3 = se3_like
        elif isinstance(se3_like, np.ndarray):
            # NOTE pin.Quaternion expects xyzw when initialized with a 4D array
            self.pin_se3 = pin.SE3(pin.Quaternion(np.roll(se3_like[3:7], -1)), se3_like[:3])
        elif isinstance(se3_like, PinocchioSE3):
            self.pin_se3 = se3_like.pin_se3
        else:
            raise ValueError(f"Unsupported type: {type(se3_like)}")

    @property
    def xyz(self) -> Float[np.ndarray, "3"]:
        return self.pin_se3.translation

    @property
    def quat_wxyz(self) -> Float[np.ndarray, "4"]:
        q = pin.Quaternion(self.pin_se3.rotation)
        return np.array([q.w, q.x, q.y, q.z])

    @property
    def quat_xyzw(self) -> Float[np.ndarray, "4"]:
        q = pin.Quaternion(self.pin_se3.rotation)
        return np.array([q.x, q.y, q.z, q.w])

    @property
    def xyz_wxyz(self) -> Float[np.ndarray, "7"]:
        return np.concatenate([self.xyz, self.quat_wxyz], axis=-1)

    def as_matrix(self) -> Float[np.ndarray, "4 4"]:
        return self.pin_se3.homogeneous

    @staticmethod
    def exp(
        log_transform: Float[np.ndarray, "6"],
        backend: Literal["numpy"] = "numpy",
    ) -> "PinocchioSE3":
        """
        Compute the exponential map of the logarithm of the SE(3) transformation.
        The input is assumed to be in [log_translation, log_rotation] format.

        Example:
            >>> log_se3 = np.array([1.0, 3.92699, 0.785398, 1.570796, 0.0, 0.0])
            >>> se3 = PinocchioSE3.exp(log_se3)
            >>> np.allclose(se3.xyz_wxyz, np.array([1.0, 2.0, 3.0, 0.70710678, 0.70710678, 0.0, 0.0]), atol=1e-6)
            True
        """
        return PinocchioSE3(pin.exp6(pin.Motion(log_transform)), backend)

    def log(self) -> Float[np.ndarray, "6"]:
        """
        Compute the logarithm of the SE(3) transformation.
        The output is in [v, omega] format (linear first, then angular).

        Example:
            >>> se3 = PinocchioSE3(np.array([1.0, 2.0, 3.0, 0.70710678, 0.70710678, 0.0, 0.0]))
            >>> log_se3 = se3.log()
            >>> np.allclose(log_se3, np.array([1.0, 3.92699, 0.785398, 1.570796, 0.0, 0.0]), atol=1e-6)
            True
        """
        return pin.log6(self.pin_se3).vector

    def inverse(self) -> "PinocchioSE3":
        return PinocchioSE3(self.pin_se3.inverse())

    def adjoint(self) -> Float[np.ndarray, "6 6"]:
        """
        Compute the 6x6 adjoint representation Ad_T of this SE(3) transform.
        This matrix maps twists between frames as: v' = Ad_T @ v.

        Convention: twist/log vectors use [v, omega] ordering (linear first, then angular).

        Example:
            >>> # 90-deg rotation about Z (no translation)
            >>> theta = np.pi / 2
            >>> se3 = PinocchioSE3(np.array([0.0, 0.0, 0.0, np.cos(theta/2), 0.0, 0.0, np.sin(theta/2)]))
            >>> Ad = se3.adjoint()
            >>> Rz = np.array([[0.0, -1.0, 0.0],
            ...               [1.0,  0.0, 0.0],
            ...               [0.0,  0.0, 1.0]])
            >>> Ad_expected = np.block([[Rz, np.zeros((3, 3))],
            ...                         [np.zeros((3, 3)), Rz]])
            >>> Ad.shape
            (6, 6)
            >>> np.allclose(Ad, Ad_expected)
            True
        """
        return self.pin_se3.action

    def jlog(self) -> Float[np.ndarray, "6 6"]:
        """Compute the Jacobian of the logarithm map."""
        return pin.Jlog6(self.pin_se3)

    def __repr__(self):
        return f"PinSE3(xyz={np.round(self.xyz, decimals=4)}, wxyz={np.round(self.quat_wxyz, decimals=4)})"

    def __mul__(self, other: Union["SE3", pin.SE3]) -> "PinocchioSE3":
        if not isinstance(other, PinocchioSE3):
            if isinstance(other, pin.SE3):
                other = PinocchioSE3(other)
            else:
                raise ValueError(f"Unsupported type: {type(other)}")
        return PinocchioSE3(self.pin_se3 * other.pin_se3)

    def __rmul__(self, other: Union[SE3, pin.SE3]) -> "PinocchioSE3":
        if not isinstance(other, PinocchioSE3):
            if isinstance(other, pin.SE3):
                other = PinocchioSE3(other)
            else:
                raise ValueError(f"Unsupported type: {type(other)}")
        return other * self


__all__ = ["PinocchioSE3"]
