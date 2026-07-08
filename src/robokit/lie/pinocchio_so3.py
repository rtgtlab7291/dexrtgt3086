from typing import Literal, Union

import numpy as np
import pinocchio as pin
from jaxtyping import Float

from robokit.lie.so3 import SO3


class PinocchioSO3(SO3):
    """
    Pinocchio SO3 backend.

    NOTE this backend does not support batching.
    """

    pin_quat: pin.Quaternion

    def __init__(
        self,
        so3_like: Union[pin.Quaternion, Float[np.ndarray, "4"]],
        backend: Literal["numpy"] = "numpy",
    ):
        if isinstance(so3_like, pin.Quaternion):
            self.pin_quat = so3_like
        elif isinstance(so3_like, np.ndarray):
            # pin.Quaternion expects xyzw when initialized with a 4D array
            # Our convention is wxyz
            xyzw = np.roll(so3_like, -1)
            self.pin_quat = pin.Quaternion(xyzw)
        elif isinstance(so3_like, PinocchioSO3):
            self.pin_quat = so3_like.pin_quat
        else:
            raise ValueError(f"Unsupported type: {type(so3_like)}")

    @property
    def wxyz(self) -> Float[np.ndarray, "4"]:
        return np.array([self.pin_quat.w, self.pin_quat.x, self.pin_quat.y, self.pin_quat.z])

    def as_matrix(self) -> Float[np.ndarray, "3 3"]:
        """
        Convert the SO3 representation to a 3x3 rotation matrix.

        Example:
            >>> so3 = PinocchioSO3(np.array([0.7071, 0.0, 0.0, 0.7071]))
            >>> expected_matrix = np.array([[0.0, -1.0, 0.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0]])
            >>> np.allclose(so3.as_matrix(), expected_matrix, atol=1e-4)
            True
        """
        return self.pin_quat.toRotationMatrix()

    @staticmethod
    def from_matrix(
        matrix: Float[np.ndarray, "3 3"],
        backend: Literal["numpy"] = "numpy",
    ) -> "PinocchioSO3":
        return PinocchioSO3(pin.Quaternion(matrix), backend)

    @staticmethod
    def exp(
        log_rot: Float[np.ndarray, "3"],
        backend: Literal["numpy"] = "numpy",
    ) -> "PinocchioSO3":
        # pin uses exponential map via rotation vector to quaternion
        # pin.exp3 returns a rotation matrix; use Quaternion from rotation vector instead
        R = pin.exp3(log_rot)
        return PinocchioSO3(pin.Quaternion(R), backend)

    def log(self) -> Float[np.ndarray, "3"]:
        return pin.log3(self.as_matrix())

    def apply(self, v: Float[np.ndarray, "... 3"]) -> Float[np.ndarray, "... 3"]:
        return self.pin_quat * v

    def inverse(self) -> "PinocchioSO3":
        return PinocchioSO3(self.pin_quat.inverse())

    def __repr__(self):
        return f"PinSO3(wxyz={np.round(self.wxyz, decimals=4)})"

    def __mul__(self, other: Union[SO3, pin.Quaternion]) -> "PinocchioSO3":
        if not isinstance(other, PinocchioSO3):
            if isinstance(other, pin.Quaternion):
                other = PinocchioSO3(other)
            else:
                raise ValueError(f"Unsupported type: {type(other)}")
        # Quaternion multiplication corresponds to rotation composition
        q = self.pin_quat * other.pin_quat
        q.normalize()
        return PinocchioSO3(q)

    def __rmul__(self, other: Union[SO3, pin.Quaternion]) -> "PinocchioSO3":
        if not isinstance(other, PinocchioSO3):
            if isinstance(other, pin.Quaternion):
                other = PinocchioSO3(other)
            else:
                raise ValueError(f"Unsupported type: {type(other)}")
        return other * self


__all__ = ["PinocchioSO3"]
