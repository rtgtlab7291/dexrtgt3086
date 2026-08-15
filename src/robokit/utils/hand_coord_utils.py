import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Tuple, Union

import numpy as np


if TYPE_CHECKING:
    import torch


AXIS = Literal["x", "y", "z", "-x", "-y", "-z"]

AXIS_TO_VEC: Dict[AXIS, Tuple[int, int, int]] = {
    "x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "y": (0, 1, 0),
    "-y": (0, -1, 0),
    "z": (0, 0, 1),
    "-z": (0, 0, -1),
}
VEC_TO_AXIS: Dict[Tuple[int, int, int], AXIS] = {v: k for k, v in AXIS_TO_VEC.items()}


@dataclass(frozen=True)
class HandCoordinateSpec:
    """
    Maps semantic hand directions to axes of the hand base-link frame (often the wrist).

    Note: in some hand URDFs the wrist is not the base link (e.g., Shadow Hand with a large base).
    If so, either:
    - define this spec in the actual base-link frame or
    - truncate the URDF so the wrist frame becomes the base link.

    Semantics:
    - palm_forward_axis: outward normal of the palm ("out of the palm").
    - four_fingers_up_axis: direction from palm toward fingertips ("index/middle fingers direction").
    - palm_right_axis: right side when looking along palm_forward_axis.

    Constraints:
    - All axes must be distinct.
    - palm_forward_axis and four_fingers_up_axis must be orthogonal.
    - Right-handed: cross(palm_forward_axis, four_fingers_up_axis) == palm_right_axis
    """

    palm_forward_axis: AXIS
    four_fingers_up_axis: AXIS

    def __post_init__(self):
        if self.palm_forward_axis[-1] == self.four_fingers_up_axis[-1]:
            raise ValueError("palm_forward_axis must be orthogonal to four_fingers_up_axis")

    @classmethod
    def from_legacy_string(cls, spec: str) -> "HandCoordinateSpec":
        """Convert a legacy coordinate string or preset to a hand coordinate specification."""
        from robokit.xform.utils import COMMON_COORDS

        spec = spec.strip().lower()
        spec = COMMON_COORDS.get(spec, spec)
        dirs = dict(re.findall(r"(x|y|z)\s*:\s*(\w+)", spec))
        axis_of = {direction: axis for axis, direction in dirs.items()}
        forward = axis_of["front"] if "front" in axis_of else "-" + axis_of["back"]
        up = axis_of["up"] if "up" in axis_of else "-" + axis_of["down"]
        return cls(palm_forward_axis=forward, four_fingers_up_axis=up)

    @property
    def palm_right_axis(self) -> AXIS:
        up = AXIS_TO_VEC[self.four_fingers_up_axis]
        fwd = AXIS_TO_VEC[self.palm_forward_axis]
        return VEC_TO_AXIS[tuple(np.cross(fwd, up))]

    @property
    def palm_forward_vec(self) -> np.ndarray:
        return np.array(AXIS_TO_VEC[self.palm_forward_axis], dtype=np.float32)

    @property
    def four_fingers_up_vec(self) -> np.ndarray:
        return np.array(AXIS_TO_VEC[self.four_fingers_up_axis], dtype=np.float32)

    @property
    def palm_right_vec(self) -> np.ndarray:
        return np.array(AXIS_TO_VEC[self.palm_right_axis], dtype=np.float32)


def hand_coord_conversion(
    src_spec: HandCoordinateSpec, dst_spec: HandCoordinateSpec, return_tensors: Literal["np", "pt"] = "np"
) -> "Union[np.ndarray, torch.Tensor]":
    """Compute R_dst_src: rotation matrix from src to dst coordinate system.

    The rotation R_dst_src transforms vectors from src frame to dst frame:
    v_dst = R_dst_src @ v_src

    Args:
        src_spec: Source hand coordinate specification
        dst_spec: Destination hand coordinate specification
        return_tensors: "np" for a NumPy array (default) or "pt" for a torch tensor.

    Returns:
        3x3 rotation matrix R_dst_src
    """
    src_basis = np.column_stack(
        [
            AXIS_TO_VEC[src_spec.palm_forward_axis],
            AXIS_TO_VEC[src_spec.four_fingers_up_axis],
            AXIS_TO_VEC[src_spec.palm_right_axis],
        ]
    ).astype(np.float32)

    dst_basis = np.column_stack(
        [
            AXIS_TO_VEC[dst_spec.palm_forward_axis],
            AXIS_TO_VEC[dst_spec.four_fingers_up_axis],
            AXIS_TO_VEC[dst_spec.palm_right_axis],
        ]
    ).astype(np.float32)

    R_dst_src = dst_basis @ src_basis.T

    if return_tensors == "pt":
        import torch

        return torch.from_numpy(R_dst_src).float()
    return R_dst_src


MANOPTH_HAND_COORD_SPEC = HandCoordinateSpec(palm_forward_axis="-y", four_fingers_up_axis="-x")
