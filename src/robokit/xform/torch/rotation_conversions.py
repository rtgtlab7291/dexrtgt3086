from typing import Dict, Literal, get_args

import torch
from jaxtyping import Float

from robokit.thirdparty.pytorch3d.rotation_conversions import (
    _axis_angle_rotation,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    quaternion_invert,
    quaternion_to_matrix,
    random_quaternions,
    random_rotation,
    random_rotations,
    rotation_6d_to_matrix,
    standardize_quaternion,
)
from robokit.thirdparty.pytorch3d.rotation_conversions import (
    matrix_to_euler_angles as matrix_to_euler_angles_pt,
)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x)) with a zero subgradient where x is 0.
    This implementation uses torch.where instead of boolean indexing for vmap compatibility.
    """
    positive_mask = x > 0
    # Replace non-positive values with 1 to avoid sqrt gradient issues
    safe_x = torch.where(positive_mask, x, torch.ones_like(x))
    sqrt_safe = torch.sqrt(safe_x)
    return torch.where(positive_mask, sqrt_safe, torch.zeros_like(x))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    This implementation is vmap compatible.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


def quaternion_raw_multiply(
    a: Float[torch.Tensor, "... 4"], b: Float[torch.Tensor, "... 4"]
) -> Float[torch.Tensor, "... 4"]:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(
    q_wxyz_1: Float[torch.Tensor, "... 4"], q_wxyz_2: Float[torch.Tensor, "... 4"]
) -> Float[torch.Tensor, "... 4"]:
    """
    Multiply two quaternions.

    Args:
        q_wxyz_1: Quaternions with real part first, as tensor of shape (..., 4).
        q_wxyz_2: Quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        The product of the two quaternions, as tensor of shape (..., 4).

    Example:
        >>> q1 = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
        >>> q2 = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
        >>> expected = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
        >>> torch.allclose(quaternion_multiply(q1, q2), expected, atol=1e-6)
        True
    """
    return standardize_quaternion(quaternion_raw_multiply(q_wxyz_1, q_wxyz_2))


def quaternion_apply(
    quaternion: Float[torch.Tensor, "... 4"], point: Float[torch.Tensor, "... 3"]
) -> Float[torch.Tensor, "... 3"]:
    """
    Apply quaternion rotation to 3D points.

    Args:
        quaternion: Quaternions with real part first, as tensor of shape (..., 4).
        point: 3D points to rotate, as tensor of shape (..., 3).

    Returns:
        Rotated 3D points with shape (..., 3).

    Example:
        >>> quaternion = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
        >>> point = torch.tensor([1.0, 0.0, 0.0])
        >>> expected = torch.tensor([0., 1., 0.])
        >>> torch.allclose(quaternion_apply(quaternion, point), expected, atol=1e-4)
        True
    """
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion), quaternion_invert(quaternion)
    )
    return out[..., 1:]


def axis_angle_to_quaternion(axis_angle: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 4"]:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Example:
        >>> axis_angle = torch.tensor([1.5708, 0.0, 0.0])
        >>> expected = torch.tensor([0.7071, 0.7071, 0.0000, 0.0000])
        >>> torch.allclose(axis_angle_to_quaternion(axis_angle), expected, atol=1e-4)
        True
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # pyright: ignore[reportArgumentType]
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps

    div_angles = torch.where(small_angles, torch.ones_like(angles), angles)
    sin_half_angles_over_angles = torch.where(
        small_angles, 0.5 - (angles * angles) / 48.0, torch.sin(half_angles) / div_angles
    )

    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions


def quaternion_to_axis_angle(quaternions: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 3"]:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    Example:
        >>> quaternions = torch.tensor([0.7071, 0.7071, 0.0, 0.0])
        >>> expected = torch.tensor([1.5708, 0.0000, 0.0000])
        >>> torch.allclose(quaternion_to_axis_angle(quaternions), expected, atol=1e-4)
        True
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)  # pyright: ignore[reportArgumentType]
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps

    div_angles = torch.where(small_angles, torch.ones_like(angles), angles)
    sin_half_angles_over_angles = torch.where(
        small_angles, 0.5 - (angles * angles) / 48.0, torch.sin(half_angles) / div_angles
    )

    return quaternions[..., 1:] / sin_half_angles_over_angles


# fmt: off
_AXES = Literal[
    "SXYZ", "SXYX", "SXZY", "SXZX", "SYZX", "SYZY", "SYXZ", "SYXY", "SZXY", "SZXZ", "SZYX", "SZYZ",
    "RZYX", "RXYX", "RYZX", "RXZX", "RXZY", "RYZY", "RZXY", "RYXY", "RYXZ", "RZXZ", "RXYZ", "RZYZ"
]
# fmt: on
_VALID_AXES: Dict[_AXES, None] = {axes: None for axes in get_args(_AXES)}


def euler_angles_to_matrix(
    euler_angles: Float[torch.Tensor, "... 3"], axes: _AXES = "SXYZ"
) -> Float[torch.Tensor, "... 3 3"]:
    """Converts Euler angles to rotation matrices.

    Args:
        euler_angles (torch.Tensor): Euler angles, the shape could be [..., 3].
        axes (str): Axis specification; one of 24 axis string sequences - e.g. `SXYZ (the default). It's recommended to use the full name of the axes, e.g. "SXYZ" instead of "XYZ", but if 3 characters are provided, it will be prefixed with "S".

    Returns:
        torch.Tensor: Rotation matrices [..., 3, 3].

    Example:
        >>> euler_angles = torch.tensor([1.0, 0.5, 2.0])
        >>> expected = torch.tensor([[-0.3652, -0.6592,  0.6574],
                                     [ 0.7980,  0.1420,  0.5857],
                                     [-0.4794,  0.7385,  0.4742]])
        >>> torch.allclose(euler_angles_to_matrix(euler_angles, axes="SXYZ"), expected, atol=1e-4)
        True
    """
    axes = axes.upper()  # pyright: ignore[reportAssignmentType]
    if len(axes) == 3:
        axes = f"S{axes}"  # pyright: ignore[reportAssignmentType]
    if axes not in _VALID_AXES:
        raise ValueError(f"Invalid axes: {axes}")

    matrices = [_axis_angle_rotation(c, e) for c, e in zip(axes[1:], torch.unbind(euler_angles, -1))]
    if axes[0] == "S":
        return torch.matmul(torch.matmul(matrices[2], matrices[1]), matrices[0])
    else:
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def matrix_to_euler_angles(
    matrix: Float[torch.Tensor, "... 3 3"], axes: _AXES = "SXYZ"
) -> Float[torch.Tensor, "... 3"]:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices with shape (..., 3, 3).
        axes: Convention string of 3/4 letters, e.g. "XYZ", "SXYZ", "RXYZ", "EXYZ".
            If the length is 3, the static rotation is assumed.
            If the length is 4, the first character is "R" (rotating), or "S" (static).

    Returns:
        Euler angles in radians with shape (..., 3).

    Example:
        >>> matrix = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        >>> expected = torch.tensor([1.5708, 0.0000, 0.0000])
        >>> torch.allclose(matrix_to_euler_angles(matrix, axes="SXYZ"), expected, atol=1e-4)
        True
    """
    axes = axes.upper()  # pyright: ignore[reportAssignmentType]
    if len(axes) == 3:
        axes = f"S{axes}"  # pyright: ignore[reportAssignmentType]
    if axes not in _VALID_AXES:
        raise ValueError(f"Invalid axes: {axes}")

    if axes[0] == "S":
        axes = axes[:0:-1]  # pyright: ignore[reportAssignmentType]
        # reverse the axes and output order for static convention
        result = matrix_to_euler_angles_pt(matrix, axes)
        return result.flip(-1)
    else:
        axes = axes[1:]  # pyright: ignore[reportAssignmentType]
        return matrix_to_euler_angles_pt(matrix, axes)


__all__ = [
    "euler_angles_to_matrix",
    "quaternion_apply",
    "quaternion_invert",
    "quaternion_raw_multiply",
    "quaternion_multiply",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "matrix_to_euler_angles",
    "matrix_to_quaternion",
    "matrix_to_rotation_6d",
    "matrix_to_axis_angle",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "euler_angles_to_matrix",
    "rotation_6d_to_matrix",
    "random_quaternions",
    "random_rotation",
    "random_rotations",
    "standardize_quaternion",
]
