import numpy as np
from jaxtyping import Float


def matrix_to_quaternion(m: Float[np.ndarray, "... 3 3"]) -> Float[np.ndarray, "... 4"]:
    """
    Convert rotation matrices to quaternions using Shepperds's method.

    Args:
        m: rotation matrices, in shape of (..., 3, 3).

    Returns:
        quaternions with real part first (wxyz), in shape of (..., 4).

    Example:
        >>> q = matrix_to_quaternion(np.eye(3))
        >>> np.allclose(q, np.array([1., 0., 0., 0.]))
        True
        >>> q = matrix_to_quaternion(np.diag([1, -1, -1]))
        >>> np.allclose(q, np.array([0., 1., 0., 0.]))
        True
        >>> rot_mat = np.array([[-0.2533, -0.6075,  0.7529],
        ...                     [ 0.8445, -0.5185, -0.1343],
        ...                     [ 0.4720,  0.6017,  0.6443]])
        >>> q = matrix_to_quaternion(rot_mat)
        >>> np.allclose(q, np.array([0.4671, 0.3940, 0.1503, 0.7772]), atol=1e-4)
        True
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    q_abs = np.stack(
        (1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22), axis=-1
    )
    q_abs = np.sqrt(np.maximum(q_abs, 0.0))

    quat_by_rijk = np.stack(
        (
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ),
        axis=-2,
    )
    flr = 0.1
    quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], flr))

    idx = np.argmax(q_abs, axis=-1)[..., None, None]
    quat = np.take_along_axis(quat_candidates, idx, axis=-2).squeeze(-2)

    return standardize_quaternion(quat)


def standardize_quaternion(quaternions: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 4"]:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: a numpy array of shape (..., 4), where the last dimension
           represents the quaternion components [w, x, y, z].

    Returns:
        A numpy array with shape of (..., 4) representing the quaternion components
        [w, x, y, z].

    Example:
        >>> quaternions = np.array([[0.0, -0.7071, 0.0, 0.7071], [-0.7071, 0.0, 0.7071, 0.0]])
        >>> expected = np.array([[0.0, -0.7071, 0.0, 0.7071], [0.7071, -0.0, -0.7071, -0.0]])
        >>> np.allclose(standardize_quaternion(quaternions), expected)
        True
    """
    return np.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_axis_angle(matrix: Float[np.ndarray, "... 3 3"]) -> Float[np.ndarray, "... 3"]:
    """
    Convert rotation matrices to axis-angle representation.

    Args:
        matrix: Rotation matrices with shape (..., 3, 3).

    Returns:
        Axis-angle representation with shape (..., 3) where the magnitude
        represents the rotation angle and the direction represents the rotation axis.

    Example:
        >>> matrix = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        >>> expected = np.array([np.pi/2, 0., 0.])
        >>> np.allclose(matrix_to_axis_angle(matrix), expected, atol=1e-4)
        True
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle: Float[np.ndarray, "... 3"]) -> Float[np.ndarray, "... 4"]:
    """
    Convert axis-angle representation to quaternion.

    Args:
        axis_angle: Axis-angle representation [..., 3] where the magnitude is the rotation angle

    Returns:
        Quaternion representation [..., 4] in wxyz format (w, x, y, z)

    Example:
        >>> axis_angle = np.array([1.5708, 0.0, 0.0])
        >>> expected = np.array([0.7071, 0.7071, 0.0000, 0.0000])
        >>> np.allclose(axis_angle_to_quaternion(axis_angle), expected, atol=1e-4)
        True
    """
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = np.abs(angles) < eps

    sin_half_angles_over_angles = np.empty_like(angles)

    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles[~small_angles]) / angles[~small_angles]

    # For small angles, use Taylor series approximation:
    # sin(x/2) ≈ x/2 - (x/2)³/6, so sin(x/2)/x ≈ 1/2 - x²/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48

    w = np.cos(half_angles)
    xyz = axis_angle * sin_half_angles_over_angles

    return np.concatenate([w, xyz], axis=-1)


def axis_angle_to_matrix(axis_angle: Float[np.ndarray, "... 3"]) -> Float[np.ndarray, "... 3 3"]:
    """
    Converts an axis-angle representation to a 3x3 rotation matrix.

    This function takes a vector in the axis-angle representation and calculates
    the corresponding rotation matrix.

    Args:
        axis_angle: A numpy array of shape (..., 3) representing a vector in
            axis-angle format where the direction indicates the axis of rotation
            and the magnitude represents the rotation angle in radians.

    Returns:
        A numpy array of shape (..., 3, 3) representing the rotation matrices
        corresponding to each axis-angle input.

    Example:
        >>> axis_angle = np.array([np.pi/2, 0., 0.])
        >>> expected = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        >>> np.allclose(axis_angle_to_matrix(axis_angle), expected, atol=1e-4)
        True
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def random_quaternions(n: int) -> Float[np.ndarray, "... 4"]:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.

    Returns:
        Quaternions as a numpy array of shape (n, 4).
    """
    o = np.random.randn(n, 4)
    s = np.sum(o * o, axis=1)
    o = o / np.copysign(np.sqrt(s), o[:, 0])[:, np.newaxis]
    return o


def quaternion_to_matrix(quaternions: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 3 3"]:
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: Quaternions [..., 4] in wxyz format (w, x, y, z)

    Returns:
        Rotation matrices [..., 3, 3]

    Example:
        >>> q = np.array([1., 0., 0., 0.])
        >>> m = quaternion_to_matrix(q)
        >>> np.allclose(m, np.eye(3))
        True
        >>> q = np.array([0., 1., 0., 0.])
        >>> m = quaternion_to_matrix(q)
        >>> np.allclose(m, np.diag([1, -1, -1]))
        True
    """
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation_matrix = np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )

    return rotation_matrix


def quaternion_raw_multiply(a: Float[np.ndarray, "... 4"], b: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 4"]:
    """
    Multiply two quaternions using raw multiplication.

    Args:
        a: Quaternions [..., 4] in wxyz format (w, x, y, z)
        b: Quaternions [..., 4] in wxyz format (w, x, y, z)

    Returns:
        The product of a and b [..., 4]
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return np.stack([ow, ox, oy, oz], axis=-1)


def quaternion_multiply(a: Float[np.ndarray, "... 4"], b: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 4"]:
    """
    This function takes two input quaternions, represented as arrays with shape (..., 4),
    and computes their Hamilton product, which results in a new quaternion.

    Args:
        a: a numpy array of shape (..., 4), where the last dimension
           represents the quaternion components [w, x, y, z].
        b: a numpy array of shape (..., 4), where the last dimension
           represents the quaternion components [w, x, y, z].

    Returns:
        A numpy array with shape of (..., 4) representing the quaternion components
        [w, x, y, z].

    Example:
        >>> a = np.array([0.9238795, 0.0, 0.0, 0.3826834])
        >>> b = np.array([0.9238795, 0.0, 0.0, 0.3826834])
        >>> expected = np.array([0.7071068, 0.0, 0.0, 0.7071068])
        >>> np.allclose(quaternion_multiply(a, b), expected, atol=1e-6)
        True
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternions: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 4"]:
    """
    Invert quaternions by negating the imaginary parts.

    Args:
        quaternions: Quaternions [..., 4] in wxyz format (w, x, y, z)

    Returns:
        Inverted quaternions [..., 4]

    Example:
        >>> q = np.array([1., 1., 1., 1.])
        >>> q_inv = quaternion_invert(q)
        >>> np.allclose(q_inv, np.array([1., -1., -1., -1.]))
        True
    """
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    return np.stack([w, -x, -y, -z], axis=-1)


def quaternion_apply(
    quaternions: Float[np.ndarray, "... 4"], points: Float[np.ndarray, "... 3"]
) -> Float[np.ndarray, "... 3"]:
    """
    Apply quaternion rotations to 3D points using raw quaternion multiplication.

    Args:
        quaternions: Quaternions [..., 4] in wxyz format (w, x, y, z)
        points: 3D points [..., 3] to rotate

    Returns:
        Rotated points [..., 3]

    Example:
        >>> quaternion = np.array([0.7071, 0.0, 0.0, 0.7071])
        >>> point = np.array([1.0, 0.0, 0.0])
        >>> expected = np.array([0., 1., 0.])
        >>> np.allclose(quaternion_apply(quaternion, point), expected, atol=1e-4)
        True
    """
    point_quaternions = np.concatenate([np.zeros(points.shape[:-1] + (1,)), points], axis=-1)

    q_inv = quaternion_invert(quaternions)
    first = quaternion_raw_multiply(quaternions, point_quaternions)
    rotated_quaternions = quaternion_raw_multiply(first, q_inv)

    return rotated_quaternions[..., 1:]


def quaternion_to_axis_angle(quaternions: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 3"]:
    """
    Convert quaternions to axis-angle representation.

    Args:
        quaternions: Quaternions [..., 4] in wxyz format (w, x, y, z)

    Returns:
        Axis-angle representation [..., 3]

    Example:
        >>> quaternions = np.array([0.7071, 0.7071, 0.0, 0.0])
        >>> expected = np.array([1.5708, 0.0000, 0.0000])
        >>> np.allclose(quaternion_to_axis_angle(quaternions), expected, atol=1e-4)
        True
    """
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles

    eps = 1e-6
    small_angles = np.abs(angles) < eps

    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48

    return quaternions[..., 1:] / sin_half_angles_over_angles


__all__ = [
    "matrix_to_quaternion",
    "matrix_to_axis_angle",
    "standardize_quaternion",
    "axis_angle_to_quaternion",
    "axis_angle_to_matrix",
    "quaternion_to_matrix",
    "quaternion_apply",
    "quaternion_raw_multiply",
    "quaternion_multiply",
    "quaternion_invert",
    "quaternion_to_axis_angle",
    "random_quaternions",
]
