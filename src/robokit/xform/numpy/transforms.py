from typing import Optional, cast

import numpy as np
from jaxtyping import Float

from robokit.xform.numpy.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix


def expand_tf_mat(tf_mat: Float[np.ndarray, "... 3 4"]) -> Float[np.ndarray, "... 4 4"]:
    """Expand transformation matrix of shape [... 3 4] to shape [... 4 4].

    Args:
        tf_mat (np.ndarray): Transformation matrix in shape [... 3 4] or [... 4 4].

    Returns:
        np.ndarray: Expanded transformation matrix in shape [... 4 4].

    Examples:
        >>> tf_mat = np.array([[0, 1, 0, 1], [0, 0, 1, 2], [1, 0, 0, 3]], dtype=np.float32)
        >>> expand_tf_mat(tf_mat)
        array([[0., 1., 0., 1.],
               [0., 0., 1., 2.],
               [1., 0., 0., 3.],
               [0., 0., 0., 1.]], dtype=float32)
    """
    if tf_mat.shape[-2:] == (3, 4):
        last_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=tf_mat.dtype)
        # broadcast last_row to match batch dimensions
        last_row = np.broadcast_to(last_row, tf_mat.shape[:-2] + (1, 4))
        tf_mat = np.concatenate([tf_mat, last_row], axis=-2)
    return tf_mat


def rot_tl_to_tf_mat(
    rot_mat: Optional[Float[np.ndarray, "... 3 3"]] = None, tl: Optional[Float[np.ndarray, "... 3"]] = None
) -> Float[np.ndarray, "... 4 4"]:
    """Build transformation matrix with rotation matrix and translation vector.

    Args:
        rot_mat (np.ndarray, optional): Rotation matrix in shape [... 3 3]. Defaults to None.
        tl (np.ndarray, optional): Translation vector in shape [... 3]. Defaults to None.

    Returns:
        np.ndarray: Transformation matrix in shape [... 4 4].

    Examples:
        >>> rot_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        >>> tl = np.array([1, 2, 3], dtype=np.float32)
        >>> rot_tl_to_tf_mat(rot_mat, tl)
        array([[0., 1., 0., 1.],
               [0., 0., 1., 2.],
               [1., 0., 0., 3.],
               [0., 0., 0., 1.]], dtype=float32)
        >>> rot_tl_to_tf_mat(tl=tl)
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]], dtype=float32)
        >>> rot_tl_to_tf_mat(rot_mat=rot_mat)
        array([[0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [1., 0., 0., 0.],
               [0., 0., 0., 1.]], dtype=float32)
    """
    if rot_mat is not None and tl is None:
        tl = np.zeros(rot_mat.shape[:-2] + (3,), dtype=rot_mat.dtype)
    elif rot_mat is None and tl is not None:
        rot_mat = np.eye(3, dtype=tl.dtype)
        # broadcast rot_mat to match batch dimensions of tl
        rot_mat = np.broadcast_to(rot_mat, tl.shape[:-1] + (3, 3))
    elif rot_mat is None and tl is None:
        raise ValueError("Either rot_mat or tl should be provided.")

    rot_mat = cast(np.ndarray, rot_mat)
    tl = cast(np.ndarray, tl)

    # broadcast shapes
    b_shape = np.broadcast_shapes(rot_mat.shape[:-2], tl.shape[:-1])
    rot_mat = np.broadcast_to(rot_mat, b_shape + (3, 3))
    tl = np.broadcast_to(tl, b_shape + (3,))

    tf_mat = np.concatenate([rot_mat, tl[..., np.newaxis]], axis=-1)
    return expand_tf_mat(tf_mat)


def inverse_tf_mat(tf_mat: Float[np.ndarray, "... 4 4"]) -> Float[np.ndarray, "... 4 4"]:
    """Invert a rigid transformation matrix (transpose the rotation, rotate the translation back).

    The numpy sibling of `xform.warp.inverse_tf_mat_func` and `xform.warp.torch_wrappers.inverse_tf_mat`.

    Args:
        tf_mat (np.ndarray): Transformation matrix in shape [... 4 4].

    Returns:
        np.ndarray: Inverted transformation matrix in shape [... 4 4].

    Examples:
        >>> tf_mat = rot_tl_to_tf_mat(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
        ...                           np.array([1, 2, 3], dtype=np.float32))
        >>> np.allclose(inverse_tf_mat(tf_mat) @ tf_mat, np.eye(4), atol=1e-6)
        True
    """
    rot_t = np.swapaxes(tf_mat[..., :3, :3], -1, -2)
    return rot_tl_to_tf_mat(rot_t, -np.einsum("...ij,...j->...i", rot_t, tf_mat[..., :3, 3]))


def pose7_to_tf_mat(pose7: Float[np.ndarray, "... 7"]) -> Float[np.ndarray, "... 4 4"]:
    """Convert translation-first, scalar-first poses `[x, y, z, qw, qx, qy, qz]` to matrices.

    Args:
        pose7 (np.ndarray): Poses in shape [... 7].

    Returns:
        np.ndarray: Transformation matrix in shape [... 4 4].

    Examples:
        >>> pose7 = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        >>> tf_mat = pose7_to_tf_mat(pose7)
        >>> tf_mat.shape, np.allclose(tf_mat[0, :3, 3], [1.0, 2.0, 3.0])
        ((1, 4, 4), True)
    """
    return rot_tl_to_tf_mat(quaternion_to_matrix(pose7[..., 3:]), pose7[..., :3]).astype(pose7.dtype, copy=False)


def tf_mat_to_pose7(tf_mat: Float[np.ndarray, "... 4 4"]) -> Float[np.ndarray, "... 7"]:
    """Convert matrices to translation-first, scalar-first poses `[x, y, z, qw, qx, qy, qz]`.

    Args:
        tf_mat (np.ndarray): Transformation matrix in shape [... 4 4].

    Returns:
        np.ndarray: Poses in shape [... 7].

    Examples:
        >>> pose7 = np.array([[0.1, 0.2, 0.3, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        >>> back = tf_mat_to_pose7(pose7_to_tf_mat(pose7))
        >>> np.allclose(np.abs(back), np.abs(pose7), atol=1e-6)
        True
    """
    return np.concatenate([tf_mat[..., :3, 3], matrix_to_quaternion(tf_mat[..., :3, :3])], axis=-1, dtype=tf_mat.dtype)


__all__ = [
    "rot_tl_to_tf_mat",
    "expand_tf_mat",
    "inverse_tf_mat",
    "pose7_to_tf_mat",
    "tf_mat_to_pose7",
]
