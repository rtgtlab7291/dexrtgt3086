from typing import Literal, Optional, Tuple, Union, cast, overload

import torch
from jaxtyping import Float
from typing_extensions import deprecated


def transform_points(
    pts: Float[torch.Tensor, "... n 3"], tf_mat: Float[torch.Tensor, "... 4 4"]
) -> Float[torch.Tensor, "... n 3"]:
    """Apply a transformation matrix on a set of 3D points.

    Args:
        pts (torch.Tensor): 3D points, could be [... n 3]
        tf_mat (torch.Tensor): Transformation matrix, could be [... 4 4]

    Returns:
        Transformed pts in shape of [... n 3]

    Examples:
        >>> pts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> tf_mat = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 2.0], [1.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
        >>> transform_points(pts, tf_mat)
        tensor([[3., 5., 4.],
                [6., 8., 7.]])

    .. note::
        The dimension number of `pts` and `tf_mat` should be the same. The batch dimensions (...) are broadcasted_ (and
        thus must be broadcastable). We don't adopt the shapes [... 3] and [... 4 4] because there is no real
        broadcasted vector-matrix multiplication in pytorch. [... 3] and [... 4 4] will be converted to [... 1 3]
        and [... 4 4] and apply a broadcasted matrix-matrix multiplication.

    .. _broadcasted: https://pytorch.org/docs/stable/notes/broadcasting.html
    """
    if pts.shape[-1] != 3:
        raise ValueError("The last dimension of pts should be 3.")
    homo_pts = to_homo(pts)
    # `homo_pts @ tf_mat.T` or `(tf_mat @ homo_pts.T).T`
    new_pts = torch.matmul(homo_pts, torch.transpose(tf_mat, -2, -1))
    return new_pts[..., :3]


@deprecated("`transform` is deprecated, use `transform_points` instead.")
def transform(
    pts: Float[torch.Tensor, "... n 3"], tf_mat: Float[torch.Tensor, "... 4 4"]
) -> Float[torch.Tensor, "... n 3"]:
    return transform_points(pts, tf_mat)  # type: ignore


def rotate_points(
    pts: Float[torch.Tensor, "... n 3"], rot_mat: Float[torch.Tensor, "... 3 3"]
) -> Float[torch.Tensor, "... n 3"]:
    """Rotate a set of 3D points by a rotation matrix.

    Args:
        pts (torch.Tensor): 3D points in shape [... n 3].
        rot_mat (torch.Tensor): Rotation matrix in shape [... 3 3].

    Returns:
        torch.Tensor: Rotated points in shape [... n 3].
    """
    if pts.ndim != rot_mat.ndim:
        raise ValueError(
            f"The dimension number of pts and rot_mat should be the same, but got {pts.ndim=} and {rot_mat.ndim=}"
        )
    # `pts @ rot_mat.T` or `(rot_mat @ pts.T).T`
    new_pts = torch.matmul(pts, torch.transpose(rot_mat, -2, -1))
    return new_pts


@overload
def project_points(pts: torch.Tensor, intr_mat: torch.Tensor, return_depth: Literal[False] = False) -> torch.Tensor: ...
@overload
def project_points(
    pts: torch.Tensor, intr_mat: torch.Tensor, return_depth: Literal[True]
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def project_points(
    pts: torch.Tensor, intr_mat: torch.Tensor, return_depth: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Project 3D points in the camera space to the image plane.

    Args:
        pts: 3D points, could be Nx3 or BxNx3.
        intr_mat: Intrinsic matrix, could be 3x3 or Bx3x3.

    Returns:
        pixels: the order is uv other than xy.
        depth (if return_depth): depth in the camera space.
    """
    new_pts = pts / pts[..., 2:3]
    new_pts = torch.matmul(new_pts, torch.transpose(intr_mat, -2, -1))

    if not return_depth:
        return new_pts[..., :2]
    else:
        return new_pts[..., :2], pts[..., 2]


def unproject_points(pixels, depth, intr_mat):
    """
    Unproject pixels in the image plane to 3D points in the camera space.

    Args:
        pixels: Pixels in the image plane, could be Nx2 or BxNx2. The order is uv rather than xy.
        depth: Depth in the camera space, could be N, Nx1, BxN or BxNx1.
        intr_mat: Intrinsic matrix, could be 3x3 or Bx3x3.
    Returns:
        pts: 3D points, Nx3 or BxNx3.
    """
    if depth.ndim < pixels.ndim:
        depth = depth[..., None]  # N -> Nx1, BxN -> BxNx1
    principal_point = torch.unsqueeze(intr_mat[..., :2, 2], dim=-2)  # 1x2, Bx1x2
    focal_length = torch.cat([intr_mat[..., 0:1, 0:1], intr_mat[..., 1:2, 1:2]], dim=-1)  # 1x2, Bx1x2
    xys = (pixels - principal_point) * depth / focal_length
    pts = torch.cat([xys, depth], dim=-1)
    return pts


def inverse_tf_mat(rot_or_tf_mat: torch.Tensor) -> torch.Tensor:
    """Inverse a rotation matrix or a transformation matrix. Reference_

    Args:
        rot_or_tf_mat (torch.Tensor): Rotation matrix (in shape [..., 3, 3]) or transformation matrix (in shape [..., 4, 4]).

    Returns:
        torch.Tensor: Inversed matrix.

    Examples:
        >>> tf_mat = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 2], [1, 0, 0, 3], [0, 0, 0, 1]], dtype=torch.float32)
        >>> torch.allclose(inverse_tf_mat(tf_mat) @ tf_mat, torch.eye(4))
        True
        >>> rot_mat = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
        >>> torch.allclose(inverse_tf_mat(rot_mat) @ rot_mat, torch.eye(3))
        True

    .. _Reference: https://math.stackexchange.com/a/1315407/757569
    """
    if rot_or_tf_mat.shape[-1] == 3:  # rotation matrix
        new_mat = torch.transpose(rot_or_tf_mat, -2, -1)
    else:  # transformation matrix
        new_rot_mat = torch.transpose(rot_or_tf_mat[..., :3, :3], -2, -1)
        ori_tl = torch.unsqueeze(rot_or_tf_mat[..., :3, 3], dim=-2)  # 1x3, Bx1x3
        new_tl = torch.squeeze(-rotate_points(ori_tl, new_rot_mat), dim=-2)  # 3, Bx3
        new_mat = rot_tl_to_tf_mat(new_rot_mat, new_tl)
    return new_mat


def swap_major(rot_or_tf_mat: torch.Tensor) -> torch.Tensor:
    """Swap the major of a rotation matrix or a transformation matrix. Reference_

    Args:
        rot_or_tf_mat (torch.Tensor): Rotation or transformation matrix in shape [..., 3, 3] or [..., 4, 4].

    Returns:
        torch.Tensor: Matrix with swapped major.

    .. _Reference: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/row-major-vs-column-major-vector # noqa
    """
    return torch.transpose(rot_or_tf_mat, -2, -1)


def to_homo(pts_3d: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 4"]:
    """Convert Cartesian 3D points to Homogeneous 4D points.

    Args:
      pts_3d (torch.Tensor): Cartesian 3D points in shape [... 3].

    Returns:
        torch.Tensor: Homogeneous 4D points in shape [... 4].

    Examples:
        >>> pts = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
        >>> to_homo(pts)
        tensor([[0., 1., 0., 1.],
                [0., 0., 1., 1.],
                [1., 0., 0., 1.]])
    """
    return torch.cat([pts_3d, torch.ones_like(pts_3d[..., :1])], dim=-1)


def rot_tl_to_tf_mat(
    rot_mat: Optional[Float[torch.Tensor, "... 3 3"]] = None, tl: Optional[Float[torch.Tensor, "... 3"]] = None
) -> Float[torch.Tensor, "... 4 4"]:
    """Build transformation matrix with rotation matrix and translation vector.

    Args:
        rot_mat (torch.Tensor, optional): Rotation matrix in shape [... 3 3]. Defaults to None.
        tl (torch.Tensor, optional): Translation vector in shape [... 3]. Defaults to None.

    Returns:
        torch.Tensor: Transformation matrix in shape [... 4 4].

    Examples:
        >>> rot_mat = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
        >>> tl = torch.tensor([1, 2, 3], dtype=torch.float32)
        >>> rot_tl_to_tf_mat(rot_mat, tl)
        tensor([[0., 1., 0., 1.],
                [0., 0., 1., 2.],
                [1., 0., 0., 3.],
                [0., 0., 0., 1.]])
        >>> rot_tl_to_tf_mat(tl=tl)
        tensor([[1., 0., 0., 1.],
                [0., 1., 0., 2.],
                [0., 0., 1., 3.],
                [0., 0., 0., 1.]])
        >>> rot_tl_to_tf_mat(rot_mat=rot_mat)
        tensor([[0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 1.]])
    """
    if rot_mat is not None and tl is None:
        tl = torch.zeros(rot_mat.shape[:-2] + (3,), device=rot_mat.device, dtype=rot_mat.dtype)
    elif rot_mat is None and tl is not None:
        rot_mat = torch.eye(3).to(tl).repeat(tl.shape[:-1] + (1, 1))
    elif rot_mat is None and tl is None:
        raise ValueError("Either rot_mat or tl should be provided.")

    rot_mat = cast(torch.Tensor, rot_mat)
    tl = cast(torch.Tensor, tl)
    b_shape = torch.broadcast_shapes(rot_mat.shape[:-2], tl.shape[:-1])
    rot_mat = rot_mat.expand(b_shape + (-1, -1))
    tl = tl.expand(b_shape + (-1,))
    tf_mat = torch.cat([rot_mat, tl.unsqueeze(-1)], dim=-1)  # type: ignore
    return expand_tf_mat(tf_mat)


def expand_tf_mat(tf_mat: Float[torch.Tensor, "... 3 4"]) -> Float[torch.Tensor, "... 4 4"]:
    """Expand transformation matrix of shape [... 3 4] to shape [... 4 4].

    Args:
        tf_mat (torch.Tensor): Transformation matrix in shape [... 3 4] or [... 4 4].

    Returns:
        torch.Tensor: Expanded transformation matrix in shape [... 4 4].

    Examples:
        >>> tf_mat = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 2], [1, 0, 0, 3]], dtype=torch.float32)
        >>> expand_tf_mat(tf_mat)
        tensor([[0., 1., 0., 1.],
                [0., 0., 1., 2.],
                [1., 0., 0., 3.],
                [0., 0., 0., 1.]])
    """
    if tf_mat.shape[-2:] == (3, 4):
        # use `expand` here should be ok, I guess
        last_row = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(tf_mat).expand(tf_mat.shape[:-2] + (1, 4))
        tf_mat = torch.cat([tf_mat, last_row], dim=-2)
    return tf_mat


__all__ = [
    "transform_points",
    "rotate_points",
    "project_points",
    "unproject_points",
    "inverse_tf_mat",
    "swap_major",
    "rot_tl_to_tf_mat",
    "to_homo",
    "expand_tf_mat",
]
