import math
import re
from typing import List, Literal, Union, overload

import numpy as np
import torch
from jaxtyping import Float
from torch.nn import functional as F


COMMON_COORDS = {
    "opencv": "x: right, y: down, z: front",
    "opengl": "x: right, y: up, z: back",
    "sapien": "x: front, y: left, z: up",  # https://sapien.ucsd.edu/docs/latest/tutorial/basic/hello_world.html#viewer
}

# NOTE it doesn't matter if `right` correspond to [1, 0, 0], the resulted matrix is the same
DIRECTIONS = {
    "right": np.array([1, 0, 0], dtype=np.float32),
    "left": np.array([-1, 0, 0], dtype=np.float32),
    "up": np.array([0, -1, 0], dtype=np.float32),
    "down": np.array([0, 1, 0], dtype=np.float32),
    "front": np.array([0, 0, 1], dtype=np.float32),
    "back": np.array([0, 0, -1], dtype=np.float32),
}


@overload
def coord_conversion(
    src_spec: str, dst_spec: str, check_handness: bool = ..., *, return_tensors: Literal["pt"]
) -> Float[torch.Tensor, "3 3"]: ...
@overload
def coord_conversion(
    src_spec: str, dst_spec: str, check_handness: bool = ..., *, return_tensors: Literal["np"]
) -> Float[np.ndarray, "3 3"]: ...
def coord_conversion(
    src_spec: str, dst_spec: str, check_handness: bool = True, return_tensors: Literal["np", "pt"] = "np"
) -> Union[Float[np.ndarray, "3 3"], Float[torch.Tensor, "3 3"]]:
    """
    Construct a rotation matrix based on given source and destination coordinate specifications.

    Args:
        src_spec: Source coordinate specification, e.g., "x: right, y: down, z: front" or "opencv".
        dst_spec: Destination coordinate specification, e.g., "x: right, y: up, z: back" or "opengl".
        check_handness: If True, checks if the rotation matrix preserves right-handedness.
        return_tensors: Return type of the rotation matrix, either "np" for NumPy array or "pt" for PyTorch tensor.

    Returns:
        A 3x3 rotation matrix converting coordinates from the source to the destination specification.

    Examples:
        >>> coord_conversion("opencv", "opengl")
        array([[ 1.,  0.,  0.],
               [ 0., -1.,  0.],
               [ 0.,  0., -1.]], dtype=float32)
        >>> coord_conversion("x: front, y: left, z: up", "x: left, y: up, z: front")
        array([[0., 1., 0.],
               [0., 0., 1.],
               [1., 0., 0.]], dtype=float32)
        >>> coord_conversion("x: right, y: down, z: front", "x: left, y: up, z: front")
        array([[-1.,  0.,  0.],
               [ 0., -1.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)
        >>> coord_conversion("x: left, y: up, z: front", "x: front, y: left, z: up", return_tensors="pt")
        tensor([[0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.]])
    """

    def parse_spec(spec: str) -> List[str]:
        spec = spec.strip().lower()
        if spec in COMMON_COORDS:
            coord = COMMON_COORDS[spec]
        else:
            coord = spec
        # Use regex to parse the coordinate specification
        pattern = r"\s*(x|y|z)\s*:\s*(\w+)\s*"
        matches = re.findall(pattern, coord)
        if len(matches) != 3:
            raise ValueError(f"Invalid coordinate specification: '{spec}'.")
        dirs = {axis: direction for axis, direction in matches}
        if set(dirs.keys()) != {"x", "y", "z"}:
            raise ValueError(f"Invalid coordinate specification: '{spec}'.")
        return [dirs["x"], dirs["y"], dirs["z"]]

    src_dirs = parse_spec(src_spec)
    dst_dirs = parse_spec(dst_spec)

    src_basis = np.stack([DIRECTIONS[dir] for dir in src_dirs])
    dst_basis = np.stack([DIRECTIONS[dir] for dir in dst_dirs])

    rot_mat = dst_basis @ src_basis.T

    if check_handness and np.linalg.det(rot_mat) < 0:
        raise RuntimeWarning("The rotation matrix is not right-handed.")

    if return_tensors == "pt":
        return torch.from_numpy(rot_mat).float()
    elif return_tensors == "np":
        return rot_mat.astype(np.float32)
    else:
        raise ValueError(f"Invalid return_tensors: '{return_tensors}'")


def compose_intr_mat(fu: float, fv: float, cu: float, cv: float, skew: float = 0.0) -> np.ndarray:
    """
    Args:
        fu: horizontal focal length (width)
        fv: vertical focal length (height)
        cu: horizontal principal point (width)
        cv: vertical principal point (height)
        skew: skew coefficient, default to 0
    """
    intr_mat = np.array([[fu, skew, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float32)
    return intr_mat


# Ref: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py
def look_at_rotation(
    camera_position: Float[torch.Tensor, "*batch 3"],
    at: Float[torch.Tensor, "*batch 3"],
    up: Float[torch.Tensor, "*batch 3"],
) -> Float[torch.Tensor, "*batch 3 3"]:
    """
    This function takes a vector `camera_position` which specifies the location of the camera in world coordinates and
    two vectors `at` and `up` which indicate the position of the object and the up directions of the world
    coordinate system respectively.

    The output is a rotation matrix representing the rotation from camera coordinates to world coordinates.

    We use the OpenGL coordinate in this function, i.e. x -> right, y -> up, z -> backward.
    Hence, z_axis: pos - at, x_axis: cross(up, z_axis), y_axis: cross(z_axis, x_axis)

    Note that our implementation differs from pytorch3d.
        1. our matrix is in the OpenGL coordinate
        2. our matrix is column-major
        3. our matrix is the camera-to-world transformation

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    Returns:
        R: rotation matrices of shape [..., 3, 3]
    """
    dtype, device = camera_position.dtype, camera_position.device
    at, up = torch.broadcast_to(at, camera_position.shape), torch.broadcast_to(up, camera_position.shape)
    z_axis = F.normalize(camera_position - at, eps=1e-5, dim=-1)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=-1), eps=1e-5, dim=-1)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), eps=1e-5, dim=-1)
    is_close = torch.isclose(x_axis, torch.tensor(0.0, dtype=dtype, device=device), atol=5e-3)
    is_close = is_close.all(dim=-1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=-1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    rot_mat = torch.cat((x_axis[..., None, :], y_axis[..., None, :], z_axis[..., None, :]), dim=-2)
    return rot_mat.transpose(-2, -1)


Device = Union[str, torch.device]


def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def format_tensor(
    input,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.

    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: Device (as str or torch.device) on which the tensor should be placed.

    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    device_ = make_device(device)
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device_)

    if input.dim() == 0:
        input = input.view(1)

    if input.device == device_:
        return input

    input = input.to(device=device)
    return input


def convert_to_tensors_and_broadcast(
    *args,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.

    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.

    Output:
        args: A list of tensors of shape (N, K_i)
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    return args_Nd


def camera_position_from_spherical_angles(
    distance: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.

    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.

    The vectors are broadcast against each other so they all have shape (N, 1).

    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    dist, elev, azim = convert_to_tensors_and_broadcast(distance, elevation, azimuth, device=device)
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)


__all__ = ["coord_conversion", "compose_intr_mat", "look_at_rotation", "camera_position_from_spherical_angles"]
