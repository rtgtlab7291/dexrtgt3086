# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
import warp as wp

from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_apply_func


if TYPE_CHECKING:
    from robokit.robo.warp_robot import WarpRobot, WarpRobotState


@wp.kernel
def _transform_keypoints_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    local_points: wp.array1d(dtype=wp.vec3),
    local_link_indices: wp.array1d(dtype=wp.int32),
    world_points: wp.array2d(dtype=wp.vec3),
) -> None:
    instance, point_idx = wp.tid()
    link_idx = local_link_indices[point_idx]
    T_world = T_world_link[instance, link_idx]
    translation = wp.vec3(T_world[0], T_world[1], T_world[2])
    quat_wxyz = wp.vec4(T_world[3], T_world[4], T_world[5], T_world[6])
    local_pt = local_points[point_idx]
    world_points[instance, point_idx] = quaternion_apply_func(quat_wxyz, local_pt) + translation


@wp.kernel
def compute_self_penetration_residual_kernel(
    world_points: wp.array2d(dtype=wp.vec3),
    pair_indices: wp.array2d(dtype=wp.int32),
    margin: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
) -> None:
    instance, pair_idx = wp.tid()
    point_idx_i = pair_indices[pair_idx, 0]
    point_idx_j = pair_indices[pair_idx, 1]
    point_i = world_points[instance, point_idx_i]
    point_j = world_points[instance, point_idx_j]
    dist = wp.length(point_i - point_j)
    penetration = wp.max(wp.float32(0.0), wp.float32(margin) - dist)
    residual = penetration
    if use_sqrt:
        residual = wp.sqrt(penetration + eps)
    residual_buffer[instance, row_offset + pair_idx] = weight * residual


@wp.kernel
def compute_self_penetration_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    world_points: wp.array2d(dtype=wp.vec3),
    point_link_indices: wp.array1d(dtype=wp.int32),
    pair_indices: wp.array2d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    margin: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
) -> None:
    instance, pair_idx, col_idx = wp.tid()
    base_dofs = 6 if has_floating_base else 0
    actuated_idx = col_idx - base_dofs

    point_idx_i = pair_indices[pair_idx, 0]
    point_idx_j = pair_indices[pair_idx, 1]
    point_i = world_points[instance, point_idx_i]
    point_j = world_points[instance, point_idx_j]
    diff = point_i - point_j
    dist = wp.length(diff)
    penetration = wp.float32(margin) - dist

    if penetration <= wp.float32(0.0):
        jacobian_buffer[instance, row_offset + pair_idx, col_idx] = wp.float32(0.0)
        return

    scale = wp.float32(-1.0)
    if use_sqrt:
        scale = wp.float32(-0.5) / wp.sqrt(penetration + eps)

    denom = wp.max(wp.float32(1e-8), dist)
    normal = diff / denom

    spatial_twist_i = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    spatial_twist_j = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if has_floating_base and col_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = wp.float32(1.0)
        spatial_twist = se3_adjoint_multiply_vec6_func(T_world_base[instance], unit_vec)
        spatial_twist_i = spatial_twist
        spatial_twist_j = spatial_twist
    else:
        link_i = point_link_indices[point_idx_i]
        link_j = point_link_indices[point_idx_j]
        num_joints = S_world.shape[2]
        for joint_idx in range(num_joints):
            weight_map = joints_to_actuated[joint_idx, actuated_idx]
            if weight_map != 0.0:
                if link_ancestor_joints_mask[link_i, joint_idx]:
                    for row in range(6):
                        spatial_twist_i[row] = spatial_twist_i[row] + S_world[instance, row, joint_idx] * weight_map
                if link_ancestor_joints_mask[link_j, joint_idx]:
                    for row in range(6):
                        spatial_twist_j[row] = spatial_twist_j[row] + S_world[instance, row, joint_idx] * weight_map

    linear_i = wp.vec3(spatial_twist_i[0], spatial_twist_i[1], spatial_twist_i[2])
    angular_i = wp.vec3(spatial_twist_i[3], spatial_twist_i[4], spatial_twist_i[5])
    linear_j = wp.vec3(spatial_twist_j[0], spatial_twist_j[1], spatial_twist_j[2])
    angular_j = wp.vec3(spatial_twist_j[3], spatial_twist_j[4], spatial_twist_j[5])

    point_vel_i = linear_i + wp.cross(angular_i, point_i)
    point_vel_j = linear_j + wp.cross(angular_j, point_j)
    rel_vel = point_vel_i - point_vel_j
    ddist_dq = wp.dot(normal, rel_vel)

    jacobian_buffer[instance, row_offset + pair_idx, col_idx] = weight * scale * ddist_dq


@dataclass
class WarpSelfPenetrationTask(WarpTask):
    robot: "WarpRobot"
    local_points: wp.array
    local_link_indices: wp.array
    weight: float = 1.0
    margin: float = 0.02
    batch_size: int = 1
    residual_mode: Literal["abs", "sqrt_abs"] = "abs"
    residual_eps: float = 1e-6
    pair_indices: Optional[Sequence[Sequence[int]]] = None

    _device: Optional[wp_device_type] = None
    _pair_indices_np: Optional[np.ndarray] = None
    _pair_indices_wp: Optional[wp.array] = None
    _num_pairs: int = 0

    def __post_init__(self) -> None:
        if self.residual_mode not in {"abs", "sqrt_abs"}:
            raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")

        num_points = int(self.local_points.shape[0])
        if num_points < 2:
            raise ValueError("Expected at least two penetration keypoints.")

        if self.pair_indices is None:
            num_pairs = num_points * (num_points - 1) // 2
            pairs = np.empty((num_pairs, 2), dtype=np.int32)
            idx = 0
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    pairs[idx, 0] = i
                    pairs[idx, 1] = j
                    idx += 1
        else:
            pairs = np.asarray(self.pair_indices, dtype=np.int32)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("pair_indices must have shape [num_pairs, 2].")

        self._pair_indices_np = pairs
        self._num_pairs = int(pairs.shape[0])

    def _init_buffers(self, device: wp_device_type) -> None:
        self._device = device
        if self._pair_indices_np is None:
            raise ValueError("pair indices were not initialized.")
        self._pair_indices_wp = wp.from_numpy(self._pair_indices_np, dtype=wp.int32, device=device)

    @property
    def residual_dim(self) -> int:
        return self._num_pairs

    def compute_weighted_residual(
        self,
        var: "WarpRobotState",
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self._device is None:
            self._init_buffers(var.q.device)
        if not var.is_fk_computed:
            var = self.robot.forward_kinematics(var)

        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        kernel_device = residual_buffer.device if residual_buffer is not None else self._device

        world_points = wp.empty((batch_size, self.local_points.shape[0]), dtype=wp.vec3, device=kernel_device)
        wp.launch(
            kernel=_transform_keypoints_kernel,
            dim=(batch_size, self.local_points.shape[0]),
            inputs=[
                var.T_world_link.xyz_wxyz.reshape((batch_size, num_links)),
                self.local_points,
                self.local_link_indices,
            ],
            outputs=[world_points],
            device=kernel_device,
        )

        if residual_buffer is None:
            residual_buffer = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self._num_pairs == 0:
            return residual_buffer

        use_sqrt = self.residual_mode == "sqrt_abs"
        wp.launch(
            kernel=compute_self_penetration_residual_kernel,
            dim=(batch_size, self._num_pairs),
            inputs=[
                world_points,
                self._pair_indices_wp,
                self.margin,
                self.weight,
                use_sqrt,
                self.residual_eps,
                row_offset,
            ],
            outputs=[residual_buffer],
            device=kernel_device,
        )

        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: "WarpRobotState",
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self._device is None:
            self._init_buffers(var.q.device)
        if not var.is_fk_computed:
            var = self.robot.forward_kinematics(var)
        if not var.is_motion_subspace_computed:
            var = self.robot.compute_motion_subspace(var)

        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        total_dofs = var.tangent_dim
        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else self._device

        world_points = wp.empty((batch_size, self.local_points.shape[0]), dtype=wp.vec3, device=kernel_device)
        wp.launch(
            kernel=_transform_keypoints_kernel,
            dim=(batch_size, self.local_points.shape[0]),
            inputs=[
                var.T_world_link.xyz_wxyz.reshape((batch_size, num_links)),
                self.local_points,
                self.local_link_indices,
            ],
            outputs=[world_points],
            device=kernel_device,
        )

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (batch_size, self.residual_dim, total_dofs),
                dtype=wp.float32,
                device=kernel_device,
            )
            row_offset = 0

        if self._num_pairs == 0:
            return jacobian_buffer

        spec_tensors = var.spec_tensors
        use_sqrt = self.residual_mode == "sqrt_abs"
        wp.launch(
            kernel=compute_self_penetration_jacobian_kernel,
            dim=(batch_size, self._num_pairs, total_dofs),
            inputs=[
                var.S_world,
                var.T_world_base.xyz_wxyz.flatten(),
                world_points,
                self.local_link_indices,
                self._pair_indices_wp,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                var.has_floating_base,
                self.margin,
                self.weight,
                use_sqrt,
                self.residual_eps,
                row_offset,
            ],
            outputs=[jacobian_buffer],
            device=kernel_device,
        )

        return jacobian_buffer


__all__ = ["WarpSelfPenetrationTask"]
