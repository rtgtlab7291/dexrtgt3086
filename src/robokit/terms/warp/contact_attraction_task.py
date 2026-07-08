# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIncompatibleVariableOverride=false
"""Contact attraction task: attract object contact points to robot collision spheres."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


if TYPE_CHECKING:
    from robokit.robo.warp_robot import WarpRobot, WarpRobotState


@wp.kernel
def _compute_contact_attraction_residual_kernel(
    object_contact_points: wp.array1d(dtype=wp.vec3),
    world_sphere_centers: wp.array2d(dtype=wp.vec3),
    sphere_radii: wp.array1d(dtype=wp.float32),
    weight: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
):
    """Compute residual for contact attraction.

    For each object contact point, find minimum distance to any robot sphere surface.
    Residual = max(0, min_dist_to_surface) to penalize points not touching spheres.
    """
    instance, contact_idx = wp.tid()
    obj_pt = object_contact_points[contact_idx]

    min_dist = wp.float32(1e6)
    num_spheres = world_sphere_centers.shape[1]

    for sphere_idx in range(num_spheres):
        center = world_sphere_centers[instance, sphere_idx]
        radius = sphere_radii[sphere_idx]
        dist_to_surface = wp.length(obj_pt - center) - radius
        min_dist = wp.min(min_dist, dist_to_surface)

    residual = wp.max(wp.float32(0.0), min_dist)
    residual_buffer[instance, row_offset + contact_idx] = wp.float32(weight) * residual


@wp.kernel
def _compute_contact_attraction_jacobian_kernel(
    object_contact_points: wp.array1d(dtype=wp.vec3),
    world_sphere_centers: wp.array2d(dtype=wp.vec3),
    sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    weight: float,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
):
    """Compute Jacobian for contact attraction.

    Jacobian dr/dq = (d(min_dist)/dq) = -(p_obj - p_sphere)/|p_obj - p_sphere| * dp_sphere/dq
    Only active when min_dist > 0 (not touching).
    """
    instance, contact_idx, col_idx = wp.tid()
    base_dofs = 6 if has_floating_base else 0
    actuated_idx = col_idx - base_dofs

    obj_pt = object_contact_points[contact_idx]
    num_spheres = world_sphere_centers.shape[1]

    min_dist = wp.float32(1e6)
    min_sphere_idx = wp.int32(0)
    min_sphere_center = wp.vec3(0.0, 0.0, 0.0)

    for sphere_idx in range(num_spheres):
        center = world_sphere_centers[instance, sphere_idx]
        radius = sphere_radii[sphere_idx]
        dist_to_surface = wp.length(obj_pt - center) - radius
        if dist_to_surface < min_dist:
            min_dist = dist_to_surface
            min_sphere_idx = wp.int32(sphere_idx)
            min_sphere_center = center

    is_active = min_dist > wp.float32(0.0)
    if not is_active:
        jacobian_buffer[instance, row_offset + contact_idx, col_idx] = wp.float32(0.0)
        return

    diff = obj_pt - min_sphere_center
    dist = wp.length(diff)
    denom = wp.max(wp.float32(1e-8), dist)
    direction = diff / denom

    link_idx = collision_spheres_link_indices[min_sphere_idx]

    spatial_twist = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if has_floating_base and col_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = wp.float32(1.0)
        spatial_twist = se3_adjoint_multiply_vec6_func(T_world_base[instance], unit_vec)
    else:
        num_joints = S_world.shape[2]
        for joint_idx in range(num_joints):
            if link_ancestor_joints_mask[link_idx, joint_idx]:
                joint_weight = joints_to_actuated[joint_idx, actuated_idx]
                if joint_weight != 0.0:
                    for row in range(6):
                        spatial_twist[row] = spatial_twist[row] + S_world[instance, row, joint_idx] * joint_weight

    linear_vel = wp.vec3(spatial_twist[0], spatial_twist[1], spatial_twist[2])
    angular_vel = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])
    sphere_vel = linear_vel + wp.cross(angular_vel, min_sphere_center)

    jacobian_value = -wp.dot(direction, sphere_vel)

    jacobian_buffer[instance, row_offset + contact_idx, col_idx] = wp.float32(weight) * jacobian_value


@dataclass
class WarpContactAttractionTask(WarpTask):
    """Task to attract robot collision spheres toward fixed object contact points.

    Object contact points are specified in world frame and remain fixed during optimization.
    For each contact point, the task finds the closest robot collision sphere and minimizes
    the distance from that sphere's surface to the contact point.

    This is useful for retargeting grasps from one hand to another while preserving contact.
    """

    robot: "WarpRobot"
    object_contact_points: np.ndarray
    weight: Union[float, Sequence[float]] = 1.0
    batch_size: int = 1

    _device: Optional[wp_device_type] = None
    _object_contact_points_wp: Optional[wp.array] = None
    _num_contact_points: int = 0
    _weight_value: float = 1.0

    def __post_init__(self):
        if not self.robot.spec.has_collision_spheres:
            raise RuntimeError("Robot must have collision spheres loaded.")

        self._num_contact_points = self.object_contact_points.shape[0]
        if isinstance(self.weight, (int, float)):
            self._weight_value = float(self.weight)
        else:
            self._weight_value = float(np.mean(self.weight))

    def _init_buffers(self, device: wp_device_type):
        self._device = device
        contact_pts = self.object_contact_points.astype(np.float32)
        self._object_contact_points_wp = wp.from_numpy(contact_pts, dtype=wp.vec3, device=device)

    @property
    def residual_dim(self) -> int:
        return self._num_contact_points

    def compute_weighted_residual(
        self,
        var: "WarpRobotState",
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self._device is None:
            self._init_buffers(var.q.device)

        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)

        kernel_device = residual_buffer.device if residual_buffer is not None else self._device

        if residual_buffer is None:
            residual_buffer = wp.empty(
                (var.batch_size, self.residual_dim),
                dtype=wp.float32,
                device=kernel_device,
            )
            row_offset = 0

        spec_tensors = var.spec_tensors

        wp.launch(
            kernel=_compute_contact_attraction_residual_kernel,
            dim=(var.batch_size, self._num_contact_points),
            inputs=[
                self._object_contact_points_wp,
                var.world_collision_sphere_centers,
                spec_tensors.local_collision_sphere_radii,
                self._weight_value,
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
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)

        total_dofs = var.tangent_dim
        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else self._device

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (var.batch_size, self.residual_dim, total_dofs),
                dtype=wp.float32,
                device=kernel_device,
            )
            row_offset = 0

        spec_tensors = var.spec_tensors

        wp.launch(
            kernel=_compute_contact_attraction_jacobian_kernel,
            dim=(var.batch_size, self._num_contact_points, total_dofs),
            inputs=[
                self._object_contact_points_wp,
                var.world_collision_sphere_centers,
                spec_tensors.local_collision_sphere_radii,
                spec_tensors.collision_spheres_link_indices,
                var.S_world,
                var.T_world_base.xyz_wxyz.flatten(),
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                var.has_floating_base,
                self._weight_value,
                row_offset,
            ],
            outputs=[jacobian_buffer],
            device=kernel_device,
        )

        return jacobian_buffer


__all__ = ["WarpContactAttractionTask"]
