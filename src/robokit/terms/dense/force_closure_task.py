# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIncompatibleVariableOverride=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportReturnType=false
# pyright: reportAssignmentType=false
# ruff: noqa: PLR0917
"""Net contact-wrench residual for grasp optimization."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.geom import WarpScene
from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_apply_func


if TYPE_CHECKING:
    from robokit.robo import Robot, RobotState


@wp.kernel
def _transform_contact_points_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    local_contact_points: wp.array2d(dtype=wp.vec3),
    contact_link_indices: wp.array2d(dtype=wp.int32),
    world_contact_points: wp.array2d(dtype=wp.vec3),
):
    instance, contact_idx = wp.tid()

    link_idx = contact_link_indices[instance, contact_idx]
    T_world = T_world_link[instance, link_idx]

    translation = wp.vec3(T_world[0], T_world[1], T_world[2])
    quat_wxyz = wp.vec4(T_world[3], T_world[4], T_world[5], T_world[6])

    local_pt = local_contact_points[instance, contact_idx]
    world_pt = quaternion_apply_func(quat_wxyz, local_pt) + translation

    world_contact_points[instance, contact_idx] = world_pt


@wp.kernel
def _stabilize_contact_normals_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    query_normals: wp.array2d(dtype=wp.vec3),
    closest_points: wp.array2d(dtype=wp.vec3),
    signed_dists: wp.array2d(dtype=wp.float32),
    use_closest_point_fallback: wp.bool,
    out_normals: wp.array2d(dtype=wp.vec3),
):
    batch, c = wp.tid()
    query_normal = query_normals[batch, c]
    normal_len = wp.length(query_normal)
    if normal_len > 1.0e-10:
        out_normals[batch, c] = query_normal / normal_len
    elif use_closest_point_fallback:
        direction = contact_points[batch, c] - closest_points[batch, c]
        direction_len = wp.length(direction)
        if direction_len > 1.0e-10:
            sign = wp.float32(1.0)
            if signed_dists[batch, c] < 0.0:
                sign = wp.float32(-1.0)
            out_normals[batch, c] = direction * (sign / direction_len)
        else:
            out_normals[batch, c] = wp.vec3(0.0, 0.0, 0.0)
    else:
        out_normals[batch, c] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _make_normal_fd_points_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    eps: wp.float32,
    out_points: wp.array3d(dtype=wp.vec3),
):
    batch, contact_idx, sample_idx = wp.tid()
    axis = sample_idx // 2
    offset = wp.vec3(0.0, 0.0, 0.0)
    offset[axis] = eps if sample_idx % 2 == 0 else -eps
    out_points[batch, contact_idx, sample_idx] = contact_points[batch, contact_idx] + offset


@wp.kernel
def _scale_scene_offsets_kernel(
    scene_offsets: wp.array1d(dtype=wp.int32),
    scale: wp.int32,
    out_offsets: wp.array1d(dtype=wp.int32),
):
    idx = wp.tid()
    out_offsets[idx] = scene_offsets[idx] * scale


@wp.kernel
def _compute_normal_jacobians_kernel(
    fd_normals: wp.array3d(dtype=wp.vec3),
    center_normals: wp.array2d(dtype=wp.vec3),
    eps: wp.float32,
    out_normal_jacobians: wp.array3d(dtype=wp.vec3),
):
    batch, contact_idx, axis = wp.tid()
    normal_plus = fd_normals[batch, contact_idx, axis * 2]
    normal_minus = fd_normals[batch, contact_idx, axis * 2 + 1]
    plus_len = wp.length(normal_plus)
    minus_len = wp.length(normal_minus)
    center_normal = center_normals[batch, contact_idx]
    if plus_len > 1.0e-10:
        normal_plus = normal_plus / plus_len
    else:
        normal_plus = center_normal
    if minus_len > 1.0e-10:
        normal_minus = normal_minus / minus_len
    else:
        normal_minus = center_normal
    out_normal_jacobians[batch, contact_idx, axis] = (normal_plus - normal_minus) / (2.0 * eps)


@wp.kernel
def compute_force_closure_residual_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    contact_normals: wp.array2d(dtype=wp.vec3),
    contact_force_weights: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    instance = wp.tid()
    num_contacts = contact_points.shape[1]

    wrench_force = wp.vec3(0.0, 0.0, 0.0)
    wrench_torque = wp.vec3(0.0, 0.0, 0.0)

    for contact_idx in range(num_contacts):
        position = contact_points[instance, contact_idx]
        normal = contact_normals[instance, contact_idx] * contact_force_weights[contact_idx]

        wrench_force = wrench_force + normal
        wrench_torque = wrench_torque + wp.cross(position, normal)

    out_residual[instance, row_offset + 0] = residual_weight[0] * wrench_force[0]
    out_residual[instance, row_offset + 1] = residual_weight[1] * wrench_force[1]
    out_residual[instance, row_offset + 2] = residual_weight[2] * wrench_force[2]
    out_residual[instance, row_offset + 3] = residual_weight[3] * wrench_torque[0]
    out_residual[instance, row_offset + 4] = residual_weight[4] * wrench_torque[1]
    out_residual[instance, row_offset + 5] = residual_weight[5] * wrench_torque[2]


@wp.kernel
def _compute_force_closure_cost_kernel(
    residual: wp.array2d(dtype=wp.float32),
    out_cost: wp.array1d(dtype=wp.float32),
):
    instance = wp.tid()
    cost = wp.float32(0.0)
    for residual_idx in range(6):
        value = residual[instance, residual_idx]
        cost = cost + wp.float32(0.5) * value * value
    wp.atomic_add(out_cost, instance, cost)


@wp.kernel
def _compute_force_closure_cost_and_gradient_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    contact_normals: wp.array2d(dtype=wp.vec3),
    contact_normal_jacobians: wp.array3d(dtype=wp.vec3),
    contact_force_weights: wp.array1d(dtype=wp.float32),
    contact_points_link_indices: wp.array2d(dtype=wp.int32),
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    out_residual: wp.array2d(dtype=wp.float32),
    col_offset: int,
    out_cost: wp.array1d(dtype=wp.float32),
    out_gradient: wp.array2d(dtype=wp.float32),
):
    instance, col_idx = wp.tid()
    num_contacts = contact_points.shape[1]
    base_dofs = wp.int32(6) if has_floating_base else wp.int32(0)
    actuated_idx = col_idx - base_dofs

    grad_value = wp.float32(0.0)
    for contact_idx in range(num_contacts):
        position = contact_points[instance, contact_idx]
        force_weight = contact_force_weights[contact_idx]
        normal = contact_normals[instance, contact_idx] * force_weight

        spatial_twist = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if has_floating_base and col_idx < 6:
            unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unit_vec[col_idx] = wp.float32(1.0)
            spatial_twist = se3_adjoint_func(T_world_base[instance]) * unit_vec
        else:
            link_idx = contact_points_link_indices[instance, contact_idx]
            num_joints = S_world.shape[2]
            for joint_idx in range(num_joints):
                if link_ancestor_joints_mask[link_idx, joint_idx]:
                    weight = joints_to_actuated[joint_idx, actuated_idx]
                    if weight != 0.0:
                        for row in range(6):
                            spatial_twist[row] = spatial_twist[row] + S_world[instance, row, joint_idx] * weight

        v = wp.vec3(spatial_twist[0], spatial_twist[1], spatial_twist[2])
        w = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])
        point_vel = v + wp.cross(w, position)
        normal_vel = (
            contact_normal_jacobians[instance, contact_idx, 0] * point_vel[0]
            + contact_normal_jacobians[instance, contact_idx, 1] * point_vel[1]
            + contact_normal_jacobians[instance, contact_idx, 2] * point_vel[2]
        ) * force_weight
        torque_vel = wp.cross(point_vel, normal) + wp.cross(position, normal_vel)

        for residual_idx in range(3):
            grad_value = (
                grad_value
                + residual_weight[residual_idx] * normal_vel[residual_idx] * out_residual[instance, residual_idx]
            )
            grad_value = (
                grad_value
                + residual_weight[residual_idx + 3]
                * torque_vel[residual_idx]
                * out_residual[instance, residual_idx + 3]
            )

    if grad_value != wp.float32(0.0):
        wp.atomic_add(out_gradient, instance, col_offset + col_idx, grad_value)

    if col_idx == 0:
        r0 = out_residual[instance, 0]
        r1 = out_residual[instance, 1]
        r2 = out_residual[instance, 2]
        r3 = out_residual[instance, 3]
        r4 = out_residual[instance, 4]
        r5 = out_residual[instance, 5]
        cost = wp.float32(0.5) * (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4 + r5 * r5)
        wp.atomic_add(out_cost, instance, cost)


@wp.kernel
def _compute_force_closure_task_weighted_jacobian_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    contact_normals: wp.array2d(dtype=wp.vec3),
    contact_normal_jacobians: wp.array3d(dtype=wp.vec3),
    contact_force_weights: wp.array1d(dtype=wp.float32),
    contact_points_link_indices: wp.array2d(dtype=wp.int32),
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    row_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    instance, residual_idx, col_idx = wp.tid()
    num_contacts = contact_points.shape[1]
    base_dofs = 6 if has_floating_base else 0
    actuated_idx = col_idx - base_dofs

    jacobian_value = wp.float32(0.0)
    for contact_idx in range(num_contacts):
        position = contact_points[instance, contact_idx]
        force_weight = contact_force_weights[contact_idx]
        normal = contact_normals[instance, contact_idx] * force_weight

        spatial_twist = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if has_floating_base and col_idx < 6:
            unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unit_vec[col_idx] = wp.float32(1.0)
            spatial_twist = se3_adjoint_func(T_world_base[instance]) * unit_vec
        else:
            link_idx = contact_points_link_indices[instance, contact_idx]
            num_joints = S_world.shape[2]
            for joint_idx in range(num_joints):
                if link_ancestor_joints_mask[link_idx, joint_idx]:
                    weight = joints_to_actuated[joint_idx, actuated_idx]
                    if weight != 0.0:
                        for row in range(6):
                            spatial_twist[row] = spatial_twist[row] + S_world[instance, row, joint_idx] * weight

        v = wp.vec3(spatial_twist[0], spatial_twist[1], spatial_twist[2])
        w = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])
        point_vel = v + wp.cross(w, position)
        normal_vel = (
            contact_normal_jacobians[instance, contact_idx, 0] * point_vel[0]
            + contact_normal_jacobians[instance, contact_idx, 1] * point_vel[1]
            + contact_normal_jacobians[instance, contact_idx, 2] * point_vel[2]
        ) * force_weight

        if residual_idx < 3:
            jacobian_value = jacobian_value + normal_vel[residual_idx]
        else:
            torque_vel = wp.cross(point_vel, normal) + wp.cross(position, normal_vel)
            jacobian_value = jacobian_value + torque_vel[residual_idx - 3]

    out_jacobian[instance, row_offset + residual_idx, col_idx] = residual_weight[residual_idx] * jacobian_value


@dataclass
class ForceClosureTask(RobotTask, ResidualTask, GradientTask):
    """Force closure task for grasp optimization.

    Ensures a grasp achieves force closure - the ability to exert forces/torques
    in all directions.

    Contact points are gathered from var.contact_point_centers_world using
    contact_point_indices. Contact normals are computed by querying the SDF of
    warp_meshes.
    """

    robot: Optional["Robot"] = None
    num_contact_points: int = 0
    warp_meshes: Optional[WarpScene] = None
    weight: Optional[Union[float, Sequence[float]]] = None
    contact_force_weights: Optional[Sequence[float]] = None
    normal_fd_eps: float = 1.0e-3
    local_contact_points: Optional[wp.array] = None
    contact_points_link_indices: Optional[wp.array] = None

    _device: Optional[wp_device_type] = None
    _cached_batch_size: int = 0
    _scene_offsets_wp: Optional[wp.array] = None
    _residual_weight_np: Optional[np.ndarray] = None
    residual_weight: Optional[wp.array] = None
    contact_force_weights_wp: Optional[wp.array] = None
    _sdf_signed_dists: Optional[wp.array] = None
    _sdf_query_normals: Optional[wp.array] = None
    _sdf_normals: Optional[wp.array] = None
    _sdf_closest_points: Optional[wp.array] = None
    _normal_fd_points: Optional[wp.array] = None
    _normal_fd_signed_dists: Optional[wp.array] = None
    _normal_fd_normals: Optional[wp.array] = None
    _normal_fd_closest_points: Optional[wp.array] = None
    _normal_jacobians: Optional[wp.array] = None
    _normal_scene_offsets_wp: Optional[wp.array] = None
    _contact_points_world: Optional[wp.array] = None
    _direct_residual_buf: Optional[wp.array] = None

    def __post_init__(self):
        if self.normal_fd_eps <= 0.0:
            raise ValueError(f"normal_fd_eps must be positive, got {self.normal_fd_eps}.")
        residual_weight = np.ones(6, dtype=np.float32)
        if self.weight is not None:
            if isinstance(self.weight, (float, int)):
                residual_weight[:] = float(self.weight)
            else:
                weight_arr = np.asarray(self.weight, dtype=np.float32)
                residual_weight[:] = weight_arr
        self._residual_weight_np = residual_weight
        contact_force_weights = np.ones(self.num_contact_points, dtype=np.float32)
        if self.contact_force_weights is not None:
            contact_force_weights = np.asarray(self.contact_force_weights, dtype=np.float32)
            if contact_force_weights.shape != (self.num_contact_points,):
                raise ValueError(
                    f"Expected contact_force_weights shape ({self.num_contact_points},), "
                    f"got {contact_force_weights.shape}."
                )
        self._contact_force_weights_np = contact_force_weights

    def set_robot(self, robot: "Robot"):
        self.robot = robot

    def _init_buffers(self, batch_size: int, device: wp_device_type):
        self._device = device
        self._cached_batch_size = batch_size
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self.contact_force_weights_wp = wp.from_numpy(
            self._contact_force_weights_np,
            dtype=wp.float32,
            device=device,
        )
        self._scene_offsets_wp = wp.from_numpy(
            np.arange(self.warp_meshes.num_scenes + 1, dtype=np.int32)
            * (batch_size // self.warp_meshes.num_scenes)
            * self.num_contact_points,
            dtype=wp.int32,
            device=device,
        )
        self._sdf_signed_dists = wp.empty((batch_size, self.num_contact_points), dtype=wp.float32, device=device)
        self._sdf_query_normals = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
        self._sdf_normals = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
        self._sdf_closest_points = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
        self._normal_fd_points = wp.empty((batch_size, self.num_contact_points, 6), dtype=wp.vec3, device=device)
        self._normal_fd_signed_dists = wp.empty(
            (batch_size, self.num_contact_points, 6), dtype=wp.float32, device=device
        )
        self._normal_fd_normals = wp.empty((batch_size, self.num_contact_points, 6), dtype=wp.vec3, device=device)
        self._normal_fd_closest_points = wp.empty(
            (batch_size, self.num_contact_points, 6), dtype=wp.vec3, device=device
        )
        self._normal_jacobians = wp.empty((batch_size, self.num_contact_points, 3), dtype=wp.vec3, device=device)
        self._normal_scene_offsets_wp = wp.empty(self.warp_meshes.num_scenes + 1, dtype=wp.int32, device=device)
        self._contact_points_world = wp.empty(
            (batch_size, self.num_contact_points), dtype=wp.vec3, device=device, requires_grad=True
        )
        self._direct_residual_buf = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return 6

    def _transform_contact_points(
        self,
        var: "RobotState",
        local_contact_points: wp.array,
        contact_points_link_indices: wp.array,
    ) -> wp.array:
        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        device = var.q.device

        wp.launch(
            kernel=_transform_contact_points_kernel,
            dim=(batch_size, self.num_contact_points),
            inputs=[
                var.T_world_link.reshape((batch_size, num_links)),
                local_contact_points,
                contact_points_link_indices,
            ],
            outputs=[self._contact_points_world],
            device=device,
        )

        return self._contact_points_world

    def _compute_contact_normals(
        self,
        contact_points_world: wp.array,
        scene_offsets: wp.array,
        precomputed_normals: Optional[wp.array] = None,
    ) -> wp.array:
        if precomputed_normals is None:
            self.warp_meshes.query_sdf(
                contact_points_world,
                scene_offsets,
                out_signed_dists=self._sdf_signed_dists,
                out_normals=self._sdf_query_normals,
                out_closest_points=self._sdf_closest_points,
            )
            query_normals = self._sdf_query_normals
        else:
            query_normals = precomputed_normals

        wp.launch(
            kernel=_stabilize_contact_normals_kernel,
            dim=(contact_points_world.shape[0], self.num_contact_points),
            inputs=[
                contact_points_world,
                query_normals,
                self._sdf_closest_points,
                self._sdf_signed_dists,
                precomputed_normals is None,
            ],
            outputs=[self._sdf_normals],
            device=contact_points_world.device,
        )
        return self._sdf_normals

    def _compute_normal_jacobians(
        self,
        contact_points_world: wp.array,
        center_normals: wp.array,
        scene_offsets: wp.array,
    ) -> wp.array:
        wp.launch(
            kernel=_make_normal_fd_points_kernel,
            dim=(contact_points_world.shape[0], self.num_contact_points, 6),
            inputs=[contact_points_world, self.normal_fd_eps],
            outputs=[self._normal_fd_points],
            device=contact_points_world.device,
        )
        wp.launch(
            kernel=_scale_scene_offsets_kernel,
            dim=self.warp_meshes.num_scenes + 1,
            inputs=[scene_offsets, 6],
            outputs=[self._normal_scene_offsets_wp],
            device=contact_points_world.device,
        )
        self.warp_meshes.query_sdf(
            self._normal_fd_points,
            self._normal_scene_offsets_wp,
            out_signed_dists=self._normal_fd_signed_dists,
            out_normals=self._normal_fd_normals,
            out_closest_points=self._normal_fd_closest_points,
        )
        wp.launch(
            kernel=_compute_normal_jacobians_kernel,
            dim=(contact_points_world.shape[0], self.num_contact_points, 3),
            inputs=[self._normal_fd_normals, center_normals, self.normal_fd_eps],
            outputs=[self._normal_jacobians],
            device=contact_points_world.device,
        )
        return self._normal_jacobians

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        local_contact_points: Optional[wp.array] = None,
        contact_points_link_indices: Optional[wp.array] = None,
        scene_offsets: Optional[wp.array] = None,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        precomputed_contact_points_world: Optional[wp.array] = None,
        precomputed_normals: Optional[wp.array] = None,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if self._device is None or self._cached_batch_size != var.batch_size:
            self._init_buffers(var.batch_size, var.q.device)

        if (precomputed_contact_points_world is None) != (precomputed_normals is None):
            raise ValueError("precomputed_contact_points_world and precomputed_normals must be provided together.")

        if precomputed_contact_points_world is not None and precomputed_normals is not None:
            contact_points_world = precomputed_contact_points_world
        else:
            if not var.is_fk_computed:
                var = self.robot.forward_kinematics(var)

            if local_contact_points is not None and contact_points_link_indices is not None:
                contact_points_world = self._transform_contact_points(
                    var, local_contact_points, contact_points_link_indices
                )
            elif self.local_contact_points is not None and self.contact_points_link_indices is not None:
                contact_points_world = self._transform_contact_points(
                    var, self.local_contact_points, self.contact_points_link_indices
                )
            else:
                raise ValueError(
                    "Contact data not provided. Pass local_contact_points and contact_points_link_indices "
                    "as arguments or set them on the task."
                )

        assert self._scene_offsets_wp is not None
        sdf_scene_offsets = scene_offsets if scene_offsets is not None else self._scene_offsets_wp
        normals = self._compute_contact_normals(
            contact_points_world,
            sdf_scene_offsets,
            precomputed_normals=precomputed_normals,
        )

        kernel_device = out_residual.device if out_residual is not None else self._device
        if out_residual is None:
            out_residual = wp.empty((var.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_force_closure_residual_kernel,
            dim=var.batch_size,
            inputs=[
                contact_points_world,
                normals,
                self.contact_force_weights_wp,
                self.residual_weight,
                row_offset,
            ],
            outputs=[out_residual],
            device=kernel_device,
        )
        return out_residual

    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
        local_contact_points: Optional[wp.array] = None,
        contact_points_link_indices: Optional[wp.array] = None,
        scene_offsets: Optional[wp.array] = None,
        precomputed_contact_points_world: Optional[wp.array] = None,
        precomputed_normals: Optional[wp.array] = None,
    ):
        """When `precomputed_contact_points_world` and `precomputed_normals` are provided, skip
        the SDF query and reuse the caller's buffers (shared with the distance task in
        `GDGraspOptHelper`)."""
        col_offset = var_values.tangent_offset(self.var_key)
        var = var_values.get(self.var_key)
        if self._device is None or self._cached_batch_size != var.batch_size:
            self._init_buffers(var.batch_size, var.q.device)
        if not var.is_motion_subspace_computed:
            var = self.robot.compute_motion_subspace(var)

        self.compute_weighted_residual(
            var_values,
            local_contact_points=local_contact_points,
            contact_points_link_indices=contact_points_link_indices,
            scene_offsets=scene_offsets,
            out_residual=self._direct_residual_buf,
            precomputed_contact_points_world=precomputed_contact_points_world,
            precomputed_normals=precomputed_normals,
        )

        if out_gradient is None:
            wp.launch(
                kernel=_compute_force_closure_cost_kernel,
                dim=var.batch_size,
                inputs=[self._direct_residual_buf],
                outputs=[out_cost],
                device=self._device,
            )
            return

        if precomputed_contact_points_world is not None:
            contact_points_world = precomputed_contact_points_world
        else:
            contact_points_world = self._contact_points_world
        normals = self._sdf_normals
        assert self._scene_offsets_wp is not None
        sdf_scene_offsets = scene_offsets if scene_offsets is not None else self._scene_offsets_wp
        normal_jacobians = self._compute_normal_jacobians(
            contact_points_world,
            normals,
            sdf_scene_offsets,
        )

        if contact_points_link_indices is not None:
            link_indices = contact_points_link_indices
        else:
            link_indices = self.contact_points_link_indices

        spec_tensors = var.spec_tensors
        wp.launch(
            kernel=_compute_force_closure_cost_and_gradient_kernel,
            dim=(var.batch_size, var.tangent_dim),
            inputs=[
                contact_points_world,
                normals,
                normal_jacobians,
                self.contact_force_weights_wp,
                link_indices,
                var.S_world,
                var.T_world_base.flatten(),
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                var.has_floating_base,
                self.residual_weight,
                self._direct_residual_buf,
                col_offset,
            ],
            outputs=[out_cost, out_gradient],
            device=self._device,
        )

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        local_contact_points: Optional[wp.array] = None,
        contact_points_link_indices: Optional[wp.array] = None,
        scene_offsets: Optional[wp.array] = None,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        assert var_values.tangent_offset(self.var_key) == 0
        if self._device is None or self._cached_batch_size != var.batch_size:
            self._init_buffers(var.batch_size, var.q.device)
        if not var.is_fk_computed:
            var = self.robot.forward_kinematics(var)
        if not var.is_motion_subspace_computed:
            var = self.robot.compute_motion_subspace(var)

        if local_contact_points is not None and contact_points_link_indices is not None:
            contact_points_world = self._transform_contact_points(
                var, local_contact_points, contact_points_link_indices
            )
        elif self.local_contact_points is not None and self.contact_points_link_indices is not None:
            contact_points_world = self._transform_contact_points(
                var, self.local_contact_points, self.contact_points_link_indices
            )
            contact_points_link_indices = self.contact_points_link_indices
        else:
            raise ValueError(
                "Contact data not provided. Pass local_contact_points and contact_points_link_indices "
                "as arguments or set them on the task."
            )

        assert self._scene_offsets_wp is not None
        sdf_scene_offsets = scene_offsets if scene_offsets is not None else self._scene_offsets_wp
        normals = self._compute_contact_normals(contact_points_world, sdf_scene_offsets)
        normal_jacobians = self._compute_normal_jacobians(
            contact_points_world,
            normals,
            sdf_scene_offsets,
        )

        link_indices = contact_points_link_indices
        total_dofs = var.tangent_dim
        kernel_device = out_jacobian.device if out_jacobian is not None else self._device
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (var.batch_size, self.residual_dim, total_dofs),
                dtype=wp.float32,
                device=kernel_device,
            )
            row_offset = 0

        spec_tensors = var.spec_tensors
        wp.launch(
            kernel=_compute_force_closure_task_weighted_jacobian_kernel,
            dim=(var.batch_size, self.residual_dim, total_dofs),
            inputs=[
                self._contact_points_world,
                normals,
                normal_jacobians,
                self.contact_force_weights_wp,
                link_indices,
                var.S_world,
                var.T_world_base.flatten(),
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                var.has_floating_base,
                self.residual_weight,
                row_offset,
            ],
            outputs=[out_jacobian],
            device=kernel_device,
        )
        return out_jacobian


__all__ = ["ForceClosureTask"]
