# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIncompatibleVariableOverride=false
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


@wp.kernel
def _query_closest_points_for_spheres_single_scene_kernel(
    world_collision_sphere_centers: wp.array2d(dtype=wp.vec3),
    sphere_indices: wp.array1d(dtype=wp.int32),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    inv_mesh_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_mesh_poses: wp.bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_mesh_scales: wp.bool,
    max_dist: float,
    closest_points_world: wp.array2d(dtype=wp.vec3),
    signs: wp.array2d(dtype=wp.float32),
    found: wp.array2d(dtype=wp.bool),
):
    instance, point_idx = wp.tid()  # type: ignore[misc]
    sphere_global_idx = sphere_indices[point_idx]
    q_pt = world_collision_sphere_centers[instance, sphere_global_idx]

    min_abs_dist = wp.float32(max_dist)
    min_closest_world = wp.vec3(0.0, 0.0, 0.0)
    min_sign = wp.float32(1.0)
    any_found = wp.bool(False)

    for m_idx in range(mesh_ids.shape[0]):
        q_pt_in_mesh_coord = q_pt
        if enable_inv_mesh_poses:
            q_pt_in_mesh_coord = wp.transform_point(inv_mesh_poses[m_idx], q_pt)
        if enable_mesh_scales:
            q_pt_in_mesh_coord = q_pt_in_mesh_coord / mesh_scales[m_idx]

        query = wp.mesh_query_point(mesh_ids[m_idx], q_pt_in_mesh_coord, max_dist)
        if query.result:
            clst_pt_in_mesh_coord = wp.mesh_eval_position(mesh_ids[m_idx], query.face, query.u, query.v)
            abs_dist = wp.length(q_pt_in_mesh_coord - clst_pt_in_mesh_coord)
            if enable_mesh_scales:
                abs_dist = abs_dist * mesh_scales[m_idx]

            if abs_dist < min_abs_dist:
                min_abs_dist = abs_dist
                any_found = wp.bool(True)
                min_sign = wp.float32(query.sign)

                clst_pt_unscaled = clst_pt_in_mesh_coord
                if enable_mesh_scales:
                    clst_pt_unscaled = clst_pt_in_mesh_coord * mesh_scales[m_idx]

                clst_world = clst_pt_unscaled
                if enable_inv_mesh_poses:
                    m_pose = wp.inverse(inv_mesh_poses[m_idx])
                    clst_world = wp.transform_point(m_pose, clst_pt_unscaled)

                min_closest_world = clst_world

    closest_points_world[instance, point_idx] = min_closest_world
    signs[instance, point_idx] = min_sign
    found[instance, point_idx] = any_found


@wp.kernel
def _query_closest_points_for_spheres_multi_scene_kernel(
    world_collision_sphere_centers: wp.array2d(dtype=wp.vec3),
    sphere_indices: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    mesh_first_idx: wp.array1d(dtype=wp.int32),
    inv_mesh_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_mesh_poses: wp.bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_mesh_scales: wp.bool,
    max_dist: float,
    closest_points_world: wp.array2d(dtype=wp.vec3),
    signs: wp.array2d(dtype=wp.float32),
    found: wp.array2d(dtype=wp.bool),
):
    """Query closest points for collision spheres - each instance queries only its scene's meshes."""
    instance, point_idx = wp.tid()
    sphere_global_idx = sphere_indices[point_idx]
    q_pt = world_collision_sphere_centers[instance, sphere_global_idx]

    scene_idx = scene_indices[instance]
    mesh_begin = mesh_first_idx[scene_idx]
    mesh_end = mesh_first_idx[scene_idx + 1]

    min_abs_dist = wp.float32(max_dist)
    min_closest_world = wp.vec3(0.0, 0.0, 0.0)
    min_sign = wp.float32(1.0)
    any_found = wp.bool(False)

    for m_idx in range(mesh_begin, mesh_end):
        q_pt_in_mesh_coord = q_pt
        if enable_inv_mesh_poses:
            q_pt_in_mesh_coord = wp.transform_point(inv_mesh_poses[m_idx], q_pt)
        if enable_mesh_scales:
            q_pt_in_mesh_coord = q_pt_in_mesh_coord / mesh_scales[m_idx]

        query = wp.mesh_query_point(mesh_ids[m_idx], q_pt_in_mesh_coord, max_dist)
        if query.result:
            clst_pt_in_mesh_coord = wp.mesh_eval_position(mesh_ids[m_idx], query.face, query.u, query.v)
            abs_dist = wp.length(q_pt_in_mesh_coord - clst_pt_in_mesh_coord)
            if enable_mesh_scales:
                abs_dist = abs_dist * mesh_scales[m_idx]

            if abs_dist < min_abs_dist:
                min_abs_dist = abs_dist
                any_found = wp.bool(True)
                min_sign = wp.float32(query.sign)

                clst_pt_unscaled = clst_pt_in_mesh_coord
                if enable_mesh_scales:
                    clst_pt_unscaled = clst_pt_in_mesh_coord * mesh_scales[m_idx]

                clst_world = clst_pt_unscaled
                if enable_inv_mesh_poses:
                    m_pose = wp.inverse(inv_mesh_poses[m_idx])
                    clst_world = wp.transform_point(m_pose, clst_pt_unscaled)

                min_closest_world = clst_world

    closest_points_world[instance, point_idx] = min_closest_world
    signs[instance, point_idx] = min_sign
    found[instance, point_idx] = any_found


@wp.kernel
def _compute_sdf_and_normal_from_closest_points_kernel(
    world_collision_sphere_centers: wp.array2d(dtype=wp.vec3),
    sphere_indices: wp.array1d(dtype=wp.int32),
    closest_points_world: wp.array2d(dtype=wp.vec3),
    signs: wp.array2d(dtype=wp.float32),
    found: wp.array2d(dtype=wp.bool),
    max_dist: float,
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
):
    instance, point_idx = wp.tid()  # type: ignore[misc]

    if not found[instance, point_idx]:
        signed_dists[instance, point_idx] = wp.float32(max_dist)
        normals[instance, point_idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    sphere_global_idx = sphere_indices[point_idx]
    q_pt = world_collision_sphere_centers[instance, sphere_global_idx]
    clst = closest_points_world[instance, point_idx]
    diff = q_pt - clst
    dist = wp.length(diff)

    denom = wp.max(wp.float32(1e-8), dist)
    sign = signs[instance, point_idx]
    normal = diff / denom
    normal = normal * sign

    signed_dists[instance, point_idx] = dist * sign
    normals[instance, point_idx] = normal


@wp.kernel
def _compute_collision_task_weighted_residual_from_sdf_kernel(
    local_collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    signed_dists: wp.array2d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    activation_dist: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
):
    instance, sphere_idx = wp.tid()  # type: ignore[misc]

    sphere_global_idx = sphere_indices[sphere_idx]
    radius = local_collision_sphere_radii[sphere_global_idx]

    signed_dist = signed_dists[instance, sphere_idx]
    gap = signed_dist - radius
    gap_clamped = wp.min(gap, wp.float32(activation_dist))
    inv_a = wp.float32(0.5) / (wp.float32(activation_dist) + wp.float32(1e-6))
    diff = gap_clamped - wp.float32(activation_dist)
    colldist = wp.where(
        gap_clamped < wp.float32(0.0),
        gap_clamped - wp.float32(0.5) * wp.float32(activation_dist),
        wp.float32(-1.0) * inv_a * diff * diff,
    )
    colldist = wp.min(colldist, wp.float32(0.0))
    residual = wp.float32(-1.0) * colldist

    residual_buffer[instance, row_offset + sphere_idx] = residual_weight[sphere_idx] * residual


@wp.kernel
def _compute_collision_task_weighted_jacobian_from_sdf_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    world_collision_sphere_centers: wp.array2d(dtype=wp.vec3),
    local_collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    activation_dist: float,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
):
    instance, sphere_idx, col_idx = wp.tid()  # type: ignore[misc]
    base_dofs = 6 if has_floating_base else 0
    actuated_idx = col_idx - base_dofs

    sphere_global_idx = sphere_indices[sphere_idx]
    center = world_collision_sphere_centers[instance, sphere_global_idx]
    radius = local_collision_sphere_radii[sphere_global_idx]

    signed_dist = signed_dists[instance, sphere_idx]
    normal = normals[instance, sphere_idx]

    gap = signed_dist - radius
    dres_dsd = wp.float32(0.0)
    if gap < wp.float32(activation_dist):
        if gap < wp.float32(0.0):
            dres_dsd = wp.float32(-1.0)
        else:
            dres_dsd = (gap - wp.float32(activation_dist)) / (wp.float32(activation_dist) + wp.float32(1e-6))

    spatial_twist = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if has_floating_base and col_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = wp.float32(1.0)
        spatial_twist = se3_adjoint_multiply_vec6_func(T_world_base[instance], unit_vec)
    else:
        link_idx = collision_spheres_link_indices[sphere_global_idx]
        num_joints = S_world.shape[2]
        for joint_idx in range(num_joints):
            if link_ancestor_joints_mask[link_idx, joint_idx]:
                weight = joints_to_actuated[joint_idx, actuated_idx]
                if weight != 0.0:
                    for row in range(6):
                        spatial_twist[row] = spatial_twist[row] + S_world[instance, row, joint_idx] * weight

    v = wp.vec3(spatial_twist[0], spatial_twist[1], spatial_twist[2])
    w = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])
    point_vel = v + wp.cross(w, center)
    dsd_dv = wp.dot(normal, point_vel)

    jacobian_buffer[instance, row_offset + sphere_idx, col_idx] = (
        residual_weight[sphere_idx] * wp.float32(dres_dsd) * dsd_dv
    )


@dataclass
class WarpCollisionTask(WarpTask):
    robot: WarpRobot
    scene_meshes: Sequence[wp.Mesh]
    weight: Optional[Union[float, Sequence[float]]] = None
    margin: float = 0.0
    beta: float = 50.0
    max_dist: float = 1e6
    batch_size: int = 1
    sphere_indices: Optional[Sequence[int]] = None
    inv_mesh_poses: Optional[wp.array] = None
    mesh_scales: Optional[wp.array] = None

    # Multi-scene batching support
    scene_indices: Optional[wp.array] = None  # [batch_size] maps instance -> scene index
    mesh_first_idx: Optional[wp.array] = None  # [n_scene + 1] mesh boundaries per scene

    _device: Optional[wp_device_type] = None
    _multi_scene_mode: bool = False
    _mesh_ids: Optional[wp.array] = None
    _sphere_indices_wp: Optional[wp.array] = None
    _inv_mesh_poses: Optional[wp.array] = None
    _mesh_scales: Optional[wp.array] = None
    _enable_inv_mesh_poses: bool = False
    _enable_mesh_scales: bool = False
    _residual_weight_np: Optional[np.ndarray] = None
    residual_weight: Optional[wp.array] = None
    _sphere_indices_np: Optional[np.ndarray] = None
    _cached_signed_dists: Optional[wp.array] = None
    _cached_normals: Optional[wp.array] = None

    def __post_init__(self) -> None:
        if len(self.scene_meshes) == 0:
            raise ValueError("scene_meshes must be non-empty.")
        if not self.robot.spec.has_collision_spheres:
            raise RuntimeError("Robot has no collision spheres. Load with load_collision_spheres=True.")

        num_spheres = int(self.robot.spec.local_collision_sphere_centers.shape[0])
        if self.sphere_indices is None:
            sphere_indices_np = np.arange(num_spheres, dtype=np.int32)
        else:
            sphere_indices_np = np.asarray(self.sphere_indices, dtype=np.int32)
            if sphere_indices_np.ndim != 1:
                raise ValueError("sphere_indices must be 1D.")
            if np.any(sphere_indices_np < 0) or np.any(sphere_indices_np >= num_spheres):
                raise ValueError("sphere_indices contains out-of-range indices.")
        self._sphere_indices_np = sphere_indices_np

        residual_weight = np.ones((sphere_indices_np.shape[0],), dtype=np.float32)
        if self.weight is not None:
            if isinstance(self.weight, (float, int)):
                residual_weight[:] = float(self.weight)
            else:
                weight_arr = np.asarray(self.weight, dtype=np.float32)
                if weight_arr.shape != (sphere_indices_np.shape[0],):
                    raise ValueError(f"Expected weight shape ({sphere_indices_np.shape[0]},), got {weight_arr.shape}.")
                residual_weight[:] = weight_arr
        self._residual_weight_np = residual_weight

        # Detect multi-scene mode
        self._multi_scene_mode = self.scene_indices is not None and self.mesh_first_idx is not None

    def _init_buffers(self, device: wp_device_type) -> None:
        self._device = device
        self._mesh_ids = wp.array([m.id for m in self.scene_meshes], dtype=wp.uint64, device=device)
        assert self._sphere_indices_np is not None
        self._sphere_indices_wp = wp.from_numpy(self._sphere_indices_np, dtype=wp.int32, device=device)
        assert self._residual_weight_np is not None
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

        if self.inv_mesh_poses is not None:
            self._inv_mesh_poses = self.inv_mesh_poses
            self._enable_inv_mesh_poses = True
        else:
            self._inv_mesh_poses = wp.zeros((len(self.scene_meshes),), dtype=wp.mat44, device=device)
            self._enable_inv_mesh_poses = False

        if self.mesh_scales is not None:
            self._mesh_scales = self.mesh_scales
            self._enable_mesh_scales = True
        else:
            ones_np = np.ones((len(self.scene_meshes),), dtype=np.float32)
            self._mesh_scales = wp.from_numpy(ones_np, dtype=wp.float32, device=device)
            self._enable_mesh_scales = False

    @property
    def residual_dim(self) -> int:
        assert self._sphere_indices_np is not None
        return int(self._sphere_indices_np.shape[0])

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if var.is_trajectory:
            raise NotImplementedError("WarpCollisionTask does not support trajectory states yet.")
        if self._device is None:
            self._init_buffers(var.q.device)
        assert self._device is not None
        assert self._mesh_ids is not None
        assert self._sphere_indices_wp is not None
        assert self.residual_weight is not None
        assert self._inv_mesh_poses is not None
        assert self._mesh_scales is not None

        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)

        if residual_buffer is None:
            residual_buffer = wp.empty((var.batch_size, self.residual_dim), dtype=wp.float32, device=self._device)
            row_offset = 0

        closest_points_world = wp.empty((var.batch_size, self.residual_dim), dtype=wp.vec3, device=self._device)
        signs = wp.empty((var.batch_size, self.residual_dim), dtype=wp.float32, device=self._device)
        found = wp.empty((var.batch_size, self.residual_dim), dtype=wp.bool, device=self._device)
        signed_dists = wp.empty((var.batch_size, self.residual_dim), dtype=wp.float32, device=self._device)
        normals = wp.empty((var.batch_size, self.residual_dim), dtype=wp.vec3, device=self._device)

        spec_t = var.spec_tensors

        if self._multi_scene_mode:
            wp.launch(
                kernel=_query_closest_points_for_spheres_multi_scene_kernel,
                dim=(var.batch_size, self.residual_dim),
                inputs=[
                    var.world_collision_sphere_centers,
                    self._sphere_indices_wp,
                    self.scene_indices,
                    self._mesh_ids,
                    self.mesh_first_idx,
                    self._inv_mesh_poses,
                    self._enable_inv_mesh_poses,
                    self._mesh_scales,
                    self._enable_mesh_scales,
                    self.max_dist,
                    closest_points_world,
                    signs,
                    found,
                ],
                device=self._device,
            )
        else:
            wp.launch(
                kernel=_query_closest_points_for_spheres_single_scene_kernel,
                dim=(var.batch_size, self.residual_dim),
                inputs=[
                    var.world_collision_sphere_centers,
                    self._sphere_indices_wp,
                    self._mesh_ids,
                    self._inv_mesh_poses,
                    self._enable_inv_mesh_poses,
                    self._mesh_scales,
                    self._enable_mesh_scales,
                    self.max_dist,
                    closest_points_world,
                    signs,
                    found,
                ],
                device=self._device,
            )

        wp.launch(
            kernel=_compute_sdf_and_normal_from_closest_points_kernel,
            dim=(var.batch_size, self.residual_dim),
            inputs=[
                var.world_collision_sphere_centers,
                self._sphere_indices_wp,
                closest_points_world,
                signs,
                found,
                self.max_dist,
                signed_dists,
                normals,
            ],
            device=self._device,
        )

        self._cached_signed_dists = signed_dists
        self._cached_normals = normals

        wp.launch(
            kernel=_compute_collision_task_weighted_residual_from_sdf_kernel,
            dim=(var.batch_size, self.residual_dim),
            inputs=[
                spec_t.local_collision_sphere_radii,
                self._sphere_indices_wp,
                signed_dists,
                self.residual_weight,
                self.margin,
                row_offset,
            ],
            outputs=[residual_buffer],
            device=self._device,
        )

        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: WarpRobotState,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if var.is_trajectory:
            raise NotImplementedError("WarpCollisionTask does not support trajectory states yet.")
        if self._device is None:
            self._init_buffers(var.q.device)
        assert self._device is not None
        assert self._mesh_ids is not None
        assert self._sphere_indices_wp is not None
        assert self.residual_weight is not None
        assert self._inv_mesh_poses is not None
        assert self._mesh_scales is not None

        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)

        total_dofs = var.tangent_dim
        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (var.batch_size, self.residual_dim, total_dofs),
                dtype=wp.float32,
                device=self._device,
            )
            row_offset = 0

        spec_t = var.spec_tensors

        signed_dists = self._cached_signed_dists
        normals = self._cached_normals

        wp.launch(
            kernel=_compute_collision_task_weighted_jacobian_from_sdf_kernel,
            dim=(var.batch_size, self.residual_dim, total_dofs),
            inputs=[
                var.S_world,
                var.T_world_base.xyz_wxyz.flatten(),
                var.world_collision_sphere_centers,
                spec_t.local_collision_sphere_radii,
                self._sphere_indices_wp,
                spec_t.collision_spheres_link_indices,
                spec_t.link_ancestor_joints_mask,
                spec_t.joints_to_actuated_mapping,
                signed_dists,
                normals,
                var.has_floating_base,
                self.residual_weight,
                self.margin,
                row_offset,
                jacobian_buffer,
            ],
            device=self._device,
        )

        return jacobian_buffer


__all__ = ["WarpCollisionTask"]
