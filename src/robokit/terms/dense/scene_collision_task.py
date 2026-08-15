# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIncompatibleVariableOverride=false
"""Sphere-based robot collision against a `WarpScene`."""

from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.geom import BaseGeom, WarpScene
from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.robot_task import RobotTask
from robokit.terms.sphere_collision_kernels import (
    PENALTY_PLAIN,
    PENALTY_SMOOTH,
    PENALTY_SURFACE_DISTANCE,
    _compute_collision_penalty_func,
    sphere_collision_residual_apply_kernel,
)
from robokit.terms.task import GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


# --- device code ------------------------------------------------------------
@wp.func
def _compute_point_jacobian_column_func(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    instance: int,
    link_idx: int,
    col_idx: int,
    has_floating_base: bool,
    point: wp.vec3,
) -> wp.vec3:
    base_dofs = wp.int32(6) if has_floating_base else wp.int32(0)
    twist = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if has_floating_base and col_idx < 6:
        unit = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit[col_idx] = wp.float32(1.0)
        twist = se3_adjoint_func(T_world_base[instance]) * unit
    else:
        actuated_idx = col_idx - base_dofs
        for joint_idx in range(S_world.shape[2]):
            if link_ancestor_joints_mask[link_idx, joint_idx]:
                joint_weight = joints_to_actuated[joint_idx, actuated_idx]
                for row in range(6):
                    twist[row] += S_world[instance, row, joint_idx] * joint_weight
    linear = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    return linear + wp.cross(angular, point)


@wp.kernel
def _sphere_collision_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    sphere_centers_world: wp.array2d(dtype=wp.vec3),
    sphere_indices: wp.array1d(dtype=wp.int32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    gap_cache: wp.array2d(dtype=wp.float32),
    normal_cache: wp.array2d(dtype=wp.vec3),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    activation_dist: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    row_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    instance, sphere_idx, col_idx = wp.tid()
    gap = gap_cache[instance, sphere_idx]
    if penalty_mode != PENALTY_SURFACE_DISTANCE and gap > wp.float32(activation_dist):
        out_jacobian[instance, row_offset + sphere_idx, col_idx] = wp.float32(0.0)
        return
    global_idx = sphere_indices[sphere_idx]
    center = sphere_centers_world[instance, global_idx]
    penalty = _compute_collision_penalty_func(gap, wp.float32(activation_dist), penalty_mode)
    scale = penalty[1]
    if use_sqrt and penalty[0] > wp.float32(0.0):
        scale *= wp.float32(0.5) / wp.sqrt(penalty[0] + eps)
    column = _compute_point_jacobian_column_func(
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joints_to_actuated,
        instance,
        collision_spheres_link_indices[global_idx],
        col_idx,
        has_floating_base,
        center,
    )
    out_jacobian[instance, row_offset + sphere_idx, col_idx] = (
        residual_weight[sphere_idx] * scale * wp.dot(normal_cache[instance, sphere_idx], column)
    )


@wp.kernel
def _compute_sphere_skip_mask_kernel(
    link_sdf: wp.array2d(dtype=wp.float32),
    link_radii: wp.array1d(dtype=wp.float32),
    sphere_link_indices: wp.array1d(dtype=wp.int32),
    activation_dist: float,
    sphere_skip_mask_flat: wp.array1d(dtype=wp.int32),
):
    """Mark sphere queries that pass the link bounding-sphere filter."""
    instance, global_idx = wp.tid()
    n_sph = sphere_link_indices.shape[0]
    link = sphere_link_indices[global_idx]
    sdf = link_sdf[instance, link]
    r = link_radii[link]
    if sdf - r > wp.float32(activation_dist):
        sphere_skip_mask_flat[instance * n_sph + global_idx] = 0
    else:
        sphere_skip_mask_flat[instance * n_sph + global_idx] = 1


@wp.kernel
def _build_gap_cache_kernel(
    raw_sdf: wp.array2d(dtype=wp.float32),
    raw_normals: wp.array2d(dtype=wp.vec3),
    sphere_indices: wp.array1d(dtype=wp.int32),
    radii_global: wp.array1d(dtype=wp.float32),
    gap_cache: wp.array2d(dtype=wp.float32),
    normal_cache: wp.array2d(dtype=wp.vec3),
):
    """Build selected-sphere gaps and normals from the scene query."""
    instance, sphere_idx = wp.tid()
    global_idx = sphere_indices[sphere_idx]
    gap_cache[instance, sphere_idx] = raw_sdf[instance, global_idx] - radii_global[global_idx]
    normal_cache[instance, sphere_idx] = raw_normals[instance, global_idx]


@wp.kernel
def _expand_scene_indices_kernel(
    scene_indices: wp.array1d(dtype=wp.int32),
    points_per_instance: int,
    scene_per_point: wp.array1d(dtype=wp.int32),
):
    point_idx = wp.tid()
    scene_per_point[point_idx] = scene_indices[point_idx // points_per_instance]


@wp.kernel
def _compute_collision_jtr_cost_and_gradient_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    sphere_centers_world: wp.array2d(dtype=wp.vec3),
    local_collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    out_residual: wp.array2d(dtype=wp.float32),
    activation_dist: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    col_offset: int,
    out_cost: wp.array1d(dtype=wp.float32),
    out_gradient: wp.array2d(dtype=wp.float32),
):
    instance, sphere_idx, col_idx = wp.tid()

    sphere_global_idx = sphere_indices[sphere_idx]
    radius = local_collision_sphere_radii[sphere_global_idx]
    signed_dist = signed_dists[instance, sphere_global_idx]
    normal = normals[instance, sphere_global_idx]
    gap = signed_dist - radius

    penalty = _compute_collision_penalty_func(gap, wp.float32(activation_dist), penalty_mode)
    dres_dgap = penalty[1]
    if use_sqrt and penalty[0] > wp.float32(0.0):
        dres_dgap = dres_dgap * wp.float32(0.5) / wp.sqrt(penalty[0] + eps)

    r = out_residual[instance, sphere_idx]

    if col_idx == 0:
        wp.atomic_add(out_cost, instance, wp.float32(0.5) * r * r)

    if dres_dgap == wp.float32(0.0):
        return

    center = sphere_centers_world[instance, sphere_global_idx]

    point_column = _compute_point_jacobian_column_func(
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joints_to_actuated,
        instance,
        collision_spheres_link_indices[sphere_global_idx],
        col_idx,
        has_floating_base,
        center,
    )
    dsd_dv = wp.dot(normal, point_column)
    weighted_jac = residual_weight[sphere_idx] * dres_dgap * dsd_dv
    contribution = weighted_jac * r
    if contribution != wp.float32(0.0):
        wp.atomic_add(out_gradient, instance, col_offset + col_idx, contribution)


@wp.kernel
def _compute_collision_direct_per_batch_sum_kernel(
    signed_dists: wp.array2d(dtype=wp.float32),
    local_collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    margin: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    num_frames: int,
    per_batch_sum: wp.array1d(dtype=wp.float32),
):
    b, t, s = wp.tid()  # type: ignore[misc]
    sphere_global_idx = sphere_indices[s]
    radius = local_collision_sphere_radii[sphere_global_idx]
    frame_global_idx = b * num_frames + t
    sd = signed_dists[frame_global_idx, sphere_global_idx]
    residual = _compute_collision_penalty_func(sd - radius, margin, penalty_mode)[0]
    if use_sqrt:
        residual = wp.sqrt(residual + eps)
    wp.atomic_add(per_batch_sum, b, residual)


@wp.kernel
def _accumulate_collision_direct_cost_kernel(
    per_batch_sum: wp.array1d(dtype=wp.float32),
    weight_over_B: float,
    out_cost: wp.array1d(dtype=wp.float32),
):
    b = wp.tid()
    s = per_batch_sum[b]
    out_cost[b] = out_cost[b] + weight_over_B * s * s


@wp.kernel
def _compute_collision_direct_gradient_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    sphere_centers_world: wp.array2d(dtype=wp.vec3),
    local_collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
    has_floating_base: wp.bool,
    margin: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    two_weight_over_B: float,
    per_batch_sum: wp.array1d(dtype=wp.float32),
    num_frames: int,
    single_tangent_dim: int,
    skip_endpoints: int,
    col_offset: int,
    out_gradient: wp.array2d(dtype=wp.float32),
):
    b, t, s, col = wp.tid()  # type: ignore[misc]
    if skip_endpoints != 0:
        if t == 0 or t == num_frames - 1:
            return

    sphere_global_idx = sphere_indices[s]
    frame_global_idx = b * num_frames + t
    radius = local_collision_sphere_radii[sphere_global_idx]
    sd = signed_dists[frame_global_idx, sphere_global_idx]
    penalty = _compute_collision_penalty_func(sd - radius, margin, penalty_mode)
    dres_dgap = penalty[1]
    if use_sqrt and penalty[0] > wp.float32(0.0):
        dres_dgap = dres_dgap * wp.float32(0.5) / wp.sqrt(penalty[0] + eps)
    if dres_dgap == wp.float32(0.0):
        return

    center = sphere_centers_world[frame_global_idx, sphere_global_idx]
    normal = normals[frame_global_idx, sphere_global_idx]
    point_column = _compute_point_jacobian_column_func(
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joints_to_actuated,
        frame_global_idx,
        collision_spheres_link_indices[sphere_global_idx],
        col,
        has_floating_base,
        center,
    )

    sum_b = per_batch_sum[b]
    grad_contrib = two_weight_over_B * sum_b * dres_dgap * wp.dot(normal, point_column)

    grad_col = col_offset + t * single_tangent_dim + col
    wp.atomic_add(out_gradient, b, grad_col, grad_contrib)


# --- task -------------------------------------------------------------------
class SceneCollisionTask(RobotTask, ResidualTask, GradientTask):
    """Penalize robot collision spheres that approach scene geometry."""

    precompute_collision_geometry = "sphere"

    def __init__(
        self,
        robot: Optional[Robot] = None,
        scene: Optional[WarpScene] = None,
        geometry: Optional[BaseGeom] = None,  # restrict queries to one scene geometry
        scene_indices: Optional[wp.array] = None,
        use_link_bounding_sphere_filter: bool = False,
        weight: Optional[Union[float, Sequence[float]]] = None,
        margin: float = 0.02,
        sphere_indices: Optional[Sequence[int]] = None,
        residual_mode: Literal["abs", "sqrt_abs"] = "abs",
        residual_eps: float = 1e-6,
        penalty: Literal["smooth", "plain", "surface_distance"] = "smooth",
        skip_trajectory_endpoints: bool = True,  # preserve pinned trajectory endpoints
    ):
        self.robot = robot
        self.scene = scene
        self.geometry = geometry
        self.scene_indices = scene_indices
        self.use_link_bounding_sphere_filter = use_link_bounding_sphere_filter
        self.weight = weight
        self.margin = margin
        self.sphere_indices = sphere_indices
        self.residual_mode = residual_mode
        self.residual_eps = residual_eps
        self.penalty = penalty
        self.skip_trajectory_endpoints = skip_trajectory_endpoints

        self._device: Optional[wp_device_type] = None
        self._sphere_indices_np: Optional[np.ndarray] = None
        self._residual_weight_np: Optional[np.ndarray] = None
        self._sphere_indices_wp: Optional[wp.array] = None
        self._residual_weight_wp: Optional[wp.array] = None
        self._scene_indices_wp: Optional[wp.array] = None
        self._sphere_scene_per_point_wp: Optional[wp.array] = None
        self._link_scene_per_point_wp: Optional[wp.array] = None
        self._raw_sdf: Optional[wp.array] = None
        self._raw_normals: Optional[wp.array] = None
        self._raw_closest_points: Optional[wp.array] = None
        self._link_sdf: Optional[wp.array] = None
        self._sphere_skip_mask_flat: Optional[wp.array] = None
        self._gap_cache: Optional[wp.array] = None
        self._normal_cache: Optional[wp.array] = None
        self._num_spheres = 0
        self._cached_batch_size = 0
        # direct GD/L-BFGS cost and gradient buffers
        self._direct_signed_dists: Optional[wp.array] = None
        self._direct_normals: Optional[wp.array] = None
        self._direct_closest_points: Optional[wp.array] = None
        self._direct_scene_per_point_wp: Optional[wp.array] = None
        self._direct_link_scene_per_point_wp: Optional[wp.array] = None
        self._direct_per_batch_sum: Optional[wp.array] = None
        self._direct_dummy_T_world_base: Optional[wp.array] = None
        self._cached_direct_shape: Tuple[int, int] = (0, 0)
        self._jtr_residual_buf: Optional[wp.array] = None
        self._direct_link_sdf: Optional[wp.array] = None
        self._direct_skip_mask_flat: Optional[wp.array] = None

        if self.penalty not in {"smooth", "plain", "surface_distance"}:
            raise ValueError(f"Unsupported penalty: {self.penalty}")
        if self.robot is None or self.scene is None:
            return  # config-time stub
        if not self.robot.spec.has_collision_spheres:
            raise RuntimeError("Robot has no collision spheres. Load with load_collision_spheres=True.")
        if not self.scene.geoms:
            raise ValueError("WarpScene must contain at least one geometry.")
        if self.scene_indices is not None and (self.scene_indices.dtype != wp.int32 or self.scene_indices.ndim != 1):
            raise TypeError("scene_indices must be a 1D wp.int32 array.")
        num_spheres = int(self.robot.spec.local_collision_sphere_centers.shape[0])
        self._num_spheres = num_spheres
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

    def set_robot(self, robot: Robot):
        self.robot = robot

    @property
    def _penalty_mode(self) -> int:
        if self.penalty == "plain":
            return PENALTY_PLAIN
        if self.penalty == "surface_distance":
            return PENALTY_SURFACE_DISTANCE
        return PENALTY_SMOOTH

    def set_scene_indices(self, scene_indices: wp.array):
        """Copy a per-query device scene mapping into stable query buffers."""
        if scene_indices.dtype != wp.int32 or scene_indices.ndim != 1:
            raise TypeError("scene_indices must be a 1D wp.int32 array.")
        batch_size = self._cached_batch_size or self._cached_direct_shape[0]
        if batch_size != 0 and scene_indices.shape != (batch_size,):
            raise ValueError("scene_indices must have shape (batch_size,).")
        self.scene_indices = scene_indices
        if self._device is not None:
            if scene_indices.device != wp.get_device(self._device):
                raise ValueError(f"scene_indices must be on {wp.get_device(self._device)}, got {scene_indices.device}.")
            if self._scene_indices_wp is None:
                self._scene_indices_wp = wp.empty(self._cached_batch_size, dtype=wp.int32, device=self._device)
                self._sphere_scene_per_point_wp = wp.empty(
                    self._cached_batch_size * self._num_spheres, dtype=wp.int32, device=self._device
                )
                if self.use_link_bounding_sphere_filter:
                    self._link_scene_per_point_wp = wp.empty(
                        self._cached_batch_size * self.robot.spec.num_links, dtype=wp.int32, device=self._device
                    )
                self._cached_direct_shape = (0, 0)
            wp.copy(self._scene_indices_wp, scene_indices)

    @property
    def residual_dim(self) -> int:
        if self.sphere_indices is None:
            return self._num_spheres
        return len(self.sphere_indices)

    def _init_buffers(self, batch_size: int, device: wp_device_type):
        self._device = device
        self._cached_batch_size = batch_size
        n_sph, n_link, n_sel = self._num_spheres, self.robot.spec.num_links, self.residual_dim

        self._scene_indices_wp = wp.zeros(batch_size, dtype=wp.int32, device=device)
        if self.scene_indices is not None:
            if self.scene_indices.shape != (batch_size,):
                raise ValueError("scene_indices must have shape (batch_size,).")
            if self.scene_indices.device != wp.get_device(device):
                raise ValueError(f"scene_indices must be on {wp.get_device(device)}, got {self.scene_indices.device}.")
            wp.copy(self._scene_indices_wp, self.scene_indices)
        self._sphere_scene_per_point_wp = wp.empty(batch_size * n_sph, dtype=wp.int32, device=device)

        self._sphere_indices_wp = wp.from_numpy(self._sphere_indices_np, dtype=wp.int32, device=device)
        self._residual_weight_wp = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self._raw_sdf = wp.empty((batch_size, n_sph), dtype=wp.float32, device=device)
        self._raw_normals = wp.empty((batch_size, n_sph), dtype=wp.vec3, device=device)
        self._raw_closest_points = wp.empty((batch_size, n_sph), dtype=wp.vec3, device=device)
        self._gap_cache = wp.empty((batch_size, n_sel), dtype=wp.float32, device=device)
        self._normal_cache = wp.empty((batch_size, n_sel), dtype=wp.vec3, device=device)
        if self.use_link_bounding_sphere_filter:
            self._link_sdf = wp.empty((batch_size, n_link), dtype=wp.float32, device=device)
            self._sphere_skip_mask_flat = wp.empty((batch_size * n_sph,), dtype=wp.int32, device=device)
            self._link_scene_per_point_wp = wp.empty(batch_size * n_link, dtype=wp.int32, device=device)

    def _run_query(self, var: RobotState):
        spec_t = var.spec_tensors
        b, n_sph = var.batch_size, self._num_spheres
        sphere_skip_mask: Optional[wp.array] = None
        wp.launch(
            _expand_scene_indices_kernel,
            dim=b * n_sph,
            inputs=[self._scene_indices_wp, n_sph],
            outputs=[self._sphere_scene_per_point_wp],
            device=self._device,
        )

        if self.use_link_bounding_sphere_filter:
            n_link = self.robot.spec.num_links
            wp.launch(
                _expand_scene_indices_kernel,
                dim=b * n_link,
                inputs=[self._scene_indices_wp, n_link],
                outputs=[self._link_scene_per_point_wp],
                device=self._device,
            )
            (self.geometry or self.scene).query_sdf(
                var.link_bounding_sphere_centers_world,
                scene_indices=self._link_scene_per_point_wp,
                out_signed_dists=self._link_sdf,
                distance_only=True,
            )
            wp.launch(
                kernel=_compute_sphere_skip_mask_kernel,
                dim=(b, n_sph),
                inputs=[
                    self._link_sdf,
                    spec_t.link_bounding_sphere_radii,
                    spec_t.collision_spheres_link_indices,
                    self.margin,
                ],
                outputs=[self._sphere_skip_mask_flat],
                device=self._device,
            )
            sphere_skip_mask = self._sphere_skip_mask_flat

        (self.geometry or self.scene).query_sdf(
            var.collision_sphere_centers_world,
            scene_indices=self._sphere_scene_per_point_wp,
            out_signed_dists=self._raw_sdf,
            out_normals=self._raw_normals,
            out_closest_points=self._raw_closest_points,
            query_mask=sphere_skip_mask,
        )
        wp.launch(
            kernel=_build_gap_cache_kernel,
            dim=(b, self.residual_dim),
            inputs=[self._raw_sdf, self._raw_normals, self._sphere_indices_wp, spec_t.collision_sphere_radii],
            outputs=[self._gap_cache, self._normal_cache],
            device=self._device,
        )

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if var.is_trajectory:
            raise NotImplementedError("SceneCollisionTask does not support trajectory states yet.")
        if self._device is None or self._cached_batch_size != var.batch_size:
            self._init_buffers(var.batch_size, var.q.device)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)

        if out_residual is None:
            out_residual = wp.empty((var.batch_size, self.residual_dim), dtype=wp.float32, device=self._device)
            row_offset = 0

        self._run_query(var)

        wp.launch(
            kernel=sphere_collision_residual_apply_kernel,
            dim=(var.batch_size, self.residual_dim),
            inputs=[
                self._gap_cache,
                self._residual_weight_wp,
                self.margin,
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                row_offset,
            ],
            outputs=[out_residual],
            device=self._device,
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        assert var_values.tangent_offset(self.var_key) == 0
        if var.is_trajectory:
            raise NotImplementedError("SceneCollisionTask does not support trajectory states yet.")
        if self._device is None or self._cached_batch_size != var.batch_size:
            self._init_buffers(var.batch_size, var.q.device)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)

        total_dofs = var.tangent_dim
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (var.batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=self._device
            )
            row_offset = 0

        spec_t = var.spec_tensors
        wp.launch(
            kernel=_sphere_collision_jacobian_kernel,
            dim=(var.batch_size, self.residual_dim, total_dofs),
            inputs=[
                var.S_world,
                var.T_world_base.flatten(),
                var.collision_sphere_centers_world,
                self._sphere_indices_wp,
                spec_t.collision_spheres_link_indices,
                self._gap_cache,
                self._normal_cache,
                spec_t.link_ancestor_joints_mask,
                spec_t.joints_to_actuated_mapping,
                var.has_floating_base,
                self._residual_weight_wp,
                self.margin,
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                row_offset,
            ],
            outputs=[out_jacobian],
            device=self._device,
        )
        return out_jacobian

    # --- analytical cost and gradient ---
    def _init_direct_buffers(self, batch_size: int, num_frames: int, device: wp_device_type):
        BT = batch_size * num_frames
        self._device = device
        if self._sphere_indices_wp is None:
            self._sphere_indices_wp = wp.from_numpy(self._sphere_indices_np, dtype=wp.int32, device=device)
        if self._residual_weight_wp is None:
            self._residual_weight_wp = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self._direct_signed_dists = wp.empty((BT, self._num_spheres), dtype=wp.float32, device=device)
        self._direct_normals = wp.empty((BT, self._num_spheres), dtype=wp.vec3, device=device)
        self._direct_closest_points = wp.empty((BT, self._num_spheres), dtype=wp.vec3, device=device)
        self._direct_per_batch_sum = wp.zeros((batch_size,), dtype=wp.float32, device=device)
        if self._scene_indices_wp is None or self._scene_indices_wp.shape != (batch_size,):
            self._scene_indices_wp = wp.zeros(batch_size, dtype=wp.int32, device=device)
            if self.scene_indices is not None:
                if self.scene_indices.shape != (batch_size,):
                    raise ValueError("scene_indices must have shape (batch_size,).")
                if self.scene_indices.device != wp.get_device(device):
                    raise ValueError(
                        f"scene_indices must be on {wp.get_device(device)}, got {self.scene_indices.device}."
                    )
                wp.copy(self._scene_indices_wp, self.scene_indices)
        self._direct_scene_per_point_wp = wp.empty(BT * self._num_spheres, dtype=wp.int32, device=device)
        self._direct_dummy_T_world_base = wp.empty((1,), dtype=wp_vec7, device=device)
        self._cached_direct_shape = (batch_size, num_frames)
        if self.use_link_bounding_sphere_filter:
            n_link = self.robot.spec.num_links
            self._direct_link_sdf = wp.empty((BT, n_link), dtype=wp.float32, device=device)
            self._direct_skip_mask_flat = wp.empty((BT * self._num_spheres,), dtype=wp.int32, device=device)
            self._direct_link_scene_per_point_wp = wp.empty(BT * n_link, dtype=wp.int32, device=device)

    def _ensure_direct_state(self, var: RobotState) -> Tuple[int, int]:
        B, T = (var.batch_size, var.shape[1]) if var.is_trajectory else (var.batch_size, 1)
        if self._cached_direct_shape != (B, T):
            self._init_direct_buffers(B, T, var.q.device)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)
        return B, T

    def _query_direct_sdf(self, var: RobotState, B: int, T: int):
        BT = B * T
        wp.launch(
            _expand_scene_indices_kernel,
            dim=BT * self._num_spheres,
            inputs=[self._scene_indices_wp, T * self._num_spheres],
            outputs=[self._direct_scene_per_point_wp],
            device=self._device,
        )
        extra_kwargs: dict = {}
        if self.use_link_bounding_sphere_filter:
            spec_t = var.spec_tensors
            n_link = self.robot.spec.num_links
            wp.launch(
                _expand_scene_indices_kernel,
                dim=BT * n_link,
                inputs=[self._scene_indices_wp, T * n_link],
                outputs=[self._direct_link_scene_per_point_wp],
                device=self._device,
            )
            (self.geometry or self.scene).query_sdf(
                var.link_bounding_sphere_centers_world,
                scene_indices=self._direct_link_scene_per_point_wp,
                out_signed_dists=self._direct_link_sdf,
                distance_only=True,
            )
            wp.launch(
                kernel=_compute_sphere_skip_mask_kernel,
                dim=(BT, self._num_spheres),
                inputs=[
                    self._direct_link_sdf,
                    spec_t.link_bounding_sphere_radii,
                    spec_t.collision_spheres_link_indices,
                    self.margin,
                ],
                outputs=[self._direct_skip_mask_flat],
                device=self._device,
            )
            extra_kwargs["query_mask"] = self._direct_skip_mask_flat
        (self.geometry or self.scene).query_sdf(
            var.collision_sphere_centers_world,
            out_signed_dists=self._direct_signed_dists,
            out_normals=self._direct_normals,
            out_closest_points=self._direct_closest_points,
            scene_indices=self._direct_scene_per_point_wp,
            **extra_kwargs,
        )

    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
        cost_form: Literal["squared_sum", "sum_squared"] = "squared_sum",
    ):
        """Compute collision cost and optional direct gradient.

        Lifecycle:
            1. Prepare trajectory-shaped collision buffers and kinematics.
            2. Query the scene SDF for every selected sphere.
            3. Accumulate the selected cost form and optional gradient.
        """
        col_offset = var_values.tangent_offset(self.var_key)
        var = var_values.get(self.var_key)
        if out_gradient is None:
            # --- cost-only step ---
            B, T = self._ensure_direct_state(var)
            self._query_direct_sdf(var, B, T)
            spec_t = var.spec_tensors
            pen_weight = float(self._residual_weight_np[0]) if self._residual_weight_np is not None else 1.0
            weight_over_B = pen_weight / float(B)
            self._direct_per_batch_sum.zero_()
            wp.launch(
                kernel=_compute_collision_direct_per_batch_sum_kernel,
                dim=(B, T, self.residual_dim),
                inputs=[
                    self._direct_signed_dists,
                    spec_t.collision_sphere_radii,
                    self._sphere_indices_wp,
                    float(self.margin),
                    self.residual_mode == "sqrt_abs",
                    self.residual_eps,
                    self._penalty_mode,
                    int(T),
                    self._direct_per_batch_sum,
                ],
                device=self._device,
            )
            wp.launch(
                kernel=_accumulate_collision_direct_cost_kernel,
                dim=B,
                inputs=[self._direct_per_batch_sum, float(weight_over_B), out_cost],
                device=self._device,
            )
            return
        if cost_form == "sum_squared":
            self._compute_weighted_cost_and_gradient_jtr(
                var_values, out_cost=out_cost, out_gradient=out_gradient, col_offset=col_offset
            )
            return
        B, T = self._ensure_direct_state(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        self._query_direct_sdf(var, B, T)
        spec_t = var.spec_tensors
        pen_weight = float(self._residual_weight_np[0]) if self._residual_weight_np is not None else 1.0
        weight_over_B = pen_weight / float(B)
        self._direct_per_batch_sum.zero_()
        wp.launch(
            kernel=_compute_collision_direct_per_batch_sum_kernel,
            dim=(B, T, self.residual_dim),
            inputs=[
                self._direct_signed_dists,
                spec_t.collision_sphere_radii,
                self._sphere_indices_wp,
                float(self.margin),
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                int(T),
                self._direct_per_batch_sum,
            ],
            device=self._device,
        )
        wp.launch(
            kernel=_accumulate_collision_direct_cost_kernel,
            dim=B,
            inputs=[self._direct_per_batch_sum, float(weight_over_B), out_cost],
            device=self._device,
        )
        single_tangent_dim = var.tangent_dim // T
        BT = B * T
        T_world_base_arg = var.T_world_base.reshape((BT,)) if var.has_floating_base else self._direct_dummy_T_world_base
        skip_endpoints = 1 if (var.is_trajectory and self.skip_trajectory_endpoints and T > 2) else 0
        wp.launch(
            kernel=_compute_collision_direct_gradient_kernel,
            dim=(B, T, self.residual_dim, single_tangent_dim),
            inputs=[
                var.S_world.reshape((BT, 6, var.spec.num_joints)),
                T_world_base_arg,
                var.collision_sphere_centers_world.reshape((BT, self._num_spheres)),
                spec_t.collision_sphere_radii,
                self._sphere_indices_wp,
                spec_t.collision_spheres_link_indices,
                spec_t.link_ancestor_joints_mask,
                spec_t.joints_to_actuated_mapping,
                self._direct_signed_dists,
                self._direct_normals,
                var.has_floating_base,
                float(self.margin),
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                float(2.0 * weight_over_B),
                self._direct_per_batch_sum,
                int(T),
                int(single_tangent_dim),
                int(skip_endpoints),
                int(col_offset),
            ],
            outputs=[out_gradient],
            device=self._device,
        )

    def _compute_weighted_cost_and_gradient_jtr(
        self,
        var_values: VarValues,
        out_cost: wp.array,
        out_gradient: wp.array,
        col_offset: int = 0,
    ):
        """Compute the per-pose Gauss-Newton cost and `Jᵀr`."""
        var = var_values.get(self.var_key)
        if var.is_trajectory:
            raise NotImplementedError("cost_form='sum_squared' requires per-pose state.")
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        if self._jtr_residual_buf is None or self._jtr_residual_buf.shape[0] != var.batch_size:
            self._jtr_residual_buf = wp.empty(
                (var.batch_size, self.residual_dim), dtype=wp.float32, device=var.q.device
            )
        self.compute_weighted_residual(var_values, out_residual=self._jtr_residual_buf)
        spec_t = var.spec_tensors
        wp.launch(
            kernel=_compute_collision_jtr_cost_and_gradient_kernel,
            dim=(var.batch_size, self.residual_dim, var.tangent_dim),
            inputs=[
                var.S_world,
                var.T_world_base.flatten(),
                var.collision_sphere_centers_world,
                spec_t.collision_sphere_radii,
                self._sphere_indices_wp,
                spec_t.collision_spheres_link_indices,
                spec_t.link_ancestor_joints_mask,
                spec_t.joints_to_actuated_mapping,
                self._raw_sdf,
                self._raw_normals,
                var.has_floating_base,
                self._residual_weight_wp,
                self._jtr_residual_buf,
                self.margin,
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                col_offset,
            ],
            outputs=[out_cost, out_gradient],
            device=self._device,
        )


__all__ = ["SceneCollisionTask"]
