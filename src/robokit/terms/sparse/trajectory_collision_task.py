# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportCallIssue=false
# pyright: reportUndefinedVariable=false
# pyright: reportOptionalSubscript=false
"""Sparse trajectory extension of `SceneCollisionTask` with optional swept sampling."""

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.geom import MeshGeom, VolumeGeom
from robokit.geom.sdf_kernels import (
    closest_on_meshes,
    closest_on_meshes_with_sdf,
    closest_on_primitives,
    closest_on_volumes,
)
from robokit.terms.robot_task import RobotTask
from robokit.terms.sphere_collision_kernels import (
    PENALTY_PLAIN,
    PENALTY_SMOOTH_ROOT,
    PENALTY_SURFACE_DISTANCE,
    _compute_collision_penalty_func,
)
from robokit.terms.task import (
    GradientTask,
    SparseTask,
    SparsityPattern,
    _sparse_cost_deterministic_kernel,
    _sparse_gradient_csc_kernel,
)
from robokit.utils.warp_utils import wp_device_type, wp_vec7


if TYPE_CHECKING:
    from robokit.geom import BaseGeom, WarpScene
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot


# --- device code ------------------------------------------------------------
@wp.func
def _transform_point_se3_func(T_world_link: wp_vec7, local_point: wp.vec3) -> wp.vec3:
    pos = wp.vec3(T_world_link[0], T_world_link[1], T_world_link[2])
    quat = wp.quat(T_world_link[4], T_world_link[5], T_world_link[6], T_world_link[3])
    return pos + wp.quat_rotate(quat, local_point)


@wp.func
def _compute_cached_gradient_func(
    world_center: wp.vec3, hit_dist: float, hit_closest: wp.vec3, hit_normal: wp.vec3
) -> wp.vec3:
    """Compute the signed closest-point direction, with the hit normal as fallback."""
    diff = world_center - hit_closest
    dlen = wp.length(diff)
    if dlen > 1.0e-8:
        if hit_dist < 0.0:
            return diff / dlen * (-1.0)
        return diff / dlen
    return hit_normal


@wp.kernel
def zero_collision_residual_kernel(
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, residual_idx = wp.tid()
    out_residual[batch_idx, row_offset + residual_idx] = 0.0


@wp.kernel
def zero_collision_jacobian_values_kernel(
    nnz_offset: int,
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, nnz_idx = wp.tid()
    values[batch_idx, nnz_offset + nnz_idx] = 0.0


@wp.kernel
def primitive_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    prim_type: wp.int32,
    primitive_params: wp.array1d(dtype=wp.vec4),
    primitive_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    primitive_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    primitive_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    penalty_mode: wp.int32,
    row_offset: int,
    num_spheres: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    radius = collision_sphere_radii[sphere_global_idx]
    world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    hit = closest_on_primitives(
        world_center,
        float(1e6),
        prim_type,
        primitive_params,
        primitive_inv_poses,
        enable_inv_poses,
        primitive_scales,
        enable_scales,
        primitive_offsets[si],
        primitive_offsets[si + 1],
        False,
    )

    margin_idx = batch_idx // (T_world_link.shape[0] // margin.shape[0])
    residual = _compute_collision_penalty_func(hit.dist - radius, margin[margin_idx], penalty_mode)[0]
    residual_idx = frame_idx * num_spheres + sphere_local_idx
    out_residual[batch_idx, row_offset + residual_idx] = residual_weight[residual_idx] * residual


@wp.kernel
def primitive_swept_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    prim_type: wp.int32,
    primitive_params: wp.array1d(dtype=wp.vec4),
    primitive_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    primitive_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    primitive_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    sweep_steps: int,
    penalty_mode: wp.int32,
    row_offset: int,
    num_spheres: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    # keep the largest residual across interpolated positions
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    radius = collision_sphere_radii[sphere_global_idx]
    center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)
    prev_frame = wp.max(frame_idx - 1, 0)
    prev_center = _transform_point_se3_func(T_world_link[batch_idx, prev_frame, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    margin_idx = batch_idx // (T_world_link.shape[0] // margin.shape[0])
    eta = margin[margin_idx]
    residual = float(0.0)
    for s in range(sweep_steps + 1):
        t = float(s + 1) / float(sweep_steps + 1)
        p = prev_center + t * (center - prev_center)
        hit = closest_on_primitives(
            p,
            float(1e6),
            prim_type,
            primitive_params,
            primitive_inv_poses,
            enable_inv_poses,
            primitive_scales,
            enable_scales,
            primitive_offsets[si],
            primitive_offsets[si + 1],
            False,
        )
        residual = wp.max(residual, _compute_collision_penalty_func(hit.dist - radius, eta, penalty_mode)[0])
    residual_idx = frame_idx * num_spheres + sphere_local_idx
    # per-sphere, not per-row: this row scores the whole segment into `frame_idx`, so the
    # free_goal_frame zeros on the last row would drop that segment rather than just the goal pose
    out_residual[batch_idx, row_offset + residual_idx] = residual_weight[sphere_local_idx] * residual


@wp.kernel
def primitive_jacobian_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    prim_type: wp.int32,
    primitive_params: wp.array1d(dtype=wp.vec4),
    primitive_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    primitive_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    primitive_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    num_spheres: int,
    cache_world_center: wp.array1d(dtype=wp.vec3),
    cache_sdf_normal: wp.array1d(dtype=wp.vec4),
):
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    hit = closest_on_primitives(
        world_center,
        float(1e6),
        prim_type,
        primitive_params,
        primitive_inv_poses,
        enable_inv_poses,
        primitive_scales,
        enable_scales,
        primitive_offsets[si],
        primitive_offsets[si + 1],
        True,
    )
    grad = _compute_cached_gradient_func(world_center, hit.dist, hit.closest, hit.normal)

    cache_idx = (batch_idx * T_world_link.shape[1] + frame_idx) * num_spheres + sphere_local_idx
    cache_world_center[cache_idx] = world_center
    cache_sdf_normal[cache_idx] = wp.vec4(hit.dist, grad[0], grad[1], grad[2])


@wp.kernel
def mesh_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    mesh_aabb_min: wp.array1d(dtype=wp.vec3),
    mesh_aabb_max: wp.array1d(dtype=wp.vec3),
    mesh_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    mesh_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    max_dist: float,
    margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    penalty_mode: wp.int32,
    row_offset: int,
    num_spheres: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    radius = collision_sphere_radii[sphere_global_idx]
    world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    hit = closest_on_meshes(
        world_center,
        float(1e6),
        mesh_ids,
        mesh_aabb_min,
        mesh_aabb_max,
        mesh_inv_poses,
        enable_inv_poses,
        mesh_scales,
        enable_scales,
        mesh_offsets[si],
        mesh_offsets[si + 1],
        max_dist,
        False,
    )

    margin_idx = batch_idx // (T_world_link.shape[0] // margin.shape[0])
    residual = _compute_collision_penalty_func(hit.dist - radius, margin[margin_idx], penalty_mode)[0]
    residual_idx = frame_idx * num_spheres + sphere_local_idx
    out_residual[batch_idx, row_offset + residual_idx] = residual_weight[residual_idx] * residual


@wp.kernel
def mesh_jacobian_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    mesh_aabb_min: wp.array1d(dtype=wp.vec3),
    mesh_aabb_max: wp.array1d(dtype=wp.vec3),
    mesh_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    mesh_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    max_dist: float,
    num_spheres: int,
    cache_world_center: wp.array1d(dtype=wp.vec3),
    cache_sdf_normal: wp.array1d(dtype=wp.vec4),
):
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    hit = closest_on_meshes(
        world_center,
        float(1e6),
        mesh_ids,
        mesh_aabb_min,
        mesh_aabb_max,
        mesh_inv_poses,
        enable_inv_poses,
        mesh_scales,
        enable_scales,
        mesh_offsets[si],
        mesh_offsets[si + 1],
        max_dist,
        True,
    )
    grad = _compute_cached_gradient_func(world_center, hit.dist, hit.closest, hit.normal)

    cache_idx = (batch_idx * T_world_link.shape[1] + frame_idx) * num_spheres + sphere_local_idx
    cache_world_center[cache_idx] = world_center
    cache_sdf_normal[cache_idx] = wp.vec4(hit.dist, grad[0], grad[1], grad[2])


@wp.kernel
def volume_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    volume_ids: wp.array1d(dtype=wp.uint64),
    volume_aabb_min: wp.array1d(dtype=wp.vec3),
    volume_aabb_max: wp.array1d(dtype=wp.vec3),
    volume_paddings: wp.array1d(dtype=wp.float32),
    volume_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    volume_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    volume_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    penalty_mode: wp.int32,
    row_offset: int,
    num_spheres: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    radius = collision_sphere_radii[sphere_global_idx]
    world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    hit = closest_on_volumes(
        world_center,
        float(1e6),
        volume_ids,
        volume_aabb_min,
        volume_aabb_max,
        volume_paddings,
        volume_inv_poses,
        enable_inv_poses,
        volume_scales,
        enable_scales,
        volume_offsets[si],
        volume_offsets[si + 1],
        False,
    )

    margin_idx = batch_idx // (T_world_link.shape[0] // margin.shape[0])
    residual = _compute_collision_penalty_func(hit.dist - radius, margin[margin_idx], penalty_mode)[0]
    residual_idx = frame_idx * num_spheres + sphere_local_idx
    out_residual[batch_idx, row_offset + residual_idx] = residual_weight[residual_idx] * residual


@wp.kernel
def volume_jacobian_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    volume_ids: wp.array1d(dtype=wp.uint64),
    volume_aabb_min: wp.array1d(dtype=wp.vec3),
    volume_aabb_max: wp.array1d(dtype=wp.vec3),
    volume_paddings: wp.array1d(dtype=wp.float32),
    volume_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    volume_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    volume_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    num_spheres: int,
    cache_world_center: wp.array1d(dtype=wp.vec3),
    cache_sdf_normal: wp.array1d(dtype=wp.vec4),
):
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    hit = closest_on_volumes(
        world_center,
        float(1e6),
        volume_ids,
        volume_aabb_min,
        volume_aabb_max,
        volume_paddings,
        volume_inv_poses,
        enable_inv_poses,
        volume_scales,
        enable_scales,
        volume_offsets[si],
        volume_offsets[si + 1],
        True,
    )
    grad = _compute_cached_gradient_func(world_center, hit.dist, hit.closest, hit.normal)

    cache_idx = (batch_idx * T_world_link.shape[1] + frame_idx) * num_spheres + sphere_local_idx
    cache_world_center[cache_idx] = world_center
    cache_sdf_normal[cache_idx] = wp.vec4(hit.dist, grad[0], grad[1], grad[2])


@wp.kernel
def hybrid_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    mesh_aabb_min: wp.array1d(dtype=wp.vec3),
    mesh_aabb_max: wp.array1d(dtype=wp.vec3),
    volume_ids: wp.array1d(dtype=wp.uint64),
    volume_paddings: wp.array1d(dtype=wp.float32),
    refine_band: float,
    mesh_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    mesh_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    max_dist: float,
    margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    penalty_mode: wp.int32,
    row_offset: int,
    num_spheres: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    radius = collision_sphere_radii[sphere_global_idx]
    world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    hit = closest_on_meshes_with_sdf(
        world_center,
        float(1e6),
        mesh_ids,
        mesh_aabb_min,
        mesh_aabb_max,
        volume_ids,
        volume_paddings,
        refine_band,
        max_dist,
        mesh_inv_poses,
        enable_inv_poses,
        mesh_scales,
        enable_scales,
        mesh_offsets[si],
        mesh_offsets[si + 1],
        False,
    )

    margin_idx = batch_idx // (T_world_link.shape[0] // margin.shape[0])
    residual = _compute_collision_penalty_func(hit.dist - radius, margin[margin_idx], penalty_mode)[0]
    residual_idx = frame_idx * num_spheres + sphere_local_idx
    out_residual[batch_idx, row_offset + residual_idx] = residual_weight[residual_idx] * residual


@wp.kernel
def hybrid_jacobian_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    mesh_aabb_min: wp.array1d(dtype=wp.vec3),
    mesh_aabb_max: wp.array1d(dtype=wp.vec3),
    volume_ids: wp.array1d(dtype=wp.uint64),
    volume_paddings: wp.array1d(dtype=wp.float32),
    refine_band: float,
    mesh_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    mesh_offsets: wp.array1d(dtype=wp.int32),
    scene_indices: wp.array1d(dtype=wp.int32),
    use_scene_indices: int,
    scene_batch_size: int,
    max_dist: float,
    num_spheres: int,
    cache_world_center: wp.array1d(dtype=wp.vec3),
    cache_sdf_normal: wp.array1d(dtype=wp.vec4),
):
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    local_center = local_collision_sphere_centers[sphere_global_idx]
    world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)

    if use_scene_indices == 1:
        si = scene_indices[batch_idx // (T_world_link.shape[0] // scene_batch_size)]
    else:
        si = 0
    hit = closest_on_meshes_with_sdf(
        world_center,
        float(1e6),
        mesh_ids,
        mesh_aabb_min,
        mesh_aabb_max,
        volume_ids,
        volume_paddings,
        refine_band,
        max_dist,
        mesh_inv_poses,
        enable_inv_poses,
        mesh_scales,
        enable_scales,
        mesh_offsets[si],
        mesh_offsets[si + 1],
        True,
    )
    grad = _compute_cached_gradient_func(world_center, hit.dist, hit.closest, hit.normal)

    cache_idx = (batch_idx * T_world_link.shape[1] + frame_idx) * num_spheres + sphere_local_idx
    cache_world_center[cache_idx] = world_center
    cache_sdf_normal[cache_idx] = wp.vec4(hit.dist, grad[0], grad[1], grad[2])


@wp.kernel
def compute_traj_collision_jacobian_pattern_kernel(
    num_spheres: int,
    num_actuated: int,
    single_tangent_dim: int,
    base_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    frame_idx, sphere_local_idx, col_local_idx = wp.tid()

    residual_idx = row_offset + frame_idx * num_spheres + sphere_local_idx
    cols_per_residual = num_actuated + base_dim
    nnz_idx = nnz_offset + (frame_idx * num_spheres + sphere_local_idx) * cols_per_residual + col_local_idx
    row_indices[nnz_idx] = residual_idx

    if col_local_idx < num_actuated:
        col_indices[nnz_idx] = frame_idx * single_tangent_dim + base_dim + col_local_idx
    else:
        base_col = col_local_idx - num_actuated
        col_indices[nnz_idx] = frame_idx * single_tangent_dim + base_col


@wp.kernel
def compute_traj_collision_jacobian_cached_kernel(
    S_world: wp.array4d(dtype=wp.float32),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    col_joint_starts: wp.array1d(dtype=wp.int32),
    col_joint_indices: wp.array1d(dtype=wp.int32),
    col_joint_weights: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    active_cols: wp.array1d(dtype=wp.int32),
    cache_world_center: wp.array1d(dtype=wp.vec3),
    cache_sdf_normal: wp.array1d(dtype=wp.vec4),
    margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    penalty_mode: wp.int32,
    num_spheres: int,
    num_actuated: int,
    n_active: int,
    base_dim: int,
    nnz_offset: int,
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, sphere_local_idx, col_grid_idx = wp.tid()

    # map the active-column grid onto real Jacobian columns
    if col_grid_idx < n_active:
        col_local_idx = active_cols[col_grid_idx]
    else:
        col_local_idx = num_actuated + (col_grid_idx - n_active)

    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    radius = collision_sphere_radii[sphere_global_idx]

    cache_idx = (batch_idx * S_world.shape[1] + frame_idx) * num_spheres + sphere_local_idx
    world_center = cache_world_center[cache_idx]
    sdf_normal = cache_sdf_normal[cache_idx]

    signed_dist = sdf_normal[0]
    normal = wp.vec3(sdf_normal[1], sdf_normal[2], sdf_normal[3])

    margin_idx = batch_idx // (S_world.shape[0] // margin.shape[0])
    dres_dsd = _compute_collision_penalty_func(signed_dist - radius, margin[margin_idx], penalty_mode)[1]

    point_velocity_x = float(0.0)
    point_velocity_y = float(0.0)
    point_velocity_z = float(0.0)

    if col_local_idx < num_actuated:
        actuated_idx = col_local_idx

        linear_velocity_x = float(0.0)
        linear_velocity_y = float(0.0)
        linear_velocity_z = float(0.0)
        angular_velocity_x = float(0.0)
        angular_velocity_y = float(0.0)
        angular_velocity_z = float(0.0)

        # visit only joints mapped to this actuated column
        col_start = col_joint_starts[actuated_idx]
        col_end = col_joint_starts[actuated_idx + 1]
        for k in range(col_start, col_end):
            joint_idx = col_joint_indices[k]
            if link_ancestor_joints_mask[link_idx, joint_idx]:
                weight = col_joint_weights[k]
                linear_velocity_x = linear_velocity_x + S_world[batch_idx, frame_idx, 0, joint_idx] * weight
                linear_velocity_y = linear_velocity_y + S_world[batch_idx, frame_idx, 1, joint_idx] * weight
                linear_velocity_z = linear_velocity_z + S_world[batch_idx, frame_idx, 2, joint_idx] * weight
                angular_velocity_x = angular_velocity_x + S_world[batch_idx, frame_idx, 3, joint_idx] * weight
                angular_velocity_y = angular_velocity_y + S_world[batch_idx, frame_idx, 4, joint_idx] * weight
                angular_velocity_z = angular_velocity_z + S_world[batch_idx, frame_idx, 5, joint_idx] * weight

        point_velocity_x = (
            linear_velocity_x + angular_velocity_y * world_center[2] - angular_velocity_z * world_center[1]
        )
        point_velocity_y = (
            linear_velocity_y + angular_velocity_z * world_center[0] - angular_velocity_x * world_center[2]
        )
        point_velocity_z = (
            linear_velocity_z + angular_velocity_x * world_center[1] - angular_velocity_y * world_center[0]
        )
    else:
        base_col = col_local_idx - num_actuated
        if base_col < 3:
            if base_col == 0:
                point_velocity_x = 1.0
            elif base_col == 1:
                point_velocity_y = 1.0
            else:
                point_velocity_z = 1.0
        else:
            axis = base_col - 3
            if axis == 0:
                point_velocity_y = world_center[2]
                point_velocity_z = -world_center[1]
            elif axis == 1:
                point_velocity_x = -world_center[2]
                point_velocity_z = world_center[0]
            else:
                point_velocity_x = world_center[1]
                point_velocity_y = -world_center[0]

    dsd_dv = normal[0] * point_velocity_x + normal[1] * point_velocity_y + normal[2] * point_velocity_z
    cols_per_residual = num_actuated + base_dim
    nnz_idx = nnz_offset + (frame_idx * num_spheres + sphere_local_idx) * cols_per_residual + col_local_idx
    values[batch_idx, nnz_idx] = residual_weight[frame_idx * num_spheres + sphere_local_idx] * dres_dsd * dsd_dv


@wp.kernel
def cached_residual_kernel(
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    cache_sdf_normal: wp.array1d(dtype=wp.vec4),
    margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    penalty_mode: wp.int32,
    num_frames: int,
    num_spheres: int,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    # reuse signed distances cached by the Jacobian pass
    batch_idx, frame_idx, sphere_local_idx = wp.tid()
    sphere_global_idx = sphere_indices[sphere_local_idx]
    radius = collision_sphere_radii[sphere_global_idx]
    cache_idx = (batch_idx * num_frames + frame_idx) * num_spheres + sphere_local_idx
    sdf_normal = cache_sdf_normal[cache_idx]
    margin_idx = batch_idx // (out_residual.shape[0] // margin.shape[0])
    residual = _compute_collision_penalty_func(sdf_normal[0] - radius, margin[margin_idx], penalty_mode)[0]
    residual_idx = frame_idx * num_spheres + sphere_local_idx
    out_residual[batch_idx, row_offset + residual_idx] = residual_weight[residual_idx] * residual


@wp.kernel
def max_penetration_kernel(
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    sphere_mask: wp.array1d(dtype=wp.float32),
    cache_sdf_normal: wp.array1d(dtype=wp.vec4),
    num_frames: int,
    num_spheres: int,
    out: wp.array1d(dtype=wp.float32),
):
    # max-reduce cached penetration after the fixed start while masking root spheres
    batch_idx, interior_idx, sphere_local_idx = wp.tid()
    frame_idx = interior_idx + 1
    radius = collision_sphere_radii[sphere_indices[sphere_local_idx]]
    cache_idx = (batch_idx * num_frames + frame_idx) * num_spheres + sphere_local_idx
    pen = sphere_mask[sphere_local_idx] * (radius - cache_sdf_normal[cache_idx][0])
    if pen > 0.0:
        wp.atomic_max(out, batch_idx, pen)


# --- task -------------------------------------------------------------------
class TrajectoryCollisionTask(RobotTask, SparseTask, GradientTask):
    """Penalize robot collision spheres across trajectory frames.

    `sweep_steps` adds primitive-scene samples between frames. `geometry` selects one
    geometry from a scene containing multiple geometry types.
    """

    def __init__(
        self,
        robot: "Robot",  # noqa: F821
        scene: "WarpScene",  # noqa: F821
        num_frames: int,
        weight: Union[float, Sequence[float]] = 20.0,
        margin: float = 0.1,
        max_dist: float = 1e6,
        sphere_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
        scene_indices: Optional[wp.array] = None,
        active_actuated_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
        sweep_steps: int = 0,
        penalty: Literal["smooth", "plain", "surface_distance"] = "plain",
        free_goal_frame: bool = False,
        geometry: Optional["BaseGeom"] = None,
    ):
        if not robot.spec.has_collision_spheres:
            raise RuntimeError("Robot has no collision spheres. Load with load_collision_spheres=True.")
        if penalty not in {"smooth", "plain", "surface_distance"}:
            raise ValueError(f"Unsupported penalty: {penalty}")
        self._geom, self._geometry = self._scene_geometry(scene, geometry)
        self.sweep_steps = sweep_steps
        self.penalty = penalty
        self.free_goal_frame = free_goal_frame
        if sweep_steps > 0:
            assert self._geom == "primitive", "sweep_steps supports primitive scenes only"

        self.robot = robot
        self._scene = scene
        # per-query buffers (scene indices, margins) broadcast over the solve batch by row count
        self._scene_batch_size = scene_indices.shape[0] if scene_indices is not None else 1
        self.num_frames = num_frames
        self.margin = margin
        self.max_dist = max_dist

        num_total_spheres = int(robot.spec.local_collision_sphere_centers.shape[0])
        if sphere_indices is None:
            self._sphere_indices_np = np.arange(num_total_spheres, dtype=np.int32)
        else:
            self._sphere_indices_np = np.asarray(sphere_indices, dtype=np.int32)
        self.num_spheres = len(self._sphere_indices_np)

        # restrict Jacobian work to optimized actuated columns
        num_actuated = robot.spec.num_actuated_joints
        if active_actuated_indices is None:
            self._active_cols_np = np.arange(num_actuated, dtype=np.int32)
        else:
            self._active_cols_np = np.asarray(active_actuated_indices, dtype=np.int32)
        self._n_active = len(self._active_cols_np)

        residual_weight = np.ones(self.num_spheres, dtype=np.float32)
        if isinstance(weight, (float, int)):
            residual_weight[:] = float(weight)
        else:
            residual_weight[:] = np.asarray(weight, dtype=np.float32)
        # optionally disable collision rows at the free goal frame
        self._residual_weight_np = np.tile(residual_weight, num_frames)
        if free_goal_frame:
            self._residual_weight_np[-self.num_spheres :] = 0.0

        if scene_indices is None:
            self._scene_indices_arg = None
            self._use_scene_indices = False
        else:
            if scene_indices.dtype != wp.int32 or scene_indices.ndim != 1:
                raise TypeError("scene_indices must be a 1D wp.int32 array.")
            self._scene_indices_arg = scene_indices
            self._use_scene_indices = True

        self.device: Optional[wp_device_type] = None
        self._sphere_indices_wp: Optional[wp.array] = None
        self.residual_weight = None
        self._scene_indices_wp: Optional[wp.array] = None
        self._margin_arg: Optional[wp.array] = None
        self._jac_cache_world_center: Optional[wp.array] = None
        self._jac_cache_sdf_normal: Optional[wp.array] = None

    @staticmethod
    def _scene_geometry(scene: "WarpScene", geometry: Optional["BaseGeom"] = None) -> Tuple[str, Optional["BaseGeom"]]:  # noqa: F821
        """Resolve the selected geometry and its query kernel family."""
        geoms = scene.geoms
        if geometry is None:
            if len(geoms) == 0:
                return "empty", None
            if len(geoms) > 1:
                raise ValueError(f"Multi-geom scene {geoms} requires geometry to pick one.")
            geometry = geoms[0]
        elif not any(g is geometry for g in geoms):
            raise ValueError(f"geometry {geometry!r} is not attached to the scene.")
        if isinstance(geometry, MeshGeom):
            return ("hybrid" if geometry.enable_sdf else "mesh"), geometry
        if isinstance(geometry, VolumeGeom):
            return "volume", geometry
        return "primitive", geometry

    @property
    def _penalty_mode(self) -> int:
        if self.penalty == "smooth":
            # this task is consumed as a residual, so it needs the root of the potential
            return PENALTY_SMOOTH_ROOT
        if self.penalty == "surface_distance":
            return PENALTY_SURFACE_DISTANCE
        return PENALTY_PLAIN

    def init_buffers(self, device: wp_device_type):
        """Build scene, sparsity, and query-cache buffers.

        Lifecycle:
            1. Upload selected sphere and actuated-column indices.
            2. Build the actuated-column joint CSR mapping.
            3. Allocate scene, margin, and Jacobian query caches.
        """
        self.device = device
        n = self._scene_batch_size * self.num_frames * self.num_spheres
        self._sphere_indices_wp = wp.from_numpy(self._sphere_indices_np, dtype=wp.int32, device=device)
        self._active_cols_wp = wp.from_numpy(self._active_cols_np, dtype=wp.int32, device=device)
        # store nonzero joint mappings for each actuated column in CSR form
        jta = self.robot.spec.joints_to_actuated_mapping  # [num_joints, num_actuated], numpy
        starts = [0]
        joint_idx_list: list[int] = []
        weight_list: list[float] = []
        for a in range(jta.shape[1]):
            nz = np.nonzero(jta[:, a])[0]
            joint_idx_list.extend(int(j) for j in nz)
            weight_list.extend(float(jta[j, a]) for j in nz)
            starts.append(len(joint_idx_list))
        self._col_joint_starts_wp = wp.from_numpy(np.asarray(starts, dtype=np.int32), dtype=wp.int32, device=device)
        self._col_joint_indices_wp = wp.from_numpy(
            np.asarray(joint_idx_list, dtype=np.int32), dtype=wp.int32, device=device
        )
        self._col_joint_weights_wp = wp.from_numpy(
            np.asarray(weight_list, dtype=np.float32), dtype=wp.float32, device=device
        )
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        movable = self.robot.spec.collision_spheres_link_indices[self._sphere_indices_np] > 0
        self._pen_sphere_mask_wp = wp.from_numpy(movable.astype(np.float32), dtype=wp.float32, device=device)
        if self._scene_indices_arg is not None:
            if self._scene_indices_arg.device != wp.get_device(device):
                raise ValueError(
                    f"scene_indices must be on {wp.get_device(device)}, got {self._scene_indices_arg.device}."
                )
            self._scene_indices_wp = wp.clone(self._scene_indices_arg)
            self._scene_batch_size = self._scene_indices_wp.shape[0]
        else:
            self._scene_indices_wp = wp.zeros(self._scene_batch_size, dtype=wp.int32, device=device)
        self._jac_cache_world_center = wp.empty((n,), dtype=wp.vec3, device=device)
        self._jac_cache_sdf_normal = wp.empty((n,), dtype=wp.vec4, device=device)
        # keep margins in stable storage for graph-safe updates
        if self._margin_arg is not None:
            if self._margin_arg.device != wp.get_device(device):
                raise ValueError(f"margin must be on {wp.get_device(device)}, got {self._margin_arg.device}.")
            self._margin_wp = wp.clone(self._margin_arg)
        else:
            self._margin_wp = wp.from_numpy(
                np.full(self._scene_batch_size, self.margin, dtype=np.float32), dtype=wp.float32, device=device
            )

    def accumulate_max_penetration(self, out: wp.array):
        """Max-accumulate penetration from the most recent Jacobian query cache."""
        if self._geom == "empty":
            return
        wp.launch(
            kernel=max_penetration_kernel,
            dim=(
                out.shape[0],
                self.num_frames - 2 if self.free_goal_frame else self.num_frames - 1,
                self.num_spheres,
            ),
            inputs=[
                self.robot.spec.get_tensors(str(self.device)).collision_sphere_radii,
                self._sphere_indices_wp,
                self._pen_sphere_mask_wp,
                self._jac_cache_sdf_normal,
                self.num_frames,
                self.num_spheres,
            ],
            outputs=[out],
            device=self.device,
        )

    def set_margin(self, margin: Union[float, wp.array]):
        """Update scalar or per-query collision margins in place without invalidating CUDA graphs."""
        if isinstance(margin, wp.array):
            if margin.dtype != wp.float32 or margin.ndim != 1:
                raise TypeError("margin must be a 1D wp.float32 array.")
            self._margin_arg = margin
            if self.device is None:
                return
            if margin.device != wp.get_device(self.device):
                raise ValueError(f"margin must be on {wp.get_device(self.device)}, got {margin.device}.")
            if margin.shape != self._margin_wp.shape:
                # resizing replaces the stable buffer; only safe before graph capture
                self._margin_wp = wp.clone(margin)
            else:
                wp.copy(self._margin_wp, margin)
        else:
            self.margin = float(margin)
            self._margin_arg = None
            if self.device is not None:
                self._margin_wp.fill_(self.margin)

    @property
    def residual_dim(self) -> int:
        return self.num_frames * self.num_spheres

    def set_scene_indices(self, scene_indices: wp.array):
        """Copy a per-query device scene mapping into the task's stable runtime buffer."""
        if scene_indices.dtype != wp.int32 or scene_indices.ndim != 1:
            raise TypeError("scene_indices must be a 1D wp.int32 array.")
        self._scene_indices_arg = scene_indices
        self._use_scene_indices = True
        if self.device is not None:
            if scene_indices.device != wp.get_device(self.device):
                raise ValueError(f"scene_indices must be on {wp.get_device(self.device)}, got {scene_indices.device}.")
            if scene_indices.shape != self._scene_indices_wp.shape:
                # resizing replaces the stable buffer; only safe before graph capture
                self._scene_indices_wp = wp.clone(scene_indices)
            else:
                wp.copy(self._scene_indices_wp, scene_indices)
            self._scene_batch_size = scene_indices.shape[0]

    def compute_weighted_cost_and_gradient(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
        **kwargs: object,
    ):
        """Compute sparse cost and optional gradient with one scene query.

        Lifecycle:
            1. Use the base cost-only path when no gradient is requested.
            2. Compute Jacobian values and cache their scene query.
            3. Rebuild residuals from the cache and reduce cost and gradient.
        """
        if out_gradient is None:
            super().compute_weighted_cost_and_gradient(var_values, *args, out_cost=out_cost, **kwargs)
            return
        self._build_sparse_gradient_buffers(var_values)
        self.compute_weighted_sparse_jacobian_values(var_values, out_jacobian_values=self._sparse_jac_buf, nnz_offset=0)
        if self._geom == "empty":
            self.compute_weighted_residual(var_values, out_residual=self._sparse_residual_buf, row_offset=0)
        else:
            spec_tensors = var_values.get("robot").spec_tensors
            wp.launch(
                kernel=cached_residual_kernel,
                dim=(var_values.batch_size, self.num_frames, self.num_spheres),
                inputs=[
                    spec_tensors.collision_sphere_radii,
                    self._sphere_indices_wp,
                    self._jac_cache_sdf_normal,
                    self._margin_wp,
                    self.residual_weight,
                    self._penalty_mode,
                    self.num_frames,
                    self.num_spheres,
                    0,
                ],
                outputs=[self._sparse_residual_buf],
                device=var_values.device,
            )
        wp.launch(
            kernel=_sparse_cost_deterministic_kernel,
            dim=var_values.batch_size,
            inputs=[self._sparse_residual_buf, out_cost, self.residual_dim],
            device=var_values.device,
        )
        col_offset = var_values.tangent_offset(self.var_key)
        wp.launch(
            kernel=_sparse_gradient_csc_kernel,
            dim=(var_values.batch_size, self._sparse_num_cols),
            inputs=[
                self._sparse_residual_buf,
                self._sparse_jac_buf,
                self._sparse_col_perm,
                self._sparse_col_nnz_starts,
                self._sparse_pattern.row_indices,
                col_offset,
            ],
            outputs=[out_gradient],
            device=var_values.device,
        )

    def fill_residual_and_jacobian(
        self,
        var_values: "VarValues",  # noqa: F821
        out_residual: wp.array,
        row_offset: int,
        out_jacobian: Optional[wp.array] = None,
        nnz_offset: int = 0,
    ):
        """Fill residuals and sparse values from one shared scene query."""
        if out_jacobian is None or self._geom == "empty":
            super().fill_residual_and_jacobian(
                var_values, out_residual, row_offset, out_jacobian=out_jacobian, nnz_offset=nnz_offset
            )
            return
        self.compute_weighted_sparse_jacobian_values(
            var_values, out_jacobian_values=out_jacobian, nnz_offset=nnz_offset
        )
        spec_tensors = var_values.get("robot").spec_tensors
        wp.launch(
            kernel=cached_residual_kernel,
            dim=(var_values.batch_size, self.num_frames, self.num_spheres),
            inputs=[
                spec_tensors.collision_sphere_radii,
                self._sphere_indices_wp,
                self._jac_cache_sdf_normal,
                self._margin_wp,
                self.residual_weight,
                self._penalty_mode,
                self.num_frames,
                self.num_spheres,
                row_offset,
            ],
            outputs=[out_residual],
            device=var_values.device,
        )

    def compute_weighted_residual(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        """Compute trajectory collision residuals.

        Lifecycle:
            1. Prepare forward kinematics and stable runtime buffers.
            2. Select the primitive, mesh, volume, or hybrid query kernel.
            3. Evaluate every frame and selected sphere.
        """
        robot_state = var_values.get("robot")
        if self.device is None:
            self.init_buffers(robot_state.q.device)
        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)

        kernel_device = out_residual.device if out_residual is not None else self.device
        spec_tensors = robot_state.spec_tensors
        actual_batch = int(robot_state.q.shape[0])
        if out_residual is None:
            out_residual = wp.empty((actual_batch, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self._geom == "empty":
            wp.launch(
                kernel=zero_collision_residual_kernel,
                dim=(actual_batch, self.residual_dim),
                inputs=[row_offset],
                outputs=[out_residual],
                device=kernel_device,
            )
            return out_residual

        dim = (actual_batch, self.num_frames, self.num_spheres)
        if self._geom == "primitive" and self.sweep_steps > 0:
            g = self._geometry
            wp.launch(
                kernel=primitive_swept_residual_kernel,
                dim=dim,
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_sphere_radii,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.TYPE_ID,
                    g.params_wp,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self._margin_wp,
                    self.residual_weight,
                    self.sweep_steps,
                    self._penalty_mode,
                    row_offset,
                    self.num_spheres,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
            return out_residual
        if self._geom == "primitive":
            g = self._geometry
            wp.launch(
                kernel=primitive_residual_kernel,
                dim=dim,
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_sphere_radii,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.TYPE_ID,
                    g.params_wp,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self._margin_wp,
                    self.residual_weight,
                    self._penalty_mode,
                    row_offset,
                    self.num_spheres,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        elif self._geom == "volume":
            g = self._geometry
            wp.launch(
                kernel=volume_residual_kernel,
                dim=dim,
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_sphere_radii,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.volume_ids_wp,
                    g.aabb_min_wp,
                    g.aabb_max_wp,
                    g.paddings_wp,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self._margin_wp,
                    self.residual_weight,
                    self._penalty_mode,
                    row_offset,
                    self.num_spheres,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        elif self._geom == "hybrid":
            g = self._geometry
            wp.launch(
                kernel=hybrid_residual_kernel,
                dim=dim,
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_sphere_radii,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.mesh_ids_wp,
                    g.aabb_min_wp,
                    g.aabb_max_wp,
                    g.volume_ids_wp,
                    g.paddings_wp,
                    g.refine_band,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self.max_dist,
                    self._margin_wp,
                    self.residual_weight,
                    self._penalty_mode,
                    row_offset,
                    self.num_spheres,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        else:
            g = self._geometry
            wp.launch(
                kernel=mesh_residual_kernel,
                dim=dim,
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_sphere_radii,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.mesh_ids_wp,
                    g.aabb_min_wp,
                    g.aabb_max_wp,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self.max_dist,
                    self._margin_wp,
                    self.residual_weight,
                    self._penalty_mode,
                    row_offset,
                    self.num_spheres,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        return out_residual

    def compute_sparse_jacobian_pattern(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> SparsityPattern:
        robot_state = var_values.get("robot")
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = robot_state.q.device
        num_actuated = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        single_tangent_dim = base_dim + num_actuated
        cols_per_residual = num_actuated + base_dim
        nnz = self.residual_dim * cols_per_residual

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=compute_traj_collision_jacobian_pattern_kernel,
            dim=(self.num_frames, self.num_spheres, cols_per_residual),
            inputs=[self.num_spheres, num_actuated, single_tangent_dim, base_dim, offset, nnz_offset],
            outputs=[row_indices, col_indices],
            device=kernel_device,
        )

        pattern = SparsityPattern()
        pattern.row_indices = row_indices
        pattern.col_indices = col_indices
        return pattern

    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        """Compute sparse trajectory collision Jacobian values.

        Lifecycle:
            1. Prepare kinematics, motion subspaces, and runtime-sized caches.
            2. Query the selected geometry and cache sphere centers and SDF normals.
            3. Assemble active joint and optional floating-base columns.
        """
        robot_state = var_values.get("robot")
        if self.device is None:
            self.init_buffers(robot_state.q.device)
        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)
        if not robot_state.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(robot_state)

        kernel_device = robot_state.q.device
        spec_tensors = robot_state.spec_tensors
        num_actuated = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        cols_per_residual = num_actuated + base_dim
        nnz = self.residual_dim * cols_per_residual
        # grow caches for the active runtime batch
        b, f, s = int(robot_state.q.shape[0]), self.num_frames, self.num_spheres
        n_cache = b * f * s
        if self._jac_cache_world_center is None or self._jac_cache_world_center.shape[0] < n_cache:
            self._jac_cache_world_center = wp.empty((n_cache,), dtype=wp.vec3, device=kernel_device)
            self._jac_cache_sdf_normal = wp.empty((n_cache,), dtype=wp.vec4, device=kernel_device)

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((b, nnz), dtype=wp.float32, device=kernel_device)

        if self._geom == "empty":
            wp.launch(
                kernel=zero_collision_jacobian_values_kernel,
                dim=(b, nnz),
                inputs=[nnz_offset],
                outputs=[out_jacobian_values],
                device=kernel_device,
            )
            return out_jacobian_values
        # leave locked columns stale because the optimizer masks them after assembly

        if self._geom == "primitive":
            g = self._geometry
            wp.launch(
                kernel=primitive_jacobian_kernel,
                dim=(b, f, s),
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.TYPE_ID,
                    g.params_wp,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self.num_spheres,
                ],
                outputs=[self._jac_cache_world_center, self._jac_cache_sdf_normal],
                device=kernel_device,
            )
        elif self._geom == "volume":
            g = self._geometry
            wp.launch(
                kernel=volume_jacobian_kernel,
                dim=(b, f, s),
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.volume_ids_wp,
                    g.aabb_min_wp,
                    g.aabb_max_wp,
                    g.paddings_wp,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self.num_spheres,
                ],
                outputs=[self._jac_cache_world_center, self._jac_cache_sdf_normal],
                device=kernel_device,
            )
        elif self._geom == "hybrid":
            g = self._geometry
            wp.launch(
                kernel=hybrid_jacobian_kernel,
                dim=(b, f, s),
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.mesh_ids_wp,
                    g.aabb_min_wp,
                    g.aabb_max_wp,
                    g.volume_ids_wp,
                    g.paddings_wp,
                    g.refine_band,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self.max_dist,
                    self.num_spheres,
                ],
                outputs=[self._jac_cache_world_center, self._jac_cache_sdf_normal],
                device=kernel_device,
            )
        else:
            g = self._geometry
            wp.launch(
                kernel=mesh_jacobian_kernel,
                dim=(b, f, s),
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    g.mesh_ids_wp,
                    g.aabb_min_wp,
                    g.aabb_max_wp,
                    g.inv_poses_wp,
                    g.enable_inv_poses,
                    g.scales_wp,
                    g.enable_scales,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    int(self._use_scene_indices),
                    self._scene_batch_size,
                    self.max_dist,
                    self.num_spheres,
                ],
                outputs=[self._jac_cache_world_center, self._jac_cache_sdf_normal],
                device=kernel_device,
            )

        wp.launch(
            kernel=compute_traj_collision_jacobian_cached_kernel,
            dim=(b, f, s, self._n_active + base_dim),
            inputs=[
                robot_state.S_world,
                spec_tensors.collision_sphere_radii,
                spec_tensors.collision_spheres_link_indices,
                spec_tensors.link_ancestor_joints_mask,
                self._col_joint_starts_wp,
                self._col_joint_indices_wp,
                self._col_joint_weights_wp,
                self._sphere_indices_wp,
                self._active_cols_wp,
                self._jac_cache_world_center,
                self._jac_cache_sdf_normal,
                self._margin_wp,
                self.residual_weight,
                self._penalty_mode,
                self.num_spheres,
                num_actuated,
                self._n_active,
                base_dim,
                nnz_offset,
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )
        return out_jacobian_values
