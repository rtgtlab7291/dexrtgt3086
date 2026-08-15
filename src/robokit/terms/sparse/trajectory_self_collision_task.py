# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportUndefinedVariable=false
"""Sparse sphere self-collision over a trajectory."""

from typing import TYPE_CHECKING, Optional

import numpy as np
import warp as wp

from robokit.terms.dense.self_collision_task import compute_active_collision_pairs
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import GradientTask, SparseTask, SparsityPattern, sparse_cost_and_gradient
from robokit.utils.warp_utils import wp_device_type, wp_vec7


# --- task -------------------------------------------------------------------
class TrajectorySelfCollisionTask(RobotTask, SparseTask, GradientTask):
    """Penalize self-collisions between robot collision spheres at all frames."""

    compute_weighted_cost_and_gradient = sparse_cost_and_gradient

    def __init__(
        self,
        robot: "Robot",  # noqa: F821
        num_frames: int,
        weight: float = 10.0,
        margin: float = 0.02,
    ):
        if not robot.spec.has_collision_spheres:
            raise RuntimeError("Robot has no collision spheres. Load with load_collision_spheres=True.")

        self.robot = robot
        self.num_frames = num_frames
        self.weight = float(weight)
        self.margin = margin

        self._active_pairs_np = compute_active_collision_pairs(robot.spec)
        ignored = {(int(a), int(b)) for a, b in robot.spec.self_collision_ignored_pairs}
        if ignored and len(self._active_pairs_np) > 0:
            ignored_both = ignored | {(j, i) for i, j in ignored}
            geom_link = robot.spec.collision_spheres_link_indices
            keep = [(int(geom_link[i]), int(geom_link[j])) not in ignored_both for i, j in self._active_pairs_np]
            self._active_pairs_np = self._active_pairs_np[np.asarray(keep, dtype=bool)]
        self.num_pairs = len(self._active_pairs_np)

        self.device: Optional[wp_device_type] = None
        self._active_pairs_wp: Optional[wp.array] = None

    def init_buffers(self, device: wp_device_type):
        self.device = device
        if self.num_pairs > 0:
            self._active_pairs_wp = wp.from_numpy(self._active_pairs_np, dtype=wp.int32, device=device)
        else:
            self._active_pairs_wp = wp.zeros((1, 2), dtype=wp.int32, device=device)

    @property
    def residual_dim(self) -> int:
        return self.num_frames * max(1, self.num_pairs)

    def compute_weighted_residual(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get("robot")
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)

        kernel_device = out_residual.device if out_residual is not None else self.device
        spec_tensors = robot_state.spec_tensors

        if out_residual is None:
            out_residual = wp.zeros((robot_state.q.shape[0], self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self.num_pairs == 0:
            return out_residual

        wp.launch(
            kernel=compute_self_collision_residual_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, self.num_pairs),
            inputs=[
                robot_state.T_world_link,
                spec_tensors.local_collision_sphere_centers,
                spec_tensors.collision_sphere_radii,
                spec_tensors.collision_spheres_link_indices,
                self._active_pairs_wp,
                self.margin,
                self.weight,
                row_offset,
                self.num_pairs,
                self.use_mask,
                self._get_frame_mask(),
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

        effective_num_pairs = max(1, self.num_pairs)
        cols_per_residual = num_actuated + base_dim
        nnz = self.num_frames * effective_num_pairs * cols_per_residual

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=compute_self_collision_jacobian_pattern_kernel,
            dim=(self.num_frames, effective_num_pairs, cols_per_residual),
            inputs=[
                effective_num_pairs,
                num_actuated,
                single_tangent_dim,
                base_dim,
                offset,
                nnz_offset,
            ],
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
        effective_num_pairs = max(1, self.num_pairs)
        cols_per_residual = num_actuated + base_dim
        nnz = self.num_frames * effective_num_pairs * cols_per_residual

        if out_jacobian_values is None:
            out_jacobian_values = wp.zeros((robot_state.q.shape[0], nnz), dtype=wp.float32, device=kernel_device)

        if self.num_pairs == 0:
            return out_jacobian_values

        wp.launch(
            kernel=compute_self_collision_jacobian_values_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, self.num_pairs, cols_per_residual),
            inputs=[
                robot_state.S_world,
                robot_state.T_world_link,
                spec_tensors.local_collision_sphere_centers,
                spec_tensors.collision_sphere_radii,
                spec_tensors.collision_spheres_link_indices,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                self._active_pairs_wp,
                self.margin,
                self.weight,
                self.num_pairs,
                num_actuated,
                base_dim,
                nnz_offset,
                self.use_mask,
                self._get_frame_mask(),
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )

        return out_jacobian_values


# --- device code ------------------------------------------------------------
@wp.func
def _transform_point_se3_func(T_world_link: wp_vec7, local_point: wp.vec3) -> wp.vec3:
    """Transform a link-local point to the world frame."""
    pos = wp.vec3(T_world_link[0], T_world_link[1], T_world_link[2])
    quat = wp.quat(T_world_link[4], T_world_link[5], T_world_link[6], T_world_link[3])
    return pos + wp.quat_rotate(quat, local_point)


@wp.kernel
def compute_self_collision_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),  # [batch, frames, num_links]
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    active_pairs: wp.array2d(dtype=wp.int32),  # [num_pairs, 2]
    margin: float,
    weight: float,
    row_offset: int,
    num_pairs: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, pair_idx = wp.tid()

    sphere_i = active_pairs[pair_idx, 0]
    sphere_j = active_pairs[pair_idx, 1]

    link_i = collision_spheres_link_indices[sphere_i]
    link_j = collision_spheres_link_indices[sphere_j]

    local_center_i = local_collision_sphere_centers[sphere_i]
    local_center_j = local_collision_sphere_centers[sphere_j]
    radius_i = collision_sphere_radii[sphere_i]
    radius_j = collision_sphere_radii[sphere_j]

    T_world_i = T_world_link[batch_idx, frame_idx, link_i]
    T_world_j = T_world_link[batch_idx, frame_idx, link_j]

    world_center_i = _transform_point_se3_func(T_world_i, local_center_i)
    world_center_j = _transform_point_se3_func(T_world_j, local_center_j)

    diff = world_center_i - world_center_j
    dist = wp.length(diff)

    surface_dist = dist - radius_i - radius_j
    penetration = wp.max(0.0, margin - surface_dist)
    mask = wp.float32(1.0)
    if use_mask:
        mask = wp.float32(frame_mask[batch_idx, frame_idx])

    residual_idx = frame_idx * num_pairs + pair_idx
    out_residual[batch_idx, row_offset + residual_idx] = weight * penetration * mask


@wp.kernel
def compute_self_collision_jacobian_pattern_kernel(
    num_pairs: int,
    num_actuated: int,
    single_tangent_dim: int,
    base_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    frame_idx, pair_idx, col_local_idx = wp.tid()

    residual_idx = row_offset + frame_idx * num_pairs + pair_idx
    cols_per_residual = num_actuated + base_dim
    nnz_idx = nnz_offset + (frame_idx * num_pairs + pair_idx) * cols_per_residual + col_local_idx

    row_indices[nnz_idx] = residual_idx

    if col_local_idx < num_actuated:
        col_indices[nnz_idx] = frame_idx * single_tangent_dim + base_dim + col_local_idx
    else:
        base_col = col_local_idx - num_actuated
        col_indices[nnz_idx] = frame_idx * single_tangent_dim + base_col


@wp.kernel
def compute_self_collision_jacobian_values_kernel(
    S_world: wp.array4d(dtype=wp.float32),  # [batch, frames, 6, num_joints]
    T_world_link: wp.array3d(dtype=wp_vec7),  # [batch, frames, num_links]
    local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    active_pairs: wp.array2d(dtype=wp.int32),
    margin: float,
    weight: float,
    num_pairs: int,
    num_actuated: int,
    base_dim: int,
    nnz_offset: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, pair_idx, col_local_idx = wp.tid()

    sphere_i = active_pairs[pair_idx, 0]
    sphere_j = active_pairs[pair_idx, 1]

    link_i = collision_spheres_link_indices[sphere_i]
    link_j = collision_spheres_link_indices[sphere_j]

    local_center_i = local_collision_sphere_centers[sphere_i]
    local_center_j = local_collision_sphere_centers[sphere_j]
    radius_i = collision_sphere_radii[sphere_i]
    radius_j = collision_sphere_radii[sphere_j]

    T_world_i = T_world_link[batch_idx, frame_idx, link_i]
    T_world_j = T_world_link[batch_idx, frame_idx, link_j]

    world_center_i = _transform_point_se3_func(T_world_i, local_center_i)
    world_center_j = _transform_point_se3_func(T_world_j, local_center_j)

    diff = world_center_i - world_center_j
    dist = wp.length(diff)
    surface_dist = dist - radius_i - radius_j

    dres_ddist = float(0.0)
    if (margin - surface_dist) > 0.0:
        dres_ddist = -1.0

    denom = wp.max(wp.float32(1e-8), dist)
    normal = diff / denom

    pv_i_x = float(0.0)
    pv_i_y = float(0.0)
    pv_i_z = float(0.0)
    pv_j_x = float(0.0)
    pv_j_y = float(0.0)
    pv_j_z = float(0.0)

    if col_local_idx < num_actuated:
        actuated_idx = col_local_idx
        num_joints = S_world.shape[3]

        for joint_idx in range(num_joints):
            jta_weight = joints_to_actuated[joint_idx, actuated_idx]
            if jta_weight != 0.0:
                if link_ancestor_joints_mask[link_i, joint_idx]:
                    v_x = S_world[batch_idx, frame_idx, 0, joint_idx] * jta_weight
                    v_y = S_world[batch_idx, frame_idx, 1, joint_idx] * jta_weight
                    v_z = S_world[batch_idx, frame_idx, 2, joint_idx] * jta_weight
                    w_x = S_world[batch_idx, frame_idx, 3, joint_idx] * jta_weight
                    w_y = S_world[batch_idx, frame_idx, 4, joint_idx] * jta_weight
                    w_z = S_world[batch_idx, frame_idx, 5, joint_idx] * jta_weight
                    pv_i_x = pv_i_x + v_x + w_y * world_center_i[2] - w_z * world_center_i[1]
                    pv_i_y = pv_i_y + v_y + w_z * world_center_i[0] - w_x * world_center_i[2]
                    pv_i_z = pv_i_z + v_z + w_x * world_center_i[1] - w_y * world_center_i[0]

                if link_ancestor_joints_mask[link_j, joint_idx]:
                    v_x = S_world[batch_idx, frame_idx, 0, joint_idx] * jta_weight
                    v_y = S_world[batch_idx, frame_idx, 1, joint_idx] * jta_weight
                    v_z = S_world[batch_idx, frame_idx, 2, joint_idx] * jta_weight
                    w_x = S_world[batch_idx, frame_idx, 3, joint_idx] * jta_weight
                    w_y = S_world[batch_idx, frame_idx, 4, joint_idx] * jta_weight
                    w_z = S_world[batch_idx, frame_idx, 5, joint_idx] * jta_weight
                    pv_j_x = pv_j_x + v_x + w_y * world_center_j[2] - w_z * world_center_j[1]
                    pv_j_y = pv_j_y + v_y + w_z * world_center_j[0] - w_x * world_center_j[2]
                    pv_j_z = pv_j_z + v_z + w_x * world_center_j[1] - w_y * world_center_j[0]
    else:
        base_col = col_local_idx - num_actuated
        if base_col < 3:
            if base_col == 0:
                pv_i_x = 1.0
                pv_j_x = 1.0
            elif base_col == 1:
                pv_i_y = 1.0
                pv_j_y = 1.0
            else:
                pv_i_z = 1.0
                pv_j_z = 1.0
        else:
            axis = base_col - 3
            if axis == 0:
                pv_i_y = world_center_i[2]
                pv_i_z = -world_center_i[1]
                pv_j_y = world_center_j[2]
                pv_j_z = -world_center_j[1]
            elif axis == 1:
                pv_i_x = -world_center_i[2]
                pv_i_z = world_center_i[0]
                pv_j_x = -world_center_j[2]
                pv_j_z = world_center_j[0]
            else:
                pv_i_x = world_center_i[1]
                pv_i_y = -world_center_i[0]
                pv_j_x = world_center_j[1]
                pv_j_y = -world_center_j[0]

    rel_vel_x = pv_i_x - pv_j_x
    rel_vel_y = pv_i_y - pv_j_y
    rel_vel_z = pv_i_z - pv_j_z

    ddist_dq = normal[0] * rel_vel_x + normal[1] * rel_vel_y + normal[2] * rel_vel_z
    mask = wp.float32(1.0)
    if use_mask:
        mask = wp.float32(frame_mask[batch_idx, frame_idx])

    cols_per_residual = num_actuated + base_dim
    nnz_idx = nnz_offset + (frame_idx * num_pairs + pair_idx) * cols_per_residual + col_local_idx
    values[batch_idx, nnz_idx] = weight * dres_ddist * ddist_dq * mask


if TYPE_CHECKING:
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot
