# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import warp as wp
from jaxtyping import Float
from torch.autograd import Function

from robokit.lie.warp_se3_kernels import (
    se3_exp_map_func,
    se3_exp_map_to_matrix_func,
    se3_inverse_func,
    se3_multiply_func,
)
from robokit.utils.warp_utils import wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_apply_func, quaternion_to_matrix_func


if TYPE_CHECKING:
    from robokit.robo.robot_spec import RobotSpec


@wp.func
def _forward_kinematics_joints_func(
    actuated_joint_positions: wp.array1d(dtype=wp.float32),  # [num_actuated_joints]
    T_world_base: wp_vec7,
    actuated_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    mimic_multipliers: wp.array1d(dtype=wp.float32),  # [num_joints]
    mimic_offsets: wp.array1d(dtype=wp.float32),  # [num_joints]
    joint_twists: wp.array1d(dtype=wp_vec6),  # [num_joints]
    parent_joint_transforms: wp.array1d(dtype=wp_vec7),  # [num_joints]
    topological_order_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    T_world_joint: wp.array1d(dtype=wp_vec7),  # [num_joints]
):
    num_joints = joint_twists.shape[0]
    for i in range(num_joints):
        joint_idx = topological_order_joint_indices[i]
        parent_joint_idx = parent_joint_indices[joint_idx]

        if mimic_actuated_joint_indices[joint_idx] == -1:  # not mimic
            if actuated_joint_indices[joint_idx] == -1:  # not actuated
                joint_pos = 0.0
            else:
                joint_pos = actuated_joint_positions[actuated_joint_indices[joint_idx]]
        else:
            mimic_idx = mimic_actuated_joint_indices[joint_idx]
            joint_pos = actuated_joint_positions[mimic_idx] * mimic_multipliers[joint_idx] + mimic_offsets[joint_idx]

        log_transform = joint_twists[joint_idx] * joint_pos
        T_delta = se3_exp_map_func(log_transform, wp.float(1e-4))
        T_parent_joint = se3_multiply_func(parent_joint_transforms[joint_idx], T_delta)

        if parent_joint_idx == -1:
            T_world_parent = T_world_base
        else:
            T_world_parent = T_world_joint[parent_joint_idx]

        T_world_joint[joint_idx] = se3_multiply_func(T_world_parent, T_parent_joint)


@wp.func
def _link_poses_from_joint_poses_func(
    T_world_joint: wp.array1d(dtype=wp_vec7),  # [num_joints]
    link_parent_joint_indices: wp.array1d(dtype=wp.Int),  # [num_links]
    T_world_base: wp_vec7,
    T_world_link: wp.array1d(dtype=wp_vec7),  # [num_links]
):
    num_links = link_parent_joint_indices.shape[0]
    for link_idx in range(num_links):
        joint_idx = link_parent_joint_indices[link_idx]
        if joint_idx == -1:
            T_world_link[link_idx] = T_world_base
        else:
            T_world_link[link_idx] = T_world_joint[joint_idx]


@wp.kernel
def forward_kinematics_kernel(
    actuated_joint_positions: wp.array2d(dtype=wp.Float),  # [num_instances, num_actuated_joints]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [num_instances]
    actuated_joint_indices: wp.array1d(dtype=wp.Int),  # [num_joints]
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.Int),  # [num_joints]
    mimic_multipliers: wp.array1d(dtype=wp.Float),  # [num_joints]
    mimic_offsets: wp.array1d(dtype=wp.Float),  # [num_joints]
    joint_twists: wp.array1d(dtype=wp_vec6),  # [num_joints]
    parent_joint_transforms: wp.array1d(dtype=wp_vec7),  # [num_joints]
    topological_order_joint_indices: wp.array1d(dtype=wp.Int),  # [num_joints]
    parent_joint_indices: wp.array1d(dtype=wp.Int),  # [num_joints]
    link_parent_joint_indices: wp.array1d(dtype=wp.Int),  # [num_links]
    T_world_joint: wp.array2d(dtype=wp_vec7),  # [num_instances, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [num_instances, num_links]
):
    """Warp kernel for forward kinematics. Each thread computes the forward kinematics for one instance."""
    idx = wp.tid()

    _forward_kinematics_joints_func(
        actuated_joint_positions[idx],
        T_world_base[idx],
        actuated_joint_indices,
        mimic_actuated_joint_indices,
        mimic_multipliers,
        mimic_offsets,
        joint_twists,
        parent_joint_transforms,
        topological_order_joint_indices,
        parent_joint_indices,
        T_world_joint[idx],
    )
    _link_poses_from_joint_poses_func(
        T_world_joint[idx],
        link_parent_joint_indices,
        T_world_base[idx],
        T_world_link[idx],
    )


@wp.func
def _forward_kinematics_joints_matrix_func(
    actuated_joint_positions: wp.array1d(dtype=wp.float32),  # [num_actuated_joints]
    T_world_base: wp.mat44,
    actuated_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    mimic_multipliers: wp.array1d(dtype=wp.float32),  # [num_joints]
    mimic_offsets: wp.array1d(dtype=wp.float32),  # [num_joints]
    joint_twists: wp.array1d(dtype=wp_vec6),  # [num_joints]
    parent_joint_transforms: wp.array1d(dtype=wp.mat44),  # [num_joints]
    topological_order_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    T_world_joint: wp.array1d(dtype=wp.mat44),  # [num_joints]
):
    num_joints = joint_twists.shape[0]
    for i in range(num_joints):
        joint_idx = topological_order_joint_indices[i]
        parent_joint_idx = parent_joint_indices[joint_idx]

        if mimic_actuated_joint_indices[joint_idx] == -1:  # not mimic
            if actuated_joint_indices[joint_idx] == -1:  # not actuated
                joint_pos = 0.0
            else:
                joint_pos = actuated_joint_positions[actuated_joint_indices[joint_idx]]
        else:
            mimic_idx = mimic_actuated_joint_indices[joint_idx]
            joint_pos = actuated_joint_positions[mimic_idx] * mimic_multipliers[joint_idx] + mimic_offsets[joint_idx]

        log_transform = joint_twists[joint_idx] * joint_pos
        T_delta = se3_exp_map_to_matrix_func(log_transform, wp.float(1e-4))
        T_parent_static = parent_joint_transforms[joint_idx]

        T_parent_joint = T_parent_static * T_delta

        if parent_joint_idx == -1:
            T_world_parent = T_world_base
        else:
            T_world_parent = T_world_joint[parent_joint_idx]

        T_world_joint[joint_idx] = T_world_parent * T_parent_joint


@wp.func
def _link_poses_from_joint_poses_matrix_func(
    T_world_joint: wp.array1d(dtype=wp.mat44),  # [num_joints]
    link_parent_joint_indices: wp.array1d(dtype=wp.Int),  # [num_links]
    T_world_base: wp.mat44,
    T_world_link: wp.array1d(dtype=wp.mat44),  # [num_links]
):
    num_links = link_parent_joint_indices.shape[0]
    for link_idx in range(num_links):
        joint_idx = link_parent_joint_indices[link_idx]
        if joint_idx == -1:
            T_world_link[link_idx] = T_world_base
        else:
            T_world_link[link_idx] = T_world_joint[joint_idx]


@wp.kernel
def forward_kinematics_matrix_kernel(
    actuated_joint_positions: wp.array2d(dtype=wp.Float),  # [num_instances, num_actuated_joints]
    T_world_base: wp.array1d(dtype=wp.mat44),  # [num_instances]
    actuated_joint_indices: wp.array1d(dtype=wp.Int),  # [num_joints]
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.Int),  # [num_joints]
    mimic_multipliers: wp.array1d(dtype=wp.Float),  # [num_joints]
    mimic_offsets: wp.array1d(dtype=wp.Float),  # [num_joints]
    joint_twists: wp.array1d(dtype=wp_vec6),  # [num_joints]
    parent_joint_transforms: wp.array1d(dtype=wp.mat44),  # [num_joints]
    topological_order_joint_indices: wp.array1d(dtype=wp.Int),  # [num_joints]
    parent_joint_indices: wp.array1d(dtype=wp.Int),  # [num_joints]
    link_parent_joint_indices: wp.array1d(dtype=wp.Int),  # [num_links]
    T_world_joint: wp.array2d(dtype=wp.mat44),  # [num_instances, num_joints]
    T_world_link: wp.array2d(dtype=wp.mat44),  # [num_instances, num_links]
):
    """Warp kernel for forward kinematics via matrix. Each thread computes the forward kinematics for one instance."""
    idx = wp.tid()

    _forward_kinematics_joints_matrix_func(
        actuated_joint_positions[idx],
        T_world_base[idx],
        actuated_joint_indices,
        mimic_actuated_joint_indices,
        mimic_multipliers,
        mimic_offsets,
        joint_twists,
        parent_joint_transforms,
        topological_order_joint_indices,
        parent_joint_indices,
        T_world_joint[idx],
    )
    _link_poses_from_joint_poses_matrix_func(
        T_world_joint[idx],
        link_parent_joint_indices,
        T_world_base[idx],
        T_world_link[idx],
    )


@wp.kernel
def integrate_joint_positions_kernel(
    q: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    velocity_offset: int,
    num_frames: int,
    single_tangent_dim: int,
    out_q: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    batch_idx = i // num_frames
    frame_idx = i % num_frames
    vel_idx = velocity_offset + frame_idx * single_tangent_dim + j
    out_q[i, j] = q[i, j] + velocity[batch_idx, vel_idx]


@wp.kernel
def integrate_floating_base_kernel(
    T_world_base: wp.array(dtype=wp_vec7),
    velocity: wp.array2d(dtype=wp.float32),
    num_frames: int,
    single_tangent_dim: int,
    eps: wp.float32,
    out_T_world_base: wp.array(dtype=wp_vec7),
):
    i = wp.tid()
    batch_idx = i // num_frames
    frame_idx = i % num_frames
    vel_offset = frame_idx * single_tangent_dim
    xi_base = wp_vec6(
        velocity[batch_idx, vel_offset + 0],
        velocity[batch_idx, vel_offset + 1],
        velocity[batch_idx, vel_offset + 2],
        velocity[batch_idx, vel_offset + 3],
        velocity[batch_idx, vel_offset + 4],
        velocity[batch_idx, vel_offset + 5],
    )
    T_delta = se3_exp_map_func(xi_base, eps)
    out_T_world_base[i] = se3_multiply_func(T_world_base[i], T_delta)


@wp.kernel
def integrate_se3_batch_kernel(
    T_world_ee: wp.array(dtype=wp_vec7),
    velocity: wp.array2d(dtype=wp.float32),
    ee_tangent_offset: int,
    num_ee: int,
    num_frames: int,
    eps: wp.float32,
    out_T_world_ee: wp.array(dtype=wp_vec7),
):
    """Integrate SE3 end-effector transforms using exponential map.

    Each thread handles one SE3 element (batch_idx, frame_idx, ee_idx).
    """
    tid = wp.tid()
    batch_idx = tid // (num_frames * num_ee)
    frame_idx = (tid // num_ee) % num_frames
    ee_idx = tid % num_ee

    vel_offset = ee_tangent_offset + (frame_idx * num_ee + ee_idx) * 6
    xi_ee = wp_vec6(
        velocity[batch_idx, vel_offset + 0],
        velocity[batch_idx, vel_offset + 1],
        velocity[batch_idx, vel_offset + 2],
        velocity[batch_idx, vel_offset + 3],
        velocity[batch_idx, vel_offset + 4],
        velocity[batch_idx, vel_offset + 5],
    )
    T_delta = se3_exp_map_func(xi_ee, eps)
    out_T_world_ee[tid] = se3_multiply_func(T_world_ee[tid], T_delta)


@wp.func
def _motion_subspace_transform(T_world_joint: wp_vec7, joint_twist: wp_vec6) -> wp_vec6:
    t = wp.vec3(T_world_joint[0], T_world_joint[1], T_world_joint[2])
    quat_wxyz = wp.vec4(T_world_joint[3], T_world_joint[4], T_world_joint[5], T_world_joint[6])
    linear = wp.vec3(joint_twist[0], joint_twist[1], joint_twist[2])
    angular = wp.vec3(joint_twist[3], joint_twist[4], joint_twist[5])

    angular_world = quaternion_apply_func(quat_wxyz, angular)
    linear_rotated = quaternion_apply_func(quat_wxyz, linear)
    linear_world = linear_rotated + wp.cross(t, angular_world)

    return wp_vec6(
        linear_world[0],
        linear_world[1],
        linear_world[2],
        angular_world[0],
        angular_world[1],
        angular_world[2],
    )


@wp.kernel
def compute_motion_subspace_kernel(
    T_world_joint: wp.array2d(dtype=wp_vec7),
    joint_twists: wp.array1d(dtype=wp_vec6),
    S_world: wp.array3d(dtype=wp.float32),
):
    instance, joint_idx = wp.tid()

    world_twist = _motion_subspace_transform(T_world_joint[instance, joint_idx], joint_twists[joint_idx])
    S_world[instance, 0, joint_idx] = world_twist[0]
    S_world[instance, 1, joint_idx] = world_twist[1]
    S_world[instance, 2, joint_idx] = world_twist[2]
    S_world[instance, 3, joint_idx] = world_twist[3]
    S_world[instance, 4, joint_idx] = world_twist[4]
    S_world[instance, 5, joint_idx] = world_twist[5]


@wp.func
def se3_adjoint_multiply_vec6_func(xyz_wxyz: wp_vec7, v: wp_vec6) -> wp_vec6:
    """Multiply SE(3) adjoint by a 6D vector inline without constructing the full matrix."""
    # Extract translation and quaternion
    t = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    q_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])

    # Extract linear and angular parts of input vector
    v_lin = wp.vec3(v[0], v[1], v[2])
    v_ang = wp.vec3(v[3], v[4], v[5])

    # Get rotation matrix
    R = quaternion_to_matrix_func(q_wxyz)

    # Compute result:
    # Top block: R * v_lin + [t]_x * R * v_ang
    R_v_ang = R * v_ang
    t_cross_R_v_ang = wp.cross(t, R_v_ang)
    result_lin = R * v_lin + t_cross_R_v_ang

    # Bottom block: R * v_ang
    result_ang = R_v_ang

    return wp_vec6(result_lin[0], result_lin[1], result_lin[2], result_ang[0], result_ang[1], result_ang[2])


@wp.kernel
def compute_link_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    ancestor_mask: wp.array1d(dtype=wp.bool),  # [num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    reference_frame: wp.int32,  # 0 = spatial, 1 = body
    has_floating_base: wp.bool,
    link_idx: wp.int32,
    J_out: wp.array3d(dtype=wp.float32),  # [batch, 6, total_dofs]
):
    instance, col_idx = wp.tid()
    batch = S_world.shape[0]
    total_dofs = J_out.shape[2]

    if instance >= batch or col_idx >= total_dofs:
        return

    num_joints = S_world.shape[2]
    base_dofs = 6 if has_floating_base else 0

    # Get the transform for this link
    T_world_frame = T_world_link[instance, link_idx]

    if col_idx < base_dofs:
        # Compute J_base column
        base_dof = col_idx
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[base_dof] = 1.0

        if reference_frame == 0:  # spatial frame
            # J_base = Adj(T_world_base)
            result_col = se3_adjoint_multiply_vec6_func(T_world_base[instance], unit_vec)
        else:  # body frame
            # J_base = Adj(T_frame_base) = Adj(T_frame_world * T_world_base)
            T_frame_world = se3_inverse_func(T_world_frame)
            T_frame_base = se3_multiply_func(T_frame_world, T_world_base[instance])
            result_col = se3_adjoint_multiply_vec6_func(T_frame_base, unit_vec)

        for row in range(6):
            J_out[instance, row, col_idx] = result_col[row]

    else:
        # Compute J_joints column
        actuated_idx = col_idx - base_dofs
        num_actuated = joints_to_actuated.shape[1]

        if actuated_idx >= num_actuated:
            return

        # Accumulate contribution from all joints
        spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for joint_idx in range(num_joints):
            if ancestor_mask[joint_idx]:
                weight = joints_to_actuated[joint_idx, actuated_idx]
                if weight != 0.0:
                    for row in range(6):
                        spatial_col[row] += S_world[instance, row, joint_idx] * weight

        # Apply reference frame transformation
        if reference_frame == 1:  # body frame
            # J_body = Adj(T_frame_world) * J_spatial
            T_frame_world = se3_inverse_func(T_world_frame)
            body_col = se3_adjoint_multiply_vec6_func(T_frame_world, spatial_col)
            for row in range(6):
                J_out[instance, row, col_idx] = body_col[row]
        else:  # spatial frame
            for row in range(6):
                J_out[instance, row, col_idx] = spatial_col[row]


@wp.kernel
def transform_link_points_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # [num_instances, num_links]
    local_points: wp.array1d(dtype=wp.vec3),  # [num_points]
    point_link_indices: wp.array1d(dtype=wp.int32),  # [num_points]
    world_points: wp.array2d(dtype=wp.vec3),  # [num_instances, num_points]
):
    """Transform local points attached to links to world frame based on T_world_link.

    Each thread transforms one point for one instance.
    """
    instance, point_idx = wp.tid()

    link_idx = point_link_indices[point_idx]

    T_world = T_world_link[instance, link_idx]
    t = wp.vec3(T_world[0], T_world[1], T_world[2])
    quat_wxyz = wp.vec4(T_world[3], T_world[4], T_world[5], T_world[6])

    local_point = local_points[point_idx]
    world_point = quaternion_apply_func(quat_wxyz, local_point) + t

    world_points[instance, point_idx] = world_point


@wp.kernel
def transform_link_points_matrix_kernel(
    T_world_link: wp.array2d(dtype=wp.mat44),  # [num_instances, num_links]
    local_points: wp.array1d(dtype=wp.vec3),  # [num_points]
    point_link_indices: wp.array1d(dtype=wp.int32),  # [num_points]
    world_points: wp.array2d(dtype=wp.vec3),  # [num_instances, num_points]
):
    """Transform local points attached to links to world frame using 4x4 matrix transforms.

    Each thread transforms one point for one instance.
    """
    instance, point_idx = wp.tid()

    link_idx = point_link_indices[point_idx]
    T_world = T_world_link[instance, link_idx]

    local_point = local_points[point_idx]
    homo_point = wp.vec4(local_point[0], local_point[1], local_point[2], 1.0)
    world_homo = T_world * homo_point

    world_points[instance, point_idx] = wp.vec3(world_homo[0], world_homo[1], world_homo[2])


# ------------------------ Torch Wrappers ------------------------ #
class WarpForwardKinematics(Function):
    @staticmethod
    def forward(
        ctx,
        robot_spec: "RobotSpec",
        q: Float[torch.Tensor, "... num_actuated_joints"],
        T_world_base: Float[torch.Tensor, "... 7"],
    ):
        from robokit.robo.warp_robot import WarpRobot

        wp.init()

        # Extract dimensions and setup
        batch_shape = q.shape[:-1]
        num_instances = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
        num_actuated = q.shape[-1]
        device = q.device
        wp_device = wp.device_from_torch(device)

        # Convert inputs to warp tensors
        requires_grad = q.requires_grad or T_world_base.requires_grad
        q_wp = wp.from_torch(
            q.contiguous().view(num_instances, num_actuated),
            dtype=wp.float32,
            requires_grad=requires_grad,
        )
        T_world_base_wp = wp.from_torch(
            T_world_base.to(device=device, dtype=torch.float32).contiguous().view(num_instances, 7),
            dtype=wp_vec7,
            requires_grad=requires_grad,
        )

        # Allocate output tensors
        T_world_joint_wp = wp.empty(
            (num_instances, robot_spec.num_joints), dtype=wp_vec7, device=wp_device, requires_grad=requires_grad
        )
        T_world_link_wp = wp.empty(
            (num_instances, robot_spec.num_links), dtype=wp_vec7, device=wp_device, requires_grad=requires_grad
        )

        # Prepare robot specification tensors
        spec_t = WarpRobot._get_spec_tensors(robot_spec, device=str(wp_device))
        spec_tensors_wp: Tuple[wp.array, ...] = (
            spec_t.actuated_joint_indices,
            spec_t.mimic_actuated_joint_indices,
            spec_t.mimic_multipliers,
            spec_t.mimic_offsets,
            spec_t.joint_twists,
            spec_t.parent_joint_transforms,
            spec_t.topological_order_joint_indices,
            spec_t.parent_joint_indices,
            spec_t.link_parent_joint_indices,
        )

        # Launch forward kinematics kernel
        wp.launch(
            kernel=forward_kinematics_kernel,
            dim=[num_instances],
            inputs=[q_wp, T_world_base_wp, *spec_tensors_wp],
            outputs=[T_world_joint_wp, T_world_link_wp],
            device=wp_device,
        )

        # Prepare return values
        if requires_grad:
            ctx.meta_info = (batch_shape, num_instances, num_actuated, robot_spec.num_joints, device)
            ctx.robot_spec = robot_spec
            ctx.joint_positions_wp = q_wp
            ctx.T_world_base_wp = T_world_base_wp
            ctx.T_world_joint_wp = T_world_joint_wp
            ctx.T_world_link_wp = T_world_link_wp
            ctx.spec_tensors_wp = spec_tensors_wp

        return wp.to_torch(T_world_link_wp).view(*batch_shape, robot_spec.num_links, 7)

    @staticmethod
    def backward(
        ctx, link_poses_grad: Float[torch.Tensor, "... num_links 7"]
    ) -> Tuple[
        None,
        Optional[Float[torch.Tensor, "... num_actuated_joints"]],
        Optional[Float[torch.Tensor, "... 7"]],
    ]:
        batch_shape, num_instances, num_actuated, num_joints, device = ctx.meta_info

        # Set output gradients
        ctx.T_world_link_wp.grad = wp.from_torch(
            link_poses_grad.contiguous().view(-1, ctx.robot_spec.num_links, 7), dtype=wp_vec7, requires_grad=False
        )
        ctx.T_world_joint_wp.grad = wp.zeros_like(ctx.T_world_joint_wp)
        ctx.joint_positions_wp.grad = wp.zeros_like(ctx.joint_positions_wp)

        base_grad_input = None
        if ctx.needs_input_grad[2]:
            ctx.T_world_base_wp.grad = wp.zeros_like(ctx.T_world_base_wp)
            base_grad_input = ctx.T_world_base_wp.grad

        wp_device = ctx.joint_positions_wp.device
        spec_tensors: Tuple[wp.array, ...] = ctx.spec_tensors_wp

        wp.launch(
            kernel=forward_kinematics_kernel,
            dim=[num_instances],
            inputs=[ctx.joint_positions_wp, ctx.T_world_base_wp, *spec_tensors],
            outputs=[ctx.T_world_joint_wp, ctx.T_world_link_wp],
            adj_inputs=[ctx.joint_positions_wp.grad, base_grad_input, *([None] * 9)],
            adj_outputs=[ctx.T_world_joint_wp.grad, ctx.T_world_link_wp.grad],
            adjoint=True,
            device=wp_device,
        )

        joint_values_grad = None
        if ctx.needs_input_grad[1]:
            joint_values_grad = wp.to_torch(ctx.joint_positions_wp.grad).view(*batch_shape, num_actuated)

        base_values_grad = None
        if ctx.needs_input_grad[2]:
            base_values_grad = wp.to_torch(ctx.T_world_base_wp.grad).view(*batch_shape, 7)

        return None, joint_values_grad, base_values_grad


class WarpForwardKinematicsMatrix(Function):
    @staticmethod
    def forward(
        ctx,
        robot_spec: "RobotSpec",
        q: Float[torch.Tensor, "... num_actuated_joints"],
        T_world_base: Float[torch.Tensor, "... 4 4"],
    ):
        from robokit.robo.warp_robot import WarpRobot

        wp.init()

        # Extract dimensions and setup
        batch_shape = q.shape[:-1]
        num_instances = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
        num_actuated = q.shape[-1]
        requires_grad = q.requires_grad or T_world_base.requires_grad
        device = q.device
        wp_device = wp.device_from_torch(device)

        # Convert inputs to warp tensors
        q_wp = wp.from_torch(
            q.contiguous().view(num_instances, num_actuated),
            dtype=wp.float32,
            requires_grad=requires_grad,
        )
        T_world_base_wp = wp.from_torch(
            T_world_base.to(device=device, dtype=torch.float32).contiguous().view(num_instances, 4, 4),
            dtype=wp.mat44,
            requires_grad=requires_grad,
        )

        # Allocate output tensors
        T_world_joint_wp = wp.empty(
            (num_instances, robot_spec.num_joints), dtype=wp.mat44, device=wp_device, requires_grad=requires_grad
        )
        T_world_link_wp = wp.empty(
            (num_instances, robot_spec.num_links), dtype=wp.mat44, device=wp_device, requires_grad=requires_grad
        )

        # Prepare robot specification tensors
        spec_t = WarpRobot._get_spec_tensors(robot_spec, device=str(wp_device))
        spec_tensors_wp: Tuple[wp.array, ...] = (
            spec_t.actuated_joint_indices,
            spec_t.mimic_actuated_joint_indices,
            spec_t.mimic_multipliers,
            spec_t.mimic_offsets,
            spec_t.joint_twists,
            spec_t.parent_joint_transforms_matrix,
            spec_t.topological_order_joint_indices,
            spec_t.parent_joint_indices,
            spec_t.link_parent_joint_indices,
        )

        # Launch forward kinematics kernel
        wp.launch(
            kernel=forward_kinematics_matrix_kernel,
            dim=[num_instances],
            inputs=[q_wp, T_world_base_wp, *spec_tensors_wp],
            outputs=[T_world_joint_wp, T_world_link_wp],
            device=wp_device,
        )

        # Prepare return values
        if requires_grad:
            ctx.meta_info = (batch_shape, num_instances, num_actuated, robot_spec.num_joints, device)
            ctx.robot_spec = robot_spec
            ctx.joint_positions_wp = q_wp
            ctx.T_world_base_wp = T_world_base_wp
            ctx.T_world_joint_wp = T_world_joint_wp
            ctx.T_world_link_wp = T_world_link_wp
            ctx.spec_tensors_wp = spec_tensors_wp

        return wp.to_torch(T_world_link_wp).view(*batch_shape, robot_spec.num_links, 4, 4)

    @staticmethod
    def backward(
        ctx, link_poses_grad: Float[torch.Tensor, "... num_links 4 4"]
    ) -> Tuple[
        None,
        Optional[Float[torch.Tensor, "... num_actuated_joints"]],
        Optional[Float[torch.Tensor, "... 4 4"]],
    ]:
        batch_shape, num_instances, num_actuated, num_joints, device = ctx.meta_info

        # Set output gradients
        ctx.T_world_link_wp.grad = wp.from_torch(
            link_poses_grad.contiguous().view(-1, ctx.robot_spec.num_links, 4, 4), dtype=wp.mat44, requires_grad=False
        )
        ctx.T_world_joint_wp.grad = wp.zeros_like(ctx.T_world_joint_wp)
        ctx.joint_positions_wp.grad = wp.zeros_like(ctx.joint_positions_wp)

        base_grad_input = None
        if ctx.needs_input_grad[2]:
            ctx.T_world_base_wp.grad = wp.zeros_like(ctx.T_world_base_wp)
            base_grad_input = ctx.T_world_base_wp.grad

        wp_device = ctx.joint_positions_wp.device
        spec_tensors: Tuple[wp.array, ...] = ctx.spec_tensors_wp

        wp.launch(
            kernel=forward_kinematics_matrix_kernel,
            dim=[num_instances],
            inputs=[ctx.joint_positions_wp, ctx.T_world_base_wp, *spec_tensors],
            outputs=[ctx.T_world_joint_wp, ctx.T_world_link_wp],
            adj_inputs=[ctx.joint_positions_wp.grad, base_grad_input, *([None] * 9)],
            adj_outputs=[ctx.T_world_joint_wp.grad, ctx.T_world_link_wp.grad],
            adjoint=True,
            device=wp_device,
        )

        joint_values_grad = None
        if ctx.needs_input_grad[1]:
            joint_values_grad = wp.to_torch(ctx.joint_positions_wp.grad).view(*batch_shape, num_actuated)

        base_values_grad = None
        if ctx.needs_input_grad[2]:
            base_values_grad = wp.to_torch(ctx.T_world_base_wp.grad).view(*batch_shape, 4, 4)

        return None, joint_values_grad, base_values_grad


class WarpTransformLinkPoints(Function):
    @staticmethod
    def forward(
        ctx,
        T_world_link: Float[torch.Tensor, "... num_links 4 4"],
        local_points: Float[torch.Tensor, "num_points 3"],
        point_link_indices: torch.Tensor,
    ) -> Float[torch.Tensor, "... num_points 3"]:
        wp.init()

        batch_shape = T_world_link.shape[:-3]
        num_instances = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
        num_links = T_world_link.shape[-3]
        num_points = local_points.shape[0]
        device = T_world_link.device
        wp_device = wp.device_from_torch(device)

        requires_grad = T_world_link.requires_grad or local_points.requires_grad

        T_world_link_wp = wp.from_torch(
            T_world_link.contiguous().view(num_instances, num_links, 4, 4),
            dtype=wp.mat44,
            requires_grad=requires_grad,
        )

        local_points_wp = wp.from_torch(
            local_points.contiguous().to(device=device, dtype=torch.float32),
            dtype=wp.vec3,
            requires_grad=requires_grad,
        )

        point_link_indices_wp = wp.from_torch(
            point_link_indices.contiguous().to(device=device, dtype=torch.int32),
            dtype=wp.int32,
            requires_grad=False,
        )

        world_points_wp = wp.empty(
            (num_instances, num_points), dtype=wp.vec3, device=wp_device, requires_grad=requires_grad
        )

        wp.launch(
            kernel=transform_link_points_matrix_kernel,
            dim=(num_instances, num_points),
            inputs=[T_world_link_wp, local_points_wp, point_link_indices_wp, world_points_wp],
            device=wp_device,
        )

        if requires_grad:
            ctx.batch_shape = batch_shape
            ctx.num_instances = num_instances
            ctx.num_links = num_links
            ctx.num_points = num_points
            ctx.T_world_link_wp = T_world_link_wp
            ctx.local_points_wp = local_points_wp
            ctx.point_link_indices_wp = point_link_indices_wp
            ctx.world_points_wp = world_points_wp

        return wp.to_torch(world_points_wp).view(*batch_shape, num_points, 3)

    @staticmethod
    def backward(
        ctx, world_points_grad: Float[torch.Tensor, "... num_points 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... num_links 4 4"]], Optional[Float[torch.Tensor, "num_points 3"]], None]:
        batch_shape = ctx.batch_shape
        num_instances = ctx.num_instances
        num_links = ctx.num_links
        num_points = ctx.num_points

        ctx.world_points_wp.grad = wp.from_torch(
            world_points_grad.contiguous().view(num_instances, num_points, 3),
            dtype=wp.vec3,
            requires_grad=False,
        )
        ctx.T_world_link_wp.grad = wp.zeros_like(ctx.T_world_link_wp)
        ctx.local_points_wp.grad = wp.zeros_like(ctx.local_points_wp)

        wp_device = ctx.T_world_link_wp.device

        wp.launch(
            kernel=transform_link_points_matrix_kernel,
            dim=(num_instances, num_points),
            inputs=[ctx.T_world_link_wp, ctx.local_points_wp, ctx.point_link_indices_wp, ctx.world_points_wp],
            adj_inputs=[ctx.T_world_link_wp.grad, ctx.local_points_wp.grad, None, ctx.world_points_wp.grad],
            adjoint=True,
            device=wp_device,
        )

        T_world_link_grad = None
        if ctx.needs_input_grad[0]:
            T_world_link_grad = wp.to_torch(ctx.T_world_link_wp.grad).view(*batch_shape, num_links, 4, 4)

        local_points_grad = None
        if ctx.needs_input_grad[1]:
            local_points_grad = wp.to_torch(ctx.local_points_wp.grad).view(num_points, 3)

        return T_world_link_grad, local_points_grad, None
