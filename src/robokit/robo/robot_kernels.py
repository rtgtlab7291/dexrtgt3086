# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportReturnType=false
import warp as wp

from robokit.lie.se3_kernels import (
    se3_adjoint_func,
    se3_exp_map_func,
    se3_exp_map_to_matrix_func,
    se3_from_matrix_func,
    se3_inverse_func,
    se3_multiply_func,
)
from robokit.utils.warp_utils import wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_apply_func, quaternion_to_matrix_func


# ---------------------------------------------------------------------------
# center of mass
# ---------------------------------------------------------------------------
@wp.func
def compute_link_com_world_func(T: wp_vec7, com_local: wp.vec3) -> wp.vec3:
    """World position of one link's CoM, given the link's world pose `T` and its link-local CoM."""
    q_wxyz = wp.vec4(T[3], T[4], T[5], T[6])
    link_pos = wp.vec3(T[0], T[1], T[2])
    return link_pos + quaternion_apply_func(q_wxyz, com_local)


@wp.func
def compute_body_com_world_func(
    T_world_link: wp.array2d(dtype=wp_vec7),  # [num_elements, num_links]
    link_masses: wp.array1d(dtype=wp.float32),
    link_local_com: wp.array1d(dtype=wp.vec3),
    total_mass_inv: float,
    idx: int,
) -> wp.vec3:
    """Whole-body CoM (mass-weighted average of link CoMs) for instance `idx`, in world frame."""
    num_links = T_world_link.shape[1]
    com = wp.vec3(0.0, 0.0, 0.0)
    for link_idx in range(num_links):
        m = link_masses[link_idx]
        if m > 0.0:
            com += m * compute_link_com_world_func(T_world_link[idx, link_idx], link_local_com[link_idx])
    return com * total_mass_inv


@wp.kernel
def compute_com_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # [num_elements, num_links]
    link_masses: wp.array1d(dtype=wp.float32),
    link_local_com: wp.array1d(dtype=wp.vec3),
    total_mass_inv: float,
    com_out: wp.array1d(dtype=wp.vec3),  # [num_elements]
):
    idx = wp.tid()
    com_out[idx] = compute_body_com_world_func(T_world_link, link_masses, link_local_com, total_mass_inv, idx)


# ---------------------------------------------------------------------------
# sequential forward kinematics
# ---------------------------------------------------------------------------
@wp.func
def _compute_joint_poses_func(
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
def _compute_link_poses_func(
    T_world_joint: wp.array1d(dtype=wp_vec7),  # [num_joints]
    link_parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_links]
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
def compute_forward_kinematics_sequential_kernel(
    actuated_joint_positions: wp.array2d(dtype=wp.float32),  # [num_instances, num_actuated_joints]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [num_instances]
    actuated_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    mimic_multipliers: wp.array1d(dtype=wp.float32),  # [num_joints]
    mimic_offsets: wp.array1d(dtype=wp.float32),  # [num_joints]
    joint_twists: wp.array1d(dtype=wp_vec6),  # [num_joints]
    parent_joint_transforms: wp.array1d(dtype=wp_vec7),  # [num_joints]
    topological_order_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    link_parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_links]
    T_world_joint: wp.array2d(dtype=wp_vec7),  # [num_instances, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [num_instances, num_links]
):
    """Warp kernel for forward kinematics. Each thread computes the forward kinematics for one instance."""
    idx = wp.tid()

    _compute_joint_poses_func(
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
    _compute_link_poses_func(
        T_world_joint[idx],
        link_parent_joint_indices,
        T_world_base[idx],
        T_world_link[idx],
    )


# ---------------------------------------------------------------------------
# sequential forward kinematics, matrix
# ---------------------------------------------------------------------------
@wp.func
def _compute_joint_poses_matrix_func(
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
def _compute_link_poses_matrix_func(
    T_world_joint: wp.array1d(dtype=wp.mat44),  # [num_joints]
    link_parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_links]
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
def compute_forward_kinematics_matrix_sequential_kernel(
    actuated_joint_positions: wp.array2d(dtype=wp.float32),  # [num_instances, num_actuated_joints]
    T_world_base: wp.array1d(dtype=wp.mat44),  # [num_instances]
    actuated_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    mimic_multipliers: wp.array1d(dtype=wp.float32),  # [num_joints]
    mimic_offsets: wp.array1d(dtype=wp.float32),  # [num_joints]
    joint_twists: wp.array1d(dtype=wp_vec6),  # [num_joints]
    parent_joint_transforms: wp.array1d(dtype=wp.mat44),  # [num_joints]
    topological_order_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_joints]
    link_parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_links]
    T_world_joint: wp.array2d(dtype=wp.mat44),  # [num_instances, num_joints]
    T_world_link: wp.array2d(dtype=wp.mat44),  # [num_instances, num_links]
):
    """Warp kernel for forward kinematics via matrix. Each thread computes the forward kinematics for one instance."""
    idx = wp.tid()

    _compute_joint_poses_matrix_func(
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
    _compute_link_poses_matrix_func(
        T_world_joint[idx],
        link_parent_joint_indices,
        T_world_base[idx],
        T_world_link[idx],
    )


# ---------------------------------------------------------------------------
# two-pass forward kinematics
# ---------------------------------------------------------------------------
@wp.func
def _compute_joint_local_transform_func(
    i: int,
    j: int,
    q: wp.array2d(dtype=wp.float32),
    actuated_joint_indices: wp.array1d(dtype=wp.int32),
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.int32),
    mimic_multipliers: wp.array1d(dtype=wp.float32),
    mimic_offsets: wp.array1d(dtype=wp.float32),
    joint_types: wp.array1d(dtype=wp.int32),
    joint_axes: wp.array1d(dtype=wp.vec3),
    parent_joint_transforms: wp.array1d(dtype=wp_vec7),
) -> wp_vec7:
    if mimic_actuated_joint_indices[j] == -1:
        if actuated_joint_indices[j] == -1:
            joint_pos = 0.0
        else:
            joint_pos = q[i, actuated_joint_indices[j]]
    else:
        mimic_idx = mimic_actuated_joint_indices[j]
        joint_pos = q[i, mimic_idx] * mimic_multipliers[j] + mimic_offsets[j]

    jtype = joint_types[j]
    axis = joint_axes[j]
    if jtype == 2:  # REVOLUTE
        half = joint_pos * 0.5
        c = wp.cos(half)
        s = wp.sin(half)
        X_j = wp_vec7(0.0, 0.0, 0.0, c, axis[0] * s, axis[1] * s, axis[2] * s)
    elif jtype == 1:  # PRISMATIC
        X_j = wp_vec7(axis[0] * joint_pos, axis[1] * joint_pos, axis[2] * joint_pos, 1.0, 0.0, 0.0, 0.0)
    else:  # FIXED
        X_j = wp_vec7(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    return se3_multiply_func(parent_joint_transforms[j], X_j)


@wp.kernel
def compute_forward_kinematics_local_kernel(
    q: wp.array2d(dtype=wp.float32),
    actuated_joint_indices: wp.array1d(dtype=wp.int32),
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.int32),
    mimic_multipliers: wp.array1d(dtype=wp.float32),
    mimic_offsets: wp.array1d(dtype=wp.float32),
    joint_types: wp.array1d(dtype=wp.int32),
    joint_axes: wp.array1d(dtype=wp.vec3),
    parent_joint_transforms: wp.array1d(dtype=wp_vec7),
    X_local: wp.array2d(dtype=wp_vec7),
):
    i, j = wp.tid()
    X_local[i, j] = _compute_joint_local_transform_func(
        i,
        j,
        q,
        actuated_joint_indices,
        mimic_actuated_joint_indices,
        mimic_multipliers,
        mimic_offsets,
        joint_types,
        joint_axes,
        parent_joint_transforms,
    )


@wp.kernel
def compute_forward_kinematics_accum_kernel(
    X_local: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    parent_joint_indices: wp.array1d(dtype=wp.int32),
    link_parent_joint_indices: wp.array1d(dtype=wp.int32),
    T_world_joint: wp.array2d(dtype=wp_vec7),
    T_world_link: wp.array2d(dtype=wp_vec7),
):
    i, link_idx = wp.tid()
    j = link_parent_joint_indices[link_idx]
    if j == -1:
        T_world_link[i, link_idx] = T_world_base[i]
        return

    Xw = X_local[i, j]
    parent = parent_joint_indices[j]
    while parent != -1:
        Xw = se3_multiply_func(X_local[i, parent], Xw)
        parent = parent_joint_indices[parent]

    Xw = se3_multiply_func(T_world_base[i], Xw)
    T_world_joint[i, j] = Xw
    T_world_link[i, link_idx] = Xw


@wp.kernel
def compute_forward_kinematics_matrix_local_kernel(
    q: wp.array2d(dtype=wp.float32),
    actuated_joint_indices: wp.array1d(dtype=wp.int32),
    mimic_actuated_joint_indices: wp.array1d(dtype=wp.int32),
    mimic_multipliers: wp.array1d(dtype=wp.float32),
    mimic_offsets: wp.array1d(dtype=wp.float32),
    joint_types: wp.array1d(dtype=wp.int32),
    joint_axes: wp.array1d(dtype=wp.vec3),
    parent_joint_transforms: wp.array1d(dtype=wp.mat44),
    X_local: wp.array2d(dtype=wp.mat44),
):
    i, j = wp.tid()
    if mimic_actuated_joint_indices[j] == -1:
        if actuated_joint_indices[j] == -1:
            joint_pos = 0.0
        else:
            joint_pos = q[i, actuated_joint_indices[j]]
    else:
        mimic_idx = mimic_actuated_joint_indices[j]
        joint_pos = q[i, mimic_idx] * mimic_multipliers[j] + mimic_offsets[j]

    jtype = joint_types[j]
    axis = joint_axes[j]
    if jtype == 2:  # REVOLUTE
        half = joint_pos * 0.5
        c = wp.cos(half)
        s = wp.sin(half)
        quat = wp.vec4(c, axis[0] * s, axis[1] * s, axis[2] * s)
        R = quaternion_to_matrix_func(quat)
        # fmt: off
        T_delta = wp.mat44(
            R[0, 0], R[0, 1], R[0, 2], 0.0,
            R[1, 0], R[1, 1], R[1, 2], 0.0,
            R[2, 0], R[2, 1], R[2, 2], 0.0,
            0.0,     0.0,     0.0,     1.0,
        )
        # fmt: on
    elif jtype == 1:  # PRISMATIC
        # fmt: off
        T_delta = wp.mat44(
            1.0, 0.0, 0.0, axis[0] * joint_pos,
            0.0, 1.0, 0.0, axis[1] * joint_pos,
            0.0, 0.0, 1.0, axis[2] * joint_pos,
            0.0, 0.0, 0.0, 1.0,
        )
        # fmt: on
    else:  # FIXED
        T_delta = wp.identity(n=4, dtype=wp.float32)
    X_local[i, j] = parent_joint_transforms[j] * T_delta


@wp.kernel
def compute_forward_kinematics_matrix_accum_kernel(
    X_local: wp.array2d(dtype=wp.mat44),
    T_world_base: wp.array1d(dtype=wp.mat44),
    parent_joint_indices: wp.array1d(dtype=wp.int32),
    link_parent_joint_indices: wp.array1d(dtype=wp.int32),
    T_world_joint: wp.array2d(dtype=wp.mat44),
    T_world_link: wp.array2d(dtype=wp.mat44),
):
    i, link_idx = wp.tid()
    j = link_parent_joint_indices[link_idx]
    if j == -1:
        T_world_link[i, link_idx] = T_world_base[i]
        return

    Xw = X_local[i, j]
    parent = parent_joint_indices[j]
    while parent != -1:
        Xw = X_local[i, parent] * Xw
        parent = parent_joint_indices[parent]

    Xw = T_world_base[i] * Xw
    T_world_joint[i, j] = Xw
    T_world_link[i, link_idx] = Xw


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------
@wp.kernel
def integrate_joint_positions_kernel(
    q: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    velocity_offset: int,
    num_frames: int,
    single_tangent_dim: int,
    tangent_mask: wp.array1d(dtype=wp.float32),
    has_tangent_mask: wp.bool,
    weight_decay: float,
    out_q: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    batch_idx = i // num_frames
    frame_idx = i % num_frames
    vel_idx = velocity_offset + frame_idx * single_tangent_dim + j
    mask = wp.float32(1.0)
    if has_tangent_mask:
        mask = tangent_mask[vel_idx]
    out_q[i, j] = q[i, j] + mask * (velocity[batch_idx, vel_idx] - weight_decay * q[i, j])


@wp.kernel
def integrate_floating_base_kernel(
    T_world_base: wp.array(dtype=wp_vec7),
    velocity: wp.array2d(dtype=wp.float32),
    tangent_mask: wp.array1d(dtype=wp.float32),
    has_tangent_mask: wp.bool,
    num_frames: int,
    single_tangent_dim: int,
    eps: wp.float32,
    out_T_world_base: wp.array(dtype=wp_vec7),
):
    i = wp.tid()
    batch_idx = i // num_frames
    frame_idx = i % num_frames
    vel_offset = frame_idx * single_tangent_dim
    # zero locked rotation twists before exp; locked world-translation axes are snapped back post-multiply instead
    xi_base = wp_vec6(
        velocity[batch_idx, vel_offset + 0],
        velocity[batch_idx, vel_offset + 1],
        velocity[batch_idx, vel_offset + 2],
        velocity[batch_idx, vel_offset + 3],
        velocity[batch_idx, vel_offset + 4],
        velocity[batch_idx, vel_offset + 5],
    )
    if has_tangent_mask:
        for axis in range(3):
            xi_base[3 + axis] = xi_base[3 + axis] * tangent_mask[vel_offset + 3 + axis]
    T_delta = se3_exp_map_func(xi_base, eps)
    new_T = se3_multiply_func(T_world_base[i], T_delta)
    if has_tangent_mask:
        for axis in range(3):
            if tangent_mask[vel_offset + axis] == 0.0:
                new_T[axis] = T_world_base[i][axis]
    out_T_world_base[i] = new_T


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
    """Integrate SE3 transforms via exponential map; one thread per element (batch_idx, frame_idx, ee_idx)."""
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


# ---------------------------------------------------------------------------
# motion subspace and jacobians
# ---------------------------------------------------------------------------
@wp.kernel
def compute_motion_subspace_kernel(
    T_world_joint: wp.array2d(dtype=wp_vec7),
    joint_twists: wp.array1d(dtype=wp_vec6),
    S_world: wp.array3d(dtype=wp.float32),
):
    instance, joint_idx = wp.tid()

    world_twist = se3_adjoint_func(T_world_joint[instance, joint_idx]) * joint_twists[joint_idx]
    S_world[instance, 0, joint_idx] = world_twist[0]
    S_world[instance, 1, joint_idx] = world_twist[1]
    S_world[instance, 2, joint_idx] = world_twist[2]
    S_world[instance, 3, joint_idx] = world_twist[3]
    S_world[instance, 4, joint_idx] = world_twist[4]
    S_world[instance, 5, joint_idx] = world_twist[5]


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

    # get the transform for this link
    T_world_frame = T_world_link[instance, link_idx]

    if col_idx < base_dofs:
        # compute J_base column
        base_dof = col_idx
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[base_dof] = 1.0

        if reference_frame == 0:  # spatial frame
            # J_base = Adj(T_world_base)
            result_col = se3_adjoint_func(T_world_base[instance]) * unit_vec
        else:  # body frame
            # J_base = Adj(T_frame_base) = Adj(T_frame_world * T_world_base)
            T_frame_world = se3_inverse_func(T_world_frame)
            T_frame_base = se3_multiply_func(T_frame_world, T_world_base[instance])
            result_col = se3_adjoint_func(T_frame_base) * unit_vec

        for row in range(6):
            J_out[instance, row, col_idx] = result_col[row]

    else:
        # compute J_joints column
        actuated_idx = col_idx - base_dofs
        num_actuated = joints_to_actuated.shape[1]

        if actuated_idx >= num_actuated:
            return

        # accumulate contribution from all joints
        spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for joint_idx in range(num_joints):
            if ancestor_mask[joint_idx]:
                weight = joints_to_actuated[joint_idx, actuated_idx]
                if weight != 0.0:
                    for row in range(6):
                        spatial_col[row] += S_world[instance, row, joint_idx] * weight

        # apply reference frame transformation
        if reference_frame == 1:  # body frame
            # J_body = Adj(T_frame_world) * J_spatial
            T_frame_world = se3_inverse_func(T_world_frame)
            body_col = se3_adjoint_func(T_frame_world) * spatial_col
            for row in range(6):
                J_out[instance, row, col_idx] = body_col[row]
        else:  # spatial frame
            for row in range(6):
                J_out[instance, row, col_idx] = spatial_col[row]


# ---------------------------------------------------------------------------
# state accept
# ---------------------------------------------------------------------------
@wp.kernel
def accept_state_forward_kinematics_kernel(
    accept_mask: wp.array1d(dtype=wp.int32),
    src_q: wp.array2d(dtype=wp.float32),
    dst_q: wp.array2d(dtype=wp.float32),
    src_joint: wp.array2d(dtype=wp_vec7),
    dst_joint: wp.array2d(dtype=wp_vec7),
    src_link: wp.array2d(dtype=wp_vec7),
    dst_link: wp.array2d(dtype=wp_vec7),
    elements_per_batch: int,
):
    idx = wp.tid()
    if accept_mask[idx // elements_per_batch] == 1:
        for j in range(src_q.shape[1]):
            dst_q[idx, j] = src_q[idx, j]
        for j in range(src_joint.shape[1]):
            dst_joint[idx, j] = src_joint[idx, j]
        for j in range(src_link.shape[1]):
            dst_link[idx, j] = src_link[idx, j]


@wp.kernel
def accept_state_forward_kinematics_floating_kernel(
    accept_mask: wp.array1d(dtype=wp.int32),
    src_q: wp.array2d(dtype=wp.float32),
    dst_q: wp.array2d(dtype=wp.float32),
    src_base: wp.array1d(dtype=wp_vec7),
    dst_base: wp.array1d(dtype=wp_vec7),
    src_joint: wp.array2d(dtype=wp_vec7),
    dst_joint: wp.array2d(dtype=wp_vec7),
    src_link: wp.array2d(dtype=wp_vec7),
    dst_link: wp.array2d(dtype=wp_vec7),
    elements_per_batch: int,
):
    idx = wp.tid()
    batch_idx = idx // elements_per_batch
    if accept_mask[batch_idx] == 1:
        for j in range(src_q.shape[1]):
            dst_q[idx, j] = src_q[idx, j]
        if idx % elements_per_batch == 0:
            dst_base[batch_idx] = src_base[batch_idx]
        for j in range(src_joint.shape[1]):
            dst_joint[idx, j] = src_joint[idx, j]
        for j in range(src_link.shape[1]):
            dst_link[idx, j] = src_link[idx, j]


# ---------------------------------------------------------------------------
# point transforms
# ---------------------------------------------------------------------------
@wp.kernel
def transform_link_points_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # [num_instances, num_links]
    local_points: wp.array1d(dtype=wp.vec3),  # [num_points]
    point_link_indices: wp.array1d(dtype=wp.int32),  # [num_points]
    world_points: wp.array2d(dtype=wp.vec3),  # [num_instances, num_points]
):
    """Transform link-local points to world frame via T_world_link; one thread per point per instance."""
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
    """Transform link-local points to world frame via 4x4 matrices; one thread per point per instance."""
    instance, point_idx = wp.tid()

    link_idx = point_link_indices[point_idx]
    T_world = T_world_link[instance, link_idx]

    local_point = local_points[point_idx]
    homo_point = wp.vec4(local_point[0], local_point[1], local_point[2], 1.0)
    world_homo = T_world * homo_point

    world_points[instance, point_idx] = wp.vec3(world_homo[0], world_homo[1], world_homo[2])


# ---------------------------------------------------------------------------
# forward kinematics matrix backward
# ---------------------------------------------------------------------------
@wp.kernel
def compute_forward_kinematics_matrix_backward_kernel(
    # forward outputs
    T_world_joint: wp.array2d(dtype=wp.mat44),  # [num_instances, num_joints]
    T_world_link: wp.array2d(dtype=wp.mat44),  # [num_instances, num_links]
    T_world_base: wp.array1d(dtype=wp.mat44),  # [num_instances]
    # grad of output
    grad_T_world_link: wp.array2d(dtype=wp.mat44),  # [num_instances, num_links]
    # robot spec arrays
    joint_twists: wp.array1d(dtype=wp_vec6),  # [num_joints]
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    link_parent_joint_indices: wp.array1d(dtype=wp.int32),  # [num_links]
    # outputs
    grad_q: wp.array2d(dtype=wp.float32),  # [num_instances, num_actuated]
    grad_base: wp.array2d(dtype=wp.float32),  # [num_instances, 16] (flattened mat44)
):
    instance = wp.tid()
    num_links = T_world_link.shape[1]
    num_joints = joint_twists.shape[0]
    num_actuated = joints_to_actuated.shape[1]

    # precompute per-link: cross_sum_l and grad_col3_l
    # cross_sum_l = sum_{k=0..3} T_col_k x grad_col_k  (where T_col_k, grad_col_k are the 3D column vectors)
    # grad_col3_l = top-3 of grad column 3 (translation gradient)
    #
    # also accumulate grad_base (sum over links of grad_T_link @ T_link^T applied to T_base)

    for a in range(num_actuated):
        grad_val = float(0.0)
        for l in range(num_links):
            g = grad_T_world_link[instance, l]
            T_l = T_world_link[instance, l]

            # cross_sum = sum_{k=0..3} T_col_k x grad_col_k
            cross_sum = wp.vec3(0.0, 0.0, 0.0)
            for k in range(4):
                t_col = wp.vec3(T_l[0, k], T_l[1, k], T_l[2, k])
                g_col = wp.vec3(g[0, k], g[1, k], g[2, k])
                cross_sum = cross_sum + wp.cross(t_col, g_col)

            grad_col3 = wp.vec3(g[0, 3], g[1, 3], g[2, 3])

            # accumulate joint contributions for this link
            for j in range(num_joints):
                if link_ancestor_joints_mask[l, j]:
                    weight = joints_to_actuated[j, a]
                    if weight != 0.0:
                        # compute S_world_j inline
                        T_j_vec7 = se3_from_matrix_func(T_world_joint[instance, j])
                        s_world = se3_adjoint_func(T_j_vec7) * joint_twists[j]
                        v_j = wp.vec3(s_world[0], s_world[1], s_world[2])
                        omega_j = wp.vec3(s_world[3], s_world[4], s_world[5])
                        grad_val = grad_val + weight * (wp.dot(omega_j, cross_sum) + wp.dot(v_j, grad_col3))  # type: ignore[call-overload]
        grad_q[instance, a] = grad_val

    # grad_base: dL/dT_base = sum_l grad_T_link_l @ (T_base^{-1} @ T_link_l)^T
    T_b = T_world_base[instance]
    # compute T_base_inv (rigid body inverse: R^T, -R^T t)
    R_b = wp.mat33(T_b[0, 0], T_b[0, 1], T_b[0, 2], T_b[1, 0], T_b[1, 1], T_b[1, 2], T_b[2, 0], T_b[2, 1], T_b[2, 2])
    t_b = wp.vec3(T_b[0, 3], T_b[1, 3], T_b[2, 3])
    R_b_inv = wp.transpose(R_b)
    t_b_inv = -(R_b_inv * t_b)
    # fmt: off
    T_b_inv = wp.mat44(
        R_b_inv[0, 0], R_b_inv[0, 1], R_b_inv[0, 2], t_b_inv[0],
        R_b_inv[1, 0], R_b_inv[1, 1], R_b_inv[1, 2], t_b_inv[1],
        R_b_inv[2, 0], R_b_inv[2, 1], R_b_inv[2, 2], t_b_inv[2],
        0.0,           0.0,           0.0,           1.0,
    )
    # fmt: on

    grad_b = wp.mat44(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for l in range(num_links):
        T_rel = T_b_inv * T_world_link[instance, l]  # T_base_inv @ T_link
        g = grad_T_world_link[instance, l]
        # grad_b += g @ T_rel^T
        grad_b = grad_b + g * wp.transpose(T_rel)

    for r in range(4):
        for c in range(4):
            grad_base[instance, r * 4 + c] = grad_b[r, c]
