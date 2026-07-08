# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
from typing import Optional, Tuple

import torch
import warp as wp
from jaxtyping import Float

from robokit.lie.warp_so3_kernels import get_epsilon, hat_func, so3_jac_left_inv_func
from robokit.utils.warp_utils import wp_mat66, wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import (
    matrix_to_quaternion_func,
    quaternion_apply_func,
    quaternion_multiply_func,
    quaternion_to_axis_angle_func,
    quaternion_to_matrix_func,
)


@wp.kernel
def se3_extract_xyz_kernel(xyz_wxyz: wp.array(dtype=wp_vec7), xyz_out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    xyz_out[tid] = wp.vec3(xyz_wxyz[tid][0], xyz_wxyz[tid][1], xyz_wxyz[tid][2])


@wp.kernel
def se3_extract_quat_kernel(xyz_wxyz: wp.array(dtype=wp_vec7), quat_out: wp.array(dtype=wp.vec4)):
    tid = wp.tid()
    quat_out[tid] = wp.vec4(xyz_wxyz[tid][3], xyz_wxyz[tid][4], xyz_wxyz[tid][5], xyz_wxyz[tid][6])


@wp.kernel
def se3_from_matrix_kernel(matrix: wp.array(dtype=wp.mat44), xyz_wxyz: wp.array(dtype=wp_vec7)):
    tid = wp.tid()
    mat = matrix[tid]
    xyz = wp.vec3(mat[0, 3], mat[1, 3], mat[2, 3])
    rotation = wp.mat33(
        mat[0, 0], mat[0, 1], mat[0, 2], mat[1, 0], mat[1, 1], mat[1, 2], mat[2, 0], mat[2, 1], mat[2, 2]
    )
    wxyz = matrix_to_quaternion_func(rotation)
    xyz_wxyz[tid] = wp_vec7(xyz[0], xyz[1], xyz[2], wxyz[0], wxyz[1], wxyz[2], wxyz[3])


@wp.kernel
def se3_to_matrix_kernel(xyz_wxyz: wp.array(dtype=wp_vec7), matrix: wp.array(dtype=wp.mat44)):
    """Fused kernel to convert SE(3) pose to 4x4 transformation matrix."""
    tid = wp.tid()
    trans = wp.vec3(xyz_wxyz[tid][0], xyz_wxyz[tid][1], xyz_wxyz[tid][2])
    quat_wxyz = wp.vec4(xyz_wxyz[tid][3], xyz_wxyz[tid][4], xyz_wxyz[tid][5], xyz_wxyz[tid][6])
    rot = quaternion_to_matrix_func(quat_wxyz)
    # fmt: off
    matrix[tid] = wp.mat44(
        rot[0, 0], rot[0, 1], rot[0, 2], trans[0],
        rot[1, 0], rot[1, 1], rot[1, 2], trans[1],
        rot[2, 0], rot[2, 1], rot[2, 2], trans[2],
        0.0, 0.0, 0.0, 1.0,
    )
    # fmt: on


@wp.func
def se3_apply_func(xyz_wxyz: wp_vec7, point: wp.vec3) -> wp.vec3:
    translation = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    quat_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])
    rotated = quaternion_apply_func(quat_wxyz, point)
    return rotated + translation


@wp.kernel
def se3_apply_kernel(
    xyz_wxyz: wp.array(dtype=wp_vec7),
    points: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    result[tid] = se3_apply_func(xyz_wxyz[tid], points[tid])


@wp.func
def se3_inverse_func(xyz_wxyz: wp_vec7) -> wp_vec7:
    xyz = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    q_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])
    q_inv_wxyz = wp.vec4(q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3])

    # Invert translation: t_inv = -q_inv * t * q
    t_inv = -quaternion_apply_func(q_inv_wxyz, xyz)

    return wp_vec7(t_inv[0], t_inv[1], t_inv[2], q_inv_wxyz[0], q_inv_wxyz[1], q_inv_wxyz[2], q_inv_wxyz[3])


@wp.kernel
def se3_inverse_kernel(xyz_wxyz: wp.array(dtype=wp_vec7), result: wp.array(dtype=wp_vec7)):
    tid = wp.tid()
    result[tid] = se3_inverse_func(xyz_wxyz[tid])


@wp.func
def se3_multiply_func(xyz_wxyz1: wp_vec7, xyz_wxyz2: wp_vec7) -> wp_vec7:
    xyz1 = wp.vec3(xyz_wxyz1[0], xyz_wxyz1[1], xyz_wxyz1[2])
    quat_wxyz1 = wp.vec4(xyz_wxyz1[3], xyz_wxyz1[4], xyz_wxyz1[5], xyz_wxyz1[6])
    xyz2 = wp.vec3(xyz_wxyz2[0], xyz_wxyz2[1], xyz_wxyz2[2])
    quat_wxyz2 = wp.vec4(xyz_wxyz2[3], xyz_wxyz2[4], xyz_wxyz2[5], xyz_wxyz2[6])

    # Multiply quaternions to get the combined rotation
    quat_wxyz_result = quaternion_multiply_func(quat_wxyz1, quat_wxyz2)

    # Apply first rotation to second translation, then add first translation
    # t_result = t1 + R1 * t2
    rotated_t2 = quaternion_apply_func(quat_wxyz1, xyz2)
    xyz_result = xyz1 + rotated_t2

    return wp_vec7(
        xyz_result[0],
        xyz_result[1],
        xyz_result[2],
        quat_wxyz_result[0],
        quat_wxyz_result[1],
        quat_wxyz_result[2],
        quat_wxyz_result[3],
    )


@wp.kernel
def se3_multiply_kernel(
    xyz_wxyz1: wp.array(dtype=wp_vec7),
    xyz_wxyz2: wp.array(dtype=wp_vec7),
    result: wp.array(dtype=wp_vec7),
):
    tid = wp.tid()
    result[tid] = se3_multiply_func(xyz_wxyz1[tid], xyz_wxyz2[tid])


@wp.func
def se3_exp_map_func(log_transform: wp_vec6, eps: wp.float32) -> wp_vec7:
    log_translation = wp.vec3(log_transform[0], log_transform[1], log_transform[2])
    log_rotation = wp.vec3(log_transform[3], log_transform[4], log_transform[5])

    # Regularized angle
    angle_sq = wp.dot(log_rotation, log_rotation)
    angle_sq_reg = angle_sq + eps * eps
    angle_reg = wp.sqrt(angle_sq_reg)

    # Quaternion computation
    half_angle = angle_reg * 0.5
    sin_half_angle = wp.sin(half_angle)
    cos_half_angle = wp.cos(half_angle)

    # k = sin(theta/2) / theta -> limit 0.5 as theta->0
    k = sin_half_angle / angle_reg

    quat_wxyz = wp.vec4(cos_half_angle, k * log_rotation[0], k * log_rotation[1], k * log_rotation[2])
    quat_wxyz = wp.normalize(quat_wxyz)

    # V matrix coefficients
    sin_angle = wp.sin(angle_reg)
    cos_angle = wp.cos(angle_reg)

    # A = sin(theta)/theta
    A = sin_angle / angle_reg
    # coeff1 = (1-cos(theta))/theta^2
    coeff1 = (1.0 - cos_angle) / angle_sq_reg
    # coeff2 = (theta-sin(theta))/theta^3 = (1-A)/theta^2
    coeff2 = (1.0 - A) / angle_sq_reg

    K = hat_func(log_rotation)
    K_squared = K * K

    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    V = identity + coeff1 * K + coeff2 * K_squared

    translation = V * log_translation

    return wp_vec7(
        translation[0], translation[1], translation[2], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    )


@wp.kernel
def se3_exp_map_kernel(
    log_transform: wp.array(dtype=wp_vec6),
    eps: wp.float32,
    xyz_wxyz_output: wp.array(dtype=wp_vec7),
):
    tid = wp.tid()
    xyz_wxyz_output[tid] = se3_exp_map_func(log_transform[tid], eps)


@wp.func
def se3_exp_map_to_matrix_func(log_transform: wp_vec6, eps: wp.float32) -> wp.mat44:
    """Fused SE(3) exponential map to matrix function.

    Uses regularized denominators to avoid division by zero and ensure
    well-defined gradients. Adding eps² to denominators ensures numerical
    stability while preserving correct limits as theta → 0.
    """
    rho = wp.vec3(log_transform[0], log_transform[1], log_transform[2])
    omega = wp.vec3(log_transform[3], log_transform[4], log_transform[5])
    theta_sq = wp.dot(omega, omega)

    # Regularize denominators to avoid division by zero
    # When theta_sq=0: theta_reg=eps, so sin(eps)/eps ≈ 1 (correct limit)
    theta_sq_reg = theta_sq + eps * eps
    theta_reg = wp.sqrt(theta_sq_reg)

    K = hat_func(omega)
    K2 = K * K

    sin_theta = wp.sin(theta_reg)
    cos_theta = wp.cos(theta_reg)

    # A = sin(θ)/θ → 1 as θ→0
    A = sin_theta / theta_reg
    # B = (1-cos(θ))/θ² → 0.5 as θ→0
    B = (1.0 - cos_theta) / theta_sq_reg
    # C = (1-A)/θ² → 1/6 as θ→0
    C = (1.0 - A) / theta_sq_reg

    I = wp.identity(n=3, dtype=wp.float32)
    R = I + A * K + B * K2
    V = I + B * K + C * K2

    t = V * rho

    # fmt: off
    return wp.mat44(
        R[0, 0], R[0, 1], R[0, 2], t[0],
        R[1, 0], R[1, 1], R[1, 2], t[1],
        R[2, 0], R[2, 1], R[2, 2], t[2],
        0.0, 0.0, 0.0, 1.0,
    )
    # fmt: on


@wp.kernel
def se3_exp_map_to_matrix_kernel(
    log_transform: wp.array(dtype=wp_vec6),
    eps: wp.float32,
    matrix_output: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    matrix_output[tid] = se3_exp_map_to_matrix_func(log_transform[tid], eps)


@wp.func
def se3_V_matrix_inverse_func(log_rotation: wp.vec3, eps: wp.float32) -> wp.mat33:
    """
    Compute the inverse of the V matrix for SE(3) logarithm.
    V^-1 = I - 0.5 * [log_rotation]_x + c2 * [log_rotation]_x^2
    where c2 = (1/(angle^2)) * (1 - (angle*sin(angle))/(2*(1-cos(angle))))
    """
    angle = wp.length(log_rotation)

    # Identity matrix
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    if angle < eps:
        # Small angle approximation: V^-1 ≈ I - 0.5 * [log_rotation]_x + (1/12) * [log_rotation]_x^2
        log_rotation_hat = wp.mat33(
            0.0,
            -log_rotation[2],
            log_rotation[1],
            log_rotation[2],
            0.0,
            -log_rotation[0],
            -log_rotation[1],
            log_rotation[0],
            0.0,
        )
        log_rotation_hat_sq = wp.mul(log_rotation_hat, log_rotation_hat)
        return identity - 0.5 * log_rotation_hat + (1.0 / 12.0) * log_rotation_hat_sq
    else:
        # Full formula with numerical stability
        sin_angle = wp.sin(angle)
        cos_angle = wp.cos(angle)

        # Skew symmetric matrix of log_rotation
        log_rotation_hat = wp.mat33(
            0.0,
            -log_rotation[2],
            log_rotation[1],
            log_rotation[2],
            0.0,
            -log_rotation[0],
            -log_rotation[1],
            log_rotation[0],
            0.0,
        )

        # log_rotation_hat^2
        log_rotation_hat_sq = wp.mul(log_rotation_hat, log_rotation_hat)

        # Coefficient for the second term - with numerical stability
        # c2 = (1/(angle^2)) * (1 - (angle*sin(angle))/(2*(1-cos(angle))))
        one_minus_cos = 1.0 - cos_angle
        if wp.abs(one_minus_cos) < eps:
            # Near zero rotation: use series expansion
            c2 = 1.0 / 12.0
        else:
            c2 = (1.0 / (angle * angle)) * (1.0 - (angle * sin_angle) / (2.0 * one_minus_cos))

        return identity - 0.5 * log_rotation_hat + c2 * log_rotation_hat_sq


@wp.func
def se3_log_translation_func(xyz_wxyz: wp_vec7, eps: wp.float32) -> wp.vec3:
    """Return only the translation part of SE(3) log map."""
    t = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    quat_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])
    log_rotation = quaternion_to_axis_angle_func(quat_wxyz)
    V_inv = se3_V_matrix_inverse_func(log_rotation, eps)
    return wp.mul(V_inv, t)


@wp.func
def se3_jlog_position_product_func(xyz_wxyz: wp_vec7, body_vel: wp_vec6, eps: wp.float32) -> wp.vec3:
    """Compute only the position (top 3) rows of Jlog(T) * v without constructing the full 6x6 matrix."""
    xyz = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    quat_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])

    w = quaternion_to_axis_angle_func(quat_wxyz)
    theta_squared = wp.dot(w, w)
    theta = wp.sqrt(theta_squared)
    use_taylor = theta_squared < eps

    jlog_so3 = so3_jac_left_inv_func(w, eps)
    jlog_so3_T = wp.transpose(jlog_so3)

    theta_safe = wp.where(use_taylor, 1.0, theta)
    theta_inv = 1.0 / theta_safe
    theta_squared_inv = theta_inv * theta_inv

    st = wp.sin(theta)
    ct = wp.cos(theta)

    one_minus_ct = 1.0 - ct
    inv_2_2ct = wp.where(use_taylor, 0.5, 1.0 / (2.0 * one_minus_ct))
    beta = theta_squared_inv - st * theta_inv * inv_2_2ct
    beta_dot_over_theta = (
        -2.0 * theta_squared_inv * theta_squared_inv + (1.0 + st * theta_inv) * theta_squared_inv * inv_2_2ct
    )

    wTp = wp.dot(w, xyz)
    v3_tmp = beta_dot_over_theta * wTp * w - (theta_squared * beta_dot_over_theta + 2.0 * beta) * xyz
    C = wp.outer(v3_tmp, w) + beta * wp.outer(w, xyz) + (wTp * beta) * wp.identity(n=3, dtype=wp.float32)
    skew_t = hat_func(xyz)
    C = C + 0.5 * skew_t
    B = C * jlog_so3_T
    B_taylor = 0.5 * skew_t
    B_final = wp.where(use_taylor, B_taylor, B)

    body_lin = wp.vec3(body_vel[0], body_vel[1], body_vel[2])
    body_ang = wp.vec3(body_vel[3], body_vel[4], body_vel[5])

    return wp.mul(jlog_so3_T, body_lin) + wp.mul(B_final, body_ang)


@wp.func
def se3_log_map_func(xyz_wxyz: wp_vec7, eps: wp.float32) -> wp_vec6:
    """SE(3) logarithm map function."""
    # Extract translation and quaternion
    t = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    quat_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])

    # Convert quaternion to axis-angle (log rotation)
    log_rotation = quaternion_to_axis_angle_func(quat_wxyz)

    # Compute V^-1
    V_inv = se3_V_matrix_inverse_func(log_rotation, eps)

    # Compute log translation: log_t = V^-1 * t
    log_translation = wp.mul(V_inv, t)

    # Return combined result
    return wp_vec6(
        log_translation[0],
        log_translation[1],
        log_translation[2],
        log_rotation[0],
        log_rotation[1],
        log_rotation[2],
    )


@wp.kernel
def se3_log_map_kernel(
    xyz_wxyz: wp.array(dtype=wp_vec7),
    eps: wp.float32,
    log_transform: wp.array(dtype=wp_vec6),
):
    """SE3 logarithm map kernel."""
    tid = wp.tid()
    log_transform[tid] = se3_log_map_func(xyz_wxyz[tid], eps)


@wp.func
def se3_adjoint_func(xyz_wxyz: wp_vec7) -> wp_mat66:
    # Extract translation and quaternion
    t = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    q_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])

    # Get rotation matrix and translation
    R = quaternion_to_matrix_func(q_wxyz)
    t_skew = wp.mat33(0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0)
    t_skew_R = t_skew * R

    # Construct 6x6 adjoint matrix
    # [[R, t_skew_R],
    #  [0, R]]
    # fmt: off
    return wp_mat66(
        R[0, 0], R[0, 1], R[0, 2], t_skew_R[0, 0], t_skew_R[0, 1], t_skew_R[0, 2],
        R[1, 0], R[1, 1], R[1, 2], t_skew_R[1, 0], t_skew_R[1, 1], t_skew_R[1, 2],
        R[2, 0], R[2, 1], R[2, 2], t_skew_R[2, 0], t_skew_R[2, 1], t_skew_R[2, 2],
        0.0, 0.0, 0.0, R[0, 0], R[0, 1], R[0, 2],
        0.0, 0.0, 0.0, R[1, 0], R[1, 1], R[1, 2],
        0.0, 0.0, 0.0, R[2, 0], R[2, 1], R[2, 2],
    )
    # fmt: on


@wp.kernel
def se3_adjoint_kernel(
    xyz_wxyz: wp.array(dtype=wp_vec7),
    result: wp.array(dtype=wp_mat66),
):
    tid = wp.tid()
    result[tid] = se3_adjoint_func(xyz_wxyz[tid])


@wp.func
def se3_jlog_func(xyz_wxyz: wp_vec7, eps: wp.float32) -> wp_mat66:
    """SE(3) jlog function.

    Args:
        xyz_wxyz: SE(3) pose in [x, y, z, qw, qx, qy, qz] format
        eps: Epsilon for numerical stability

    Returns:
        6x6 jlog matrix
    """
    # Extract translation and quaternion
    xyz = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    quat_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])

    w = quaternion_to_axis_angle_func(quat_wxyz)
    theta_squared = wp.dot(w, w)
    theta = wp.sqrt(theta_squared)
    use_taylor = theta_squared < eps

    # Compute SO(3) jlog
    jlog_so3 = so3_jac_left_inv_func(w, eps)
    jlog_so3_T = wp.transpose(jlog_so3)

    # Safe versions for division
    theta_safe = wp.where(use_taylor, 1.0, theta)
    theta_inv = 1.0 / theta_safe
    theta_squared_inv = theta_inv * theta_inv

    # Trigonometric functions
    st = wp.sin(theta)
    ct = wp.cos(theta)

    # Compute intermediate terms
    one_minus_ct = 1.0 - ct
    inv_2_2ct = wp.where(use_taylor, 0.5, 1.0 / (2.0 * one_minus_ct))
    beta = theta_squared_inv - st * theta_inv * inv_2_2ct
    beta_dot_over_theta = (
        -2.0 * theta_squared_inv * theta_squared_inv + (1.0 + st * theta_inv) * theta_squared_inv * inv_2_2ct
    )

    # Cross term w^T * p
    wTp = wp.dot(w, xyz)

    # Compute v3_tmp term
    v3_tmp = beta_dot_over_theta * wTp * w - (theta_squared * beta_dot_over_theta + 2.0 * beta) * xyz

    # Compute C matrix components
    C = wp.outer(v3_tmp, w) + beta * wp.outer(w, xyz) + (wTp * beta) * wp.identity(n=3, dtype=wp.float32)
    # Add skew-symmetric part
    skew_t = hat_func(xyz)
    C = C + 0.5 * skew_t
    # Compute B = C @ jlog_so3
    B = C * jlog_so3_T
    # For small angles, use Taylor expansion
    B_taylor = 0.5 * skew_t
    B_final = wp.where(use_taylor, B_taylor, B)

    # Construct 6x6 matrix from blocks in row-major order
    # Top-left 3x3: jlog_so3^T, Top-right 3x3: B_final
    # Bottom-left 3x3: zeros, Bottom-right 3x3: jlog_so3^T
    # fmt: off
    return wp_mat66(
        jlog_so3_T[0, 0], jlog_so3_T[0, 1], jlog_so3_T[0, 2], B_final[0, 0], B_final[0, 1], B_final[0, 2],
        jlog_so3_T[1, 0], jlog_so3_T[1, 1], jlog_so3_T[1, 2], B_final[1, 0], B_final[1, 1], B_final[1, 2],
        jlog_so3_T[2, 0], jlog_so3_T[2, 1], jlog_so3_T[2, 2], B_final[2, 0], B_final[2, 1], B_final[2, 2],
        0.0, 0.0, 0.0, jlog_so3_T[0, 0], jlog_so3_T[0, 1], jlog_so3_T[0, 2],
        0.0, 0.0, 0.0, jlog_so3_T[1, 0], jlog_so3_T[1, 1], jlog_so3_T[1, 2],
        0.0, 0.0, 0.0, jlog_so3_T[2, 0], jlog_so3_T[2, 1], jlog_so3_T[2, 2]
    )
    # fmt: on


@wp.kernel
def se3_jlog_kernel(
    xyz_wxyz: wp.array(dtype=wp_vec7),
    eps: wp.float32,
    result: wp.array(dtype=wp_mat66),
):
    """SE(3) jlog kernel."""
    tid = wp.tid()
    result[tid] = se3_jlog_func(xyz_wxyz[tid], eps)


# ------------------------ Torch Wrappers ------------------------ #
class SE3Multiply(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz_wxyz1: Float[torch.Tensor, "... 7"],
        xyz_wxyz2: Float[torch.Tensor, "... 7"],
    ) -> Float[torch.Tensor, "... 7"]:
        """
        Forward pass for SE(3) multiplication.

        Args:
            xyz_wxyz1: First SE(3) transform in [x, y, z, qw, qx, qy, qz] format
            xyz_wxyz2: Second SE(3) transform in [x, y, z, qw, qx, qy, qz] format

        Returns:
            Result of SE(3) multiplication in [x, y, z, qw, qx, qy, qz] format
        """
        wp.init()

        # Create warp arrays from input tensors
        xyz_wxyz1_wp = wp.from_torch(
            xyz_wxyz1.contiguous().view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz1.requires_grad,
        )
        xyz_wxyz2_wp = wp.from_torch(
            xyz_wxyz2.contiguous().view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz2.requires_grad,
        )

        # Create output array
        result_wp = wp.from_torch(
            torch.empty_like(xyz_wxyz1).view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz1.requires_grad or xyz_wxyz2.requires_grad,
        )

        # Launch kernel
        wp.launch(
            kernel=se3_multiply_kernel,
            dim=(xyz_wxyz1_wp.shape[0],),
            inputs=[xyz_wxyz1_wp, xyz_wxyz2_wp, result_wp],
            device=xyz_wxyz1_wp.device,
        )

        # Store for backward pass
        if xyz_wxyz1.requires_grad or xyz_wxyz2.requires_grad:
            ctx.xyz_wxyz1_wp = xyz_wxyz1_wp
            ctx.xyz_wxyz2_wp = xyz_wxyz2_wp
            ctx.result_wp = result_wp
            ctx.input_shape = xyz_wxyz1.shape

        # Convert back to torch
        result = wp.to_torch(result_wp).view(xyz_wxyz1.shape)

        return result

    @staticmethod
    def backward(  # type: ignore
        ctx, xyz_wxyz_grad: Float[torch.Tensor, "... 7"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 7"]], Optional[Float[torch.Tensor, "... 7"]]]:
        """
        Backward pass for SE(3) multiplication.
        """
        wp.init()

        # Set output gradient
        ctx.result_wp.grad = wp.from_torch(
            xyz_wxyz_grad.contiguous().view(-1, 7),
            dtype=wp_vec7,
        )

        # Run adjoint
        wp.launch(
            kernel=se3_multiply_kernel,
            dim=(ctx.xyz_wxyz1_wp.shape[0],),
            inputs=[ctx.xyz_wxyz1_wp, ctx.xyz_wxyz2_wp, ctx.result_wp],
            adj_inputs=[ctx.xyz_wxyz1_wp.grad, ctx.xyz_wxyz2_wp.grad, ctx.result_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz1_wp.device,
        )

        # Collect gradients
        xyz_wxyz1_grad = None
        xyz_wxyz2_grad = None
        if ctx.xyz_wxyz1_wp.requires_grad:
            xyz_wxyz1_grad = wp.to_torch(ctx.xyz_wxyz1_wp.grad).view(ctx.input_shape)

        if ctx.xyz_wxyz2_wp.requires_grad:
            xyz_wxyz2_grad = wp.to_torch(ctx.xyz_wxyz2_wp.grad).view(ctx.input_shape)

        return xyz_wxyz1_grad, xyz_wxyz2_grad


class SE3Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"]) -> Float[torch.Tensor, "... 6 6"]:
        """
        Forward pass for SE(3) adjoint computation.

        Args:
            xyz_wxyz: SE(3) transform in [x, y, z, qw, qx, qy, qz] format

        Returns:
            6x6 adjoint matrix
        """
        wp.init()

        # Convert to warp array
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )

        # Create output array using pre-allocated approach
        adjoint_output = torch.zeros(
            xyz_wxyz.shape[:-1] + (6, 6),
            dtype=xyz_wxyz.dtype,
            device=xyz_wxyz.device,
            requires_grad=xyz_wxyz.requires_grad,
        )
        adjoint_output_wp = wp.from_torch(adjoint_output.view(-1, 6, 6), dtype=wp_mat66)

        # Launch kernel
        wp.launch(
            kernel=se3_adjoint_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, adjoint_output_wp],
            device=xyz_wxyz_wp.device,
        )

        # Store for backward pass
        if xyz_wxyz.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.adjoint_output_wp = adjoint_output_wp
            ctx.input_shape = xyz_wxyz.shape

        return adjoint_output

    @staticmethod
    def backward(  # type: ignore
        ctx, adjoint_grad: Float[torch.Tensor, "... 6 6"]
    ) -> Optional[Float[torch.Tensor, "... 7"]]:
        """
        Backward pass for SE(3) adjoint computation.
        """
        wp.init()

        # Initialize gradients to zero
        ctx.xyz_wxyz_wp.grad.zero_()

        # Set output gradient with correct dtype
        ctx.adjoint_output_wp.grad = wp.from_torch(adjoint_grad.contiguous().view(-1, 6, 6), dtype=wp_mat66)

        # Run adjoint computation
        wp.launch(
            kernel=se3_adjoint_kernel,
            dim=(ctx.xyz_wxyz_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp, ctx.adjoint_output_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad, ctx.adjoint_output_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )

        # Return gradient
        if ctx.xyz_wxyz_wp.requires_grad:
            xyz_wxyz_grad = wp.to_torch(ctx.xyz_wxyz_wp.grad).view(ctx.input_shape)
            return xyz_wxyz_grad

        return None


class SE3Jlog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"]) -> Float[torch.Tensor, "... 6 6"]:
        wp.init()
        eps = get_epsilon(xyz_wxyz.dtype)

        # Store for backward
        ctx.save_for_backward(xyz_wxyz)
        ctx.eps = eps

        # Convert to warp array
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )

        # Create output tensor with shape (batch, 6, 6)
        jlog_output = torch.zeros(
            xyz_wxyz.shape[:-1] + (6, 6),
            dtype=xyz_wxyz.dtype,
            device=xyz_wxyz.device,
            requires_grad=xyz_wxyz.requires_grad,
        )
        jlog_output_wp = wp.from_torch(jlog_output.view(-1, 6, 6), dtype=wp_mat66)

        # Launch kernel
        wp.launch(
            kernel=se3_jlog_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, eps, jlog_output_wp],
            device=xyz_wxyz_wp.device,
        )

        return jlog_output

    @staticmethod
    def backward(ctx, grad_output):
        (xyz_wxyz,) = ctx.saved_tensors
        eps = ctx.eps

        # Initialize gradients
        grad_xyz_wxyz = torch.zeros_like(xyz_wxyz)

        # Convert to warp arrays with gradients
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7),
            dtype=wp_vec7,
            grad=wp.from_torch(grad_xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7),
        )

        # Create output array and its gradient
        jlog_output = torch.zeros(xyz_wxyz.shape[:-1] + (6, 6), dtype=xyz_wxyz.dtype, device=xyz_wxyz.device)
        jlog_output_wp = wp.from_torch(
            jlog_output.view(-1, 6, 6),
            dtype=wp_mat66,
            grad=wp.from_torch(grad_output.contiguous().view(-1, 6, 6), dtype=wp_mat66),
        )

        # Launch forward kernel with adjoint=True for backward pass
        wp.launch(
            kernel=se3_jlog_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, eps, jlog_output_wp],
            adj_inputs=[None, eps, None],  # eps is not differentiable
            adjoint=True,
            device=xyz_wxyz_wp.device,
        )

        return grad_xyz_wxyz


class SE3ExpMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_transforms: Float[torch.Tensor, "... 6"], eps: float = 1e-4) -> Float[torch.Tensor, "... 7"]:
        wp.init()

        # Convert log_transforms to warp array (batch, 6)
        log_transform_wp = wp.from_torch(
            log_transforms.contiguous().view(-1, 6), dtype=wp_vec6, requires_grad=log_transforms.requires_grad
        )

        # Create output array for xyz_wxyz (batch, 7)
        xyz_wxyz_output_wp = wp.from_torch(
            torch.empty(
                log_transforms.shape[:-1] + (7,), dtype=log_transforms.dtype, device=log_transforms.device
            ).view(-1, 7),
            dtype=wp_vec7,
            requires_grad=log_transforms.requires_grad,
        )

        wp.launch(
            kernel=se3_exp_map_kernel,
            dim=(log_transform_wp.shape[0],),
            inputs=[log_transform_wp, eps],
            outputs=[xyz_wxyz_output_wp],
            device=log_transform_wp.device,
        )

        if log_transforms.requires_grad:
            ctx.log_transform_wp = log_transform_wp
            ctx.xyz_wxyz_output_wp = xyz_wxyz_output_wp
            ctx.eps = eps

        # Convert result back to torch
        xyz_wxyz_result = wp.to_torch(xyz_wxyz_output_wp).view(log_transforms.shape[:-1] + (7,))

        return xyz_wxyz_result

    @staticmethod
    def backward(  # type: ignore
        ctx, xyz_wxyz_grad: Float[torch.Tensor, "... 7"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 6"]], None]:
        wp.init()

        # Convert gradient to warp array
        ctx.xyz_wxyz_output_wp.grad = wp.from_torch(xyz_wxyz_grad.contiguous().view(-1, 7), dtype=wp_vec7)

        wp.launch(
            kernel=se3_exp_map_kernel,
            dim=(ctx.log_transform_wp.shape[0],),
            inputs=[ctx.log_transform_wp, ctx.eps],
            outputs=[ctx.xyz_wxyz_output_wp],
            adj_inputs=[ctx.log_transform_wp.grad, None],
            adj_outputs=[ctx.xyz_wxyz_output_wp.grad],
            adjoint=True,
            device=ctx.log_transform_wp.device,
        )

        # Convert gradient back to torch
        log_transform_grad = wp.to_torch(ctx.log_transform_wp.grad).view(xyz_wxyz_grad.shape[:-1] + (6,))

        return log_transform_grad, None


class SE3ExpMapToMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_transforms: Float[torch.Tensor, "... 6"], eps: float = 1e-4) -> Float[torch.Tensor, "... 4 4"]:
        wp.init()

        # Convert log_transforms to warp array (batch, 6)
        log_transform_wp = wp.from_torch(
            log_transforms.contiguous().view(-1, 6), dtype=wp_vec6, requires_grad=log_transforms.requires_grad
        )

        # Create output array for mat44 (batch, 4, 4)
        matrix_output_wp = wp.from_torch(
            torch.empty(
                log_transforms.shape[:-1] + (4, 4), dtype=log_transforms.dtype, device=log_transforms.device
            ).view(-1, 4, 4),
            dtype=wp.mat44,
            requires_grad=log_transforms.requires_grad,
        )

        wp.launch(
            kernel=se3_exp_map_to_matrix_kernel,
            dim=(log_transform_wp.shape[0],),
            inputs=[log_transform_wp, eps],
            outputs=[matrix_output_wp],
            device=log_transform_wp.device,
        )

        if log_transforms.requires_grad:
            ctx.log_transform_wp = log_transform_wp
            ctx.matrix_output_wp = matrix_output_wp
            ctx.eps = eps

        # Convert result back to torch
        matrix_result = wp.to_torch(matrix_output_wp).view(log_transforms.shape[:-1] + (4, 4))

        return matrix_result

    @staticmethod
    def backward(  # type: ignore
        ctx, matrix_grad: Float[torch.Tensor, "... 4 4"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 6"]], None]:
        wp.init()

        # Convert gradient to warp array
        ctx.matrix_output_wp.grad = wp.from_torch(matrix_grad.contiguous().view(-1, 4, 4), dtype=wp.mat44)

        wp.launch(
            kernel=se3_exp_map_to_matrix_kernel,
            dim=(ctx.log_transform_wp.shape[0],),
            inputs=[ctx.log_transform_wp, ctx.eps],
            outputs=[ctx.matrix_output_wp],
            adj_inputs=[ctx.log_transform_wp.grad, None],
            adj_outputs=[ctx.matrix_output_wp.grad],
            adjoint=True,
            device=ctx.log_transform_wp.device,
        )

        # Convert gradient back to torch
        log_transform_grad = wp.to_torch(ctx.log_transform_wp.grad).view(matrix_grad.shape[:-2] + (6,))

        return log_transform_grad, None


class SE3LogMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"], eps: float = 1e-4) -> Float[torch.Tensor, "... 6"]:
        wp.init()

        # Convert to warp array
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )

        log_transform_wp = wp.from_torch(
            torch.empty(xyz_wxyz.shape[:-1] + (6,), dtype=xyz_wxyz.dtype, device=xyz_wxyz.device).view(-1, 6),
            dtype=wp_vec6,
            requires_grad=xyz_wxyz.requires_grad,
        )

        wp.launch(
            kernel=se3_log_map_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, eps],
            outputs=[log_transform_wp],
            device=xyz_wxyz_wp.device,
        )

        if xyz_wxyz.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.log_transform_wp = log_transform_wp
            ctx.eps = eps
            ctx.input_shape = xyz_wxyz.shape

        # Convert result back to torch
        log_transforms = wp.to_torch(log_transform_wp)

        return log_transforms.view(xyz_wxyz.shape[:-1] + (6,))

    @staticmethod
    def backward(  # type: ignore
        ctx, log_transforms_grad: Float[torch.Tensor, "... 6"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 7"]], None]:
        wp.init()

        # Set gradient
        ctx.log_transform_wp.grad = wp.from_torch(log_transforms_grad.contiguous().view(-1, 6), dtype=wp_vec6)

        wp.launch(
            kernel=se3_log_map_kernel,
            dim=(ctx.xyz_wxyz_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp, ctx.eps],
            outputs=[ctx.log_transform_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad, ctx.eps],
            adj_outputs=[ctx.log_transform_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )

        # Convert gradient back to torch
        xyz_wxyz_grad = wp.to_torch(ctx.xyz_wxyz_wp.grad)

        return xyz_wxyz_grad.view(ctx.input_shape), None


class SE3Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"]) -> Float[torch.Tensor, "... 7"]:
        """
        Forward pass for SE(3) inverse.

        Args:
            xyz_wxyz: SE(3) transform in [x, y, z, qw, qx, qy, qz] format

        Returns:
            Inverse SE(3) transform in [x, y, z, qw, qx, qy, qz] format
        """
        wp.init()

        # Convert to warp array
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz.requires_grad,
        )

        # Create output array
        result_wp = wp.from_torch(
            torch.empty_like(xyz_wxyz).view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz.requires_grad,
        )

        # Launch kernel
        wp.launch(
            kernel=se3_inverse_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, result_wp],
            device=xyz_wxyz_wp.device,
        )

        # Store for backward pass
        if xyz_wxyz.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.result_wp = result_wp
            ctx.input_shape = xyz_wxyz.shape

        # Convert back to torch
        result = wp.to_torch(result_wp).view(xyz_wxyz.shape)

        return result

    @staticmethod
    def backward(ctx, result_grad: Float[torch.Tensor, "... 7"]) -> Optional[Float[torch.Tensor, "... 7"]]:
        """
        Backward pass for SE(3) inverse.
        """
        wp.init()

        # Set output gradient
        ctx.result_wp.grad = wp.from_torch(
            result_grad.contiguous().view(-1, 7),
            dtype=wp_vec7,
        )

        # Run adjoint
        wp.launch(
            kernel=se3_inverse_kernel,
            dim=(ctx.xyz_wxyz_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp, ctx.result_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad, ctx.result_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )

        # Collect gradient
        if ctx.xyz_wxyz_wp.requires_grad:
            xyz_wxyz_grad = wp.to_torch(ctx.xyz_wxyz_wp.grad).view(ctx.input_shape)
            return xyz_wxyz_grad

        return None


class SE3Apply(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz_wxyz: Float[torch.Tensor, "... 7"],
        points: Float[torch.Tensor, "... 3"],
    ) -> Float[torch.Tensor, "... 3"]:
        """
        Forward pass for applying SE(3) transform to points.

        Args:
            xyz_wxyz: SE(3) transform in [x, y, z, qw, qx, qy, qz] format
            points: 3D points to transform

        Returns:
            Transformed 3D points
        """
        wp.init()

        # Convert to warp arrays
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz.requires_grad,
        )
        points_wp = wp.from_torch(
            points.contiguous().view(-1, 3),
            dtype=wp.vec3,
            requires_grad=points.requires_grad,
        )

        # Create output array
        result_wp = wp.from_torch(
            torch.empty_like(points).view(-1, 3),
            dtype=wp.vec3,
            requires_grad=xyz_wxyz.requires_grad or points.requires_grad,
        )

        # Launch kernel
        wp.launch(
            kernel=se3_apply_kernel,
            dim=(points_wp.shape[0],),
            inputs=[xyz_wxyz_wp, points_wp, result_wp],
            device=xyz_wxyz_wp.device,
        )

        # Store for backward pass
        if xyz_wxyz.requires_grad or points.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.points_wp = points_wp
            ctx.result_wp = result_wp
            ctx.input_shape_xyz_wxyz = xyz_wxyz.shape
            ctx.input_shape_points = points.shape

        # Convert back to torch
        result = wp.to_torch(result_wp).view(points.shape)

        return result

    @staticmethod
    def backward(
        ctx, result_grad: Float[torch.Tensor, "... 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 7"]], Optional[Float[torch.Tensor, "... 3"]]]:
        """
        Backward pass for SE(3) apply.
        """
        wp.init()

        # Set output gradient
        ctx.result_wp.grad = wp.from_torch(
            result_grad.contiguous().view(-1, 3),
            dtype=wp.vec3,
        )

        # Run adjoint
        wp.launch(
            kernel=se3_apply_kernel,
            dim=(ctx.points_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp, ctx.points_wp, ctx.result_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad, ctx.points_wp.grad, ctx.result_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )

        # Collect gradients
        xyz_wxyz_grad = None
        points_grad = None

        if ctx.xyz_wxyz_wp.requires_grad:
            xyz_wxyz_grad = wp.to_torch(ctx.xyz_wxyz_wp.grad).view(ctx.input_shape_xyz_wxyz)

        if ctx.points_wp.requires_grad:
            points_grad = wp.to_torch(ctx.points_wp.grad).view(ctx.input_shape_points)

        return xyz_wxyz_grad, points_grad
