# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
import warp as wp


@wp.kernel
def axis_angle_to_matrix_kernel(
    axis: wp.array(dtype=wp.vec3), angle: wp.array(dtype=wp.float32), rot_mat: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()

    axis_elem = axis[tid]
    x, y, z = axis_elem[0], axis_elem[1], axis_elem[2]
    s, c = wp.sin(angle[tid]), wp.cos(angle[tid])
    C = 1.0 - c

    xs, ys, zs = x * s, y * s, z * s
    xC, yC, zC = x * C, y * C, z * C
    xyC, yzC, zxC = x * yC, y * zC, z * xC

    rot_mat[tid] = wp.mat33(
        x * xC + c, xyC - zs, zxC + ys, xyC + zs, y * yC + c, yzC - xs, zxC - ys, yzC + xs, z * zC + c
    )


@wp.kernel
def axis_angle_to_quaternion_kernel(axis_angle: wp.array(dtype=wp.vec3), quat_wxyz: wp.array(dtype=wp.vec4)):
    """Warp kernel to convert axis-angle to quaternion."""
    tid = wp.tid()

    axis_angle_vec = axis_angle[tid]
    angle = wp.length(axis_angle_vec)
    eps = 1e-6

    if angle < eps:
        # small angle approximation: q ≈ [1, axis_angle/2]
        # use Taylor series: sin(x/2) ≈ x/2 - x³/48 for small x
        angle_sq = angle * angle
        scale = 0.5 - angle_sq / 48.0
        quat_wxyz[tid] = wp.vec4(
            1.0 - angle_sq / 8.0,  # cos(angle/2) ≈ 1 - x²/8
            axis_angle_vec[0] * scale,
            axis_angle_vec[1] * scale,
            axis_angle_vec[2] * scale,
        )
    else:
        # standard conversion
        axis = axis_angle_vec / angle
        half_angle = angle * 0.5
        sin_half = wp.sin(half_angle)
        cos_half = wp.cos(half_angle)

        quat_wxyz[tid] = wp.vec4(cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half)


@wp.func
def euler_angle_to_matrix_func(angle: wp.float32, axis: wp.int32) -> wp.mat33:
    c, s = wp.cos(angle), wp.sin(angle)
    if axis == 0:
        return wp.mat33(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
    elif axis == 1:
        return wp.mat33(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
    else:
        return wp.mat33(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def euler_angles_to_matrix_kernel(
    euler_angles: wp.array(dtype=wp.vec3), axes: wp.vec4i, rot_mat: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()
    euler_angles_elem = euler_angles[tid]

    if axes[0] == 0:  # static/extrinsic rotation
        rot_mat[tid] = wp.mul(
            wp.mul(
                euler_angle_to_matrix_func(euler_angles_elem[2], axes[3]),
                euler_angle_to_matrix_func(euler_angles_elem[1], axes[2]),
            ),
            euler_angle_to_matrix_func(euler_angles_elem[0], axes[1]),
        )
    else:  # rotating/intrinsic rotation
        rot_mat[tid] = wp.mul(
            wp.mul(
                euler_angle_to_matrix_func(euler_angles_elem[0], axes[1]),
                euler_angle_to_matrix_func(euler_angles_elem[1], axes[2]),
            ),
            euler_angle_to_matrix_func(euler_angles_elem[2], axes[3]),
        )


@wp.kernel
def matrix_to_euler_angles_kernel(rot_mat: wp.array(dtype=wp.mat33), euler_angles: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    rot_mat_elem = rot_mat[tid]

    sy = wp.clamp(rot_mat_elem[0, 2], -1.0, 1.0)
    pitch = wp.asin(sy)
    roll = wp.atan2(-rot_mat_elem[1, 2], rot_mat_elem[2, 2])
    yaw = wp.atan2(-rot_mat_elem[0, 1], rot_mat_elem[0, 0])

    euler_angles[tid] = wp.vec3(roll, pitch, yaw)


@wp.func
def matrix_to_quaternion_func(rot_mat: wp.mat33) -> wp.vec4:
    m00, m01, m02 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2]
    m10, m11, m12 = rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2]
    m20, m21, m22 = rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]

    t0 = 1.0 + m00 + m11 + m22
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22

    q0 = wp.sqrt(t0) if t0 > 0.0 else 0.0
    q1 = wp.sqrt(t1) if t1 > 0.0 else 0.0
    q2 = wp.sqrt(t2) if t2 > 0.0 else 0.0
    q3 = wp.sqrt(t3) if t3 > 0.0 else 0.0

    c0 = wp.vec4(q0 * q0, m21 - m12, m02 - m20, m10 - m01)
    c1 = wp.vec4(m21 - m12, q1 * q1, m10 + m01, m02 + m20)
    c2 = wp.vec4(m02 - m20, m10 + m01, q2 * q2, m12 + m21)
    c3 = wp.vec4(m10 - m01, m20 + m02, m21 + m12, q3 * q3)

    qmax, qb = q0, c0
    if q1 > qmax:
        qmax, qb = q1, c1
    if q2 > qmax:
        qmax, qb = q2, c2
    if q3 > qmax:
        qmax, qb = q3, c3

    denom = 2.0 * (qmax if qmax > 0.1 else 0.1)
    q = qb / denom
    return -q if q[0] < 0.0 else q


@wp.kernel
def matrix_to_quaternion_kernel(rot_mat: wp.array(dtype=wp.mat33), quat_wxyz: wp.array(dtype=wp.vec4)):
    tid = wp.tid()
    quat_wxyz[tid] = matrix_to_quaternion_func(rot_mat[tid])  # wxyz format


@wp.func
def quaternion_to_matrix_func(quat_wxyz: wp.vec4) -> wp.mat33:
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    two_s = 2.0 / (w * w + x * x + y * y + z * z)
    return wp.mat33(
        1.0 - two_s * (y * y + z * z),
        two_s * (x * y - z * w),
        two_s * (x * z + y * w),
        two_s * (x * y + z * w),
        1.0 - two_s * (x * x + z * z),
        two_s * (y * z - x * w),
        two_s * (x * z - y * w),
        two_s * (y * z + x * w),
        1.0 - two_s * (x * x + y * y),
    )


@wp.kernel
def quaternion_to_matrix_kernel(quat_wxyz: wp.array(dtype=wp.vec4), rot_mat: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    rot_mat[tid] = quaternion_to_matrix_func(quat_wxyz[tid])


@wp.func
def quaternion_apply_func(quat_wxyz: wp.vec4, vector: wp.vec3):
    # apply quaternion rotation using the formula:
    # v' = v + 2 * w * (qv × v) + 2 * qv × (qv × v)
    w = quat_wxyz[0]
    qv = wp.vec3(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])
    uv = wp.cross(qv, vector)
    uuv = wp.cross(qv, uv)

    return vector + 2.0 * (w * uv + uuv)


@wp.kernel
def quaternion_apply_kernel(
    quat_wxyz: wp.array(dtype=wp.vec4),
    vectors: wp.array(dtype=wp.vec3),
    out_vectors: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_vectors[tid] = quaternion_apply_func(quat_wxyz[tid], vectors[tid])


@wp.func
def quaternion_invert_func(quat_wxyz: wp.vec4) -> wp.vec4:
    return wp.vec4(quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3])


@wp.kernel
def quaternion_invert_kernel(
    quat_wxyz: wp.array(dtype=wp.vec4),
    quat_wxyz_inv: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    quat_wxyz_inv[tid] = quaternion_invert_func(quat_wxyz[tid])


@wp.func
def quaternion_to_axis_angle_func(quat_wxyz: wp.vec4) -> wp.vec3:
    w = quat_wxyz[0]
    xyz = wp.vec3(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])
    norm = wp.length(xyz)

    half_angle = wp.atan2(norm, w)
    angle = 2.0 * half_angle
    eps = 1e-6
    small_angle = wp.abs(angle) < eps

    if small_angle:
        # for small angles, sin(half_angle) ≈ half_angle - half_angle^3/6
        # so sin(half_angle)/angle ≈ 0.5 - angle^2/48
        sin_half_angle_over_angle = 0.5 - (angle * angle) / 48.0
    else:
        sin_half_angle = wp.sin(half_angle)
        sin_half_angle_over_angle = sin_half_angle / angle

    return xyz / sin_half_angle_over_angle


@wp.kernel
def quaternion_to_axis_angle_kernel(
    quat_wxyz: wp.array(dtype=wp.vec4),
    axis_angle: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    axis_angle[tid] = quaternion_to_axis_angle_func(quat_wxyz[tid])


@wp.func
def standarize_quaternion_func(quat_wxyz: wp.vec4) -> wp.vec4:
    """Standardize quaternion to have non-negative real part."""
    # assumes wxyz format
    if quat_wxyz[0] < 0.0:
        return -quat_wxyz
    return quat_wxyz


@wp.func
def quaternion_multiply_func(quat_wxyz1: wp.vec4, quat_wxyz2: wp.vec4) -> wp.vec4:
    """Multiply two quaternions (assumes wxyz format)."""
    w1, x1, y1, z1 = quat_wxyz1[0], quat_wxyz1[1], quat_wxyz1[2], quat_wxyz1[3]
    w2, x2, y2, z2 = quat_wxyz2[0], quat_wxyz2[1], quat_wxyz2[2], quat_wxyz2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return standarize_quaternion_func(wp.vec4(w, x, y, z))


@wp.kernel
def quaternion_multiply_kernel(
    quat_wxyz1: wp.array(dtype=wp.vec4), quat_wxyz2: wp.array(dtype=wp.vec4), output: wp.array(dtype=wp.vec4)
):
    tid = wp.tid()
    output[tid] = quaternion_multiply_func(quat_wxyz1[tid], quat_wxyz2[tid])


@wp.kernel
def standardize_quaternion_kernel(quat_wxyz: wp.array(dtype=wp.vec4), output: wp.array(dtype=wp.vec4)):
    """Standardize quaternions to have non-negative real part and unit norm."""
    tid = wp.tid()
    q = quat_wxyz[tid]
    q_normalized = wp.normalize(q)
    output[tid] = standarize_quaternion_func(q_normalized)
