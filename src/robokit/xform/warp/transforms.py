import warp as wp


@wp.kernel
def transform_points_kernel(
    points: wp.array(dtype=wp.vec3),
    tf_mat: wp.array(dtype=wp.mat44),
    n_pts: wp.int32,
    out_pt: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    b_idx = tid / (n_pts)
    out_pt[tid] = wp.transform_point(tf_mat[b_idx], points[tid])


@wp.kernel
def rotate_points_kernel(
    points: wp.array(dtype=wp.vec3),
    rot_mat: wp.array(dtype=wp.mat33),
    n_pts: wp.int32,
    out_pt: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    b_idx = tid / (n_pts)
    out_pt[tid] = wp.mul(rot_mat[b_idx], points[tid])


@wp.kernel
def rot_tl_to_tf_mat_kernel(
    rotation: wp.array(dtype=wp.mat33),
    translation: wp.array(dtype=wp.vec3),
    tf_mat: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    rot = rotation[tid]
    tl = translation[tid]

    # Create 4x4 transformation matrix from 3x3 rotation and 3D translation
    # fmt: off
    tf_mat[tid] = wp.mat44(
        rot[0, 0], rot[0, 1], rot[0, 2], tl[0],
        rot[1, 0], rot[1, 1], rot[1, 2], tl[1],
        rot[2, 0], rot[2, 1], rot[2, 2], tl[2],
        0.0, 0.0, 0.0, 1.0  # type: ignore
    )
    # fmt: on
