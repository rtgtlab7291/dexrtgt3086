# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
import warp as wp

from robokit.xform.warp.transforms import inverse_tf_mat_func, tf_mat_to_rot_mat_func


# --- segment closest-point utilities ---------------------------------------
_SEG_EPS = wp.constant(1.0e-6)


@wp.struct
class SegmentClosest:
    """Witness points of the closest approach between two segments."""

    c1: wp.vec3
    c2: wp.vec3


@wp.func
def closest_point_on_segment_func(a: wp.vec3, b: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Closest point on segment `[a, b]` to point `p`."""
    ab = b - a
    t = wp.clamp(wp.dot(p - a, ab) / (wp.dot(ab, ab) + _SEG_EPS), 0.0, 1.0)
    return a + ab * t


@wp.func
def closest_segment_to_segment_func(a1: wp.vec3, b1: wp.vec3, a2: wp.vec3, b2: wp.vec3) -> SegmentClosest:
    """Closest points between segments `[a1, b1]` and `[a2, b2]` (Gram solve, then clamp to `[0, 1]` and refine)."""
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2

    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    c = wp.dot(d1, r)
    b = wp.dot(d1, d2)
    denom = a * e - b * b

    s = wp.float32(0.0)
    t = wp.float32(0.0)
    if denom < _SEG_EPS:  # parallel / degenerate segments
        s = -c / (a + _SEG_EPS)
        t = f / (e + _SEG_EPS)
    else:
        s = (b * f - c * e) / denom
        t = (a * f - b * c) / denom

    s_clamped = wp.clamp(s, 0.0, 1.0)
    t_clamped = wp.clamp(t, 0.0, 1.0)

    t_recomp = wp.dot(d2, (a1 + d1 * s_clamped) - a2) / (e + _SEG_EPS)
    t_final = t_clamped
    if wp.abs(s - s_clamped) > _SEG_EPS:
        t_final = wp.clamp(t_recomp, 0.0, 1.0)

    s_recomp = wp.dot(d1, (a2 + d2 * t_final) - a1) / (a + _SEG_EPS)
    s_final = s_clamped
    if wp.abs(t - t_final) > _SEG_EPS:
        s_final = wp.clamp(s_recomp, 0.0, 1.0)

    out = SegmentClosest()
    out.c1 = a1 + d1 * s_final
    out.c2 = a2 + d2 * t_final
    return out


# --- per-geometry closest-point functions ----------------------------------
PRIM_BOX = wp.constant(0)
PRIM_SPHERE = wp.constant(1)
PRIM_PLANE = wp.constant(2)
PRIM_CAPSULE = wp.constant(3)


@wp.struct
class SdfHit:
    """Best hit for one point: dist, normal, closest (world) + local closest and element index; `index < 0` = none."""

    dist: wp.float32
    normal: wp.vec3
    closest: wp.vec3
    closest_local: wp.vec3
    index: wp.int32


@wp.func
def closest_on_meshes(
    q_pt: wp.vec3,
    start_dist: wp.float32,
    mesh_ids: wp.array(dtype=wp.uint64),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
    inv_poses: wp.array(dtype=wp.mat44),
    enable_inv_poses: bool,
    scales: wp.array(dtype=wp.float32),
    enable_scales: bool,
    begin: wp.int32,
    end: wp.int32,
    max_dist: wp.float32,
    compute_normal: bool,
) -> SdfHit:
    """Closest of meshes `[begin, end)`, starting from `start_dist`; AABBs cull the BVH walk (result unchanged)."""
    hit = SdfHit()
    hit.dist = start_dist
    hit.index = wp.int32(-1)
    for m_idx in range(begin, end):
        if enable_inv_poses:
            q_local = wp.transform_point(inv_poses[m_idx], q_pt)
        else:
            q_local = q_pt
        if enable_scales:
            q_local = q_local / scales[m_idx]

        aabb_lo = aabb_min[m_idx]
        aabb_hi = aabb_max[m_idx]
        aabb_dist = wp.length(
            wp.vec3(
                wp.max(wp.max(aabb_lo[0] - q_local[0], q_local[0] - aabb_hi[0]), 0.0),
                wp.max(wp.max(aabb_lo[1] - q_local[1], q_local[1] - aabb_hi[1]), 0.0),
                wp.max(wp.max(aabb_lo[2] - q_local[2], q_local[2] - aabb_hi[2]), 0.0),
            )
        )
        radius = max_dist
        if aabb_dist > 0.0:  # inside the AABB the signed distance is unbounded below - never cull
            scale = wp.where(enable_scales, scales[m_idx], wp.float32(1.0))
            if aabb_dist * scale >= hit.dist:
                continue
            radius = wp.min(hit.dist / scale, max_dist)

        query = wp.mesh_query_point(mesh_ids[m_idx], q_local, radius)
        if query.result:
            clst_local = wp.mesh_eval_position(mesh_ids[m_idx], query.face, query.u, query.v)
            unscaled = wp.length(clst_local - q_local) * query.sign
            if enable_scales:
                dist = unscaled * scales[m_idx]
            else:
                dist = unscaled
            if dist < hit.dist:
                hit.dist = dist
                hit.index = m_idx
                if compute_normal:
                    normal_local = wp.mesh_eval_face_normal(mesh_ids[m_idx], query.face)
                    if enable_scales:
                        clst = clst_local * scales[m_idx]
                    else:
                        clst = clst_local
                    if enable_inv_poses:
                        fwd = inverse_tf_mat_func(inv_poses[m_idx])
                        hit.closest = wp.transform_point(fwd, clst)
                        hit.normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                    else:
                        hit.closest = clst
                        hit.normal = normal_local
                    hit.closest_local = clst_local
    return hit


@wp.func
def closest_on_primitives(
    q_pt: wp.vec3,
    start_dist: wp.float32,
    prim_type: wp.int32,
    params: wp.array(dtype=wp.vec4),
    inv_poses: wp.array(dtype=wp.mat44),
    enable_inv_poses: bool,
    scales: wp.array(dtype=wp.float32),
    enable_scales: bool,
    begin: wp.int32,
    end: wp.int32,
    compute_normal: bool,
) -> SdfHit:
    """Closest of primitives `[begin, end)`, starting from `start_dist` (shared with the trajectory kernel)."""
    hit = SdfHit()
    hit.dist = start_dist
    hit.index = wp.int32(-1)
    for p_idx in range(begin, end):
        if enable_inv_poses:
            q_local = wp.transform_point(inv_poses[p_idx], q_pt)
        else:
            q_local = q_pt
        if enable_scales:
            q_local = q_local / scales[p_idx]

        prm = params[p_idx]
        unscaled = wp.float32(0.0)
        normal_local = wp.vec3(0.0, 0.0, 1.0)
        clst_local = q_local

        if prim_type == PRIM_BOX:
            half_extents = wp.vec3(prm[0], prm[1], prm[2])
            abs_p = wp.vec3(wp.abs(q_local[0]), wp.abs(q_local[1]), wp.abs(q_local[2]))
            q_vec = abs_p - half_extents
            q_max = wp.max(wp.max(q_vec[0], q_vec[1]), q_vec[2])
            q_pos = wp.vec3(wp.max(q_vec[0], 0.0), wp.max(q_vec[1], 0.0), wp.max(q_vec[2], 0.0))
            unscaled = wp.length(q_pos) + wp.min(q_max, 0.0)
            if q_max < 0.0:  # inside: normal of the nearest face
                if q_vec[0] >= q_vec[1] and q_vec[0] >= q_vec[2]:
                    normal_local = wp.vec3(wp.where(q_local[0] < 0.0, wp.float32(-1.0), wp.float32(1.0)), 0.0, 0.0)
                elif q_vec[1] >= q_vec[2]:
                    normal_local = wp.vec3(0.0, wp.where(q_local[1] < 0.0, wp.float32(-1.0), wp.float32(1.0)), 0.0)
                else:
                    normal_local = wp.vec3(0.0, 0.0, wp.where(q_local[2] < 0.0, wp.float32(-1.0), wp.float32(1.0)))
            else:
                clamped = wp.vec3(
                    wp.clamp(q_local[0], -half_extents[0], half_extents[0]),
                    wp.clamp(q_local[1], -half_extents[1], half_extents[1]),
                    wp.clamp(q_local[2], -half_extents[2], half_extents[2]),
                )
                diff = q_local - clamped
                dlen = wp.length(diff)
                if dlen > 1.0e-10:
                    normal_local = diff / dlen
            clst_local = q_local - unscaled * normal_local
        elif prim_type == PRIM_SPHERE:
            r = prm[0]
            qlen = wp.length(q_local)
            unscaled = qlen - r
            if qlen > 1.0e-10:
                normal_local = q_local / qlen
            else:
                normal_local = wp.vec3(1.0, 0.0, 0.0)
            clst_local = normal_local * r
        elif prim_type == PRIM_PLANE:
            unscaled = q_local[2]
            normal_local = wp.vec3(0.0, 0.0, 1.0)
            clst_local = wp.vec3(q_local[0], q_local[1], 0.0)
        else:  # PRIM_CAPSULE
            r = prm[0]
            half_h = prm[1]
            t = wp.clamp(q_local[2], -half_h, half_h)
            axis_pt = wp.vec3(0.0, 0.0, t)
            diff = q_local - axis_pt
            dlen = wp.length(diff)
            unscaled = dlen - r
            if dlen > 1.0e-10:
                normal_local = diff / dlen
            else:
                normal_local = wp.vec3(1.0, 0.0, 0.0)
            clst_local = axis_pt + normal_local * r

        if enable_scales:
            dist = unscaled * scales[p_idx]
        else:
            dist = unscaled
        if dist < hit.dist:
            hit.dist = dist
            hit.index = p_idx
            if compute_normal:
                if enable_scales:
                    clst = clst_local * scales[p_idx]
                else:
                    clst = clst_local
                if enable_inv_poses:
                    fwd = inverse_tf_mat_func(inv_poses[p_idx])
                    hit.closest = wp.transform_point(fwd, clst)
                    hit.normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                else:
                    hit.closest = clst
                    hit.normal = normal_local
                hit.closest_local = clst_local
    return hit


@wp.func
def closest_on_volumes(
    q_pt: wp.vec3,
    start_dist: wp.float32,
    volume_ids: wp.array(dtype=wp.uint64),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
    paddings: wp.array(dtype=wp.float32),
    inv_poses: wp.array(dtype=wp.mat44),
    enable_inv_poses: bool,
    scales: wp.array(dtype=wp.float32),
    enable_scales: bool,
    begin: wp.int32,
    end: wp.int32,
    compute_normal: bool,
) -> SdfHit:
    """Closest of SDF volumes `[begin, end)`, starting from `start_dist` (AABB-gated NanoVDB sample)."""
    hit = SdfHit()
    hit.dist = start_dist
    hit.index = wp.int32(-1)
    for v_idx in range(begin, end):
        if enable_inv_poses:
            q_local = wp.transform_point(inv_poses[v_idx], q_pt)
        else:
            q_local = q_pt
        if enable_scales:
            q_local = q_local / scales[v_idx]

        aabb_lo = aabb_min[v_idx]
        aabb_hi = aabb_max[v_idx]
        grad_f = wp.vec3(0.0, 0.0, 0.0)
        inside_aabb = (
            q_local[0] >= aabb_lo[0]
            and q_local[0] <= aabb_hi[0]
            and q_local[1] >= aabb_lo[1]
            and q_local[1] <= aabb_hi[1]
            and q_local[2] >= aabb_lo[2]
            and q_local[2] <= aabb_hi[2]
        )
        if inside_aabb:
            uvw = wp.volume_world_to_index(volume_ids[v_idx], q_local)
            unscaled = wp.volume_sample_grad_f(volume_ids[v_idx], uvw, wp.Volume.LINEAR, grad_f)
        else:
            closest_local = wp.vec3(
                wp.clamp(q_local[0], aabb_lo[0], aabb_hi[0]),
                wp.clamp(q_local[1], aabb_lo[1], aabb_hi[1]),
                wp.clamp(q_local[2], aabb_lo[2], aabb_hi[2]),
            )
            outside = q_local - closest_local
            aabb_dist = wp.length(outside)
            unscaled = aabb_dist + paddings[v_idx]
            grad_f = outside / aabb_dist

        if enable_scales:
            dist = unscaled * scales[v_idx]
        else:
            dist = unscaled
        if dist < hit.dist:
            hit.dist = dist
            hit.index = v_idx
            if compute_normal:
                grad_len = wp.length(grad_f)
                if grad_len > 1.0e-8:
                    normal_local = grad_f / grad_len
                else:
                    normal_local = wp.vec3(0.0, 0.0, 1.0)
                clst_local = q_local - unscaled * normal_local
                if enable_scales:
                    clst = clst_local * scales[v_idx]
                else:
                    clst = clst_local
                if enable_inv_poses:
                    fwd = inverse_tf_mat_func(inv_poses[v_idx])
                    hit.closest = wp.transform_point(fwd, clst)
                    hit.normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                else:
                    hit.closest = clst
                    hit.normal = normal_local
                hit.closest_local = clst_local
    return hit


@wp.func
def closest_on_meshes_with_sdf(
    q_pt: wp.vec3,
    start_dist: wp.float32,
    mesh_ids: wp.array(dtype=wp.uint64),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
    volume_ids: wp.array(dtype=wp.uint64),
    paddings: wp.array(dtype=wp.float32),
    refine_band: wp.float32,
    max_dist: wp.float32,
    inv_poses: wp.array(dtype=wp.mat44),
    enable_inv_poses: bool,
    scales: wp.array(dtype=wp.float32),
    enable_scales: bool,
    begin: wp.int32,
    end: wp.int32,
    compute_normal: bool,
) -> SdfHit:
    """LOD cascade for meshes with a baked SDF mid-tier (`volume_ids[i]` baked from `mesh_ids[i]`):

    outside the padded AABB       -> aabb_dist + padding
    inside                        -> O(1) volume sample
    within refine_band of surface -> exact mesh BVH
    """
    hit = SdfHit()
    hit.dist = start_dist
    hit.index = wp.int32(-1)
    for v_idx in range(begin, end):
        if enable_inv_poses:
            q_local = wp.transform_point(inv_poses[v_idx], q_pt)
        else:
            q_local = q_pt
        if enable_scales:
            q_local = q_local / scales[v_idx]

        aabb_lo = aabb_min[v_idx]
        aabb_hi = aabb_max[v_idx]
        grad_f = wp.vec3(0.0, 0.0, 0.0)
        inside_aabb = (
            q_local[0] >= aabb_lo[0]
            and q_local[0] <= aabb_hi[0]
            and q_local[1] >= aabb_lo[1]
            and q_local[1] <= aabb_hi[1]
            and q_local[2] >= aabb_lo[2]
            and q_local[2] <= aabb_hi[2]
        )
        if inside_aabb:
            uvw = wp.volume_world_to_index(volume_ids[v_idx], q_local)
            unscaled = wp.volume_sample_grad_f(volume_ids[v_idx], uvw, wp.Volume.LINEAR, grad_f)
        else:
            closest_local = wp.vec3(
                wp.clamp(q_local[0], aabb_lo[0], aabb_hi[0]),
                wp.clamp(q_local[1], aabb_lo[1], aabb_hi[1]),
                wp.clamp(q_local[2], aabb_lo[2], aabb_hi[2]),
            )
            outside = q_local - closest_local
            aabb_dist = wp.length(outside)
            unscaled = aabb_dist + paddings[v_idx]
            grad_f = outside / aabb_dist

        if enable_scales:
            vol_dist = unscaled * scales[v_idx]
        else:
            vol_dist = unscaled

        grad_len = wp.length(grad_f)
        if grad_len > 1.0e-8:
            normal_local = grad_f / grad_len
        else:
            normal_local = wp.vec3(0.0, 0.0, 1.0)
        clst_local = q_local - unscaled * normal_local
        use_dist = vol_dist

        # gate the mesh BVH on this object's own volume reading (near-surface only)
        if vol_dist <= refine_band:
            query = wp.mesh_query_point(mesh_ids[v_idx], q_local, max_dist)
            if query.result:
                clst_mesh_local = wp.mesh_eval_position(mesh_ids[v_idx], query.face, query.u, query.v)
                unscaled_mesh = wp.length(clst_mesh_local - q_local) * query.sign
                if enable_scales:
                    mesh_dist = unscaled_mesh * scales[v_idx]
                else:
                    mesh_dist = unscaled_mesh
                use_dist = mesh_dist
                clst_local = clst_mesh_local
                normal_local = wp.mesh_eval_face_normal(mesh_ids[v_idx], query.face)

        if use_dist < hit.dist:
            hit.dist = use_dist
            hit.index = v_idx
            if compute_normal:
                if enable_scales:
                    clst = clst_local * scales[v_idx]
                else:
                    clst = clst_local
                if enable_inv_poses:
                    fwd = inverse_tf_mat_func(inv_poses[v_idx])
                    hit.closest = wp.transform_point(fwd, clst)
                    hit.normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                else:
                    hit.closest = clst
                    hit.normal = normal_local
                hit.closest_local = clst_local
    return hit


# --- scene query kernels ---------------------------------------------------
@wp.kernel
def build_scene_per_point_kernel(
    scene_offsets: wp.array(dtype=wp.int32),
    scene_per_point: wp.array(dtype=wp.int32),
):
    """For each point, binary-search CSR `scene_offsets` for its scene id (avoids a host sync)."""
    pt_idx = wp.tid()
    lo = wp.int32(0)
    hi = scene_offsets.shape[0] - wp.int32(2)
    while lo < hi:
        mid = (lo + hi + wp.int32(1)) // wp.int32(2)
        if scene_offsets[mid] <= pt_idx:
            lo = mid
        else:
            hi = mid - wp.int32(1)
    scene_per_point[pt_idx] = lo


@wp.kernel
def query_sdf_on_meshes_kernel(
    points: wp.array(dtype=wp.vec3),
    scene_per_point: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_aabb_min: wp.array(dtype=wp.vec3),
    mesh_aabb_max: wp.array(dtype=wp.vec3),
    scene_offsets: wp.array(dtype=wp.int32),
    inv_mesh_poses: wp.array(dtype=wp.mat44),
    enable_inv_mesh_poses: bool,
    mesh_scales: wp.array(dtype=wp.float32),
    enable_mesh_scales: bool,
    query_mask: wp.array(dtype=wp.int32),
    enable_query_mask: bool,
    enable_normals: bool,
    max_dist: float,
    signed_dists: wp.array(dtype=wp.float32),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    closest_points_in_mesh_coord: wp.array(dtype=wp.vec3),
    closest_mesh_indices: wp.array(dtype=wp.int32),
):
    pt_idx = wp.tid()
    if enable_query_mask:
        if query_mask[pt_idx] == 0:
            return
    batch_idx = scene_per_point[pt_idx]
    q_pt = points[pt_idx]

    m_begin_idx = scene_offsets[batch_idx]
    m_end_idx = scene_offsets[batch_idx + 1]

    # start from the current per-point min so geom kernels can run in any order
    hit = closest_on_meshes(
        q_pt,
        signed_dists[pt_idx],
        mesh_ids,
        mesh_aabb_min,
        mesh_aabb_max,
        inv_mesh_poses,
        enable_inv_mesh_poses,
        mesh_scales,
        enable_mesh_scales,
        m_begin_idx,
        m_end_idx,
        max_dist,
        enable_normals,
    )
    if hit.index >= 0:
        signed_dists[pt_idx] = hit.dist
        if enable_normals:
            normals[pt_idx] = hit.normal
            closest_points[pt_idx] = hit.closest
            if enable_inv_mesh_poses or enable_mesh_scales:
                closest_points_in_mesh_coord[pt_idx] = hit.closest_local
                closest_mesh_indices[pt_idx] = hit.index


@wp.kernel
def query_sdf_on_volumes_kernel(
    points: wp.array(dtype=wp.vec3),
    scene_per_point: wp.array(dtype=wp.int32),
    volume_ids: wp.array(dtype=wp.uint64),
    volume_aabb_min: wp.array(dtype=wp.vec3),
    volume_aabb_max: wp.array(dtype=wp.vec3),
    volume_paddings: wp.array(dtype=wp.float32),
    scene_offsets: wp.array(dtype=wp.int32),
    inv_volume_poses: wp.array(dtype=wp.mat44),
    enable_inv_volume_poses: bool,
    volume_scales: wp.array(dtype=wp.float32),
    enable_volume_scales: bool,
    query_mask: wp.array(dtype=wp.int32),
    enable_query_mask: bool,
    enable_normals: bool,
    max_dist: float,
    signed_dists: wp.array(dtype=wp.float32),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    closest_points_in_volume_coord: wp.array(dtype=wp.vec3),
    closest_volume_indices: wp.array(dtype=wp.int32),
):
    pt_idx = wp.tid()
    if enable_query_mask:
        if query_mask[pt_idx] == 0:
            return
    batch_idx = scene_per_point[pt_idx]
    q_pt = points[pt_idx]

    v_begin_idx = scene_offsets[batch_idx]
    v_end_idx = scene_offsets[batch_idx + 1]

    min_dist = signed_dists[pt_idx]

    for v_idx in range(v_begin_idx, v_end_idx):
        if enable_inv_volume_poses:
            q_pt_local = wp.transform_point(inv_volume_poses[v_idx], q_pt)
        else:
            q_pt_local = q_pt
        if enable_volume_scales:
            q_pt_local /= volume_scales[v_idx]

        # outside the padded AABB the volume sample is invalid; use the distance to the AABB instead
        aabb_lo = volume_aabb_min[v_idx]
        aabb_hi = volume_aabb_max[v_idx]
        grad_f = wp.vec3(0.0, 0.0, 0.0)
        # per-axis inside test avoids a sqrt on the common in-grid path
        inside_aabb = (
            q_pt_local[0] >= aabb_lo[0]
            and q_pt_local[0] <= aabb_hi[0]
            and q_pt_local[1] >= aabb_lo[1]
            and q_pt_local[1] <= aabb_hi[1]
            and q_pt_local[2] >= aabb_lo[2]
            and q_pt_local[2] <= aabb_hi[2]
        )
        if inside_aabb:
            uvw = wp.volume_world_to_index(volume_ids[v_idx], q_pt_local)
            unscaled_dist = wp.volume_sample_grad_f(volume_ids[v_idx], uvw, wp.Volume.LINEAR, grad_f)
        else:
            closest_local = wp.vec3(
                wp.clamp(q_pt_local[0], aabb_lo[0], aabb_hi[0]),
                wp.clamp(q_pt_local[1], aabb_lo[1], aabb_hi[1]),
                wp.clamp(q_pt_local[2], aabb_lo[2], aabb_hi[2]),
            )
            outside = q_pt_local - closest_local
            aabb_dist = wp.length(outside)
            # the mesh sits at least `padding` inside the AABB walls, so this never overestimates
            unscaled_dist = aabb_dist + volume_paddings[v_idx]
            grad_f = outside / aabb_dist

        if enable_volume_scales:
            dist = unscaled_dist * volume_scales[v_idx]
        else:
            dist = unscaled_dist

        if dist < min_dist:
            min_dist = dist

            grad_len = wp.length(grad_f)
            if grad_len > 1.0e-8:
                normal_local = grad_f / grad_len
            else:
                normal_local = wp.vec3(0.0, 0.0, 1.0)

            clst_pt_local = q_pt_local - unscaled_dist * normal_local

            if enable_volume_scales:
                clst_pt = clst_pt_local * volume_scales[v_idx]
            else:
                clst_pt = clst_pt_local

            signed_dists[pt_idx] = dist
            if enable_normals:
                if enable_inv_volume_poses:
                    fwd = inverse_tf_mat_func(inv_volume_poses[v_idx])
                    clst_pt = wp.transform_point(fwd, clst_pt)
                    normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                else:
                    normal = normal_local
                normals[pt_idx] = normal
                closest_points[pt_idx] = clst_pt
                if enable_inv_volume_poses or enable_volume_scales:
                    closest_points_in_volume_coord[pt_idx] = clst_pt_local
                    closest_volume_indices[pt_idx] = wp.int32(v_idx)


@wp.kernel
def query_sdf_on_meshes_with_sdf_kernel(
    points: wp.array(dtype=wp.vec3),
    scene_per_point: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_aabb_min: wp.array(dtype=wp.vec3),
    mesh_aabb_max: wp.array(dtype=wp.vec3),
    volume_ids: wp.array(dtype=wp.uint64),
    volume_paddings: wp.array(dtype=wp.float32),
    refine_band: float,
    scene_offsets: wp.array(dtype=wp.int32),
    inv_mesh_poses: wp.array(dtype=wp.mat44),
    enable_inv_mesh_poses: bool,
    mesh_scales: wp.array(dtype=wp.float32),
    enable_mesh_scales: bool,
    query_mask: wp.array(dtype=wp.int32),
    enable_query_mask: bool,
    enable_normals: bool,
    max_dist: float,
    signed_dists: wp.array(dtype=wp.float32),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    closest_points_in_mesh_coord: wp.array(dtype=wp.vec3),
    closest_mesh_indices: wp.array(dtype=wp.int32),
):
    """Mesh geom with a baked SDF mid-tier; see `closest_on_meshes_with_sdf` for the cascade."""
    pt_idx = wp.tid()
    if enable_query_mask:
        if query_mask[pt_idx] == 0:
            return
    batch_idx = scene_per_point[pt_idx]
    q_pt = points[pt_idx]

    m_begin_idx = scene_offsets[batch_idx]
    m_end_idx = scene_offsets[batch_idx + 1]

    hit = closest_on_meshes_with_sdf(
        q_pt,
        signed_dists[pt_idx],
        mesh_ids,
        mesh_aabb_min,
        mesh_aabb_max,
        volume_ids,
        volume_paddings,
        refine_band,
        max_dist,
        inv_mesh_poses,
        enable_inv_mesh_poses,
        mesh_scales,
        enable_mesh_scales,
        m_begin_idx,
        m_end_idx,
        enable_normals,
    )
    if hit.index >= 0:
        signed_dists[pt_idx] = hit.dist
        if enable_normals:
            normals[pt_idx] = hit.normal
            closest_points[pt_idx] = hit.closest
            if enable_inv_mesh_poses or enable_mesh_scales:
                closest_points_in_mesh_coord[pt_idx] = hit.closest_local
                closest_mesh_indices[pt_idx] = hit.index


@wp.kernel
def query_sdf_on_primitives_kernel(
    points: wp.array(dtype=wp.vec3),
    scene_per_point: wp.array(dtype=wp.int32),
    prim_type: wp.int32,
    primitive_params: wp.array(dtype=wp.vec4),
    scene_offsets: wp.array(dtype=wp.int32),
    inv_primitive_poses: wp.array(dtype=wp.mat44),
    enable_inv_primitive_poses: bool,
    primitive_scales: wp.array(dtype=wp.float32),
    enable_primitive_scales: bool,
    query_mask: wp.array(dtype=wp.int32),
    enable_query_mask: bool,
    enable_normals: bool,
    max_dist: float,
    signed_dists: wp.array(dtype=wp.float32),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    closest_points_in_primitive_coord: wp.array(dtype=wp.vec3),
    closest_primitive_indices: wp.array(dtype=wp.int32),
):
    """Fused query over one primitive shape; params: BOX (hx,hy,hz) / SPHERE (r) / PLANE () / CAPSULE (r, half_h)."""
    pt_idx = wp.tid()
    if enable_query_mask:
        if query_mask[pt_idx] == 0:
            return
    batch_idx = scene_per_point[pt_idx]
    q_pt = points[pt_idx]

    p_begin_idx = scene_offsets[batch_idx]
    p_end_idx = scene_offsets[batch_idx + 1]

    hit = closest_on_primitives(
        q_pt,
        signed_dists[pt_idx],
        prim_type,
        primitive_params,
        inv_primitive_poses,
        enable_inv_primitive_poses,
        primitive_scales,
        enable_primitive_scales,
        p_begin_idx,
        p_end_idx,
        enable_normals,
    )
    if hit.index >= 0:
        signed_dists[pt_idx] = hit.dist
        if enable_normals:
            normals[pt_idx] = hit.normal
            closest_points[pt_idx] = hit.closest
            if enable_inv_primitive_poses or enable_primitive_scales:
                closest_points_in_primitive_coord[pt_idx] = hit.closest_local
                closest_primitive_indices[pt_idx] = hit.index
