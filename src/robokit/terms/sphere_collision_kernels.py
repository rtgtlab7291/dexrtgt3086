# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
"""Shared sphere-collision penalty and residual kernel."""

import warp as wp


PENALTY_SMOOTH = wp.constant(wp.int32(0))
PENALTY_PLAIN = wp.constant(wp.int32(1))
PENALTY_SURFACE_DISTANCE = wp.constant(wp.int32(2))
# square root of PENALTY_SMOOTH: a residual whose 0.5*r^2 is that potential, for terms
# consumed as least squares rather than as a cost
PENALTY_SMOOTH_ROOT = wp.constant(wp.int32(3))


# --- device code ------------------------------------------------------------
@wp.func
def _compute_collision_penalty_func(gap: float, margin: float, penalty_mode: wp.int32) -> wp.vec2:
    """Return the collision residual and its derivative with respect to `gap`."""
    if penalty_mode == PENALTY_SURFACE_DISTANCE:
        return wp.vec2(wp.abs(gap), wp.sign(gap))
    if penalty_mode == PENALTY_PLAIN:
        if gap < margin:
            return wp.vec2(margin - gap, wp.float32(-1.0))
        return wp.vec2(wp.float32(0.0), wp.float32(0.0))
    if penalty_mode == PENALTY_SMOOTH_ROOT:
        if gap < wp.float32(0.0):
            root = wp.sqrt(wp.float32(0.5) * margin - gap)
            return wp.vec2(root, wp.float32(-0.5) / root)
        if gap <= margin:
            scale = wp.sqrt(wp.float32(0.5) / (margin + wp.float32(1e-6)))
            return wp.vec2(scale * (margin - gap), -scale)
        return wp.vec2(wp.float32(0.0), wp.float32(0.0))
    if gap < wp.float32(0.0):
        return wp.vec2(wp.float32(0.5) * margin - gap, wp.float32(-1.0))
    if gap <= margin:
        diff = gap - margin
        inv_margin = wp.float32(1.0) / (margin + wp.float32(1e-6))
        return wp.vec2(wp.float32(0.5) * inv_margin * diff * diff, inv_margin * diff)
    return wp.vec2(wp.float32(0.0), wp.float32(0.0))


@wp.kernel
def sphere_collision_residual_apply_kernel(
    gap_cache: wp.array2d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    activation_dist: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    instance, sphere_idx = wp.tid()
    gap = gap_cache[instance, sphere_idx]
    residual = _compute_collision_penalty_func(gap, wp.float32(activation_dist), penalty_mode)[0]
    if use_sqrt:
        residual = wp.sqrt(residual + eps)
    out_residual[instance, row_offset + sphere_idx] = residual_weight[sphere_idx] * residual


__all__ = ["sphere_collision_residual_apply_kernel"]
