# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalSubscript=false
# pyright: reportOperatorIssue=false
# pyright: reportMissingImports=false
"""Mask and depth alignment for hand-eye calibration."""

from typing import Any, List, Optional, Tuple

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import warp as wp

from robokit.lie.se3 import SE3Var, se3_to_matrix
from robokit.lie.se3_kernels import se3_adjoint_kernel
from robokit.opt.var_values import VarValues
from robokit.terms.task import EagerTask
from robokit.utils.warp_utils import wp_mat66, wp_vec6
from robokit.xform.warp.torch_wrappers import intr_to_proj_mat


_OPENCV2GL = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)


def render_mask(
    glctx: dr.RasterizeCudaContext,
    verts: torch.Tensor,
    faces: torch.Tensor,
    intrinsic: torch.Tensor,
    T_camera_link: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """Render a differentiable binary mask of a mesh seen from a camera.

    Args:
        glctx: nvdiffrast CUDA rasterisation context.
        verts: Mesh vertices `[V, 3]`.
        faces: Triangle indices `[F, 3]` (int32).
        intrinsic: Camera intrinsic matrix `[3, 3]`.
        T_camera_link: 4x4 SE3 from link-local frame to camera frame.
        H, W: Output image height and width.

    Returns:
        Anti-aliased mask of shape `[H, W]` with values in `[0, 1]`.
    """
    proj = intr_to_proj_mat(intrinsic, H, W)
    pose = _OPENCV2GL.to(T_camera_link.device) @ T_camera_link
    mvp = proj @ pose  # [4, 4]
    # homogeneous clip-space vertices
    ones = torch.ones(verts.shape[0], 1, dtype=torch.float32, device=verts.device)
    pos_homo = torch.cat([verts, ones], dim=1)
    pos_clip = (pos_homo @ mvp.t()).unsqueeze(0)

    rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=(H, W))
    vtx_color = torch.ones(1, verts.shape[0], 3, dtype=torch.float32, device=verts.device)
    color, _ = dr.interpolate(vtx_color, rast_out, faces)
    color = dr.antialias(color, rast_out, pos_clip, faces)
    mask: torch.Tensor = color[0, :, :, 0]
    return torch.flip(mask, dims=[0])


# --- device code ------------------------------------------------------------
@wp.func
def _compute_huber_residual_signed_func(error: float, delta: float) -> float:
    abs_error = wp.abs(error)
    if abs_error <= delta:
        return error
    scale = wp.sqrt(wp.max(2.0 * delta * abs_error - delta * delta, 0.0))
    return scale if error >= 0.0 else -scale


@wp.func
def _compute_huber_gradient_scale_func(error: float, delta: float) -> float:
    abs_error = wp.abs(error)
    if abs_error <= delta:
        return 1.0
    denom = wp.sqrt(wp.max(2.0 * delta * abs_error - delta * delta, 1e-12))
    return delta / denom


@wp.kernel
def _accumulate_normal_equations_kernel(
    rendered: wp.array3d(dtype=wp.float32),
    target: wp.array3d(dtype=wp.float32),
    positions: wp.array(dtype=wp.float32, ndim=4),
    fx: wp.float32,
    fy: wp.float32,
    img_h: wp.int32,
    img_w: wp.int32,
    n_obs: wp.int32,
    adjoint: wp.array(dtype=wp_mat66),
    JtJ: wp.array3d(dtype=wp.float32),
    Jtr: wp.array3d(dtype=wp.float32),
    cost: wp.array1d(dtype=wp.float32),
):
    """Per-pixel residual + analytical Jacobian, accumulated per seed."""
    sn, v, u = wp.tid()
    seed = sn // n_obs

    r = rendered[sn, v, u] - target[sn, v, u]
    wp.atomic_add(cost, seed, wp.float32(0.5) * r * r)

    du = wp.float32(0.0)
    dv = wp.float32(0.0)
    if u > 0 and u < img_w - 1:
        du = wp.float32(0.5) * (rendered[sn, v, u + 1] - rendered[sn, v, u - 1])
    if v > 0 and v < img_h - 1:
        dv = wp.float32(0.5) * (rendered[sn, v + 1, u] - rendered[sn, v - 1, u])

    if du == wp.float32(0.0) and dv == wp.float32(0.0):
        return

    X = positions[sn, v, u, 0]
    Y = positions[sn, v, u, 1]
    Z = positions[sn, v, u, 2]
    if Z <= wp.float32(0.0):
        return

    inv_Z = wp.float32(1.0) / Z
    inv_Z2 = inv_Z * inv_Z

    j = wp_vec6(
        -(du * fx * inv_Z),
        -(dv * fy * inv_Z),
        (du * fx * X + dv * fy * Y) * inv_Z2,
        du * fx * X * Y * inv_Z2 - dv * (-fy - fy * Y * Y * inv_Z2),
        -(du * (fx + fx * X * X * inv_Z2) + dv * fy * X * Y * inv_Z2),
        du * fx * Y * inv_Z - dv * fy * X * inv_Z,
    )
    j = wp.transpose(adjoint[seed]) * j

    for i in range(6):
        wp.atomic_add(Jtr, seed, i, 0, j[i] * r)
        for k in range(6):
            wp.atomic_add(JtJ, seed, i, k, j[i] * j[k])


@wp.kernel
def _compute_cost_kernel(
    rendered: wp.array3d(dtype=wp.float32),
    target: wp.array3d(dtype=wp.float32),
    n_obs: wp.int32,
    cost: wp.array1d(dtype=wp.float32),
):
    sn, v, u = wp.tid()
    seed = sn // n_obs
    r = rendered[sn, v, u] - target[sn, v, u]
    wp.atomic_add(cost, seed, wp.float32(0.5) * r * r)


# reject depth-image gradients across occlusion boundaries
_DEPTH_FLOW_GRAD_MAX = wp.constant(0.05)


# combine direct depth motion with image flow so depth constrains all six directions
@wp.kernel
def _accumulate_depth_normal_equations_kernel(
    positions: wp.array(dtype=wp.float32, ndim=4),
    target_depth: wp.array3d(dtype=wp.float32),
    fx: wp.float32,
    fy: wp.float32,
    img_h: wp.int32,
    img_w: wp.int32,
    huber_delta: wp.float32,
    weight: wp.float32,
    n_obs: wp.int32,
    adjoint: wp.array(dtype=wp_mat66),
    JtJ: wp.array3d(dtype=wp.float32),
    Jtr: wp.array3d(dtype=wp.float32),
    cost: wp.array1d(dtype=wp.float32),
):
    sn, v, u = wp.tid()
    seed = sn // n_obs

    Z = positions[sn, v, u, 2]
    Dt = target_depth[sn, v, u]
    # require rendered and observed depth
    if Z <= wp.float32(0.0) or Dt <= wp.float32(0.0):
        return

    X = positions[sn, v, u, 0]
    Y = positions[sn, v, u, 1]

    # compute depth gradients away from occlusion edges
    du = wp.float32(0.0)
    dv = wp.float32(0.0)
    if u > 0 and u < img_w - 1:
        zp = positions[sn, v, u + 1, 2]
        zm = positions[sn, v, u - 1, 2]
        if zp > wp.float32(0.0) and zm > wp.float32(0.0):
            g = wp.float32(0.5) * (zp - zm)
            if wp.abs(g) <= _DEPTH_FLOW_GRAD_MAX:
                du = g
    if v > 0 and v < img_h - 1:
        zp = positions[sn, v + 1, u, 2]
        zm = positions[sn, v - 1, u, 2]
        if zp > wp.float32(0.0) and zm > wp.float32(0.0):
            g = wp.float32(0.5) * (zp - zm)
            if wp.abs(g) <= _DEPTH_FLOW_GRAD_MAX:
                dv = g

    inv_Z = wp.float32(1.0) / Z
    inv_Z2 = inv_Z * inv_Z
    r = Z - Dt
    sw = wp.sqrt(weight)
    rw = sw * _compute_huber_residual_signed_func(r, huber_delta)
    scale = sw * _compute_huber_gradient_scale_func(r, huber_delta)
    j = wp_vec6(
        scale * (-(du * fx * inv_Z)),
        scale * (-(dv * fy * inv_Z)),
        scale * (wp.float32(1.0) + (du * fx * X + dv * fy * Y) * inv_Z2),
        scale * (Y + du * fx * X * Y * inv_Z2 - dv * (-fy - fy * Y * Y * inv_Z2)),
        scale * (-X - (du * (fx + fx * X * X * inv_Z2) + dv * fy * X * Y * inv_Z2)),
        scale * (du * fx * Y * inv_Z - dv * fy * X * inv_Z),
    )
    j = wp.transpose(adjoint[seed]) * j

    wp.atomic_add(cost, seed, wp.float32(0.5) * rw * rw)
    for i in range(6):
        wp.atomic_add(Jtr, seed, i, 0, j[i] * rw)
        for k in range(6):
            wp.atomic_add(JtJ, seed, i, k, j[i] * j[k])


@wp.kernel
def _compute_depth_cost_kernel(
    positions: wp.array(dtype=wp.float32, ndim=4),
    target_depth: wp.array3d(dtype=wp.float32),
    huber_delta: wp.float32,
    weight: wp.float32,
    n_obs: wp.int32,
    cost: wp.array1d(dtype=wp.float32),
):
    sn, v, u = wp.tid()
    seed = sn // n_obs
    Z = positions[sn, v, u, 2]
    Dt = target_depth[sn, v, u]
    if Z <= wp.float32(0.0) or Dt <= wp.float32(0.0):
        return
    rw = _compute_huber_residual_signed_func(Z - Dt, huber_delta)
    wp.atomic_add(cost, seed, weight * wp.float32(0.5) * rw * rw)


# --- rendering --------------------------------------------------------------
def _precompute_merged_mesh(
    link_vertices: List[torch.Tensor],
    link_faces: List[torch.Tensor],
    link_poses: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute merged faces and per-observation base-frame vertices.

    Returns:
        merged_faces: `[F_total, 3]` int32 on CUDA.
        v_base: `[N, V_total, 3]` float32 on CUDA.
    """
    all_faces: List[torch.Tensor] = []
    vert_offset = 0
    for mi in range(len(link_vertices)):
        all_faces.append(link_faces[mi] + vert_offset)
        vert_offset += link_vertices[mi].shape[0]
    merged_faces = torch.cat(all_faces, dim=0)

    N = link_poses.shape[0]
    per_obs: List[torch.Tensor] = []
    for n in range(N):
        parts: List[torch.Tensor] = []
        for mi, v in enumerate(link_vertices):
            T = link_poses[n, mi]
            parts.append(v @ T[:3, :3].t() + T[:3, 3])
        per_obs.append(torch.cat(parts, dim=0))
    v_base = torch.stack(per_obs)
    return merged_faces, v_base


def _render_batch(
    glctx: dr.RasterizeCudaContext,
    T_cam_base: torch.Tensor,
    intrinsic: torch.Tensor,
    merged_faces: torch.Tensor,
    v_base: torch.Tensor,
    H: int,
    W: int,
    with_positions: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Batch-render S seeds × N observations in a single nvdiffrast call.

    Args:
        T_cam_base: `[S, 4, 4]` or `[4, 4]` camera extrinsic(s).
        v_base: `[N, V_total, 3]` base-frame vertices per observation.

    Returns:
        masks `[S*N, H, W]` and (optionally) cam-space positions `[S*N, H, W, 3]`.
    """
    if T_cam_base.ndim == 2:
        T_cam_base = T_cam_base.unsqueeze(0)

    S = T_cam_base.shape[0]
    R = T_cam_base[:, :3, :3]  # [S, 3, 3]
    t = T_cam_base[:, :3, 3]  # [S, 3]

    # v_cam[s, n] = v_base[n] @ R[s].T + t[s]
    v_cam = torch.einsum("nvi,sji->snvj", v_base, R) + t[:, None, None, :]  # [S, N, V, 3]
    N, V = v_base.shape[:2]
    v_cam = v_cam.reshape(S * N, V, 3)

    ones = torch.ones(S * N, V, 1, device=v_cam.device, dtype=torch.float32)
    v_homo = torch.cat([v_cam, ones], dim=2)
    mvp = (intr_to_proj_mat(intrinsic, H, W) @ _OPENCV2GL.to(v_cam.device)).to(v_cam.device)
    v_clip = v_homo @ mvp.t()

    rast_out, _ = dr.rasterize(glctx, v_clip, merged_faces, resolution=(H, W))  # type: ignore[reportGeneralTypeIssues]

    vtx_white = torch.ones(1, V, 3, device=v_cam.device, dtype=torch.float32)
    color, _ = dr.interpolate(vtx_white, rast_out, merged_faces)  # type: ignore[reportGeneralTypeIssues]
    color = dr.antialias(color, rast_out, v_clip, merged_faces)
    masks = torch.flip(color[:, :, :, 0], dims=[1])

    pos_cam = None
    if with_positions:
        pos_interp, _ = dr.interpolate(v_cam, rast_out, merged_faces)  # type: ignore[reportGeneralTypeIssues]
        pos_cam = torch.flip(pos_interp, dims=[1])

    return masks, pos_cam


# --- task -------------------------------------------------------------------
class MaskAlignmentNormalEqTask(EagerTask):
    """Stream mask and depth normal equations without storing pixel Jacobians.

    Lifecycle:
        1. `prepare(..., "proposal", ...)` renders the current pose.
        2. `accumulate_normal_equations(...)` accumulates the build or cost step.
        3. `prepare(..., "acceptance", ...)` renders the proposed pose.
        4. `on_step(...)` updates accepted buffers and stopping state.
    """

    num_dofs = 6
    residual_weight = None
    batch_size: int = 0

    @property
    def residual_dim(self) -> int:
        return self.N * self.H * self.W

    def compute_weighted_residual(self, var_values: VarValues, *args: Any, **kwargs: Any) -> wp.array:
        raise NotImplementedError("MaskAlignmentNormalEqTask streams residuals directly into normal equations.")

    def __init__(
        self,
        camera_intrinsic: torch.Tensor,
        target_masks: Optional[torch.Tensor],
        link_poses: torch.Tensor,
        link_vertices: List[torch.Tensor],
        link_faces: List[torch.Tensor],
        height: int,
        width: int,
        blur_sigma: float = 8.0,
        patience: int = 20,
        target_depths: Optional[torch.Tensor] = None,
        depth_weight: float = 0.0,
        depth_huber_delta: float = 0.02,
    ):
        if target_masks is None and (target_depths is None or depth_weight <= 0.0):
            raise ValueError("requires target_masks and/or (target_depths with depth_weight > 0)")
        if target_masks is None:
            assert target_depths is not None
            # use valid depth as a silhouette so zero overlap is not a zero-cost solution
            target_masks = (target_depths > 0).float()
        self.camera_intrinsic = camera_intrinsic
        self.target_masks = target_masks  # [N, H, W]
        self.target_depths = target_depths if depth_weight > 0.0 else None  # [N, H, W], meters, 0 = invalid
        self.depth_weight = depth_weight
        self.depth_huber_delta = depth_huber_delta
        self.H = height
        self.W = width
        self.blur_sigma = blur_sigma
        self.patience = patience
        self.torch_device = target_masks.device
        self.N = target_masks.shape[0]
        self.glctx = dr.RasterizeCudaContext()
        self.fx = camera_intrinsic[0, 0].item()
        self.fy = camera_intrinsic[1, 1].item()
        self.merged_faces, self.v_base = _precompute_merged_mesh(link_vertices, link_faces, link_poses)

    def _alloc_buffers(self, batch_size: int, device):
        self.batch_size = batch_size
        S, N, H, W = batch_size, self.N, self.H, self.W
        self.wp_device = device
        dev = self.torch_device
        self._rendered_curr = torch.empty(S * N, H, W, dtype=torch.float32, device=dev)
        self._positions_curr = torch.empty(S * N, H, W, 3, dtype=torch.float32, device=dev)
        self._rendered_prop = torch.empty(S * N, H, W, dtype=torch.float32, device=dev)
        self._positions_prop = torch.empty(S * N, H, W, 3, dtype=torch.float32, device=dev)
        target_exp = self.target_masks.unsqueeze(0).expand(S, -1, -1, -1).reshape(S * N, H, W).contiguous()
        # alias render buffers when blur is disabled
        if self.blur_sigma < 0.5:
            self._blurred_curr = self._rendered_curr
            self._blurred_prop = self._rendered_prop
            self._target_blurred = target_exp
        else:
            self._blurred_curr = torch.empty(S * N, H, W, dtype=torch.float32, device=dev)
            self._blurred_prop = torch.empty(S * N, H, W, dtype=torch.float32, device=dev)
            self._target_blurred = self._blur(target_exp, self.blur_sigma).contiguous()
        self._wp_blurred_curr = wp.from_torch(self._blurred_curr)
        self._wp_blurred_prop = wp.from_torch(self._blurred_prop)
        self._wp_target_blurred = wp.from_torch(self._target_blurred)
        if self.target_depths is not None:
            depth_exp = self.target_depths.unsqueeze(0).expand(S, -1, -1, -1).reshape(S * N, H, W).contiguous()
            self._wp_target_depth = wp.from_torch(depth_exp)
        self._wp_positions_curr = wp.from_torch(self._positions_curr)
        self._wp_positions_prop = wp.from_torch(self._positions_prop)
        # pixel Jacobians are camera-frame (left) perturbations; Ad(T) maps them onto the
        # body-frame (right) twist that VarValues.integrate applies
        self._wp_adjoint_curr = wp.empty(S, dtype=wp_mat66, device=device)
        self._best_cost = float("inf")
        self._last_improvement = 0
        self._current_var: Optional[SE3Var] = None
        self._best_xyz_wxyz: Optional[np.ndarray] = None
        # mark whether current buffers already contain the accepted render
        self._curr_is_fresh = False

    def _blur(self, masks: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma < 0.5:
            return masks
        ks = int(6 * sigma + 1) | 1
        x = torch.arange(ks, device=masks.device, dtype=torch.float32) - ks // 2
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        m = masks.unsqueeze(1)
        m = F.conv2d(m, k.view(1, 1, -1, 1), padding=(ks // 2, 0))
        m = F.conv2d(m, k.view(1, 1, 1, -1), padding=(0, ks // 2))
        return m.squeeze(1)

    def prepare(self, var_values: VarValues, proposed: bool, iter_idx: int):
        """Render the current or proposed camera transform into persistent buffers."""
        var = var_values.get(self.var_key)
        if proposed:
            with torch.no_grad():
                T_torch = wp.to_torch(se3_to_matrix(var.xyz_wxyz))
                prop_masks, prop_positions = _render_batch(
                    self.glctx, T_torch, self.camera_intrinsic, self.merged_faces, self.v_base, self.H, self.W
                )
                self._rendered_prop.copy_(prop_masks)
                self._positions_prop.copy_(prop_positions)
                if self.blur_sigma >= 0.5:
                    self._blurred_prop.copy_(self._blur(self._rendered_prop, self.blur_sigma))
            return
        # allocate stable buffers once per solve shape
        if self.batch_size != var.batch_size:
            self._alloc_buffers(var.batch_size, var.device)
        self._current_var = var
        wp.launch(
            se3_adjoint_kernel,
            dim=var.batch_size,
            inputs=[var.xyz_wxyz],
            outputs=[self._wp_adjoint_curr],
            device=self.wp_device,
        )
        # discard cached renders at the start of a solve
        if iter_idx == 0:
            self._curr_is_fresh = False
        if self._curr_is_fresh:
            # reuse the accepted proposal render
            self._curr_is_fresh = False
            return
        with torch.no_grad():
            T_torch = wp.to_torch(se3_to_matrix(var.xyz_wxyz))
            rendered, positions = _render_batch(
                self.glctx, T_torch, self.camera_intrinsic, self.merged_faces, self.v_base, self.H, self.W
            )
            self._rendered_curr.copy_(rendered)
            self._positions_curr.copy_(positions)
            if self.blur_sigma >= 0.5:
                self._blurred_curr.copy_(self._blur(self._rendered_curr, self.blur_sigma))

    def accumulate_normal_equations(
        self,
        var_values: VarValues,
        *,
        costs: wp.array,
        JtJ: Optional[wp.array] = None,
        Jtr: Optional[wp.array] = None,
    ):
        """Accumulate a build step or proposed-pose cost step."""
        S, N, H, W = self.batch_size, self.N, self.H, self.W
        if JtJ is None:
            # --- cost step ---
            wp.launch(
                _compute_cost_kernel,
                dim=(S * N, H, W),
                inputs=[self._wp_blurred_prop, self._wp_target_blurred, N],
                outputs=[costs],
                device=self.wp_device,
            )
            if self.target_depths is not None:
                wp.launch(
                    _compute_depth_cost_kernel,
                    dim=(S * N, H, W),
                    inputs=[
                        self._wp_positions_prop,
                        self._wp_target_depth,
                        self.depth_huber_delta,
                        self.depth_weight,
                        N,
                    ],
                    outputs=[costs],
                    device=self.wp_device,
                )
            return
        # --- build step ---
        wp.launch(
            _accumulate_normal_equations_kernel,
            dim=(S * N, H, W),
            inputs=[
                self._wp_blurred_curr,
                self._wp_target_blurred,
                self._wp_positions_curr,
                self.fx,
                self.fy,
                H,
                W,
                N,
                self._wp_adjoint_curr,
            ],
            outputs=[JtJ, Jtr, costs],
            device=self.wp_device,
        )
        if self.target_depths is not None:
            wp.launch(
                _accumulate_depth_normal_equations_kernel,
                dim=(S * N, H, W),
                inputs=[
                    self._wp_positions_curr,
                    self._wp_target_depth,
                    self.fx,
                    self.fy,
                    H,
                    W,
                    self.depth_huber_delta,
                    self.depth_weight,
                    N,
                    self._wp_adjoint_curr,
                ],
                outputs=[JtJ, Jtr, costs],
                device=self.wp_device,
            )

    def on_step(self, accept_mask: wp.array, iter_idx: int, costs: wp.array) -> bool:
        """Update accepted render buffers and return whether patience expired."""
        # select accepted proposal buffers without a CPU synchronization
        S, N = self.batch_size, self.N
        # broadcast each seed decision across its observations
        w = wp.to_torch(accept_mask).to(torch.float32).view(S, 1).expand(S, N).reshape(S * N, 1, 1)
        self._rendered_curr.lerp_(self._rendered_prop, w)
        self._positions_curr.lerp_(self._positions_prop, w.unsqueeze(-1))
        if self.blur_sigma >= 0.5:
            self._blurred_curr.lerp_(self._blurred_prop, w)
        self._curr_is_fresh = True
        # update early-stopping state
        costs_np = costs.numpy()
        m = float(costs_np.min())
        best_idx = int(np.argmin(costs_np))
        if m < self._best_cost:
            self._best_cost = m
            self._best_xyz_wxyz = self._current_var.xyz_wxyz.numpy()[best_idx].copy()
            self._last_improvement = iter_idx
        return iter_idx - self._last_improvement >= self.patience
