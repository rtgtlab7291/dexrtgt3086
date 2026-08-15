# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import quaternion_to_matrix_func
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec7


_EXP_BARRIER_ACTIVATION = wp.constant(0.8)
_EXP_BARRIER_SCALE = wp.constant(5.0)
_EXP_BARRIER_POWER = wp.constant(4.0)


# --- base kernels -----------------------------------------------------------
@wp.kernel
def _base_position_limit_residual_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),
    axis: int,
    lo: float,
    hi: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    row: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    p = T_world_base[batch_idx][axis]
    error = wp.max(wp.float32(0.0), lo - p) + wp.max(wp.float32(0.0), p - hi)
    residual = error
    if use_sqrt and error > wp.float32(0.0):
        residual = wp.sqrt(error + eps)
    out_residual[batch_idx, row] = weight * residual


@wp.kernel
def _base_position_limit_jacobian_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),
    axis: int,
    lo: float,
    hi: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    row: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    p = T_world_base[batch_idx][axis]
    sign = wp.where(p > hi, wp.float32(1.0), wp.where(p < lo, wp.float32(-1.0), wp.float32(0.0)))
    error = wp.max(wp.float32(0.0), lo - p) + wp.max(wp.float32(0.0), p - hi)
    scale = wp.float32(1.0)
    if use_sqrt and sign != wp.float32(0.0):
        scale = wp.float32(0.5) / wp.sqrt(error + eps)
    T = T_world_base[batch_idx]
    R = quaternion_to_matrix_func(wp.vec4(T[3], T[4], T[5], T[6]))
    for col in range(3):
        out_jacobian[batch_idx, row, col_offset + col] = weight * sign * scale * R[axis, col]


@wp.kernel
def _base_position_limit_cost_and_gradient_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),
    axis: int,
    lo: float,
    hi: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    col_offset: int,
    out_cost: wp.array1d(dtype=wp.float32),
    out_gradient: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    p = T_world_base[batch_idx][axis]
    sign = wp.where(p > hi, wp.float32(1.0), wp.where(p < lo, wp.float32(-1.0), wp.float32(0.0)))
    if sign == wp.float32(0.0):
        return
    error = wp.max(wp.float32(0.0), lo - p) + wp.max(wp.float32(0.0), p - hi)
    residual = weight * error
    jac_scale = weight * sign
    if use_sqrt:
        residual = weight * wp.sqrt(error + eps)
        jac_scale *= wp.float32(0.5) / wp.sqrt(error + eps)
    wp.atomic_add(out_cost, batch_idx, wp.float32(0.5) * residual * residual)
    if out_gradient:
        T = T_world_base[batch_idx]
        R = quaternion_to_matrix_func(wp.vec4(T[3], T[4], T[5], T[6]))
        for col in range(3):
            wp.atomic_add(out_gradient, batch_idx, col_offset + col, jac_scale * R[axis, col] * residual)


# --- joint barrier kernels --------------------------------------------------
@wp.kernel
def compute_position_limit_exp_barrier_residual_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, n_dofs]
    joint_limits: wp.array2d(dtype=wp.float32),  # [n_dofs, 2]
    residual_weight: wp.array1d(dtype=wp.float32),  # [n_dofs]
    dof_mask: wp.array1d(dtype=wp.float32),  # [n_dofs]
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_dim]
):
    """Exponential barrier with a dead zone around the range center."""
    batch_idx, dof_idx = wp.tid()  # pyright: ignore

    lower = joint_limits[dof_idx, 0]
    upper = joint_limits[dof_idx, 1]
    half = (upper - lower) * wp.float32(0.5)
    center = lower + half

    q_val = q[batch_idx, dof_idx]
    norm = (q_val - center) / (half + wp.float32(1e-8))
    abs_norm = wp.where(norm >= wp.float32(0.0), norm, -norm)
    violation = wp.max(wp.float32(0.0), abs_norm - _EXP_BARRIER_ACTIVATION)
    penalty = wp.float32(1.0) - wp.exp(-wp.pow(violation * _EXP_BARRIER_SCALE, _EXP_BARRIER_POWER))

    out_residual[batch_idx, row_offset + dof_idx] = (
        (q_val - center) * penalty * residual_weight[dof_idx] * dof_mask[dof_idx]
    )


@wp.kernel
def compute_position_limit_exp_barrier_jacobian_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, n_dofs]
    joint_limits: wp.array2d(dtype=wp.float32),  # [n_dofs, 2]
    residual_weight: wp.array1d(dtype=wp.float32),  # [n_dofs]
    dof_mask: wp.array1d(dtype=wp.float32),  # [n_dofs]
    row_offset: int,
    col_offset: int,
    base_col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_res, total_dofs]
):
    """Analytic diagonal Jacobian for the exponential barrier."""
    batch_idx, dof_idx = wp.tid()  # pyright: ignore

    lower = joint_limits[dof_idx, 0]
    upper = joint_limits[dof_idx, 1]
    half = (upper - lower) * wp.float32(0.5)
    center = lower + half
    inv_half = wp.float32(1.0) / (half + wp.float32(1e-8))

    q_val = q[batch_idx, dof_idx]
    norm = (q_val - center) * inv_half
    abs_norm = wp.where(norm >= wp.float32(0.0), norm, -norm)
    violation = wp.max(wp.float32(0.0), abs_norm - _EXP_BARRIER_ACTIVATION)

    scaled_violation = violation * _EXP_BARRIER_SCALE
    exp_term = wp.exp(-wp.pow(scaled_violation, _EXP_BARRIER_POWER))
    penalty = wp.float32(1.0) - exp_term

    sign_norm = wp.where(norm >= wp.float32(0.0), wp.float32(1.0), wp.float32(-1.0))
    active = wp.where(violation > wp.float32(0.0), wp.float32(1.0), wp.float32(0.0))
    d_penalty = (
        exp_term
        * _EXP_BARRIER_POWER
        * scaled_violation
        * scaled_violation
        * scaled_violation
        * _EXP_BARRIER_SCALE
        * sign_norm
        * inv_half
        * active
    )

    diag = (penalty + (q_val - center) * d_penalty) * residual_weight[dof_idx] * dof_mask[dof_idx]
    out_jacobian[batch_idx, row_offset + dof_idx, col_offset + base_col_offset + dof_idx] = diag


# --- task -------------------------------------------------------------------
class PositionLimit(RobotTask, ResidualTask, GradientTask):
    """Penalize joint and optional floating-base translation limit violations.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarValues
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"))
        >>> joint_var = robot.spec.actuated_joint_limits[:, 0] - 0.1
        >>> state = robot.state(q=wp.from_numpy(joint_var[None].astype(np.float32), dtype=wp.float32))
        >>> position_limit = PositionLimit(robot=robot)
        >>> weighted_residual = position_limit.compute_weighted_residual(VarValues(robot=state))
        >>> np.allclose(weighted_residual.numpy()[0], np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32))
        True
        >>> weighted_jacobian = position_limit.compute_weighted_jacobian(VarValues(robot=state))
        >>> np.allclose(np.diag(weighted_jacobian.numpy()[0]), np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32))
        True
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        residual_mode: Literal["abs", "sqrt_abs", "exp_barrier"] = "abs",
        residual_eps: float = 1e-6,
        mask: Optional[np.ndarray] = None,
        base_axis: Optional[int] = None,
        base_bounds: Tuple[float, float] = (0.3, 0.76),
        base_weight: float = 50.0,
        include_joints: bool = True,
    ):
        if residual_mode not in {"abs", "sqrt_abs", "exp_barrier"}:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")
        self.weight = weight
        self.residual_mode = residual_mode
        self.residual_eps = residual_eps
        self.mask = mask
        self.base_axis = base_axis
        self.base_bounds = (float(base_bounds[0]), float(base_bounds[1]))
        self.base_weight = float(base_weight)
        self.include_joints = include_joints
        if base_axis is not None and base_axis not in {0, 1, 2}:
            raise ValueError("base_axis must be 0, 1, or 2.")
        if self.base_bounds[0] >= self.base_bounds[1]:
            raise ValueError("base_bounds must satisfy lo < hi.")
        if not include_joints and base_axis is None:
            raise ValueError("PositionLimit must include joints or a base bound.")
        self.robot: Optional[Robot] = None
        self.device = None
        self.dof_mask: Optional[wp.array] = None
        if robot is not None:
            self.set_robot(robot)

    def set_robot(self, robot: Robot):
        self.robot = robot
        self.n_dofs = robot.spec.num_actuated_joints
        self._joint_rows = self.n_dofs if self.include_joints else 0
        self._base_rows = 1 if self.base_axis is not None else 0
        self._residual_dim = self._joint_rows + self._base_rows

        residual_weight = np.zeros(self.n_dofs, dtype=np.float32)
        residual_weight[:] = self.weight if self.weight is not None else 1.0
        self._residual_weight_np = residual_weight
        self.residual_weight = wp.from_numpy(residual_weight, dtype=wp.float32)

        self._dof_mask_np = (
            self.mask.astype(np.float32) if self.mask is not None else np.ones(self.n_dofs, dtype=np.float32)
        )

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self.dof_mask = wp.from_numpy(self._dof_mask_np, dtype=wp.float32, device=device)

    def set_weight(self, weight: Union[float, Sequence[float]]):
        """Update residual weights in-place. Accepts a scalar or per-joint sequence."""
        self._residual_weight_np[:] = weight
        self.weight = weight
        if self.device is not None:
            wp.copy(self.residual_weight, wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device="cpu"))

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: wp.array = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if self.device is None:
            self.init_buffers(var.q.device)
        batch_size = var.q.shape[0]
        if out_residual is None:
            out_residual = wp.zeros((batch_size, self.residual_dim), dtype=wp.float32, device=self.device)
            row_offset = 0
        if self.include_joints and self.residual_mode == "exp_barrier":
            wp.launch(
                kernel=compute_position_limit_exp_barrier_residual_kernel,
                dim=(batch_size, self.n_dofs),
                inputs=[
                    var.q,
                    var.spec_tensors.actuated_joint_limits,
                    self.residual_weight,
                    self.dof_mask,
                    row_offset,
                ],
                outputs=[out_residual],
                device=self.device,
            )
        elif self.include_joints:
            wp.launch(
                kernel=compute_position_limit_weighted_residual_kernel,
                dim=(batch_size, self.n_dofs),
                inputs=[
                    var.q,
                    var.spec_tensors.actuated_joint_limits,
                    self.residual_weight,
                    self.residual_mode == "sqrt_abs",
                    self.residual_eps,
                    row_offset,
                ],
                outputs=[out_residual],
                device=self.device,
            )
        if self.base_axis is not None and var.has_floating_base:
            wp.launch(
                kernel=_base_position_limit_residual_kernel,
                dim=batch_size,
                inputs=[
                    var.T_world_base,
                    self.base_axis,
                    self.base_bounds[0],
                    self.base_bounds[1],
                    self.base_weight,
                    self.residual_mode == "sqrt_abs",
                    self.residual_eps,
                    row_offset + self._joint_rows,
                ],
                outputs=[out_residual],
                device=self.device,
            )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: wp.array = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        col_offset = var_values.tangent_offset(self.var_key)
        if self.device is None:
            self.init_buffers(var.q.device)
        total_dofs = var.tangent_dim
        base_col_offset = 6 if var.has_floating_base else 0
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (var.batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=self.device
            )
            row_offset = 0
            col_offset = 0
        if self.include_joints and self.residual_mode == "exp_barrier":
            wp.launch(
                kernel=compute_position_limit_exp_barrier_jacobian_kernel,
                dim=(var.batch_size, self.n_dofs),
                inputs=[
                    var.q,
                    var.spec_tensors.actuated_joint_limits,
                    self.residual_weight,
                    self.dof_mask,
                    row_offset,
                    col_offset,
                    base_col_offset,
                ],
                outputs=[out_jacobian],
                device=self.device,
            )
        elif self.include_joints:
            wp.launch(
                kernel=compute_position_limit_weighted_jacobian_kernel,
                dim=(var.batch_size, self.n_dofs),
                inputs=[
                    var.q,
                    var.spec_tensors.actuated_joint_limits,
                    self.residual_weight,
                    self.residual_mode == "sqrt_abs",
                    self.residual_eps,
                    row_offset,
                    col_offset,
                    base_col_offset,
                ],
                outputs=[out_jacobian],
                device=self.device,
            )
        if self.base_axis is not None and var.has_floating_base:
            wp.launch(
                kernel=_base_position_limit_jacobian_kernel,
                dim=var.batch_size,
                inputs=[
                    var.T_world_base,
                    self.base_axis,
                    self.base_bounds[0],
                    self.base_bounds[1],
                    self.base_weight,
                    self.residual_mode == "sqrt_abs",
                    self.residual_eps,
                    row_offset + self._joint_rows,
                    col_offset,
                ],
                outputs=[out_jacobian],
                device=self.device,
            )
        return out_jacobian

    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
    ):
        col_offset = var_values.tangent_offset(self.var_key)
        var = var_values.get(self.var_key)
        if self.device is None:
            self.init_buffers(var.q.device)
        base_col_offset = 6 if var.has_floating_base else 0
        if self.include_joints:
            wp.launch(
                kernel=_compute_position_limit_cost_and_gradient_kernel,
                dim=(var.batch_size, self.n_dofs),
                inputs=[
                    var.q,
                    var.spec_tensors.actuated_joint_limits,
                    self.residual_weight,
                    self.dof_mask,
                    self.residual_mode == "exp_barrier",
                    self.residual_mode == "sqrt_abs",
                    self.residual_eps,
                    col_offset,
                    base_col_offset,
                ],
                outputs=[out_cost, out_gradient],
                device=self.device,
            )
        if self.base_axis is not None and var.has_floating_base:
            wp.launch(
                kernel=_base_position_limit_cost_and_gradient_kernel,
                dim=var.batch_size,
                inputs=[
                    var.T_world_base,
                    self.base_axis,
                    self.base_bounds[0],
                    self.base_bounds[1],
                    self.base_weight,
                    self.residual_mode == "sqrt_abs",
                    self.residual_eps,
                    col_offset,
                ],
                outputs=[out_cost, out_gradient],
                device=self.device,
            )


# --- joint limit kernels ----------------------------------------------------
@wp.kernel
def compute_position_limit_weighted_residual_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, n_dofs]
    joint_limits: wp.array2d(dtype=wp.float32),  # [n_dofs, 2]
    residual_weight: wp.array1d(dtype=wp.float32),  # [n_dofs]
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, dof_idx = wp.tid()  # pyright: ignore

    joint_pos = q[batch_idx, dof_idx]
    lower_limit = joint_limits[dof_idx, 0]
    upper_limit = joint_limits[dof_idx, 1]

    error_upper = wp.max(0.0, joint_pos - upper_limit)
    error_lower = wp.max(0.0, lower_limit - joint_pos)

    error = error_upper + error_lower
    residual = error
    if use_sqrt:
        residual = wp.where(error > wp.float32(0.0), wp.sqrt(error + eps), wp.float32(0.0))
    out_residual[batch_idx, row_offset + dof_idx] = residual_weight[dof_idx] * residual


@wp.kernel
def compute_position_limit_weighted_jacobian_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, n_dofs]
    joint_limits: wp.array2d(dtype=wp.float32),  # [n_dofs, 2]
    residual_weight: wp.array1d(dtype=wp.float32),  # [n_dofs]
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    col_offset: int,
    base_col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, dof_idx = wp.tid()  # pyright: ignore

    joint_pos = q[batch_idx, dof_idx]
    lower_limit = joint_limits[dof_idx, 0]
    upper_limit = joint_limits[dof_idx, 1]

    sign = 0.0
    if joint_pos > upper_limit:
        sign = 1.0
    elif joint_pos < lower_limit:
        sign = -1.0

    error_upper = wp.max(0.0, joint_pos - upper_limit)
    error_lower = wp.max(0.0, lower_limit - joint_pos)
    error = error_upper + error_lower
    scale = wp.float32(1.0)
    if use_sqrt and sign != 0.0:
        scale = wp.float32(0.5) / wp.sqrt(error + eps)

    out_jacobian[batch_idx, row_offset + dof_idx, col_offset + base_col_offset + dof_idx] = (
        residual_weight[dof_idx] * sign * scale
    )


@wp.kernel
def _compute_position_limit_cost_and_gradient_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, n_dofs]
    joint_limits: wp.array2d(dtype=wp.float32),  # [n_dofs, 2]
    residual_weight: wp.array1d(dtype=wp.float32),  # [n_dofs]
    dof_mask: wp.array1d(dtype=wp.float32),  # [n_dofs]
    use_exp_barrier: wp.bool,
    use_sqrt: wp.bool,
    eps: float,
    col_offset: int,
    base_col_offset: int,
    out_cost: wp.array1d(dtype=wp.float32),  # [batch]
    out_gradient: wp.array2d(dtype=wp.float32),  # [batch, total_dofs]
):
    batch_idx, dof_idx = wp.tid()  # pyright: ignore

    joint_pos = q[batch_idx, dof_idx]
    lower_limit = joint_limits[dof_idx, 0]
    upper_limit = joint_limits[dof_idx, 1]

    if use_exp_barrier:
        half = (upper_limit - lower_limit) * wp.float32(0.5)
        center = lower_limit + half
        inv_half = wp.float32(1.0) / (half + wp.float32(1e-8))
        offset = joint_pos - center
        norm = offset * inv_half
        abs_norm = wp.where(norm >= wp.float32(0.0), norm, -norm)
        violation = wp.max(wp.float32(0.0), abs_norm - _EXP_BARRIER_ACTIVATION)
        if violation == wp.float32(0.0):
            return

        scaled_violation = violation * _EXP_BARRIER_SCALE
        exp_term = wp.exp(-wp.pow(scaled_violation, _EXP_BARRIER_POWER))
        penalty = wp.float32(1.0) - exp_term
        sign_norm = wp.where(norm >= wp.float32(0.0), wp.float32(1.0), wp.float32(-1.0))
        d_penalty = (
            exp_term
            * _EXP_BARRIER_POWER
            * scaled_violation
            * scaled_violation
            * scaled_violation
            * _EXP_BARRIER_SCALE
            * sign_norm
            * inv_half
        )
        w = residual_weight[dof_idx] * dof_mask[dof_idx]
        residual = w * offset * penalty
        jdd = w * (penalty + offset * d_penalty)
        wp.atomic_add(out_cost, batch_idx, wp.float32(0.5) * residual * residual)
        if out_gradient:
            wp.atomic_add(out_gradient, batch_idx, col_offset + base_col_offset + dof_idx, jdd * residual)
        return

    sign = wp.float32(0.0)
    if joint_pos > upper_limit:
        sign = wp.float32(1.0)
    elif joint_pos < lower_limit:
        sign = wp.float32(-1.0)

    if sign == wp.float32(0.0):
        return

    error = wp.max(0.0, joint_pos - upper_limit) + wp.max(0.0, lower_limit - joint_pos)
    w = residual_weight[dof_idx]

    if use_sqrt:
        residual = w * wp.sqrt(error + eps)
        jdd = w * sign * (wp.float32(0.5) / wp.sqrt(error + eps))
    else:
        residual = w * error
        jdd = w * sign

    wp.atomic_add(out_cost, batch_idx, wp.float32(0.5) * residual * residual)
    if out_gradient:  # null on cost-only (line-search) calls
        wp.atomic_add(out_gradient, batch_idx, col_offset + base_col_offset + dof_idx, jdd * residual)
