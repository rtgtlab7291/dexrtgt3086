# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import (
    se3_adjoint_func,
    se3_compose_func,
    se3_inverse_func,
    se3_jlog_func,
    se3_log_map_func,
)
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot, RobotState
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


# --- task -------------------------------------------------------------------
class SmoothnessTask(ResidualTask):
    """Penalize change from a previous joint and optional base state.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"))
        >>> state_prev = robot.state(q=robot.spec.zero_q)
        >>> state_curr = robot.state(q=robot.spec.zero_q + 0.1)
        >>> smooth_task = SmoothnessTask(robot=robot, weight=0.5)
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        prev_var: Optional[RobotState] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        base_weight: Optional[Union[float, Sequence[float]]] = None,
    ):
        self.weight = weight
        self.prev_var = prev_var.clone() if prev_var is not None else None
        self._base_weight = base_weight
        self._include_base_residual = base_weight is not None or (prev_var is not None and prev_var.has_floating_base)
        self.robot: Optional[Robot] = None
        self.device = None
        self.residual_weight = None
        if robot is not None:
            self.set_robot(robot)

    def set_robot(self, robot: Robot):
        self.robot = robot
        num_joints = robot.spec.num_actuated_joints

        residual_weight = np.ones(num_joints, dtype=np.float32)
        if self.weight is not None:
            if isinstance(self.weight, (float, int)):
                residual_weight[:] = self.weight
            else:
                residual_weight[:] = self.weight

        self._joint_weight_np = residual_weight.copy()
        if self._include_base_residual:
            self._residual_dim = num_joints + 6
            residual_weight_full = np.ones(self._residual_dim, dtype=np.float32)
            residual_weight_full[:num_joints] = self._joint_weight_np
            if self._base_weight is not None:
                if isinstance(self._base_weight, (float, int)):
                    residual_weight_full[num_joints:] = self._base_weight
                else:
                    residual_weight_full[num_joints:] = self._base_weight
            else:
                residual_weight_full[num_joints:] = self._joint_weight_np[0] if len(self._joint_weight_np) > 0 else 1.0
            self._residual_weight_np = residual_weight_full
        else:
            self._residual_dim = num_joints
            self._residual_weight_np = residual_weight

    def set_weight(self, weight: Union[float, Sequence[float]]):
        """Update joint-block residual weights in-place. Base-block weights are preserved."""
        num_joints = self.robot.spec.num_actuated_joints
        self._joint_weight_np[:] = weight
        self._residual_weight_np[:num_joints] = weight
        self.weight = weight
        if self.residual_weight is not None:
            wp.copy(self.residual_weight, wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device="cpu"))

    def set_prev_state(self, prev_state: RobotState):
        if self.prev_var is None:
            self.prev_var = prev_state.clone()
            return
        if self.prev_var.q.shape != prev_state.q.shape:
            self.prev_var = prev_state.clone()
            return
        wp.copy(self.prev_var.q, prev_state.q)
        if self.prev_var.has_floating_base and prev_state.has_floating_base:
            wp.copy(self.prev_var.T_world_base, prev_state.T_world_base)

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        prev_var: Optional[RobotState] = None,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if prev_var is None:
            prev_var = self.prev_var

        batch_size = var.batch_size
        if prev_var is None:
            if out_residual is None:
                return wp.zeros((batch_size, self.residual_dim), dtype=wp.float32, device=var.q.device)
            else:
                return out_residual
        if self._include_base_residual and (not var.has_floating_base or not prev_var.has_floating_base):
            raise ValueError("SmoothnessTask is configured with base residuals but var/prev_var has no floating base.")

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = out_residual.device if out_residual is not None else self.device
        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self._include_base_residual:
            wp.launch(
                kernel=compute_smoothness_task_weighted_residual_with_base_kernel,
                dim=batch_size,
                inputs=[
                    var.q,
                    prev_var.q,
                    var.T_world_base,
                    prev_var.T_world_base,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_smoothness_task_weighted_residual_kernel,
                dim=(batch_size, self.robot.spec.num_actuated_joints),
                inputs=[
                    var.q,
                    prev_var.q,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )

        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        prev_var: Optional[RobotState] = None,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        col_offset = var_values.tangent_offset(self.var_key)
        if prev_var is None:
            prev_var = self.prev_var

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = out_jacobian.device if out_jacobian is not None else self.device
        total_dofs = var.tangent_dim

        batch_size = var.batch_size
        if out_jacobian is None:
            out_jacobian = wp.zeros((batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device)
            row_offset = 0
            col_offset = 0

        if prev_var is None:
            return out_jacobian
        if self._include_base_residual and (not var.has_floating_base or not prev_var.has_floating_base):
            raise ValueError("SmoothnessTask is configured with base residuals but var/prev_var has no floating base.")

        if self._include_base_residual:
            wp.launch(
                kernel=compute_smoothness_task_weighted_jacobian_with_base_kernel,
                dim=(batch_size, total_dofs),
                inputs=[
                    var.T_world_base,
                    prev_var.T_world_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                    self.robot.spec.num_actuated_joints,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_smoothness_task_weighted_jacobian_kernel,
                dim=(batch_size, self.robot.spec.num_actuated_joints),
                inputs=[
                    var.has_floating_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )

        return out_jacobian


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_smoothness_task_weighted_residual_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [prev_batch, num_joints]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore
    n_repeat = q_curr.shape[0] // q_prev.shape[0]
    prev_idx = batch_idx // n_repeat
    diff = q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]
    out_residual[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * diff


@wp.kernel
def compute_smoothness_task_weighted_residual_with_base_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    T_world_base_curr: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_prev: wp.array1d(dtype=wp_vec7),  # [prev_batch]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx = wp.tid()  # type: ignore
    num_joints = q_curr.shape[1]
    n_repeat = q_curr.shape[0] // q_prev.shape[0]
    prev_idx = batch_idx // n_repeat

    # joint residuals
    for joint_idx in range(num_joints):
        diff = q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]
        out_residual[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * diff

    # base residuals
    T_prev_inv = se3_inverse_func(T_world_base_prev[prev_idx])
    T_error = se3_compose_func(T_world_base_curr[batch_idx], T_prev_inv)
    r_base = se3_log_map_func(T_error, wp.float32(1e-4))

    out_residual[batch_idx, row_offset + num_joints + 0] = residual_weight[num_joints + 0] * r_base[0]
    out_residual[batch_idx, row_offset + num_joints + 1] = residual_weight[num_joints + 1] * r_base[1]
    out_residual[batch_idx, row_offset + num_joints + 2] = residual_weight[num_joints + 2] * r_base[2]
    out_residual[batch_idx, row_offset + num_joints + 3] = residual_weight[num_joints + 3] * r_base[3]
    out_residual[batch_idx, row_offset + num_joints + 4] = residual_weight[num_joints + 4] * r_base[4]
    out_residual[batch_idx, row_offset + num_joints + 5] = residual_weight[num_joints + 5] * r_base[5]


@wp.kernel
def compute_smoothness_task_weighted_jacobian_kernel(
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore

    base_col_offset = 6 if has_floating_base else 0

    # joint identity block
    out_jacobian[batch_idx, row_offset + joint_idx, col_offset + base_col_offset + joint_idx] = residual_weight[
        joint_idx
    ]


@wp.kernel
def compute_smoothness_task_weighted_jacobian_with_base_kernel(
    T_world_base_curr: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_prev: wp.array1d(dtype=wp_vec7),  # [prev_batch]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    row_offset: int,
    col_offset: int,
    num_joints: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, col_idx = wp.tid()  # type: ignore
    n_repeat = T_world_base_curr.shape[0] // T_world_base_prev.shape[0]
    prev_idx = batch_idx // n_repeat

    # joint identity block starts after the base columns
    if col_idx >= 6 and col_idx < (6 + num_joints):
        joint_idx = col_idx - 6
        out_jacobian[batch_idx, row_offset + joint_idx, col_offset + col_idx] = residual_weight[joint_idx]

    # base block is Jlog @ Ad
    if col_idx < 6:
        T_prev_inv = se3_inverse_func(T_world_base_prev[prev_idx])
        T_error = se3_compose_func(T_world_base_curr[batch_idx], T_prev_inv)

        # compute Jlog @ Ad column by column
        jlog = se3_jlog_func(T_error, wp.float32(1e-4))

        # adjoint column for the selected basis vector
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = 1.0
        ad_col = se3_adjoint_func(T_world_base_prev[prev_idx]) * unit_vec

        # apply Jlog
        result_col = wp.mul(jlog, ad_col)

        out_jacobian[batch_idx, row_offset + num_joints + 0, col_offset + col_idx] = (
            residual_weight[num_joints + 0] * result_col[0]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 1, col_offset + col_idx] = (
            residual_weight[num_joints + 1] * result_col[1]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 2, col_offset + col_idx] = (
            residual_weight[num_joints + 2] * result_col[2]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 3, col_offset + col_idx] = (
            residual_weight[num_joints + 3] * result_col[3]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 4, col_offset + col_idx] = (
            residual_weight[num_joints + 4] * result_col[4]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 5, col_offset + col_idx] = (
            residual_weight[num_joints + 5] * result_col[5]
        )
