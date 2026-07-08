from typing import Optional, Sequence, cast

import numpy as np
import warp as wp

from robokit.lie.warp_se3 import WarpSE3
from robokit.lie.warp_se3_kernels import se3_inverse_func, se3_jlog_func, se3_log_map_func, se3_multiply_func
from robokit.robo.warp_robot import WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


class WarpBaseStepLimit(WarpTask):
    """
    Soft constraint to limit floating-base step components (Warp backend).

    Unlike PinocchioBaseStepLimit which uses hard QP constraints, this uses
    soft penalties. Set high weight (e.g., 10.0) to strongly discourage movement
    in locked base DOF components.

    When T_world_base_ref is provided:
        Residual: r = log(T_world_base @ T_ref^-1)[lock_indices]
        Jacobian: J = (Jlog @ Ad)[lock_indices, :6]
    This actively pulls the locked DOFs back to the reference pose.

    When T_world_base_ref is None (legacy behavior):
        Residual: r = 0 (only prevents movement via Jacobian regularization)
        Jacobian: J = identity for locked base columns

    Example:
        >>> import warp as wp
        >>> import numpy as np
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> from robokit.lie.warp_se3 import WarpSE3
        >>> from robokit.utils.warp_utils import wp_vec7
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="warp")
        >>> T0_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        >>> T0 = WarpSE3(wp.from_numpy(T0_np, dtype=wp_vec7))
        >>> q = wp.from_numpy(robot.spec.zero_q.reshape(1, -1), dtype=wp.float32)
        >>> state = robot.state(q=q, T_world_base=T0)
        >>> # Lock z, roll, pitch for planar base with reference pose
        >>> limit = WarpBaseStepLimit(lock_indices=[2, 3, 4], T_world_base_ref=T0, weight=10.0, batch_size=1)
    """

    JACOBIAN_MODE = "analytic"

    def __init__(
        self,
        lock_indices: Sequence[int],
        T_world_base_ref: Optional[WarpSE3] = None,
        weight: float = 10.0,
        batch_size: int = 1,
    ):
        """
        Args:
            lock_indices: Base twist indices to lock [0-5] for [vx,vy,vz,wx,wy,wz]
            T_world_base_ref: Reference base pose. If provided, actively corrects deviations.
            weight: Penalty weight (higher = stronger constraint, try 10.0-100.0)
            batch_size: Batch size
        """
        self.lock_indices = list(lock_indices)
        self._T_world_base_ref = T_world_base_ref
        self.batch_size = batch_size

        residual_weight = np.ones(len(lock_indices), dtype=np.float32) * weight
        self._residual_weight_np = residual_weight
        self._lock_indices_np = np.array(lock_indices, dtype=np.int32)
        self.residual_weight = None  # type: ignore[assignment]
        self.lock_indices_wp: Optional[wp.array] = None
        self.device: Optional[wp_device_type] = None

    def set_reference(self, T_world_base_ref: WarpSE3) -> None:
        self._T_world_base_ref = T_world_base_ref

    def init_buffers(self, device: wp_device_type) -> None:
        self.device = device
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self.lock_indices_wp = wp.from_numpy(self._lock_indices_np, dtype=wp.int32, device=device)

    @property
    def residual_dim(self) -> int:
        return len(self.lock_indices)

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self.device is None:
            self.init_buffers(cast(wp_device_type, var.q.device))

        kernel_device = residual_buffer.device if residual_buffer is not None else self.device
        if residual_buffer is None:
            residual_buffer = wp.zeros((self.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)

        if not var.has_floating_base:
            return residual_buffer

        if self._T_world_base_ref is not None:
            wp.launch(
                kernel=compute_base_step_limit_residual_with_ref_kernel,
                dim=(self.batch_size, len(self.lock_indices)),
                inputs=[
                    var.T_world_base.xyz_wxyz,
                    self._T_world_base_ref.xyz_wxyz,
                    self.lock_indices_wp,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[residual_buffer],
                device=kernel_device,
            )
        # else: residual remains zero (legacy behavior for backward compatibility)

        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: WarpRobotState,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self.device is None:
            self.init_buffers(cast(wp_device_type, var.q.device))

        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else self.device
        total_dofs = var.tangent_dim

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (self.batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device
            )
            row_offset = 0

        if not var.has_floating_base:
            return jacobian_buffer

        if self._T_world_base_ref is not None:
            wp.launch(
                kernel=compute_base_step_limit_jacobian_with_ref_kernel,
                dim=(self.batch_size, len(self.lock_indices), 6),
                inputs=[
                    var.T_world_base.xyz_wxyz,
                    self._T_world_base_ref.xyz_wxyz,
                    self.lock_indices_wp,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_base_step_limit_jacobian_kernel,
                dim=(self.batch_size, len(self.lock_indices)),
                inputs=[
                    self.lock_indices_wp,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )

        return jacobian_buffer


@wp.kernel
def compute_base_step_limit_jacobian_kernel(
    lock_indices: wp.array1d(dtype=wp.int32),  # [num_locked]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_locked]
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    """Set identity for locked base DOF columns (legacy kernel without reference)."""
    batch_idx, lock_idx = wp.tid()  # type: ignore

    base_dof_idx = lock_indices[lock_idx]  # Which base DOF to lock (0-5)

    # Set J[lock_idx, base_dof_idx] = weight
    # This makes the optimizer minimize movement in this DOF
    jacobian_buffer[batch_idx, row_offset + lock_idx, base_dof_idx] = residual_weight[lock_idx]


@wp.kernel
def compute_base_step_limit_residual_with_ref_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_ref: wp.array1d(dtype=wp_vec7),  # [batch] or [1]
    lock_indices: wp.array1d(dtype=wp.int32),  # [num_locked]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_locked]
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    """Compute residual as SE(3) log error for locked DOFs only."""
    batch_idx, lock_idx = wp.tid()  # type: ignore

    ref_idx = 0 if T_world_base_ref.shape[0] == 1 else batch_idx
    T_ref_inv = se3_inverse_func(T_world_base_ref[ref_idx])
    T_error = se3_multiply_func(T_world_base[batch_idx], T_ref_inv)

    r_twist = se3_log_map_func(T_error, 1e-4)

    dof_idx = lock_indices[lock_idx]
    residual_buffer[batch_idx, row_offset + lock_idx] = residual_weight[lock_idx] * r_twist[dof_idx]  # type: ignore[index]


@wp.kernel
def compute_base_step_limit_jacobian_with_ref_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_ref: wp.array1d(dtype=wp_vec7),  # [batch] or [1]
    lock_indices: wp.array1d(dtype=wp.int32),  # [num_locked]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_locked]
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    """Compute Jacobian as (Jlog @ Ad)[locked_row, col] for base columns only."""
    batch_idx, lock_idx, col_idx = wp.tid()  # type: ignore

    ref_idx = 0 if T_world_base_ref.shape[0] == 1 else batch_idx
    T_ref_inv = se3_inverse_func(T_world_base_ref[ref_idx])
    T_error = se3_multiply_func(T_world_base[batch_idx], T_ref_inv)

    jlog = se3_jlog_func(T_error, 1e-4)

    unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unit_vec[col_idx] = 1.0
    ad_col = se3_adjoint_multiply_vec6_func(T_world_base_ref[ref_idx], unit_vec)

    result_col = wp.mul(jlog, ad_col)

    dof_idx = lock_indices[lock_idx]
    jacobian_buffer[batch_idx, row_offset + lock_idx, col_idx] = residual_weight[lock_idx] * result_col[dof_idx]  # type: ignore[index]
