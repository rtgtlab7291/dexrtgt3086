# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportIncompatibleVariableOverride=false
from typing import List, Optional, Union

import numpy as np
import warp as wp

from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


@wp.func
def huber_residual_signed(error: float, delta: float) -> float:
    abs_error = wp.abs(error)
    if abs_error <= delta:
        return error
    scale = wp.sqrt(wp.max(2.0 * delta * abs_error - delta * delta, 0.0))
    return scale if error >= 0.0 else -scale


@wp.func
def huber_grad_scale(error: float, delta: float) -> float:
    abs_error = wp.abs(error)
    if abs_error <= delta:
        return 1.0
    denom = wp.sqrt(wp.max(2.0 * delta * abs_error - delta * delta, 1e-12))
    return delta / denom


class WarpFramePositionHuberTask(WarpTask):
    def __init__(
        self,
        robot: WarpRobot,
        link_indices: List[int],
        target_positions: np.ndarray,
        huber_delta: float = 0.02,
        weight: float = 1.0,
        batch_size: int = 1,
    ):
        self.robot = robot
        self.link_indices_list = list(link_indices)
        self.num_links = len(link_indices)
        self.huber_delta = float(huber_delta)
        self.weight = float(weight)
        self.batch_size = batch_size

        self.device: Optional[wp_device_type] = None
        self.target_positions_wp: Optional[wp.array] = None
        self.link_indices_wp: Optional[wp.array] = None
        self.residual_weight: Optional[wp.array] = None

        self._target_positions_np = target_positions.astype(np.float32)
        if self._target_positions_np.ndim == 2:
            self._target_positions_np = self._target_positions_np[None, :, :]

        scale = np.sqrt(1.0 / float(self.num_links))
        residual_weight = np.full(self.num_links * 3, self.weight * scale, dtype=np.float32)
        self._residual_weight_np = residual_weight

    @property
    def residual_dim(self) -> int:
        return self.num_links * 3

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.link_indices_wp = wp.from_numpy(
            np.array(self.link_indices_list, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self.target_positions_wp = wp.from_numpy(self._target_positions_np, dtype=wp.float32, device=device)

    def set_targets(self, target_positions: Union[np.ndarray, wp.array]) -> None:
        if isinstance(target_positions, np.ndarray):
            target_np = target_positions.astype(np.float32)
            self._target_positions_np = target_np[None, :, :] if target_np.ndim == 2 else target_np
            if self.device is not None and self.target_positions_wp is not None:
                target_wp = wp.from_numpy(self._target_positions_np, dtype=wp.float32, device="cpu")
                wp.copy(self.target_positions_wp, target_wp)
                return
        else:
            if self.device is not None and self.target_positions_wp is not None:
                wp.copy(self.target_positions_wp, target_positions)
                return
            self._target_positions_np = target_positions.numpy()

        if self.device is not None:
            self.target_positions_wp = wp.from_numpy(self._target_positions_np, dtype=wp.float32, device=self.device)

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)

        kernel_device = residual_buffer.device if residual_buffer is not None else var.q.device
        if self.device is None or self.device != kernel_device:
            self.init_buffers(kernel_device)

        if residual_buffer is None:
            residual_buffer = wp.empty((self.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_position_huber_residual_kernel,
            dim=(self.batch_size, self.num_links),
            inputs=[
                var.T_world_link.xyz_wxyz,
                self.link_indices_wp,
                self.target_positions_wp,
                self.residual_weight,
                self.huber_delta,
                row_offset,
            ],
            outputs=[residual_buffer],
            device=kernel_device,
        )
        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: WarpRobotState,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)

        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else var.q.device
        if self.device is None or self.device != kernel_device:
            self.init_buffers(kernel_device)

        total_dofs = var.tangent_dim
        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (self.batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device
            )
            row_offset = 0

        base_dim = 6 if var.has_floating_base else 0
        spec_tensors = var.spec_tensors

        wp.launch(
            kernel=compute_position_huber_jacobian_kernel,
            dim=(self.batch_size, self.num_links, self.robot.num_actuated_joints),
            inputs=[
                var.S_world,
                var.T_world_link.xyz_wxyz,
                self.link_indices_wp,
                self.target_positions_wp,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                self.residual_weight,
                self.huber_delta,
                base_dim,
                row_offset,
            ],
            outputs=[jacobian_buffer],
            device=kernel_device,
        )

        if var.has_floating_base:
            wp.launch(
                kernel=compute_position_huber_jacobian_base_kernel,
                dim=(self.batch_size, self.num_links, 6),
                inputs=[
                    var.T_world_link.xyz_wxyz,
                    var.T_world_base.xyz_wxyz,
                    self.link_indices_wp,
                    self.target_positions_wp,
                    self.residual_weight,
                    self.huber_delta,
                    row_offset,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )
        return jacobian_buffer


@wp.kernel
def compute_position_huber_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    link_indices: wp.array1d(dtype=wp.int32),
    target_positions: wp.array3d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    huber_delta: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
):
    batch_idx, link_idx = wp.tid()

    link_id = link_indices[link_idx]
    link_pose = T_world_link[batch_idx, link_id]
    link_pos = wp.vec3(link_pose[0], link_pose[1], link_pose[2])

    target_pos = wp.vec3(
        target_positions[batch_idx, link_idx, 0],
        target_positions[batch_idx, link_idx, 1],
        target_positions[batch_idx, link_idx, 2],
    )
    error = link_pos - target_pos

    res_base_idx = row_offset + link_idx * 3
    for i in range(3):
        res = huber_residual_signed(error[i], huber_delta)
        residual_buffer[batch_idx, res_base_idx + i] = residual_weight[link_idx * 3 + i] * res


@wp.kernel
def compute_position_huber_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_link: wp.array2d(dtype=wp_vec7),
    link_indices: wp.array1d(dtype=wp.int32),
    target_positions: wp.array3d(dtype=wp.float32),
    link_ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    huber_delta: float,
    base_dim: int,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
):
    batch_idx, link_idx, actuated_idx = wp.tid()

    link_id = link_indices[link_idx]
    link_pose = T_world_link[batch_idx, link_id]
    link_pos = wp.vec3(link_pose[0], link_pose[1], link_pose[2])

    target_pos = wp.vec3(
        target_positions[batch_idx, link_idx, 0],
        target_positions[batch_idx, link_idx, 1],
        target_positions[batch_idx, link_idx, 2],
    )
    error = link_pos - target_pos

    num_joints = S_world.shape[2]
    dp_dq = wp.vec3(0.0, 0.0, 0.0)

    for joint_idx in range(num_joints):
        weight_val = joints_to_actuated[joint_idx, actuated_idx]
        if weight_val == 0.0:
            continue

        v = wp.vec3(
            S_world[batch_idx, 0, joint_idx], S_world[batch_idx, 1, joint_idx], S_world[batch_idx, 2, joint_idx]
        )
        omega = wp.vec3(
            S_world[batch_idx, 3, joint_idx],
            S_world[batch_idx, 4, joint_idx],
            S_world[batch_idx, 5, joint_idx],
        )

        if link_ancestor_mask[link_id, joint_idx]:
            dp_dq = dp_dq + weight_val * (v + wp.cross(omega, link_pos))

    res_base_idx = row_offset + link_idx * 3
    col_idx = base_dim + actuated_idx
    for i in range(3):
        scale = huber_grad_scale(error[i], huber_delta)
        jacobian_buffer[batch_idx, res_base_idx + i, col_idx] = residual_weight[link_idx * 3 + i] * scale * dp_dq[i]


@wp.kernel
def compute_position_huber_jacobian_base_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    link_indices: wp.array1d(dtype=wp.int32),
    target_positions: wp.array3d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    huber_delta: float,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
):
    batch_idx, link_idx, base_idx = wp.tid()

    link_id = link_indices[link_idx]
    link_pose = T_world_link[batch_idx, link_id]
    link_pos = wp.vec3(link_pose[0], link_pose[1], link_pose[2])

    target_pos = wp.vec3(
        target_positions[batch_idx, link_idx, 0],
        target_positions[batch_idx, link_idx, 1],
        target_positions[batch_idx, link_idx, 2],
    )
    error = link_pos - target_pos

    unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unit_vec[base_idx] = 1.0
    base_twist = se3_adjoint_multiply_vec6_func(T_world_base[batch_idx], unit_vec)
    v = wp.vec3(base_twist[0], base_twist[1], base_twist[2])
    omega = wp.vec3(base_twist[3], base_twist[4], base_twist[5])

    d_pos = v + wp.cross(omega, link_pos)

    res_base_idx = row_offset + link_idx * 3
    for i in range(3):
        scale = huber_grad_scale(error[i], huber_delta)
        jacobian_buffer[batch_idx, res_base_idx + i, base_idx] = residual_weight[link_idx * 3 + i] * scale * d_pos[i]
