# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.warp_se3 import WarpSE3
from robokit.lie.warp_se3_kernels import se3_inverse_func, se3_jlog_func, se3_log_map_func, se3_multiply_func
from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_vec6, wp_vec7


class WarpFrameTask(WarpTask):
    """
    Warp-based frame task for computing weighted SE(3) residuals and Jacobians.
    Supports both single-frame and multi-frame targets.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
        >>> target_link_index = robot.link_names.index("ee_link")
        >>> state = robot.state(q=robot.zero_q)
        >>> target_link_pose = robot.forward_kinematics(state).get_T_world_link(target_link_index)
        >>> frame_task = WarpFrameTask(robot, target_link_index, target_link_pose, position_weight=1.0, orientation_weight=1.0)
        >>> random_q = wp.from_numpy(np.array([[-0.5, 0.3, -0.2, 0.4, -0.1, 0.25]], dtype=np.float32), dtype=wp.float32)
        >>> state = robot.state()
        >>> state.set_configuration(q=random_q)
        >>> robot.forward_kinematics(state)
        >>> robot.compute_motion_subspace(state)
        >>> weighted_residual = frame_task.compute_weighted_residual(state)
        >>> weighted_jacobian = frame_task.compute_weighted_jacobian(state)
        >>> np.allclose(weighted_residual.numpy()[0, :3], np.array([0.5839, -0.0795, -0.1784], dtype=np.float32), atol=1e-3)
        True
        >>> np.allclose(weighted_jacobian.numpy()[0, :, 0], np.array([-1.1628, 0.1626, -0.167, -0.0765, 0.3698, 0.9521], dtype=np.float32), atol=1e-3)
        True
    """

    def __init__(
        self,
        robot: WarpRobot,
        frame_index: Union[int, Sequence[int]],
        T_world_target: Union[WarpSE3, Sequence[WarpSE3]],
        position_weight: Union[float, Sequence[float]],
        orientation_weight: Union[float, Sequence[float]],
        num_seeds: int = 1,
    ):
        self.robot = robot
        self.num_seeds = num_seeds

        if isinstance(frame_index, int):
            frame_indices_list = [frame_index]
        else:
            frame_indices_list = list(frame_index)

        if isinstance(T_world_target, WarpSE3):
            targets_list = [T_world_target]
        else:
            targets_list = list(T_world_target)

        if len(frame_indices_list) != len(targets_list):
            raise ValueError(
                f"Number of frame indices ({len(frame_indices_list)}) must match "
                f"number of targets ({len(targets_list)})"
            )

        self.num_frames = len(frame_indices_list)
        self._frame_indices_np = np.array(frame_indices_list, dtype=np.int32)

        first_target = targets_list[0]
        self.device = first_target.xyz_wxyz.device
        self._batch_size = first_target.batch_size

        self.frame_indices = wp.from_numpy(self._frame_indices_np, dtype=wp.int32, device=self.device)

        self.T_world_target = WarpSE3.stack(targets_list, axis=1)
        if self.num_frames == 1:
            self.frame_index = frame_indices_list[0]
        else:
            self.frame_index = -1

        if self.num_seeds == 1:
            self.dim = self._batch_size
        else:
            self.dim = self._batch_size * self.num_seeds

        if isinstance(position_weight, (int, float)):
            position_weights = [float(position_weight)] * self.num_frames
        else:
            position_weights = list(position_weight)

        if isinstance(orientation_weight, (int, float)):
            orientation_weights = [float(orientation_weight)] * self.num_frames
        else:
            orientation_weights = list(orientation_weight)

        if len(position_weights) != self.num_frames:
            raise ValueError(
                f"Number of position weights ({len(position_weights)}) must match "
                f"number of frames ({self.num_frames})"
            )
        if len(orientation_weights) != self.num_frames:
            raise ValueError(
                f"Number of orientation weights ({len(orientation_weights)}) must match "
                f"number of frames ({self.num_frames})"
            )

        residual_weight = np.zeros(self.num_frames * 6, dtype=np.float32)
        for i in range(self.num_frames):
            residual_weight[i * 6 : i * 6 + 3] = position_weights[i]
            residual_weight[i * 6 + 3 : i * 6 + 6] = orientation_weights[i]
        self._residual_weight_np = residual_weight
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=self.device)

    def set_target(self, target: Union[WarpSE3, Sequence[WarpSE3]]) -> None:
        """Update target poses in-place. Batch size must match the original."""
        targets = [target] if isinstance(target, WarpSE3) else list(target)
        WarpSE3.stack(targets, axis=1, dest=self.T_world_target)

    @property
    def residual_dim(self) -> int:
        return 6 * self.num_frames

    def compute_weighted_residual(
        self, var: WarpRobotState, residual_buffer: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        kernel_device = residual_buffer.device if residual_buffer is not None else self.device
        if residual_buffer is None:
            residual_buffer = wp.empty((self.dim, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self.num_frames == 1:
            wp.launch(
                kernel=compute_frame_task_weighted_residual_kernel,
                dim=[self.dim],
                inputs=[
                    var.T_world_link.xyz_wxyz,
                    self.frame_index,
                    self.T_world_target,
                    self.num_seeds,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[residual_buffer],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_multi_frame_task_weighted_residual_kernel,
                dim=[self.dim],
                inputs=[
                    var.T_world_link.xyz_wxyz,
                    self.frame_indices,
                    self.T_world_target,
                    self.num_frames,
                    self.num_seeds,
                    self.residual_weight,
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
        col_offset: int = 0,
    ) -> wp.array:
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        spec_tensor = var.spec_tensors
        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else self.device
        if jacobian_buffer is None:
            total_dofs = var.tangent_dim
            jacobian_buffer = wp.zeros(
                (self.dim, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device
            )
            row_offset = 0
        num_actuated = spec_tensor.joints_to_actuated_mapping.shape[1]

        if self.num_frames == 1:
            wp.launch(
                kernel=compute_frame_task_weighted_jacobian_kernel,
                dim=(self.dim, num_actuated),
                inputs=[
                    var.S_world,
                    var.T_world_link.xyz_wxyz,
                    var.T_world_base.xyz_wxyz,
                    self.T_world_target,
                    self.num_seeds,
                    self.frame_index,
                    spec_tensor.link_ancestor_joints_mask,
                    spec_tensor.joints_to_actuated_mapping,
                    var.has_floating_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_multi_frame_task_weighted_jacobian_kernel,
                dim=(self.dim, num_actuated),
                inputs=[
                    var.S_world,
                    var.T_world_link.xyz_wxyz,
                    var.T_world_base.xyz_wxyz,
                    self.T_world_target,
                    self.frame_indices,
                    self.num_frames,
                    self.num_seeds,
                    spec_tensor.link_ancestor_joints_mask,
                    spec_tensor.joints_to_actuated_mapping,
                    var.has_floating_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )
        return jacobian_buffer


@wp.kernel
def compute_frame_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_target: wp.array2d(dtype=wp_vec7),  # [batch_size, num_frames]
    num_seeds: int,
    frame_index: int,
    ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # (6,)
    row_offset: int,
    col_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    instance, actuated_idx = wp.tid()  # pyright: ignore

    num_joints = S_world.shape[2]

    T_world_frame = T_world_link[instance, frame_index]

    target_batch_idx = instance // num_seeds
    T_target_world = se3_inverse_func(T_world_target[target_batch_idx, 0])
    T_target_frame = se3_multiply_func(T_target_world, T_world_frame)

    jlog = se3_jlog_func(T_target_frame, wp.float32(1e-4))

    T_frame_world = se3_inverse_func(T_world_frame)

    base_col_offset = 6 if has_floating_base else 0

    spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for joint_idx in range(num_joints):
        if ancestor_mask[frame_index, joint_idx]:
            weight = joints_to_actuated[joint_idx, actuated_idx]
            if weight != 0.0:
                for row in range(6):
                    spatial_col[row] += S_world[instance, row, joint_idx] * weight

    body_col = se3_adjoint_multiply_vec6_func(T_frame_world, spatial_col)

    jlog_times_body_col = wp.mul(jlog, body_col)

    for row in range(6):
        jacobian_buffer[instance, row_offset + row, col_offset + base_col_offset + actuated_idx] = (
            -residual_weight[row] * jlog_times_body_col[row]
        )

    if has_floating_base and actuated_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[actuated_idx] = 1.0

        # Right perturbation: T_world_base_new = T_world_base * exp(xi)
        # de/dxi = -Jlog(T_target_frame) * Ad(T_frame_base)
        T_frame_base = se3_multiply_func(T_frame_world, T_world_base[instance])
        result_col = se3_adjoint_multiply_vec6_func(T_frame_base, unit_vec)

        jlog_times_result_col = wp.mul(jlog, result_col)

        for row in range(6):
            jacobian_buffer[instance, row_offset + row, col_offset + actuated_idx] = (
                -residual_weight[row] * jlog_times_result_col[row]
            )


@wp.kernel
def compute_frame_task_weighted_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_index: int,
    T_world_target: wp.array2d(dtype=wp_vec7),  # (batch_size, num_frames)
    num_seeds: int,
    residual_weight: wp.array1d(dtype=wp.float32),  # (6,)
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # (num_instances, total_residual_dim)
):
    idx = wp.tid()

    target_batch_idx = idx // num_seeds

    T_world_frame = T_world_link[idx, frame_index]
    T_frame_world = se3_inverse_func(T_world_frame)
    T_frame_target = se3_multiply_func(T_frame_world, T_world_target[target_batch_idx, 0])
    error_in_frame = se3_log_map_func(T_frame_target, wp.float32(1e-4))

    for i in range(6):
        residual_buffer[idx, row_offset + i] = residual_weight[i] * error_in_frame[i]


@wp.kernel
def compute_multi_frame_task_weighted_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_indices: wp.array1d(dtype=wp.int32),  # (num_frames,)
    T_world_target: wp.array2d(dtype=wp_vec7),  # (batch_size, num_frames)
    num_frames: int,
    num_seeds: int,
    residual_weight: wp.array1d(dtype=wp.float32),  # (num_frames * 6,) per-frame weights
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # (num_instances, total_residual_dim)
):
    idx = wp.tid()
    target_batch_idx = idx // num_seeds

    for frame_num in range(num_frames):
        frame_index = frame_indices[frame_num]

        T_world_frame = T_world_link[idx, frame_index]
        T_frame_world = se3_inverse_func(T_world_frame)
        T_frame_target = se3_multiply_func(T_frame_world, T_world_target[target_batch_idx, frame_num])
        error_in_frame = se3_log_map_func(T_frame_target, wp.float32(1e-4))

        residual_offset = row_offset + frame_num * 6
        weight_offset = frame_num * 6
        for i in range(6):
            residual_buffer[idx, residual_offset + i] = residual_weight[weight_offset + i] * error_in_frame[i]


@wp.kernel
def compute_multi_frame_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_target: wp.array2d(dtype=wp_vec7),  # [batch_size, num_frames]
    frame_indices: wp.array1d(dtype=wp.int32),  # [num_frames]
    num_frames: int,
    num_seeds: int,
    ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # (num_frames * 6,) per-frame weights
    row_offset: int,
    col_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    instance, actuated_idx = wp.tid()  # pyright: ignore

    num_joints = S_world.shape[2]
    target_batch_idx = instance // num_seeds
    base_col_offset = 6 if has_floating_base else 0

    for frame_num in range(num_frames):
        frame_index = frame_indices[frame_num]
        residual_offset = row_offset + frame_num * 6
        weight_offset = frame_num * 6

        T_world_frame = T_world_link[instance, frame_index]
        T_target_world = se3_inverse_func(T_world_target[target_batch_idx, frame_num])
        T_target_frame = se3_multiply_func(T_target_world, T_world_frame)

        jlog = se3_jlog_func(T_target_frame, wp.float32(1e-4))
        T_frame_world = se3_inverse_func(T_world_frame)

        spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for joint_idx in range(num_joints):
            if ancestor_mask[frame_index, joint_idx]:
                weight = joints_to_actuated[joint_idx, actuated_idx]
                if weight != 0.0:
                    for row in range(6):
                        spatial_col[row] += S_world[instance, row, joint_idx] * weight

        body_col = se3_adjoint_multiply_vec6_func(T_frame_world, spatial_col)
        jlog_times_body_col = wp.mul(jlog, body_col)

        for row in range(6):
            jacobian_buffer[instance, residual_offset + row, col_offset + base_col_offset + actuated_idx] = (
                -residual_weight[weight_offset + row] * jlog_times_body_col[row]
            )

        if has_floating_base and actuated_idx < 6:
            unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unit_vec[actuated_idx] = 1.0

            # Right perturbation: T_world_base_new = T_world_base * exp(xi)
            # de/dxi = -Jlog(T_target_frame) * Ad(T_frame_base)
            T_frame_base = se3_multiply_func(T_frame_world, T_world_base[instance])
            result_col = se3_adjoint_multiply_vec6_func(T_frame_base, unit_vec)
            jlog_times_result_col = wp.mul(jlog, result_col)

            for row in range(6):
                jacobian_buffer[instance, residual_offset + row, col_offset + actuated_idx] = (
                    -residual_weight[weight_offset + row] * jlog_times_result_col[row]
                )
