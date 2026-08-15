# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import (
    se3_adjoint_func,
    se3_inverse_func,
    se3_jlog_func,
    se3_log_map_func,
    se3_multiply_func,
)
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import stack, wp_vec6, wp_vec7


# --- task -------------------------------------------------------------------
class FrameTask(RobotTask, ResidualTask):
    """Match one or more link frames to world-frame targets.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarValues
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"))
        >>> target_link_index = robot.spec.link_names.index("ee_link")
        >>> state = robot.state(q=robot.spec.zero_q)
        >>> target_link_pose = robot.forward_kinematics(state).get_T_world_link(target_link_index)
        >>> frame_task = FrameTask(robot, target_link_index, target_link_pose, position_weight=1.0, orientation_weight=1.0)
        >>> random_q = wp.from_numpy(np.array([[-0.5, 0.3, -0.2, 0.4, -0.1, 0.25]], dtype=np.float32), dtype=wp.float32, device="cpu")
        >>> state = robot.state(q=random_q)
        >>> robot.forward_kinematics(state)
        >>> robot.compute_motion_subspace(state)
        >>> weighted_residual = frame_task.compute_weighted_residual(VarValues(robot=state))
        >>> weighted_jacobian = frame_task.compute_weighted_jacobian(VarValues(robot=state))
        >>> assert np.allclose(weighted_residual.numpy()[0, :3], np.array([0.5839, -0.0795, -0.1784], dtype=np.float32), atol=1e-3)
        >>> assert np.allclose(weighted_jacobian.numpy()[0, :, 0], np.array([-1.1628, 0.1626, -0.167, -0.0765, 0.3698, 0.9521], dtype=np.float32), atol=1e-3)
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        frame_index: Optional[Union[int, Sequence[int]]] = None,
        T_world_target: Optional[Union[wp.array, Sequence[wp.array]]] = None,
        position_weight: Union[float, Sequence[float]] = 1.0,
        orientation_weight: Union[float, Sequence[float]] = 1.0,
    ):
        self.robot: Optional[Robot] = robot
        if frame_index is None or T_world_target is None:
            return  # config-time stub

        if isinstance(frame_index, int):
            frame_indices_list = [frame_index]
        else:
            frame_indices_list = list(frame_index)

        if isinstance(T_world_target, wp.array) and T_world_target.ndim > 1:
            if len(frame_indices_list) != T_world_target.shape[1]:
                raise ValueError(
                    f"Number of frame indices ({len(frame_indices_list)}) must match "
                    f"number of targets ({T_world_target.shape[1]})"
                )
            self.device = T_world_target.device
            self.T_world_target = wp.clone(T_world_target)
        else:
            # a bare [batch] array is one frame's target; anything else is a sequence of them
            targets_list = [T_world_target] if isinstance(T_world_target, wp.array) else list(T_world_target)
            if len(frame_indices_list) != len(targets_list):
                raise ValueError(
                    f"Number of frame indices ({len(frame_indices_list)}) must match "
                    f"number of targets ({len(targets_list)})"
                )
            self.device = targets_list[0].device
            self.T_world_target = stack(targets_list, axis=1)

        self.num_frames = len(frame_indices_list)
        self._frame_indices_np = np.array(frame_indices_list, dtype=np.int32)

        self.frame_indices = wp.from_numpy(self._frame_indices_np, dtype=wp.int32, device=self.device)
        if self.num_frames == 1:
            self.frame_index = frame_indices_list[0]
        else:
            self.frame_index = -1

        pos_w = (
            np.full(self.num_frames, position_weight, dtype=np.float32)
            if isinstance(position_weight, (int, float))
            else np.asarray(position_weight, dtype=np.float32)
        )
        ori_w = (
            np.full(self.num_frames, orientation_weight, dtype=np.float32)
            if isinstance(orientation_weight, (int, float))
            else np.asarray(orientation_weight, dtype=np.float32)
        )

        if pos_w.shape[0] != self.num_frames:
            raise ValueError(
                f"Number of position weights ({pos_w.shape[0]}) must match number of frames ({self.num_frames})"
            )
        if ori_w.shape[0] != self.num_frames:
            raise ValueError(
                f"Number of orientation weights ({ori_w.shape[0]}) must match number of frames ({self.num_frames})"
            )

        residual_weight = np.empty(self.num_frames * 6, dtype=np.float32)
        residual_weight.reshape(-1, 6)[:, :3] = np.repeat(pos_w, 3).reshape(-1, 3)
        residual_weight.reshape(-1, 6)[:, 3:] = np.repeat(ori_w, 3).reshape(-1, 3)
        self._residual_weight_np = residual_weight
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=self.device)

    def set_target(self, target: Union[wp.array, Sequence[wp.array]]):
        """Update target poses in-place. Batch size must match the original."""
        if isinstance(target, wp.array) and target.ndim > 1:
            wp.copy(self.T_world_target, target)
            return
        targets = [target] if isinstance(target, wp.array) else list(target)
        stack(targets, axis=1, out=self.T_world_target)

    def set_robot(self, robot: Robot):
        self.robot = robot

    @property
    def residual_dim(self) -> int:
        return 6 * self.num_frames

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        kernel_device = out_residual.device if out_residual is not None else self.device
        batch_size = var.batch_size
        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self.num_frames == 1:
            wp.launch(
                kernel=compute_frame_task_weighted_residual_kernel,
                dim=[batch_size],
                inputs=[
                    var.T_world_link,
                    self.frame_index,
                    self.T_world_target,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_multi_frame_task_weighted_residual_kernel,
                dim=[batch_size],
                inputs=[
                    var.T_world_link,
                    self.frame_indices,
                    self.T_world_target,
                    self.num_frames,
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
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        col_offset = var_values.tangent_offset(self.var_key)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        spec_tensor = var.spec_tensors
        kernel_device = out_jacobian.device if out_jacobian is not None else self.device
        batch_size = var.batch_size
        if out_jacobian is None:
            total_dofs = var.tangent_dim
            out_jacobian = wp.zeros((batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device)
            row_offset = 0
        num_actuated = spec_tensor.joints_to_actuated_mapping.shape[1]

        if self.num_frames == 1:
            wp.launch(
                kernel=compute_frame_task_weighted_jacobian_kernel,
                dim=(batch_size, num_actuated),
                inputs=[
                    var.S_world,
                    var.T_world_link,
                    var.T_world_base,
                    self.T_world_target,
                    self.frame_index,
                    spec_tensor.link_ancestor_joints_mask,
                    spec_tensor.joints_to_actuated_mapping,
                    var.has_floating_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_multi_frame_task_weighted_jacobian_kernel,
                dim=(batch_size, num_actuated),
                inputs=[
                    var.S_world,
                    var.T_world_link,
                    var.T_world_base,
                    self.T_world_target,
                    self.frame_indices,
                    self.num_frames,
                    spec_tensor.link_ancestor_joints_mask,
                    spec_tensor.joints_to_actuated_mapping,
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
def compute_frame_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_target: wp.array2d(dtype=wp_vec7),  # [batch_size, num_frames]
    frame_index: int,
    ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # (6,)
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    instance, actuated_idx = wp.tid()  # pyright: ignore

    num_joints = S_world.shape[2]

    T_world_frame = T_world_link[instance, frame_index]

    n_repeat = T_world_link.shape[0] // T_world_target.shape[0]
    target_batch_idx = instance // n_repeat
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

    body_col = se3_adjoint_func(T_frame_world) * spatial_col

    jlog_times_body_col = wp.mul(jlog, body_col)

    for row in range(6):
        out_jacobian[instance, row_offset + row, col_offset + base_col_offset + actuated_idx] = (
            -residual_weight[row] * jlog_times_body_col[row]
        )

    if has_floating_base and actuated_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[actuated_idx] = 1.0

        # right perturbation gives de/dxi = -Jlog(T_target_frame) * Ad(T_frame_base)
        T_frame_base = se3_multiply_func(T_frame_world, T_world_base[instance])
        result_col = se3_adjoint_func(T_frame_base) * unit_vec

        jlog_times_result_col = wp.mul(jlog, result_col)

        for row in range(6):
            out_jacobian[instance, row_offset + row, col_offset + actuated_idx] = (
                -residual_weight[row] * jlog_times_result_col[row]
            )


@wp.kernel
def compute_frame_task_weighted_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_index: int,
    T_world_target: wp.array2d(dtype=wp_vec7),  # (batch_size, num_frames)
    residual_weight: wp.array1d(dtype=wp.float32),  # (6,)
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # (num_instances, total_residual_dim)
):
    idx = wp.tid()

    n_repeat = T_world_link.shape[0] // T_world_target.shape[0]
    target_batch_idx = idx // n_repeat

    T_world_frame = T_world_link[idx, frame_index]
    T_frame_world = se3_inverse_func(T_world_frame)
    T_frame_target = se3_multiply_func(T_frame_world, T_world_target[target_batch_idx, 0])
    error_in_frame = se3_log_map_func(T_frame_target, wp.float32(1e-4))

    for i in range(6):
        out_residual[idx, row_offset + i] = residual_weight[i] * error_in_frame[i]


@wp.kernel
def compute_multi_frame_task_weighted_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_indices: wp.array1d(dtype=wp.int32),  # (num_frames,)
    T_world_target: wp.array2d(dtype=wp_vec7),  # (batch_size, num_frames)
    num_frames: int,
    residual_weight: wp.array1d(dtype=wp.float32),  # (num_frames * 6,) per-frame weights
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # (num_instances, total_residual_dim)
):
    idx = wp.tid()
    n_repeat = T_world_link.shape[0] // T_world_target.shape[0]
    target_batch_idx = idx // n_repeat

    for frame_num in range(num_frames):
        frame_index = frame_indices[frame_num]

        T_world_frame = T_world_link[idx, frame_index]
        T_frame_world = se3_inverse_func(T_world_frame)
        T_frame_target = se3_multiply_func(T_frame_world, T_world_target[target_batch_idx, frame_num])
        error_in_frame = se3_log_map_func(T_frame_target, wp.float32(1e-4))

        residual_offset = row_offset + frame_num * 6
        weight_offset = frame_num * 6
        for i in range(6):
            out_residual[idx, residual_offset + i] = residual_weight[weight_offset + i] * error_in_frame[i]


@wp.kernel
def compute_multi_frame_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_target: wp.array2d(dtype=wp_vec7),  # [batch_size, num_frames]
    frame_indices: wp.array1d(dtype=wp.int32),  # [num_frames]
    num_frames: int,
    ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # (num_frames * 6,) per-frame weights
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    instance, actuated_idx = wp.tid()  # pyright: ignore

    num_joints = S_world.shape[2]
    n_repeat = T_world_link.shape[0] // T_world_target.shape[0]
    target_batch_idx = instance // n_repeat
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

        body_col = se3_adjoint_func(T_frame_world) * spatial_col
        jlog_times_body_col = wp.mul(jlog, body_col)

        for row in range(6):
            out_jacobian[instance, residual_offset + row, col_offset + base_col_offset + actuated_idx] = (
                -residual_weight[weight_offset + row] * jlog_times_body_col[row]
            )

        if has_floating_base and actuated_idx < 6:
            unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unit_vec[actuated_idx] = 1.0

            # right perturbation gives de/dxi = -Jlog(T_target_frame) * Ad(T_frame_base)
            T_frame_base = se3_multiply_func(T_frame_world, T_world_base[instance])
            result_col = se3_adjoint_func(T_frame_base) * unit_vec
            jlog_times_result_col = wp.mul(jlog, result_col)

            for row in range(6):
                out_jacobian[instance, residual_offset + row, col_offset + actuated_idx] = (
                    -residual_weight[weight_offset + row] * jlog_times_result_col[row]
                )
