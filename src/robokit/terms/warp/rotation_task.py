# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.warp_se3 import WarpSE3
from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import (
    quaternion_invert_func,
    quaternion_multiply_func,
    quaternion_to_axis_angle_func,
)


class WarpRotationTask(WarpTask):
    """
    Warp-based rotation task for computing weighted orientation residuals and Jacobians.
    Computes a world-frame rotation error: residual = axis_angle(q_actual * inv(q_target)).
    Jacobian uses the geometric angular velocity: J = weight * omega.
    Supports both single-frame and multi-frame targets.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
        >>> target_link_index = robot.link_names.index("ee_link")
        >>> state = robot.state(q=robot.zero_q)
        >>> target_link_pose = robot.forward_kinematics(state).get_T_world_link(target_link_index)
        >>> rotation_task = WarpRotationTask(robot, target_link_index, target_link_pose, weight=1.0)
        >>> random_q = wp.from_numpy(np.array([[-0.5, 0.3, -0.2, 0.4, -0.1, 0.25]], dtype=np.float32), dtype=wp.float32)
        >>> state = robot.state()
        >>> state.set_configuration(q=random_q)
        >>> robot.forward_kinematics(state)
        >>> robot.compute_motion_subspace(state)
        >>> weighted_residual = rotation_task.compute_weighted_residual(state)
        >>> weighted_jacobian = rotation_task.compute_weighted_jacobian(state)
        >>> weighted_residual.numpy().shape[1] == 3
        True
        >>> weighted_jacobian.numpy().shape[1] == 3
        True
    """

    def __init__(
        self,
        robot: WarpRobot,
        frame_index: Union[int, Sequence[int]],
        T_world_target: Union[WarpSE3, Sequence[WarpSE3]],
        weight: float,
    ):
        self.robot = robot

        frame_indices_list = [frame_index] if isinstance(frame_index, int) else list(frame_index)
        targets_list = [T_world_target] if isinstance(T_world_target, WarpSE3) else list(T_world_target)

        if len(frame_indices_list) != len(targets_list):
            raise ValueError(
                f"Number of frame indices ({len(frame_indices_list)}) must match "
                f"number of targets ({len(targets_list)})"
            )

        self.num_frames = len(frame_indices_list)
        self.device = targets_list[0].xyz_wxyz.device
        batch_size = targets_list[0].batch_size

        self.frame_indices = wp.from_numpy(
            np.array(frame_indices_list, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self.T_world_target = WarpSE3.stack(targets_list, axis=1)
        self.dim = batch_size
        self.residual_weight = wp.from_numpy(np.full(3, weight, dtype=np.float32), dtype=wp.float32, device=self.device)

    def set_target(self, target: Union[WarpSE3, Sequence[WarpSE3]]) -> None:
        """Update target poses in-place. Batch size must match the original."""
        targets = [target] if isinstance(target, WarpSE3) else list(target)
        WarpSE3.stack(targets, axis=1, dest=self.T_world_target)

    @property
    def residual_dim(self) -> int:
        return 3 * self.num_frames

    def compute_weighted_residual(
        self, var: WarpRobotState, residual_buffer: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        kernel_device = residual_buffer.device if residual_buffer is not None else self.device
        if residual_buffer is None:
            residual_buffer = wp.empty((self.dim, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_rotation_task_weighted_residual_kernel,
            dim=(self.dim, self.num_frames),
            inputs=[
                var.T_world_link.xyz_wxyz,
                self.frame_indices,
                self.T_world_target,
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

        wp.launch(
            kernel=compute_rotation_task_weighted_jacobian_kernel,
            dim=(self.dim, self.num_frames, num_actuated),
            inputs=[
                var.S_world,
                var.T_world_base.xyz_wxyz,
                self.frame_indices,
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
def compute_rotation_task_weighted_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_indices: wp.array1d(dtype=wp.int32),  # (num_frames,)
    T_world_target: wp.array2d(dtype=wp_vec7),  # (num_instances, num_frames)
    residual_weight: wp.array1d(dtype=wp.float32),  # (3,)
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # (num_instances, total_residual_dim)
):
    idx, frame_num = wp.tid()  # pyright: ignore

    frame_index = frame_indices[frame_num]
    T_world_frame = T_world_link[idx, frame_index]
    target = T_world_target[idx, frame_num]

    q_world_frame = wp.vec4(T_world_frame[3], T_world_frame[4], T_world_frame[5], T_world_frame[6])
    q_world_target = wp.vec4(target[3], target[4], target[5], target[6])

    q_target_inv = quaternion_invert_func(q_world_target)
    q_err = quaternion_multiply_func(q_world_frame, q_target_inv)
    error = quaternion_to_axis_angle_func(q_err)

    residual_offset = row_offset + frame_num * 3
    for i in range(3):
        residual_buffer[idx, residual_offset + i] = residual_weight[i] * error[i]


@wp.kernel
def compute_rotation_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    frame_indices: wp.array1d(dtype=wp.int32),  # [num_frames]
    ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # (3,)
    row_offset: int,
    col_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    instance, frame_num, actuated_idx = wp.tid()  # pyright: ignore

    num_joints = S_world.shape[2]
    base_col_offset = 6 if has_floating_base else 0

    frame_index = frame_indices[frame_num]
    residual_offset = row_offset + frame_num * 3

    # Accumulate angular velocity column from motion subspace
    omega_col = wp.vec3(0.0, 0.0, 0.0)
    for joint_idx in range(num_joints):
        if ancestor_mask[frame_index, joint_idx]:
            weight = joints_to_actuated[joint_idx, actuated_idx]
            if weight != 0.0:
                omega_col += weight * wp.vec3(
                    S_world[instance, 3, joint_idx],
                    S_world[instance, 4, joint_idx],
                    S_world[instance, 5, joint_idx],
                )

    # Jacobian = weight * omega (geometric angular velocity, consistent with position task)
    for row in range(3):
        jacobian_buffer[instance, residual_offset + row, col_offset + base_col_offset + actuated_idx] = (
            residual_weight[row] * omega_col[row]
        )

    if has_floating_base and actuated_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[actuated_idx] = 1.0
        spatial_twist = se3_adjoint_multiply_vec6_func(T_world_base[instance], unit_vec)
        world_ang_fb = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])

        for row in range(3):
            jacobian_buffer[instance, residual_offset + row, col_offset + actuated_idx] = (
                residual_weight[row] * world_ang_fb[row]
            )
