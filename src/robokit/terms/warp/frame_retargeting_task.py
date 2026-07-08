# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportUndefinedVariable=false
# pyright: reportCallIssue=false
from typing import List, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


class WarpFrameRetargetingTask(WarpTask):
    """Frame-level retargeting task with position and angle residuals.

    Computes pairwise position and angle residuals between robot links and target keypoints.
    Uses fixed scale (default 1.0) for position scaling:
        position_residual = delta_target - delta_robot * scale
    The angle residual uses unscaled delta_robot.

    This is a simplified dense WarpTask for frame-level solving without position_scale optimization.
    """

    def __init__(
        self,
        robot: WarpRobot,
        robot_link_indices: List[int],
        target_keypoint_indices: List[int],
        pair_indices: Union[np.ndarray, List[List[int]]],
        target_keypoints: Union[np.ndarray, wp.array],
        position_weight: Union[float, Sequence[float], np.ndarray] = 1.0,
        angle_weight: float = 1.0,
        scale: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        self.robot = robot
        self.robot_link_indices_list = robot_link_indices
        self.target_keypoint_indices_list = target_keypoint_indices
        self.angle_weight = angle_weight
        self.scale = scale
        self.batch_size = batch_size
        self.residual_weight = None

        self.target_keypoints_input = target_keypoints
        self.num_selected = len(robot_link_indices)

        pair_indices_array = np.asarray(pair_indices, dtype=np.int32)
        if pair_indices_array.ndim != 2 or pair_indices_array.shape[1] != 2:
            raise ValueError("pair_indices must have shape [num_pairs, 2]")
        self.pair_rows_input = pair_indices_array[:, 0]
        self.pair_cols_input = pair_indices_array[:, 1]
        self.num_pairs = int(pair_indices_array.shape[0])

        if isinstance(position_weight, (int, float)):
            self.position_weights_np = np.full(self.num_pairs, float(position_weight), dtype=np.float32)
        else:
            self.position_weights_np = np.asarray(position_weight, dtype=np.float32)
            if self.position_weights_np.shape[0] != self.num_pairs:
                raise ValueError(
                    f"position_weight length {self.position_weights_np.shape[0]} != num_pairs {self.num_pairs}"
                )

        self.device: Optional[wp_device_type] = None
        self.target_keypoints: Optional[wp.array] = None
        self.robot_link_indices: Optional[wp.array] = None
        self.target_keypoint_indices: Optional[wp.array] = None
        self.pair_rows: Optional[wp.array] = None
        self.pair_cols: Optional[wp.array] = None

        self.position_weights: Optional[wp.array] = None
        self.link_ancestor_masks: Optional[wp.array] = None
        self.all_ancestor_dof_indices: Optional[wp.array] = None
        self.num_unique_dofs: int = 0

    def set_targets(self, target_keypoints: wp.array) -> None:
        if self.target_keypoints is None:
            self.target_keypoints = target_keypoints
        else:
            wp.copy(self.target_keypoints, target_keypoints)

    def init_buffers(self, device: wp_device_type) -> None:
        self.device = device

        if isinstance(self.target_keypoints_input, np.ndarray):
            target_np = self.target_keypoints_input
            if target_np.ndim == 2:
                target_np = target_np[None, ...]
            self.target_keypoints = wp.from_numpy(target_np.astype(np.float32), dtype=wp.float32, device=device)
        else:
            self.target_keypoints = self.target_keypoints_input.to(device)

        self.robot_link_indices = wp.from_numpy(
            np.array(self.robot_link_indices_list, dtype=np.int32), dtype=wp.int32, device=device
        )
        self.target_keypoint_indices = wp.from_numpy(
            np.array(self.target_keypoint_indices_list, dtype=np.int32), dtype=wp.int32, device=device
        )
        self.pair_rows = wp.from_numpy(self.pair_rows_input, dtype=wp.int32, device=device)
        self.pair_cols = wp.from_numpy(self.pair_cols_input, dtype=wp.int32, device=device)
        self.position_weights = wp.from_numpy(self.position_weights_np, dtype=wp.float32, device=device)

        num_joints = self.robot.spec.num_joints
        link_ancestor_masks_np = np.zeros((self.num_selected, num_joints), dtype=np.bool_)
        for idx, link_idx in enumerate(self.robot_link_indices_list):
            link_ancestor_masks_np[idx] = self.robot.spec.link_ancestor_joints_mask[link_idx]
        self.link_ancestor_masks = wp.from_numpy(link_ancestor_masks_np, dtype=wp.bool, device=device)

        joints_to_actuated = self.robot.spec.joints_to_actuated_mapping
        all_ancestor_mask = np.any(link_ancestor_masks_np, axis=0)
        ancestor_indices = np.where(all_ancestor_mask)[0]

        unique_dofs = set()
        for joint_idx in ancestor_indices:
            for actuated_idx in range(joints_to_actuated.shape[1]):
                if joints_to_actuated[joint_idx, actuated_idx] != 0.0:
                    unique_dofs.add(actuated_idx)

        unique_dofs_arr = np.array(sorted(unique_dofs), dtype=np.int32)
        self.num_unique_dofs = len(unique_dofs_arr)
        self.all_ancestor_dof_indices = wp.from_numpy(unique_dofs_arr, dtype=wp.int32, device=device)

    @property
    def residual_dim(self) -> int:
        return self.num_pairs * 4

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        *_args: object,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
        **_kwargs: object,
    ) -> wp.array:
        if self.device is None:
            self.init_buffers(var.q.device)

        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)

        kernel_device = residual_buffer.device if residual_buffer is not None else self.device

        if residual_buffer is None:
            residual_buffer = wp.empty((self.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_frame_retargeting_residual_kernel,
            dim=(self.batch_size, self.num_pairs),
            inputs=[
                var.T_world_link.xyz_wxyz,
                self.target_keypoints,
                self.robot_link_indices,
                self.target_keypoint_indices,
                self.pair_rows,
                self.pair_cols,
                self.scale,
                self.position_weights,
                self.angle_weight,
                row_offset,
            ],
            outputs=[residual_buffer],
            device=kernel_device,
        )

        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: WarpRobotState,
        *_args: object,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
        **_kwargs: object,
    ) -> wp.array:
        if self.device is None:
            self.init_buffers(var.q.device)

        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)

        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)

        kernel_device = var.q.device
        tangent_dim = var.tangent_dim

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (self.batch_size, self.residual_dim, tangent_dim), dtype=wp.float32, device=kernel_device
            )

        spec_tensors = var.spec_tensors
        num_dofs = self.robot.num_actuated_joints
        base_dim = 6 if bool(var.has_floating_base) else 0

        wp.launch(
            kernel=compute_frame_retargeting_jacobian_kernel,
            dim=(self.batch_size, self.num_pairs),
            inputs=[
                var.S_world,
                var.T_world_link.xyz_wxyz,
                var.T_world_base.xyz_wxyz if var.has_floating_base else None,
                self.target_keypoints,
                self.robot_link_indices,
                self.target_keypoint_indices,
                self.pair_rows,
                self.pair_cols,
                self.link_ancestor_masks,
                spec_tensors.joints_to_actuated_mapping,
                self.scale,
                self.position_weights,
                self.angle_weight,
                base_dim,
                num_dofs,
                row_offset,
            ],
            outputs=[jacobian_buffer],
            device=kernel_device,
        )

        return jacobian_buffer


@wp.kernel
def compute_frame_retargeting_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    target_keypoints: wp.array3d(dtype=wp.float32),
    robot_link_indices: wp.array1d(dtype=wp.int32),
    target_keypoint_indices: wp.array1d(dtype=wp.int32),
    pair_rows: wp.array1d(dtype=wp.int32),
    pair_cols: wp.array1d(dtype=wp.int32),
    scale: float,
    position_weights: wp.array1d(dtype=wp.float32),
    angle_weight: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
):
    batch_idx, pair_idx = wp.tid()
    pw = position_weights[pair_idx]

    row = pair_rows[pair_idx]
    col = pair_cols[pair_idx]

    base_idx = row_offset + pair_idx * 4

    if row == col:
        residual_buffer[batch_idx, base_idx + 0] = 0.0
        residual_buffer[batch_idx, base_idx + 1] = 0.0
        residual_buffer[batch_idx, base_idx + 2] = 0.0
        residual_buffer[batch_idx, base_idx + 3] = 0.0
        return

    link_idx_i = robot_link_indices[row]
    link_idx_j = robot_link_indices[col]
    joint_idx_i = target_keypoint_indices[row]
    joint_idx_j = target_keypoint_indices[col]

    t_i = T_world_link[batch_idx, link_idx_i]
    t_j = T_world_link[batch_idx, link_idx_j]
    pos_robot_i = wp.vec3(t_i[0], t_i[1], t_i[2])
    pos_robot_j = wp.vec3(t_j[0], t_j[1], t_j[2])
    delta_robot = pos_robot_i - pos_robot_j

    pos_target_i = wp.vec3(
        target_keypoints[batch_idx, joint_idx_i, 0],
        target_keypoints[batch_idx, joint_idx_i, 1],
        target_keypoints[batch_idx, joint_idx_i, 2],
    )
    pos_target_j = wp.vec3(
        target_keypoints[batch_idx, joint_idx_j, 0],
        target_keypoints[batch_idx, joint_idx_j, 1],
        target_keypoints[batch_idx, joint_idx_j, 2],
    )
    delta_target = pos_target_i - pos_target_j

    diff = delta_target - delta_robot * scale
    residual_buffer[batch_idx, base_idx + 0] = diff[0] * pw
    residual_buffer[batch_idx, base_idx + 1] = diff[1] * pw
    residual_buffer[batch_idx, base_idx + 2] = diff[2] * pw

    eps = 1e-6
    len_robot = wp.length(delta_robot) + eps
    len_target = wp.length(delta_target) + eps
    dir_robot = delta_robot / len_robot
    dir_target = delta_target / len_target
    angle_err = 1.0 - wp.dot(dir_robot, dir_target)
    residual_buffer[batch_idx, base_idx + 3] = angle_err * angle_weight


@wp.kernel
def compute_frame_retargeting_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    target_keypoints: wp.array3d(dtype=wp.float32),
    robot_link_indices: wp.array1d(dtype=wp.int32),
    target_keypoint_indices: wp.array1d(dtype=wp.int32),
    pair_rows: wp.array1d(dtype=wp.int32),
    pair_cols: wp.array1d(dtype=wp.int32),
    link_ancestor_masks: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    scale: float,
    position_weights: wp.array1d(dtype=wp.float32),
    angle_weight: float,
    base_dim: int,
    num_dofs: int,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
):
    batch_idx, pair_idx = wp.tid()
    pw = position_weights[pair_idx]

    row = pair_rows[pair_idx]
    col = pair_cols[pair_idx]

    base_res_idx = row_offset + pair_idx * 4

    if row == col:
        return

    link_idx_i = robot_link_indices[row]
    link_idx_j = robot_link_indices[col]
    joint_idx_i = target_keypoint_indices[row]
    joint_idx_j = target_keypoint_indices[col]

    t_i = T_world_link[batch_idx, link_idx_i]
    t_j = T_world_link[batch_idx, link_idx_j]
    p_i = wp.vec3(t_i[0], t_i[1], t_i[2])
    p_j = wp.vec3(t_j[0], t_j[1], t_j[2])
    delta_robot = p_i - p_j

    pos_target_i = wp.vec3(
        target_keypoints[batch_idx, joint_idx_i, 0],
        target_keypoints[batch_idx, joint_idx_i, 1],
        target_keypoints[batch_idx, joint_idx_i, 2],
    )
    pos_target_j = wp.vec3(
        target_keypoints[batch_idx, joint_idx_j, 0],
        target_keypoints[batch_idx, joint_idx_j, 1],
        target_keypoints[batch_idx, joint_idx_j, 2],
    )
    delta_target = pos_target_i - pos_target_j

    eps = 1e-6
    len_robot = wp.length(delta_robot) + eps
    len_target = wp.length(delta_target) + eps
    dir_robot = delta_robot / len_robot
    dir_target = delta_target / len_target

    dot_val = wp.dot(dir_target, dir_robot)
    d_angle_d_delta = -(dir_target - dot_val * dir_robot) / len_robot

    num_joints = joints_to_actuated.shape[0]

    for actuated_idx in range(num_dofs):
        d_delta_dq = wp.vec3(0.0, 0.0, 0.0)

        for joint_idx in range(num_joints):
            affects_i = link_ancestor_masks[row, joint_idx]
            affects_j = link_ancestor_masks[col, joint_idx]

            if not affects_i and not affects_j:
                continue

            weight_val = joints_to_actuated[joint_idx, actuated_idx]
            if weight_val == 0.0:
                continue

            v = wp.vec3(
                S_world[batch_idx, 0, joint_idx],
                S_world[batch_idx, 1, joint_idx],
                S_world[batch_idx, 2, joint_idx],
            )
            omega = wp.vec3(
                S_world[batch_idx, 3, joint_idx],
                S_world[batch_idx, 4, joint_idx],
                S_world[batch_idx, 5, joint_idx],
            )

            dp_i_dq = wp.vec3(0.0, 0.0, 0.0)
            dp_j_dq = wp.vec3(0.0, 0.0, 0.0)

            if affects_i:
                dp_i_dq = weight_val * (v + wp.cross(omega, p_i))
            if affects_j:
                dp_j_dq = weight_val * (v + wp.cross(omega, p_j))

            d_delta_dq = d_delta_dq + (dp_i_dq - dp_j_dq)

        col_idx = base_dim + actuated_idx
        jacobian_buffer[batch_idx, base_res_idx + 0, col_idx] = -d_delta_dq[0] * scale * pw
        jacobian_buffer[batch_idx, base_res_idx + 1, col_idx] = -d_delta_dq[1] * scale * pw
        jacobian_buffer[batch_idx, base_res_idx + 2, col_idx] = -d_delta_dq[2] * scale * pw

        angle_jac = wp.dot(d_angle_d_delta, d_delta_dq)
        jacobian_buffer[batch_idx, base_res_idx + 3, col_idx] = angle_jac * angle_weight

    for b in range(base_dim):
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[b] = 1.0
        base_twist = se3_adjoint_multiply_vec6_func(T_world_base[batch_idx], unit_vec)
        v = wp.vec3(base_twist[0], base_twist[1], base_twist[2])
        omega = wp.vec3(base_twist[3], base_twist[4], base_twist[5])

        dp_i_dq = v + wp.cross(omega, p_i)
        dp_j_dq = v + wp.cross(omega, p_j)
        d_delta_dq = dp_i_dq - dp_j_dq

        jacobian_buffer[batch_idx, base_res_idx + 0, b] = -d_delta_dq[0] * scale * pw
        jacobian_buffer[batch_idx, base_res_idx + 1, b] = -d_delta_dq[1] * scale * pw
        jacobian_buffer[batch_idx, base_res_idx + 2, b] = -d_delta_dq[2] * scale * pw

        angle_jac = wp.dot(d_angle_d_delta, d_delta_dq)
        jacobian_buffer[batch_idx, base_res_idx + 3, b] = angle_jac * angle_weight
