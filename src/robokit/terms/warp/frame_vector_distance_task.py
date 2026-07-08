# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalOperand=false
# pyright: reportCallIssue=false
from typing import List, Optional, Union

import numpy as np
import warp as wp

from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


@wp.func
def huber_residual_pos(error: float, delta: float) -> float:
    if error <= delta:
        return error
    return wp.sqrt(wp.max(2.0 * delta * error - delta * delta, 0.0))


@wp.func
def huber_grad_scale_pos(error: float, delta: float) -> float:
    if error <= delta:
        return 1.0
    denom = wp.sqrt(wp.max(2.0 * delta * error - delta * delta, 1e-12))
    return delta / denom


class WarpFrameVectorDistanceTask(WarpTask):
    def __init__(
        self,
        robot: WarpRobot,
        origin_link_indices: List[int],
        task_link_indices: List[int],
        target_vectors: np.ndarray,
        huber_delta: float = 0.02,
        vector_weights: Optional[np.ndarray] = None,
        batch_size: int = 1,
    ):
        self.robot = robot
        self.origin_link_indices_list = list(origin_link_indices)
        self.task_link_indices_list = list(task_link_indices)
        self.num_vectors = len(origin_link_indices)
        self.huber_delta = float(huber_delta)
        self.batch_size = batch_size

        if len(origin_link_indices) != len(task_link_indices):
            raise ValueError("origin_link_indices and task_link_indices must have same length")

        self.device: Optional[wp_device_type] = None
        self.target_vectors_wp: Optional[wp.array] = None
        self.weight_scale_wp: Optional[wp.array] = None
        self.origin_indices_wp: Optional[wp.array] = None
        self.task_indices_wp: Optional[wp.array] = None

        self._target_vectors_np = target_vectors.astype(np.float32)
        if self._target_vectors_np.ndim == 2:
            self._target_vectors_np = self._target_vectors_np[None, :, :]

        self._vector_weights_np = None if vector_weights is None else vector_weights.astype(np.float32)
        if self._vector_weights_np is None:
            self._vector_weights_np = np.ones(self.num_vectors, dtype=np.float32)
        if self._vector_weights_np.ndim == 1:
            self._vector_weights_np = np.tile(self._vector_weights_np[None, :], (self.batch_size, 1))
        self._update_weight_scale()

    @property
    def residual_dim(self) -> int:
        return self.num_vectors

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.origin_indices_wp = wp.from_numpy(
            np.array(self.origin_link_indices_list, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        self.task_indices_wp = wp.from_numpy(
            np.array(self.task_link_indices_list, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        self.target_vectors_wp = wp.from_numpy(self._target_vectors_np, dtype=wp.float32, device=device)
        self.weight_scale_wp = wp.from_numpy(self._weight_scale_np, dtype=wp.float32, device=device)

    def _update_weight_scale(self) -> None:
        denom = float(self.num_vectors)
        self._weight_scale_np = np.sqrt(self._vector_weights_np / denom).astype(np.float32)

    def set_targets(
        self,
        target_vectors: Union[np.ndarray, wp.array],
        vector_weights: Optional[np.ndarray] = None,
    ) -> None:
        if isinstance(target_vectors, np.ndarray):
            target_np = target_vectors.astype(np.float32)
            self._target_vectors_np = target_np[None, :, :] if target_np.ndim == 2 else target_np
            if self.device is not None and self.target_vectors_wp is not None:
                target_wp = wp.from_numpy(self._target_vectors_np, dtype=wp.float32, device="cpu")
                wp.copy(self.target_vectors_wp, target_wp)
                target_vectors = self.target_vectors_wp
        else:
            if self.device is not None and self.target_vectors_wp is not None:
                wp.copy(self.target_vectors_wp, target_vectors)
                target_vectors = self.target_vectors_wp
            else:
                self._target_vectors_np = target_vectors.numpy()

        if vector_weights is not None:
            weight_np = vector_weights.astype(np.float32)
            if weight_np.ndim == 1:
                weight_np = np.tile(weight_np[None, :], (self.batch_size, 1))
            self._vector_weights_np = weight_np
            self._update_weight_scale()

        if self.device is not None:
            if self.target_vectors_wp is None:
                self.target_vectors_wp = wp.from_numpy(self._target_vectors_np, dtype=wp.float32, device=self.device)
            if self.weight_scale_wp is None:
                self.weight_scale_wp = wp.from_numpy(self._weight_scale_np, dtype=wp.float32, device=self.device)

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
            kernel=compute_vector_distance_residual_kernel,
            dim=(self.batch_size, self.num_vectors),
            inputs=[
                var.T_world_link.xyz_wxyz,
                self.origin_indices_wp,
                self.task_indices_wp,
                self.target_vectors_wp,
                self.weight_scale_wp,
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
            kernel=compute_vector_distance_jacobian_kernel,
            dim=(self.batch_size, self.num_vectors, self.robot.num_actuated_joints),
            inputs=[
                var.S_world,
                var.T_world_link.xyz_wxyz,
                self.origin_indices_wp,
                self.task_indices_wp,
                self.target_vectors_wp,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                self.weight_scale_wp,
                self.huber_delta,
                base_dim,
                row_offset,
            ],
            outputs=[jacobian_buffer],
            device=kernel_device,
        )

        if var.has_floating_base:
            wp.launch(
                kernel=compute_vector_distance_jacobian_base_kernel,
                dim=(self.batch_size, self.num_vectors, 6),
                inputs=[
                    var.T_world_link.xyz_wxyz,
                    var.T_world_base.xyz_wxyz,
                    self.origin_indices_wp,
                    self.task_indices_wp,
                    self.target_vectors_wp,
                    self.weight_scale_wp,
                    self.huber_delta,
                    row_offset,
                ],
                outputs=[jacobian_buffer],
                device=kernel_device,
            )
        return jacobian_buffer


@wp.kernel
def compute_vector_distance_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    origin_indices: wp.array1d(dtype=wp.int32),
    task_indices: wp.array1d(dtype=wp.int32),
    target_vectors: wp.array3d(dtype=wp.float32),
    weight_scale: wp.array2d(dtype=wp.float32),
    huber_delta: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
):
    batch_idx, vec_idx = wp.tid()

    origin_link_idx = origin_indices[vec_idx]
    origin_pose = T_world_link[batch_idx, origin_link_idx]
    origin_pos = wp.vec3(origin_pose[0], origin_pose[1], origin_pose[2])

    task_link_idx = task_indices[vec_idx]
    task_pose = T_world_link[batch_idx, task_link_idx]
    task_pos = wp.vec3(task_pose[0], task_pose[1], task_pose[2])

    robot_vector = task_pos - origin_pos
    target_vector = wp.vec3(
        target_vectors[batch_idx, vec_idx, 0],
        target_vectors[batch_idx, vec_idx, 1],
        target_vectors[batch_idx, vec_idx, 2],
    )

    diff = robot_vector - target_vector
    dist = wp.length(diff)
    residual = huber_residual_pos(dist, huber_delta)

    residual_buffer[batch_idx, row_offset + vec_idx] = weight_scale[batch_idx, vec_idx] * residual


@wp.kernel
def compute_vector_distance_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_link: wp.array2d(dtype=wp_vec7),
    origin_indices: wp.array1d(dtype=wp.int32),
    task_indices: wp.array1d(dtype=wp.int32),
    target_vectors: wp.array3d(dtype=wp.float32),
    link_ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    weight_scale: wp.array2d(dtype=wp.float32),
    huber_delta: float,
    base_dim: int,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
):
    batch_idx, vec_idx, actuated_idx = wp.tid()

    origin_link_idx = origin_indices[vec_idx]
    task_link_idx = task_indices[vec_idx]

    origin_pose = T_world_link[batch_idx, origin_link_idx]
    origin_pos = wp.vec3(origin_pose[0], origin_pose[1], origin_pose[2])

    task_pose = T_world_link[batch_idx, task_link_idx]
    task_pos = wp.vec3(task_pose[0], task_pose[1], task_pose[2])

    robot_vector = task_pos - origin_pos
    target_vector = wp.vec3(
        target_vectors[batch_idx, vec_idx, 0],
        target_vectors[batch_idx, vec_idx, 1],
        target_vectors[batch_idx, vec_idx, 2],
    )
    diff = robot_vector - target_vector
    dist = wp.length(diff)

    if dist < 1e-8:
        jacobian_buffer[batch_idx, row_offset + vec_idx, base_dim + actuated_idx] = 0.0
        return

    diff_unit = diff / dist
    scale = huber_grad_scale_pos(dist, huber_delta)

    num_joints = S_world.shape[2]
    d_vector_dq = wp.vec3(0.0, 0.0, 0.0)

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

        dp_origin_dq = wp.vec3(0.0, 0.0, 0.0)
        dp_task_dq = wp.vec3(0.0, 0.0, 0.0)

        if link_ancestor_mask[origin_link_idx, joint_idx]:
            dp_origin_dq = weight_val * (v + wp.cross(omega, origin_pos))
        if link_ancestor_mask[task_link_idx, joint_idx]:
            dp_task_dq = weight_val * (v + wp.cross(omega, task_pos))

        d_vector_dq = d_vector_dq + (dp_task_dq - dp_origin_dq)

    d_dist_dq = wp.dot(diff_unit, d_vector_dq)
    jacobian_buffer[batch_idx, row_offset + vec_idx, base_dim + actuated_idx] = (
        weight_scale[batch_idx, vec_idx] * scale * d_dist_dq
    )


@wp.kernel
def compute_vector_distance_jacobian_base_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    origin_indices: wp.array1d(dtype=wp.int32),
    task_indices: wp.array1d(dtype=wp.int32),
    target_vectors: wp.array3d(dtype=wp.float32),
    weight_scale: wp.array2d(dtype=wp.float32),
    huber_delta: float,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
):
    batch_idx, vec_idx, base_idx = wp.tid()

    origin_link_idx = origin_indices[vec_idx]
    task_link_idx = task_indices[vec_idx]

    origin_pose = T_world_link[batch_idx, origin_link_idx]
    origin_pos = wp.vec3(origin_pose[0], origin_pose[1], origin_pose[2])

    task_pose = T_world_link[batch_idx, task_link_idx]
    task_pos = wp.vec3(task_pose[0], task_pose[1], task_pose[2])

    robot_vector = task_pos - origin_pos
    target_vector = wp.vec3(
        target_vectors[batch_idx, vec_idx, 0],
        target_vectors[batch_idx, vec_idx, 1],
        target_vectors[batch_idx, vec_idx, 2],
    )
    diff = robot_vector - target_vector
    dist = wp.length(diff)

    if dist < 1e-8:
        jacobian_buffer[batch_idx, row_offset + vec_idx, base_idx] = 0.0
        return

    diff_unit = diff / dist
    scale = huber_grad_scale_pos(dist, huber_delta)

    unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unit_vec[base_idx] = 1.0
    base_twist = se3_adjoint_multiply_vec6_func(T_world_base[batch_idx], unit_vec)
    v = wp.vec3(base_twist[0], base_twist[1], base_twist[2])
    omega = wp.vec3(base_twist[3], base_twist[4], base_twist[5])

    d_vector = (v + wp.cross(omega, task_pos)) - (v + wp.cross(omega, origin_pos))
    d_dist_dq = wp.dot(diff_unit, d_vector)
    jacobian_buffer[batch_idx, row_offset + vec_idx, base_idx] = weight_scale[batch_idx, vec_idx] * scale * d_dist_dq
