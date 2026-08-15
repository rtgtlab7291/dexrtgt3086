# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
from typing import Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_kernels import compute_body_com_world_func, compute_link_com_world_func
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec6, wp_vec7


# --- task -------------------------------------------------------------------
class ComPositionTask(RobotTask, ResidualTask):
    """Penalizes deviation of the whole-body center of mass from a target position.

    Computes CoM as the mass-weighted average of link CoM positions in world frame.
    Analytic Jacobian uses mass-weighted sum of link position Jacobians evaluated
    at each link's CoM.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("g1_description"))
        >>> task = ComPositionTask(robot=robot, target_com_position=np.zeros(3), weight=1.0)
        >>> task.residual_dim
        3
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        target_com_position: Optional[np.ndarray] = None,
        weight: Union[float, Sequence[float]] = 1.0,
    ):
        self.weight = weight
        self.target_com_position = target_com_position
        target_arr = np.asarray(target_com_position, dtype=np.float32)
        self._batched_target = target_arr.ndim == 2
        self._target_com_np = target_arr.reshape(-1, 3) if self._batched_target else target_arr.ravel()[:3]
        weight_arr = (
            np.full(3, weight, dtype=np.float32) if np.isscalar(weight) else np.asarray(weight, dtype=np.float32)
        )
        self._weight_np = weight_arr.ravel()[:3]
        self.robot: Optional[Robot] = None
        self.device = None
        if robot is not None:
            self.set_robot(robot)

    def set_robot(self, robot: Robot):
        self.robot = robot
        self._total_mass_inv = 1.0 / robot.spec.total_mass if robot.spec.total_mass > 0 else 0.0

    def _init_buffers(self, device):
        self.device = device
        if self._batched_target:
            self.target_com = wp.from_numpy(self._target_com_np, dtype=wp.float32, device=device)
        else:
            self.target_com = wp.from_numpy(self._target_com_np, dtype=wp.float32, device=device)
        self.residual_weight = wp.from_numpy(self._weight_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return 3

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if self.device is None:
            self._init_buffers(var.q.device)
        batch_size = var.batch_size
        kernel_device = out_residual.device if out_residual is not None else self.device
        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        spec_tensor = var.spec_tensors
        kernel = _com_position_residual_batched_kernel if self._batched_target else _com_position_residual_kernel
        wp.launch(
            kernel=kernel,
            dim=batch_size,
            inputs=[
                var.T_world_link,
                spec_tensor.link_masses,
                spec_tensor.link_local_com_positions,
                self.target_com,
                self.residual_weight,
                wp.float32(self._total_mass_inv),
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
        if self.device is None:
            self._init_buffers(var.q.device)
        spec_tensor = var.spec_tensors
        kernel_device = out_jacobian.device if out_jacobian is not None else self.device
        batch_size = var.batch_size
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (batch_size, self.residual_dim, var.tangent_dim), dtype=wp.float32, device=kernel_device
            )
            row_offset = 0
        num_actuated = spec_tensor.joints_to_actuated_mapping.shape[1]

        wp.launch(
            kernel=_com_position_jacobian_kernel,
            dim=(batch_size, num_actuated),
            inputs=[
                var.S_world,
                var.T_world_link,
                var.T_world_base,
                spec_tensor.link_masses,
                spec_tensor.link_local_com_positions,
                spec_tensor.link_ancestor_joints_mask,
                spec_tensor.joints_to_actuated_mapping,
                var.has_floating_base,
                self.residual_weight,
                wp.float32(self._total_mass_inv),
                row_offset,
                col_offset,
            ],
            outputs=[out_jacobian],
            device=kernel_device,
        )
        return out_jacobian


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _com_position_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    link_masses: wp.array1d(dtype=wp.float32),
    link_local_com: wp.array1d(dtype=wp.vec3),
    target_com: wp.array1d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    total_mass_inv: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    idx = wp.tid()
    com = compute_body_com_world_func(T_world_link, link_masses, link_local_com, total_mass_inv, idx)
    target = wp.vec3(target_com[0], target_com[1], target_com[2])
    error = target - com

    for i in range(3):
        out_residual[idx, row_offset + i] = weight[i] * error[i]


@wp.kernel
def _com_position_residual_batched_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    link_masses: wp.array1d(dtype=wp.float32),
    link_local_com: wp.array1d(dtype=wp.vec3),
    target_com: wp.array2d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    total_mass_inv: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    idx = wp.tid()
    com = compute_body_com_world_func(T_world_link, link_masses, link_local_com, total_mass_inv, idx)
    n_repeat = T_world_link.shape[0] // target_com.shape[0]
    target_idx = idx // n_repeat
    target = wp.vec3(target_com[target_idx, 0], target_com[target_idx, 1], target_com[target_idx, 2])
    error = target - com

    for i in range(3):
        out_residual[idx, row_offset + i] = weight[i] * error[i]


@wp.kernel
def _com_position_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    link_masses: wp.array1d(dtype=wp.float32),
    link_local_com: wp.array1d(dtype=wp.vec3),
    ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    weight: wp.array1d(dtype=wp.float32),
    total_mass_inv: float,
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    instance, actuated_idx = wp.tid()  # pyright: ignore

    num_joints = S_world.shape[2]
    num_links = T_world_link.shape[1]
    base_col_offset = 6 if has_floating_base else 0

    # accumulate the mass-weighted position Jacobian
    j_col = wp.vec3(0.0, 0.0, 0.0)
    for link_idx in range(num_links):
        m = link_masses[link_idx]
        if m > 0.0:
            # link center of mass in the world frame
            com_world = compute_link_com_world_func(T_world_link[instance, link_idx], link_local_com[link_idx])

            # motion-subspace column for this link and actuated joint
            spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for joint_idx in range(num_joints):
                if ancestor_mask[link_idx, joint_idx]:
                    w = joints_to_actuated[joint_idx, actuated_idx]
                    if w != 0.0:
                        for r in range(6):
                            spatial_col[r] += S_world[instance, r, joint_idx] * w

            v = wp.vec3(spatial_col[0], spatial_col[1], spatial_col[2])
            omega = wp.vec3(spatial_col[3], spatial_col[4], spatial_col[5])
            v_com = v + wp.cross(omega, com_world)
            j_col += m * v_com

    j_col = j_col * total_mass_inv

    col = col_offset + base_col_offset + actuated_idx
    for row in range(3):
        out_jacobian[instance, row_offset + row, col] = -weight[row] * j_col[row]

    # floating base columns
    if has_floating_base and actuated_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[actuated_idx] = 1.0
        spatial_twist = se3_adjoint_func(T_world_base[instance]) * unit_vec

        j_col_fb = wp.vec3(0.0, 0.0, 0.0)
        for link_idx in range(num_links):
            m = link_masses[link_idx]
            if m > 0.0:
                com_world = compute_link_com_world_func(T_world_link[instance, link_idx], link_local_com[link_idx])

                v_fb = wp.vec3(spatial_twist[0], spatial_twist[1], spatial_twist[2])
                omega_fb = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])
                v_com_fb = v_fb + wp.cross(omega_fb, com_world)
                j_col_fb += m * v_com_fb

        j_col_fb = j_col_fb * total_mass_inv

        fb_col = col_offset + actuated_idx
        for row in range(3):
            out_jacobian[instance, row_offset + row, fb_col] = -weight[row] * j_col_fb[row]
