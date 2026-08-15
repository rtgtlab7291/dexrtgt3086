# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
"""Whole-trajectory hand retargeting from named point targets."""

from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import numpy as np
import warp as wp

from robokit.helpers.hand_retargeting._kernels import (
    convert_root_quaternions_kernel,
    set_default_trajectory_state_kernel,
    set_trajectory_solution_kernel,
)
from robokit.helpers.hand_retargeting.config import HandRetargetingOfflineConfig, HandSpec
from robokit.helpers.hand_retargeting.kinematics_utils import build_retargeting_pairs
from robokit.lie.se3 import se3_identity
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizer
from robokit.opt.var_values import VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryContactTask, TrajectoryPositionTask
from robokit.terms.sparse.trajectory_retargeting_task import TrajectoryRetargetingTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.task import SparseTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.hand_coord_utils import hand_coord_conversion
from robokit.utils.warp_utils import wp_device_type, wp_vec7
from robokit.xform.numpy.rotation_conversions import matrix_to_quaternion


class HandRetargetingOffline:
    """Retarget fixed-shape point trajectories with reusable Warp solvers.

    Named correspondences from HandSpec support arbitrary point topologies.
    Normal, contact-aware, and anchored-contact solves share one canonical device point buffer and return packed base-plus-joint trajectories.

    Args:
        robot: Loaded robot model.
        spec: Ordered semantic point topology.
        config: Trajectory, contact, and solver settings.
        scene: Optional caller-owned collision scene.
        device: Warp device. The first CUDA device is used when available.
    """

    def __init__(
        self,
        robot: Robot,
        spec: HandSpec,
        config: HandRetargetingOfflineConfig,
        scene: Optional["WarpScene"] = None,  # noqa: F821
        device: Optional[wp_device_type] = None,
    ) -> None:
        """Compile named topology and shape-independent robot state."""
        self.robot = robot
        self.spec = spec
        self.config = config
        self.scene = scene
        self.device = wp.get_device(device if device is not None else ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self._shape: Optional[Tuple[int, int]] = None
        self._num_dofs = self.robot.spec.num_actuated_joints

        # --- compile named topology ---
        target_index = {name: index for index, name in enumerate(spec.target_names)}
        link_index = {name: index for index, name in enumerate(self.robot.spec.link_names)}
        selected_names = [name for name in spec.target_names if name in spec.target_link_names]
        if not config.include_root_target:
            selected_names.remove(spec.root_target_name)
        self._target_indices = [target_index[name] for name in selected_names]
        self._link_indices = [link_index[spec.target_link_names[name]] for name in selected_names]
        self._root_target_index = target_index[spec.root_target_name]
        self._root_link_index = link_index[spec.target_link_names[spec.root_target_name]]
        self._contact_target_indices = [target_index[name] for name in spec.contact_target_names]
        self._contact_link_indices = np.asarray(
            [link_index[spec.target_link_names[name]] for name in spec.contact_target_names], dtype=np.int32
        )

        self._pair_indices = (
            build_retargeting_pairs(spec, selected_names, config.pair_mode, config.include_root_target)
            if config.vector_weight > 0 or config.direction_weight > 0
            else np.empty((0, 2), dtype=np.int32)
        )

        # --- compile joint state ---
        joint_index = {name: index for index, name in enumerate(self.robot.spec.actuated_joint_names)}
        self._active_dofs = (
            None
            if config.active_joint_names is None
            else np.asarray([joint_index[name] for name in config.active_joint_names], dtype=np.int32)
        )

        root_ancestors = self.robot.spec.link_ancestor_joints_mask[self._root_link_index]
        hand_joints = self.robot.spec.link_ancestor_joints_mask[self._link_indices].any(axis=0) & ~root_ancestors
        hand_mapping = self.robot.spec.joints_to_actuated_mapping[hand_joints]
        hand_dofs = np.flatnonzero(np.any(hand_mapping != 0.0, axis=0)).astype(np.int32)

        limits = self.robot.spec.actuated_joint_limits.astype(np.float32, copy=False)
        self._q_rest_np = np.clip(self.robot.spec.zero_q.astype(np.float32, copy=True), limits[:, 0], limits[:, 1])
        for name, value in spec.rest_q_by_name.items():
            self._q_rest_np[joint_index[name]] = value
        self._q_init_np = self._q_rest_np.copy()
        if hand_dofs.size:
            self._q_init_np[hand_dofs] = np.mean(limits[hand_dofs], axis=1)
        for name, value in spec.init_q_by_name.items():
            self._q_init_np[joint_index[name]] = value
        self._q_init_np = np.clip(self._q_init_np, limits[:, 0], limits[:, 1])
        self._rest_weight_np = None
        if config.rest_weight > 0:
            self._rest_weight_np = np.full(self._num_dofs, config.rest_weight, dtype=np.float32)
            if not spec.floating_base and len(hand_dofs) != self._num_dofs:
                self._rest_weight_np.fill(0.0)
                self._rest_weight_np[hand_dofs] = config.rest_weight
            if not np.any(self._rest_weight_np):
                self._rest_weight_np = None
        self._joint_limits_wp = self.robot.spec.get_tensors(str(self.device)).actuated_joint_limits
        self._q_init_wp = wp.from_numpy(self._q_init_np, dtype=wp.float32, device=self.device)

        R_root_target = hand_coord_conversion(spec.target_coord_spec, spec.root_link_coord_spec)
        self._q_wxyz_target_root = wp.vec4(*matrix_to_quaternion(R_root_target).astype(np.float32))
        self._T_base_root_rest_wp = wp_vec7(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        if spec.floating_base:
            rest_state = self.robot.state(q=wp.from_numpy(self._q_rest_np[None], dtype=wp.float32, device=self.device))
            self.robot.forward_kinematics(rest_state)
            T_base_root_rest = cast(wp.array, rest_state.T_world_link[:, self._root_link_index]).numpy()[0]
            self._T_base_root_rest_wp = wp_vec7(*T_base_root_rest.astype(np.float32))

        self._collision_sphere_indices = None
        if len(self._contact_link_indices) and scene is not None and config.collision_weight > 0:
            contact_links = np.unique(self._contact_link_indices)
            contact_branches = self.robot.spec.link_ancestor_links_mask[contact_links].any(axis=0)
            contact_branches &= ~self.robot.spec.link_ancestor_links_mask[contact_links].all(axis=0)
            contact_branches[contact_links] = True
            self._collision_sphere_indices = np.flatnonzero(
                contact_branches[self.robot.spec.collision_spheres_link_indices]
            )

    def warmup(self, batch_size: int, num_frames: int) -> None:
        """Build buffers and all enabled solver modes for one trajectory shape.

        Args:
            batch_size: Number of independent trajectories.
            num_frames: Frames in each fixed-shape trajectory.

        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if num_frames < 3:
            raise ValueError("num_frames must be at least 3.")
        if self._shape == (batch_size, num_frames):
            return
        self._shape = None
        config = self.config
        num_contacts = len(self._contact_link_indices)

        # --- allocate stable solve buffers ---
        self._target_points_wp = wp.empty(
            (batch_size, num_frames, len(self.spec.target_names), 3), dtype=wp.float32, device=self.device
        )
        # translation-first pose buffer: the rotation task reads it as wp_vec7, the two
        # trajectory kernels read the quaternion tail as a strided [batch, frames, 4] view
        root_poses = np.zeros((batch_size, num_frames, 7), dtype=np.float32)
        root_poses[:, :, 3] = 1.0
        self._root_poses_wp = wp.from_numpy(root_poses, dtype=wp.float32, device=self.device)
        self._root_quaternions_wp = cast(wp.array, self._root_poses_wp[:, :, 3:7])
        self._root_target_wp = wp.array(
            ptr=self._root_poses_wp.ptr,
            shape=(batch_size * num_frames, 1),
            dtype=wp_vec7,
            device=self.device,
            copy=False,
        )
        q = wp.from_numpy(np.tile(self._q_init_np, (batch_size, num_frames, 1)), dtype=wp.float32, device=self.device)
        T_world_base = se3_identity((batch_size, num_frames), self.device) if self.spec.floating_base else None
        self._state = self.robot.state(q=q, T_world_base=T_world_base)
        self._var = VarValues(robot=self._state)
        self._output_wp = wp.empty((batch_size, num_frames, 7 + self._num_dofs), dtype=wp.float32, device=self.device)

        # --- build tracking objectives ---
        tracking_tasks: List[SparseTask] = []
        normal_position_links: List[int] = []
        normal_position_targets: List[int] = []
        normal_position_weights: List[float] = []
        if config.global_position_weight > 0:
            normal_position_links.extend(self._contact_link_indices.tolist())
            normal_position_targets.extend(self._contact_target_indices)
            normal_position_weights.extend([config.global_position_weight] * num_contacts)
        if config.root_position_weight > 0:
            normal_position_links.append(self._root_link_index)
            normal_position_targets.append(self._root_target_index)
            normal_position_weights.append(config.root_position_weight)
        if normal_position_links:
            tracking_tasks.append(
                TrajectoryPositionTask(
                    robot=self.robot,
                    frame_index=normal_position_links,
                    target_positions=self._target_points_wp,
                    target_point_indices=normal_position_targets,
                    weight=normal_position_weights,
                )
            )
        self._root_rotation_task = None
        if config.root_orientation_weight > 0:
            self._root_rotation_task = TrajectoryTask(
                RotationTask(
                    robot=self.robot,
                    frame_index=self._root_link_index,
                    T_world_target=self._root_target_wp,
                    weight=config.root_orientation_weight,
                ),
                num_frames=num_frames,
            )
            tracking_tasks.append(self._root_rotation_task)

        retargeting_task = None
        if len(self._pair_indices) and (config.vector_weight > 0 or config.direction_weight > 0):
            retargeting_task = TrajectoryRetargetingTask(
                robot=self.robot,
                target_keypoints=self._target_points_wp,
                robot_link_indices=self._link_indices,
                target_joint_indices=self._target_indices,
                pair_indices=self._pair_indices,
                position_weight=config.vector_weight,
                angle_weight=config.direction_weight,
                target_scale=config.target_scale,
            )
            tracking_tasks.append(retargeting_task)
        contact_tracking_tasks = [] if retargeting_task is None else [retargeting_task]
        if config.global_position_weight > 0:
            contact_tracking_tasks.append(
                TrajectoryPositionTask(
                    robot=self.robot,
                    frame_index=self._link_indices,
                    target_positions=self._target_points_wp,
                    target_point_indices=self._target_indices,
                    weight=config.global_position_weight,
                )
            )

        # --- build shared regularization ---
        reference_tasks: List[SparseTask] = []
        self._reference_q_wp = None
        if config.velocity_weight > 0 or config.acceleration_weight > 0:
            self._reference_q_wp = wp.zeros(
                (batch_size, num_frames, self._num_dofs), dtype=wp.float32, device=self.device
            )
        if config.velocity_weight > 0:
            task = TrajectorySmoothnessTask(self.robot, num_frames, config.velocity_weight)
            task.reference_q = self._reference_q_wp
            task.init_buffers(self.device)
            reference_tasks.append(task)
        if config.acceleration_weight > 0:
            task = TrajectorySmoothnessTask(self.robot, num_frames, config.acceleration_weight, order=2)
            task.reference_q = self._reference_q_wp
            task.init_buffers(self.device)
            reference_tasks.append(task)
        normal_regularization: List[SparseTask] = list(reference_tasks)
        refinement_regularization: List[SparseTask] = list(reference_tasks)
        refine_base_is_active = self.spec.floating_base and config.optimize_base
        if self.spec.floating_base and (
            config.root_position_velocity_weight > 0 or config.root_orientation_velocity_weight > 0
        ):
            root_smoothness = TrajectorySmoothnessTask(
                robot=self.robot,
                num_frames=num_frames,
                weight=0.0,
                base_weight=np.array(
                    [config.root_position_velocity_weight] * 3 + [config.root_orientation_velocity_weight] * 3,
                    dtype=np.float32,
                ),
            )
            normal_regularization.append(root_smoothness)
            if refine_base_is_active:
                refinement_regularization.append(root_smoothness)
        if config.limit_weight > 0:
            limit_task = TrajectoryTask(
                PositionLimit(robot=self.robot, weight=config.limit_weight),
                num_frames=num_frames,
            )
            normal_regularization.append(limit_task)
            refinement_regularization.append(limit_task)
        if self._rest_weight_np is not None:
            normal_regularization.append(
                TrajectoryTask(
                    RestTask(robot=self.robot, rest_q=self._q_rest_np, weight=self._rest_weight_np),
                    num_frames=num_frames,
                )
            )

        # --- build contact and anchor objectives ---
        self._contact_task = None
        self._anchor_task = None
        self._base_anchor_task = None
        collision_task = None
        if num_contacts:
            if config.contact_weight > 0:
                self._contact_task = TrajectoryContactTask(
                    robot=self.robot,
                    contact_points=np.zeros((num_frames, num_contacts, 3), dtype=np.float32),
                    contact_link_indices=self._contact_link_indices,
                    contact_mask=np.zeros((num_frames, num_contacts), dtype=np.bool_),
                    contact_margin=config.contact_margin,
                    weight=config.contact_weight,
                )
            if config.anchor_q_weight > 0:
                self._anchor_task = TrajectoryTask(
                    RestTask(
                        robot=self.robot,
                        rest_q=np.tile(self._q_init_np, (batch_size * num_frames, 1)),
                        weight=config.anchor_q_weight,
                    ),
                    num_frames=num_frames,
                )
            if refine_base_is_active and config.anchor_base_weight > 0:
                self._base_anchor_task = TrajectoryTask(
                    RestTask(
                        robot=self.robot,
                        T_world_base_rest=se3_identity(shape=(batch_size * num_frames,), device=self.device),
                        base_weight=config.anchor_base_weight,
                        include_joints=False,
                    ),
                    num_frames=num_frames,
                )
            if self._collision_sphere_indices is not None:
                collision_task = TrajectoryCollisionTask(
                    robot=self.robot,
                    scene=cast("WarpScene", self.scene),
                    num_frames=num_frames,
                    weight=config.collision_weight,
                    margin=config.collision_margin,
                    sphere_indices=self._collision_sphere_indices,
                )

        # --- build solver modes ---
        solver_config = replace(config.solver, use_early_stopping=False)

        def optimizer(
            tasks: List[SparseTask], values: VarValues, active_mask: Optional[wp.array]
        ) -> Optional[SparseLMOptimizer]:
            if not tasks:
                return None
            return SparseLMOptimizer(
                term=tasks,
                batch_size=batch_size,
                total_tangent_dim=values.tangent_dim,
                device=self.device,
                config=solver_config,
                active_dof_mask=active_mask,
            )

        self._optimizer = optimizer(tracking_tasks + normal_regularization, self._var, None)
        self._contact_optimizer = None
        self._refinement_optimizer = None
        if num_contacts:
            base_dim = 6 if self.spec.floating_base else 0
            frame_dim = base_dim + self._num_dofs
            refine_mask = np.ones(self._var.tangent_dim, dtype=np.float32)
            refine_robot_mask = refine_mask.reshape(num_frames, frame_dim)
            refine_robot_mask[:, :base_dim] *= float(refine_base_is_active)
            if self._active_dofs is not None:
                joint_mask = np.zeros(self._num_dofs, dtype=np.float32)
                joint_mask[self._active_dofs] = 1.0
                refine_robot_mask[:, base_dim:] *= joint_mask
            if config.locked_prefix_frames:
                self._refine_unlocked_mask_np = refine_mask.copy()
            refine_robot_mask[: config.locked_prefix_frames, base_dim:] = 0.0
            self._refine_mask_wp = (
                None if np.all(refine_mask == 1.0) else wp.from_numpy(refine_mask, dtype=wp.float32, device=self.device)
            )
            contact_extras = [task for task in (self._contact_task, collision_task) if task is not None]
            self._contact_optimizer = optimizer(
                contact_tracking_tasks + normal_regularization + contact_extras, self._var, None
            )
            refinement_tasks = contact_extras + refinement_regularization
            if self._anchor_task is not None:
                refinement_tasks.insert(1, self._anchor_task)
            if self._base_anchor_task is not None:
                refinement_tasks.append(self._base_anchor_task)
            self._refinement_optimizer = optimizer(refinement_tasks, self._var, self._refine_mask_wp)
        self._shape = (batch_size, num_frames)

        # --- compile reusable hot paths ---
        self._target_points_wp.zero_()
        for solver, values in (
            (self._optimizer, self._var),
            (self._contact_optimizer, self._var),
            (self._refinement_optimizer, self._var),
        ):
            if solver is not None:
                solver.solve(values)
        wp.synchronize_device(self.device)

    def solve(
        self,
        target_points: wp.array,
        root_quat_wxyz: Optional[wp.array] = None,
        init_state: Optional[RobotState] = None,
        out: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget one fixed-shape Warp trajectory batch.

        Lifecycle:
            1. Validate input and output buffers against the warmed shape and device.
            2. Update tracking targets and initialize the trajectory state.
            3. Solve, clamp joints, and write packed output.

        Args:
            target_points: Float points shaped (batch, frames, targets, 3) in
                spec.target_names order.
            root_quat_wxyz: Optional root rotations shaped (batch, frames, 4).
            init_state: Optional joint and base warm start.
            out: Optional packed output shaped (batch, frames, 7 + dofs).

        Returns:
            Packed base transforms and joint positions.
        """
        # --- validate inputs ---
        if self._shape is None:
            raise RuntimeError("call warmup(batch_size, num_frames) before solve().")
        batch_size, num_frames = self._shape
        expected_points = (batch_size, num_frames, len(self.spec.target_names), 3)
        if (target_points.shape, target_points.dtype, target_points.device) != (
            expected_points,
            wp.float32,
            self.device,
        ):
            raise ValueError(f"target_points must be float32 on {self.device} with shape {expected_points}.")
        if self.config.root_orientation_weight > 0 and root_quat_wxyz is None:
            raise ValueError("root_quat_wxyz is required when root_orientation_weight is nonzero.")
        if root_quat_wxyz is not None:
            expected_quaternions = (batch_size, num_frames, 4)
            if (root_quat_wxyz.shape, root_quat_wxyz.dtype, root_quat_wxyz.device) != (
                expected_quaternions,
                wp.float32,
                self.device,
            ):
                raise ValueError(f"root_quat_wxyz must be float32 on {self.device} with shape {expected_quaternions}.")
        if init_state is not None:
            expected_state = (batch_size, num_frames, self._num_dofs), (batch_size, num_frames), self.device
            if (init_state.q.shape, init_state.T_world_base.shape, init_state.device) != expected_state:
                raise ValueError("init_state must match the warmed trajectory shape, robot DOFs, and device.")
        expected_output = (batch_size, num_frames, 7 + self._num_dofs)
        if out is not None and (out.shape != expected_output or out.dtype != wp.float32 or out.device != self.device):
            raise ValueError(f"out must be float32 on {self.device} with shape {expected_output}.")

        # --- update targets and state ---
        wp.copy(self._target_points_wp, target_points)
        if root_quat_wxyz is not None:
            wp.launch(
                convert_root_quaternions_kernel,
                dim=(batch_size, num_frames),
                inputs=[root_quat_wxyz, self._q_wxyz_target_root, self._root_quaternions_wp],
                device=self.device,
            )
        if self._root_rotation_task is not None:
            # RotationTask keeps its own target copy
            self._root_rotation_task.dense_task.set_target(self._root_target_wp)
        if self._reference_q_wp is not None:
            self._reference_q_wp.zero_()
        if init_state is None:
            wp.launch(
                set_default_trajectory_state_kernel,
                dim=(batch_size, num_frames),
                inputs=[
                    self._target_points_wp,
                    self._root_target_index,
                    self._root_quaternions_wp,
                    int(root_quat_wxyz is not None),
                    self._T_base_root_rest_wp,
                    int(self.spec.floating_base),
                    self._q_init_wp,
                    self._state.q,
                    self._state.T_world_base,
                ],
                device=self.device,
            )
        else:
            wp.copy(self._state.q, init_state.q)
            wp.copy(self._state.T_world_base, init_state.T_world_base)
        self._state.invalidate()

        # --- solve and write output ---
        optimized_var, _ = self._optimizer.solve(self._var)
        optimized_state = cast(RobotState, optimized_var.get("robot"))
        output = self._output_wp if out is None else out
        wp.launch(
            set_trajectory_solution_kernel,
            dim=expected_output,
            inputs=[optimized_state.q, optimized_state.T_world_base, self._joint_limits_wp, output],
            device=self.device,
        )
        return output

    def solve_numpy(
        self,
        target_points: np.ndarray,
        root_quat_wxyz: Optional[np.ndarray] = None,
        init_state: Optional[RobotState] = None,
    ) -> np.ndarray:
        """NumPy wrapper for `solve()`."""
        points = np.asarray(target_points, dtype=np.float32)
        squeeze_batch = points.ndim == 3
        if squeeze_batch:
            points = points[None]
        batch_size, num_frames = points.shape[:2]
        quaternions = None if root_quat_wxyz is None else np.asarray(root_quat_wxyz, dtype=np.float32)
        if quaternions is not None and squeeze_batch and quaternions.ndim == 2:
            quaternions = quaternions[None]
        self.warmup(batch_size, num_frames)
        result = (
            self.solve(
                wp.from_numpy(np.ascontiguousarray(points), dtype=wp.float32, device=self.device),
                (
                    wp.from_numpy(np.ascontiguousarray(quaternions), dtype=wp.float32, device=self.device)
                    if quaternions is not None
                    else None
                ),
                init_state,
            )
            .numpy()
            .copy()
        )
        return result[0] if squeeze_batch else result

    def solve_with_contact(
        self,
        target_points: wp.array,
        contact_points: wp.array,
        contact_mask: wp.array,
        init_state: Optional[RobotState] = None,
        local_contact_points: Optional[wp.array] = None,
        out: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget with contacts.

        Lifecycle:
            1. Validate input and output buffers against the warmed shape and device.
            2. Update contact targets and initialize from target points or `init_state`.
            3. Solve, clamp joints, and write packed output.

        Args:
            target_points: Named points shaped (batch, frames, targets, 3).
            contact_points: World contacts shaped (batch, frames, contacts, 3).
            contact_mask: Active contacts shaped (batch, frames, contacts).
            init_state: Optional state that anchors the solution around that trajectory.
            local_contact_points: Optional contacts in each link frame.
            out: Optional caller-owned packed output.

        Returns:
            Packed base transforms and joint positions.
        """
        # --- validate inputs ---
        if self._shape is None:
            raise RuntimeError("call warmup(batch_size, num_frames) before solve_with_contact().")
        batch_size, num_frames = self._shape
        expected_points = (batch_size, num_frames, len(self.spec.target_names), 3)
        if (target_points.shape, target_points.dtype, target_points.device) != (
            expected_points,
            wp.float32,
            self.device,
        ):
            raise ValueError(f"target_points must be float32 on {self.device} with shape {expected_points}.")
        if init_state is not None:
            expected_state = (batch_size, num_frames, self._num_dofs), (batch_size, num_frames), self.device
            if (init_state.q.shape, init_state.T_world_base.shape, init_state.device) != expected_state:
                raise ValueError("init_state must match the warmed trajectory shape, robot DOFs, and device.")
        expected_output = (batch_size, num_frames, 7 + self._num_dofs)
        if out is not None and (out.shape != expected_output or out.dtype != wp.float32 or out.device != self.device):
            raise ValueError(f"out must be float32 on {self.device} with shape {expected_output}.")

        # --- update targets and state ---
        anchored = init_state is not None
        if anchored:
            if self._reference_q_wp is not None:
                wp.copy(self._reference_q_wp, init_state.q)
            wp.copy(self._state.q, init_state.q)
            wp.copy(self._state.T_world_base, init_state.T_world_base)
        else:
            wp.copy(self._target_points_wp, target_points)
            if self._reference_q_wp is not None:
                self._reference_q_wp.zero_()
            wp.launch(
                set_default_trajectory_state_kernel,
                dim=(batch_size, num_frames),
                inputs=[
                    self._target_points_wp,
                    self._root_target_index,
                    self._root_quaternions_wp,
                    0,
                    wp_vec7(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    int(self.spec.floating_base),
                    self._q_init_wp,
                    self._state.q,
                    self._state.T_world_base,
                ],
                device=self.device,
            )
        self._state.invalidate()
        if self._contact_task is not None:
            self._contact_task.set_contacts(contact_points, contact_mask, local_contact_points)

        # --- solve and write output ---
        optimizer = self._contact_optimizer
        if anchored:
            if self._anchor_task is not None:
                self._anchor_task.dense_task.set_rest_state(rest_q=init_state.q.reshape((-1, self._num_dofs)))
            if self._base_anchor_task is not None:
                self._base_anchor_task.dense_task.set_rest_state(T_world_base_rest=init_state.T_world_base.flatten())
            optimizer = self._refinement_optimizer
        optimized_var, _ = optimizer.solve(self._var)
        optimized_state = cast(RobotState, optimized_var.get("robot"))
        output = self._output_wp if out is None else out
        wp.launch(
            set_trajectory_solution_kernel,
            dim=expected_output,
            inputs=[optimized_state.q, optimized_state.T_world_base, self._joint_limits_wp, output],
            device=self.device,
        )
        return output

    def solve_with_contact_numpy(
        self,
        target_points: np.ndarray,
        contact_points: np.ndarray,
        contact_mask: np.ndarray,
        init_state: Optional[RobotState] = None,
        local_contact_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """NumPy wrapper for `solve_with_contact()`."""
        points = np.asarray(target_points, dtype=np.float32)
        squeeze_batch = points.ndim == 3
        if squeeze_batch:
            points = points[None]
        batch_size, num_frames = points.shape[:2]
        contacts = np.asarray(contact_points, dtype=np.float32)
        masks = np.asarray(contact_mask, dtype=np.bool_)
        local_contacts = None if local_contact_points is None else np.asarray(local_contact_points, dtype=np.float32)
        if squeeze_batch:
            if contacts.ndim == 3:
                contacts = contacts[None]
            if masks.ndim == 2:
                masks = masks[None]
            if local_contacts is not None and local_contacts.ndim == 3:
                local_contacts = local_contacts[None]
        self.warmup(batch_size, num_frames)
        result = (
            self.solve_with_contact(
                wp.from_numpy(np.ascontiguousarray(points), dtype=wp.float32, device=self.device),
                wp.from_numpy(np.ascontiguousarray(contacts), dtype=wp.float32, device=self.device),
                wp.from_numpy(np.ascontiguousarray(masks), dtype=wp.bool, device=self.device),
                init_state,
                (
                    wp.from_numpy(np.ascontiguousarray(local_contacts), dtype=wp.float32, device=self.device)
                    if local_contacts is not None
                    else None
                ),
            )
            .numpy()
            .copy()
        )
        return result[0] if squeeze_batch else result


if TYPE_CHECKING:
    from robokit.geom import WarpScene


__all__ = ["HandRetargetingOffline"]
