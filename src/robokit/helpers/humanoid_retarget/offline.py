"""Offline (whole-trajectory) humanoid retargeting."""

from typing import Optional, Tuple, cast

import numpy as np
import warp as wp

from robokit.helpers.humanoid_retarget._kernels import compute_targets_kernel, set_trajectory_solution_kernel
from robokit.helpers.humanoid_retarget.config import (
    HumanoidRetargetingOfflineConfig,
    HumanoidRetargetingOnlineConfig,
)
from robokit.helpers.ik import IK, IKConfig
from robokit.lie.se3 import se3_identity
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizer, SparseLMOptimizerConfig
from robokit.opt.var_values import VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryPositionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.sparse.trajectory_velocity_limit_task import TrajectoryVelocityLimitTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.warp_utils import wp_vec7


class HumanoidRetargetingOffline:
    """Whole-clip trajectory retargeting.

    Lifecycle:
        1. ``__init__`` stores the robot and configuration.
        2. ``warmup(batch_size, num_frames)`` builds shape-bound buffers and optimizers.
        3. ``solve(T_world_human)`` retargets one fixed-shape Warp trajectory batch.

    Input joints follow ``human_joint_names``; ``solve_numpy(...)`` converts NumPy inputs and calls ``warmup(...)`` for their shape.
    """

    def __init__(
        self,
        config: HumanoidRetargetingOnlineConfig,
        offline_config: HumanoidRetargetingOfflineConfig,
        robot: Optional[Robot] = None,
        device: str = "cuda:0",
    ) -> None:
        online_only = [
            name
            for name in ("interaction", "correspondence_edges", "scene_collision_weight", "self_collision_weight")
            if getattr(config, name)
        ]
        if online_only:
            raise ValueError(
                f"{', '.join(online_only)} are online-only; offline retargeting would ignore them. "
                "Use HumanoidRetargetingOfflineConfig.self_collision_weight for offline self-collision."
            )
        self.retarget_config = config
        self.offline_config = offline_config
        self.robot = robot if robot is not None else config.load_robot()
        self._wp_device = wp.get_device(device)
        self._shape = None
        if offline_config.self_collision_weight > 0 and not self.robot.spec.has_collision_spheres:
            raise ValueError("offline self collision requires robot collision spheres")

        self.human_joint_names = config.human_joint_names
        self._targets = config.target_arrays(self.human_joint_names)
        self._num_targets = len(self._targets.joint_indices)

    def warmup(self, batch_size: int, num_frames: int):
        """Build and cache all state bound to ``(batch_size, num_frames)``."""
        if batch_size < 1 or num_frames < 1:
            raise ValueError("batch_size and num_frames must be positive")
        shape = (batch_size, num_frames)
        if self._shape == shape:
            return
        self._shape = None

        robot = self.robot
        mapping = self.retarget_config.link_mapping
        links = list(mapping)
        link_indices = [robot.spec.link_names.index(name) for name in links]
        flat = batch_size * num_frames

        self._target_joint_indices_wp = wp.from_numpy(
            self._targets.joint_indices, dtype=wp.int32, device=self._wp_device
        )
        self._target_scales_wp = wp.from_numpy(self._targets.scales, dtype=wp.float32, device=self._wp_device)
        self._rotation_offsets_wp = wp.from_numpy(self._targets.rotation_offsets, dtype=wp.vec4, device=self._wp_device)
        self._position_offsets_wp = wp.from_numpy(self._targets.position_offsets, dtype=wp.vec3, device=self._wp_device)
        # One packed target buffer: float32 for the alignment terms, wp_vec7 alias for the target
        # kernel and IK. Two whole-buffer slices give the per-link position/orientation views.
        self._target_buf_wp = wp.zeros(
            (batch_size, num_frames, len(links), 7), dtype=wp.float32, device=self._wp_device
        )
        # wp_vec7 alias over the same memory: the buffer is contiguous, so the default strides for
        # (flat, links) of 28-byte elements land exactly on the float32 (batch, frames, links, 7) layout.
        self._T_world_target_wp = wp.array(
            ptr=self._target_buf_wp.ptr,
            shape=(flat, len(links)),
            dtype=wp_vec7,
            device=self._wp_device,
            copy=False,
        )
        self._T_world_base_target_wp = wp.from_numpy(
            np.empty((flat, 7), dtype=np.float32), dtype=wp_vec7, device=self._wp_device
        )
        self._unused_indices_wp = wp.zeros((1,), dtype=wp.int32, device=self._wp_device)
        self._unused_scales_wp = wp.zeros((1,), dtype=wp.float32, device=self._wp_device)
        self._unused_vectors_wp = wp.zeros((flat, 1, 3), dtype=wp.float32, device=self._wp_device)
        self._default_valid_lengths_wp = wp.from_numpy(
            np.full(batch_size, num_frames, dtype=np.int32), dtype=wp.int32, device=self._wp_device
        )
        self._default_human_heights_wp = wp.from_numpy(
            np.full(batch_size, self.retarget_config.human_height_assumption, dtype=np.float32),
            dtype=wp.float32,
            device=self._wp_device,
        )
        self._joint_limits_wp = wp.from_numpy(
            robot.spec.actuated_joint_limits.astype(np.float32, copy=False),
            dtype=wp.float32,
            device=self._wp_device,
        )
        velocity_limits = (
            np.array(self.offline_config.velocity_limit_override, dtype=np.float32)
            if self.offline_config.velocity_limit_override is not None
            else robot.spec.actuated_joint_velocity_limits.astype(np.float32, copy=False)
        )
        self._max_dq_wp = wp.from_numpy(
            velocity_limits * self.offline_config.velocity_limit_dt * self.offline_config.velocity_clamp_scale,
            dtype=wp.float32,
            device=self._wp_device,
        )
        self._output_wp = wp.empty(
            (batch_size, num_frames, 7 + robot.spec.num_actuated_joints),
            dtype=wp.float32,  # type: ignore[reportArgumentType]
            device=self._wp_device,
        )

        target_positions = cast(wp.array, self._target_buf_wp[:, :, :, 0:3])

        ik_config = (
            IKConfig(
                enable_T_world_base=True,
                solver=MultiSeedSolverConfig(
                    stages=[
                        StageConfig(num_seeds=4, iters=6, lm_lambda=1.5),
                        StageConfig(num_seeds=1, iters=3, lm_lambda=1.5),
                    ],
                ),
            )
            .add(PositionTask(weight=10.0))
            .add(RotationTask(weight=5.0))
            .add(PositionLimit(weight=30.0))
        )
        self._ik = IK(ik_config, robot=robot, link=link_indices, device=self._wp_device)
        self._ik.warmup(flat)
        ik_init_q = wp.from_numpy(
            np.tile(robot.spec.midrange_q.astype(np.float32), (flat, 1)),
            dtype=wp.float32,
            device=self._wp_device,
        )
        self._ik_init_state = robot.state(
            q=ik_init_q,
            T_world_base=self._T_world_base_target_wp,
        )

        self._position_task = TrajectoryPositionTask(
            robot=robot,
            frame_index=link_indices,
            target_positions=target_positions,
            weight=[mapping[name].position_weight for name in links],
        )
        self._orientation_task = TrajectoryTask(
            RotationTask(
                robot=robot,
                frame_index=link_indices,
                T_world_target=self._T_world_target_wp,
                weight=[mapping[name].orientation_weight for name in links],
            ),
            num_frames=num_frames,
        )
        config = self.offline_config
        self._tasks = [
            self._position_task,
            self._orientation_task,
            TrajectorySmoothnessTask(
                robot=robot,
                num_frames=num_frames,
                weight=(
                    np.array(config.joint_smoothness_weight, dtype=np.float32)
                    if isinstance(config.joint_smoothness_weight, list)
                    else config.joint_smoothness_weight
                ),
                base_weight=np.array(
                    [config.root_position_smoothness_weight] * 3 + [config.root_orientation_smoothness_weight] * 3,
                    dtype=np.float32,
                ),
            ),
            TrajectoryTask(
                PositionLimit(robot=robot, weight=config.limit_weight),
                num_frames=num_frames,
            ),
        ]
        rest_weight = (
            np.array(config.rest_weight, dtype=np.float32)
            if isinstance(config.rest_weight, list)
            else config.rest_weight
        )
        if np.any(np.asarray(rest_weight) > 0):
            self._tasks.append(
                TrajectoryTask(
                    RestTask(robot=robot, rest_q=robot.spec.midrange_q, weight=rest_weight),
                    num_frames=num_frames,
                )
            )
        if config.base_rest_weight is not None:
            self._tasks.append(
                TrajectoryTask(
                    RestTask(
                        robot=robot,
                        T_world_base_rest=se3_identity(shape=(1,), device=self._wp_device),
                        base_weight=(
                            np.array(config.base_rest_weight, dtype=np.float32)
                            if isinstance(config.base_rest_weight, list)
                            else config.base_rest_weight
                        ),
                        include_joints=False,
                    ),
                    num_frames=num_frames,
                )
            )
        if config.velocity_limit_weight > 0:
            velocity_limits = (
                np.array(config.velocity_limit_override, dtype=np.float32)
                if config.velocity_limit_override is not None
                else None
            )
            self._tasks.append(
                TrajectoryVelocityLimitTask(
                    robot=robot,
                    num_frames=num_frames,
                    dt=config.velocity_limit_dt,
                    velocity_limits=velocity_limits,
                    weight=config.velocity_limit_weight,
                )
            )
        if config.self_collision_weight > 0:
            self._tasks.append(
                TrajectorySelfCollisionTask(
                    robot=robot,
                    num_frames=num_frames,
                    weight=config.self_collision_weight,
                    margin=config.self_collision_margin,
                )
            )

        for task in self._tasks:
            task.init_buffers(self._wp_device)
            task.set_valid_lengths(self._default_valid_lengths_wp)

        q_init = np.tile(
            robot.spec.midrange_q.astype(np.float32),
            (batch_size, num_frames, 1),
        )
        self._state = robot.state(
            q=wp.from_numpy(q_init, dtype=wp.float32, device=self._wp_device),
            T_world_base=se3_identity(shape=(batch_size, num_frames), device=self._wp_device),
        )
        self._var = VarValues(robot=self._state)
        self._optimizer = SparseLMOptimizer(
            term=self._tasks,
            batch_size=batch_size,
            total_tangent_dim=self._var.tangent_dim,
            device=self._wp_device,
            config=SparseLMOptimizerConfig(
                lm_lambda=config.lm_lambda,
                max_iter=config.max_iter,
                lambda_factor=config.lambda_factor,
                lambda_min=config.lambda_min,
                lambda_max=config.lambda_max,
                rho_min=config.rho_min,
                use_cuda_graph=config.cuda_graph_mode != "none",
            ),
        )
        self._shape = shape

    def _compute_targets(
        self,
        T_world_human: wp.array,
        human_heights: wp.array,
    ) -> wp.array:
        """Compute every trajectory target with one Warp launch."""
        batch_size, num_frames = cast(Tuple[int, int], self._shape)
        wp.launch(
            compute_targets_kernel,
            dim=(batch_size * num_frames, self._num_targets + 1),
            inputs=[
                T_world_human.reshape((batch_size * num_frames, len(self.human_joint_names))),
                human_heights,
                self._target_joint_indices_wp,
                self._target_scales_wp,
                self._rotation_offsets_wp,
                self._position_offsets_wp,
                self._unused_indices_wp,
                self._unused_indices_wp,
                self._unused_scales_wp,
                self._unused_scales_wp,
                self._T_world_target_wp,
                self._T_world_base_target_wp,
                self._unused_vectors_wp,
                self._num_targets,
                0,
                self._targets.root_joint_index,
                self._targets.root_target_index,
                self._targets.root_scale,
                float(self.retarget_config.human_height_assumption),
                num_frames,
                float(self.retarget_config.ground_height),
            ],
            device=self._wp_device,
        )
        # RotationTask keeps its own target copy
        self._orientation_task.dense_task.set_target(self._T_world_target_wp)
        return self._T_world_target_wp

    def solve(
        self,
        T_world_human: wp.array,
        valid_lengths: Optional[wp.array] = None,
        human_heights: Optional[wp.array] = None,
        *,
        out: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget one padded ``[B, T, J]`` Warp trajectory batch.

        Lifecycle:
            1. Validate the shape fixed by ``warmup(...)``.
            2. Compute all frame targets and independent IK warm starts.
            3. Reset the cached trajectory state and valid-frame masks.
            4. Solve once and write the requested padded output buffer.

        Args:
            T_world_human: Human joint transforms with shape ``(batch, frames, joints)``.
            valid_lengths: Optional valid frame count for each padded batch row.
            human_heights: Optional human height in meters for each batch row.
            out: Optional caller-owned output with shape ``(batch, frames, 7 + dofs)``.

        Returns:
            ``out`` when provided; otherwise a reusable helper-owned buffer overwritten by the next call.
        """
        # --- validate inputs ---
        if self._shape is None:
            raise RuntimeError("call warmup() before solve()")
        batch, length = cast(Tuple[int, int], self._shape)
        expected_shape = (batch, length, len(self.human_joint_names))
        if T_world_human.shape != expected_shape:
            raise ValueError(f"expected T_world_human shape {expected_shape}, got {T_world_human.shape}")
        lengths = self._default_valid_lengths_wp if valid_lengths is None else valid_lengths
        heights = self._default_human_heights_wp if human_heights is None else human_heights
        if lengths.shape != (batch,):
            raise ValueError(f"expected valid_lengths shape ({batch},), got {lengths.shape}")
        if heights.shape != (batch,):
            raise ValueError(f"expected human_heights shape ({batch},), got {heights.shape}")

        # --- compute targets ---
        self._compute_targets(T_world_human, heights)

        # --- compute IK warm starts ---
        ik_state = self._ik.solve(self._T_world_target_wp, init_state=self._ik_init_state)

        # --- reset trajectory state ---
        for task in self._tasks:
            task.set_valid_lengths(lengths)
        wp.copy(
            self._state.q,
            ik_state.q.reshape((batch, length, self.robot.spec.num_actuated_joints)),
        )
        wp.copy(
            self._state.T_world_base,
            self._T_world_base_target_wp.reshape((batch, length)),
        )
        self._state.invalidate()

        # --- solve and finalize ---
        optimized_var, _ = self._optimizer.solve(self._var)
        optimized_state = cast(RobotState, optimized_var.get("robot"))
        output = self._output_wp if out is None else out
        wp.launch(
            set_trajectory_solution_kernel,
            dim=(batch, 7 + self.robot.spec.num_actuated_joints),
            inputs=[
                optimized_state.q,
                optimized_state.T_world_base,
                self._joint_limits_wp,
                self._max_dq_wp,
                output,
                1 if self.offline_config.velocity_clamp_scale > 0 else 0,
            ],
            device=self._wp_device,
        )
        return output

    def solve_numpy(
        self,
        T_world_human: np.ndarray,
        valid_lengths: Optional[np.ndarray] = None,
        human_heights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """NumPy boundary for padded ``[B, T, J, 7]`` trajectories."""
        motions = np.ascontiguousarray(T_world_human, dtype=np.float32)
        if motions.ndim != 4 or motions.shape[-1] != 7:
            raise ValueError("T_world_human must have shape [B, T, J, 7]")
        batch, length, joints, _ = motions.shape
        if joints != len(self.human_joint_names):
            raise ValueError(f"expected J={len(self.human_joint_names)}, got {joints}")
        lengths = (
            np.full(batch, length, dtype=np.int32)
            if valid_lengths is None
            else np.asarray(valid_lengths, dtype=np.int32)
        )
        heights = (
            np.full(batch, self.retarget_config.human_height_assumption, dtype=np.float32)
            if human_heights is None
            else np.asarray(human_heights, dtype=np.float32)
        )
        if lengths.shape != (batch,) or not np.all((1 <= lengths) & (lengths <= length)):
            raise ValueError(f"valid_lengths must contain {batch} values in [1, {length}]")
        if heights.shape != (batch,) or np.any(heights <= 0):
            raise ValueError(f"human_heights must contain {batch} positive values")
        self.warmup(batch, length)
        output = self.solve(
            wp.from_numpy(motions, dtype=wp_vec7, device=self._wp_device),
            wp.from_numpy(lengths, dtype=wp.int32, device=self._wp_device),
            wp.from_numpy(heights, dtype=wp.float32, device=self._wp_device),
        )
        return output.numpy().copy()


__all__ = ["HumanoidRetargetingOffline"]
