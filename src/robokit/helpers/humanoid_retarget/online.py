"""Dense-array online humanoid retargeting."""

from typing import List, Optional, cast

import numpy as np
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.humanoid_retarget._kernels import (
    compute_scaled_weights_kernel,
    compute_seed_states_kernel,
    compute_targets_kernel,
    set_solution_state_kernel,
)
from robokit.helpers.humanoid_retarget.config import HumanoidRetargetingOnlineConfig
from robokit.lie.se3 import se3_identity
from robokit.opt.multi_seed_solver import MultiSeedSolver, MultiSeedSolverConfig
from robokit.opt.var_values import VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms import FrameTask, FrameVectorTask
from robokit.terms.dense.interaction_mesh_task import InteractionMeshTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec7


class HumanoidRetargetingOnline:
    """Online humanoid retargeter.

    Lifecycle:
        1. ``__init__`` stores the robot, configuration, and caller-owned scene.
        2. ``warmup(batch_size)`` builds batch-shaped buffers and solvers.
        3. ``solve(T_world_human)`` advances one batched frame from Warp inputs.

    Input joints follow ``human_joint_names``
    """

    def __init__(
        self,
        config: HumanoidRetargetingOnlineConfig,
        robot: Optional[Robot] = None,
        scene: Optional[WarpScene] = None,
        device: str = "cuda:0",
        seed: int = 0,
    ):
        assert config.urdf_path is not None, "config.urdf_path must be set"
        self.config = config
        self.robot = robot if robot is not None else config.load_robot()
        self.device = wp.get_device(device)
        self._scene = scene
        if scene is not None and scene.device != self.device:
            raise ValueError(f"scene must be on {self.device}, got {scene.device}")
        if config.uses_collision_spheres and not self.robot.spec.has_collision_spheres:
            raise ValueError("scene and self collision require robot collision spheres")
        if config.scene_collision_weight > 0:
            if scene is None:
                raise ValueError("scene_collision_weight > 0 requires scene")
            if not scene.geoms:
                raise ValueError("scene collision requires at least one geometry")
        self._seed = seed
        self._num_seeds0 = config.stages[0].num_seeds
        self._num_dofs = self.robot.spec.num_actuated_joints
        self.link_name_to_index = {name: i for i, name in enumerate(self.robot.spec.link_names)}
        assert config.stages and config.stages[-1].num_seeds == 1, "stages must end with num_seeds=1"
        if config.interaction is not None:
            assert len(config.stages) == 1 and config.stages[0].num_seeds == 1, (
                "interaction retargeting currently requires one stage with one seed"
            )

        self.human_joint_names = config.human_joint_names
        joint_index = {name: i for i, name in enumerate(self.human_joint_names)}
        self._targets = config.target_arrays(self.human_joint_names)
        self._num_targets = len(self._targets.joint_indices)

        edges = config.correspondence_edges or []
        self._corr_origin_indices_np = np.array([joint_index[e.origin_joint] for e in edges], dtype=np.int32)
        self._corr_task_indices_np = np.array([joint_index[e.task_joint] for e in edges], dtype=np.int32)
        self._corr_origin_scales_np = np.array(
            [config.scale_table.get(e.origin_joint, 1.0) for e in edges], dtype=np.float32
        )
        self._corr_task_scales_np = np.array(
            [config.scale_table.get(e.task_joint, 1.0) for e in edges], dtype=np.float32
        )
        self._num_vector_targets = len(edges)
        self._batch_size: Optional[int] = None
        self._interaction_task: Optional[InteractionMeshTask] = None

        if config.interaction is not None:
            if scene is None:
                raise ValueError("interaction retargeting requires scene")
            mesh_geoms = [geom for geom in scene.geoms if isinstance(geom, MeshGeom)]
            if len(mesh_geoms) != 1 or mesh_geoms[0].count != 1:
                raise ValueError("interaction retargeting requires one scene mesh")
            key_link_indices = [self.link_name_to_index[name] for name in config.link_mapping]
            self._interaction_task = InteractionMeshTask(
                self.robot,
                np.array(key_link_indices, dtype=np.int32),
                weight=config.interaction.laplacian_weight,
            )

    @property
    def interaction_task(self) -> InteractionMeshTask:
        """Return the interaction task; set its first frame before warmup and update it before each solve."""
        if self._interaction_task is None:
            raise RuntimeError("interaction is not enabled")
        return self._interaction_task

    def _build_solver(self, batch_size: int):
        """Build the multi-seed solver + per-stage term lists (sets the ``_solver``/``*_tasks`` attrs)."""
        mapping = self.config.link_mapping
        stages = self.config.stages
        wp_device = self.device

        frame_indices: List[int] = []
        position_weights: List[float] = []
        orientation_weights: List[float] = []
        for robot_link, entry in mapping.items():
            frame_indices.append(self.link_name_to_index[robot_link])
            position_weights.append(entry.position_weight)
            orientation_weights.append(entry.orientation_weight)

        placeholder_targets = se3_identity(shape=(batch_size, len(frame_indices)), device=wp_device)
        placeholder_base = se3_identity(shape=(batch_size,), device=wp_device)

        n_dofs = self.robot.spec.num_actuated_joints
        mask_np = np.ones(n_dofs, dtype=np.float32)
        if self.config.per_dof_limit_mask:
            joint_names = self.robot.spec.actuated_joint_names
            for joint_name, val in self.config.per_dof_limit_mask.items():
                mask_np[joint_names.index(joint_name)] = float(val)

        corr_edges = self.config.correspondence_edges or []
        corr_origin_idx = [self.link_name_to_index[e.origin_link] for e in corr_edges]
        corr_task_idx = [self.link_name_to_index[e.task_link] for e in corr_edges]
        corr_weights = [e.weight for e in corr_edges]

        stage_terms: List[List[ResidualTask]] = []
        self._frame_tasks: List[FrameTask] = []
        self._smoothness_tasks: List[Optional[SmoothnessTask]] = []
        self._velocity_tasks: List[Optional[VelocityLimitTask]] = []
        self._position_limit_tasks: List[PositionLimit] = []
        self._correspondence_tasks: List[FrameVectorTask] = []
        self._scene_collision_tasks: List[SceneCollisionTask] = []

        for _ in stages:
            terms_for_stage: List[ResidualTask] = []
            frame_task = FrameTask(
                robot=self.robot,
                frame_index=frame_indices,
                T_world_target=placeholder_targets,
                position_weight=position_weights,
                orientation_weight=orientation_weights,
            )
            terms_for_stage.append(frame_task)
            self._frame_tasks.append(frame_task)

            if self._interaction_task is not None:
                terms_for_stage.append(self._interaction_task)

            position_limit = PositionLimit(
                robot=self.robot,
                weight=self.config.position_limit_weight,
                residual_mode=self.config.position_limit_mode,
                mask=mask_np,
            )
            position_limit.init_buffers(wp_device)
            terms_for_stage.append(position_limit)
            self._position_limit_tasks.append(position_limit)

            if np.any(self.config.rest_weight):
                rest_task = RestTask(
                    robot=self.robot,
                    rest_q=self.robot.spec.midrange_q,
                    weight=self.config.rest_weight,
                )
                rest_task.init_buffers(wp_device)
                terms_for_stage.append(rest_task)

            smoothness_task: Optional[SmoothnessTask] = None
            if np.any(self.config.smoothness_weight) or np.any(self.config.base_smoothness_weight):
                prev_base_placeholder = wp.clone(placeholder_base)
                prev_q_placeholder = wp.zeros(
                    (batch_size, self.robot.spec.num_actuated_joints), dtype=wp.float32, device=wp_device
                )
                prev_state_placeholder = self.robot.state(q=prev_q_placeholder, T_world_base=prev_base_placeholder)
                smoothness_task = SmoothnessTask(
                    robot=self.robot,
                    prev_var=prev_state_placeholder,
                    weight=self.config.smoothness_weight,
                    base_weight=self.config.base_smoothness_weight,
                )
                smoothness_task.init_buffers(wp_device)
                terms_for_stage.append(smoothness_task)
            self._smoothness_tasks.append(smoothness_task)

            velocity_task: Optional[VelocityLimitTask] = None
            if self.config.velocity_limit_weight > 0:
                prev_q_placeholder_v = wp.zeros(
                    (batch_size, self.robot.spec.num_actuated_joints), dtype=wp.float32, device=wp_device
                )
                prev_base_placeholder_v = wp.clone(placeholder_base)
                prev_state_placeholder_v = self.robot.state(
                    q=prev_q_placeholder_v, T_world_base=prev_base_placeholder_v
                )
                vel_limits_np = (
                    np.array(self.config.velocity_limit_override, dtype=np.float32)
                    if self.config.velocity_limit_override is not None
                    else None
                )
                velocity_task = VelocityLimitTask(
                    robot=self.robot,
                    dt=self.config.velocity_limit_dt,
                    prev_state_var=prev_state_placeholder_v,
                    velocity_limits=vel_limits_np,
                    weight=self.config.velocity_limit_weight,
                )
                velocity_task.init_buffers(wp_device)
                terms_for_stage.append(velocity_task)
            self._velocity_tasks.append(velocity_task)

            if corr_edges:
                corr_task = FrameVectorTask(
                    robot=self.robot,
                    origin_link_indices=corr_origin_idx,
                    task_link_indices=corr_task_idx,
                    targets=np.zeros((batch_size, len(corr_edges), 3), dtype=np.float32),
                    weight=np.sqrt(np.asarray(corr_weights, dtype=np.float32)),
                    scale=1.0,
                )
                corr_task.init_buffers(wp_device)
                terms_for_stage.append(corr_task)
                self._correspondence_tasks.append(corr_task)

            if self.config.scene_collision_weight > 0:
                scene_collision = SceneCollisionTask(
                    robot=self.robot,
                    scene=self._scene,
                    margin=self.config.scene_collision_margin,
                    weight=self.config.scene_collision_weight,
                )
                terms_for_stage.append(scene_collision)
                self._scene_collision_tasks.append(scene_collision)

            if self.config.self_collision_weight > 0:
                terms_for_stage.append(
                    SelfCollisionTask(
                        robot=self.robot,
                        weight=self.config.self_collision_weight,
                        margin=self.config.self_collision_margin,
                        max_active_pairs=self.config.self_collision_max_active_pairs,
                    )
                )

            stage_terms.append(terms_for_stage)

        solver_config = MultiSeedSolverConfig(
            stages=list(stages),
            cuda_graph_mode=self.config.cuda_graph_mode,
        )
        self._solver = MultiSeedSolver(
            terms=stage_terms,
            config=solver_config,
            device=wp_device,
            score_terms=None,
        )

        first_stage_seeds = stages[0].num_seeds
        expanded_q = wp.zeros(
            (batch_size * first_stage_seeds, self.robot.spec.num_actuated_joints),
            dtype=wp.float32,
            device=wp_device,
        )
        expanded_base = se3_identity(shape=(batch_size * first_stage_seeds,), device=wp_device)
        self._initial_state = self.robot.state(q=expanded_q, T_world_base=expanded_base)
        self._initial_var = VarValues(robot=self._initial_state)
        self._solver.setup(self._initial_var)

    def warmup(self, batch_size: int):
        """Fix the online batch size and build its reusable solver state."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self._batch_size is not None:
            if batch_size != self._batch_size:
                raise ValueError(f"batch size mismatch: expected {self._batch_size}, got {batch_size}")
            return

        if self.config.interaction is not None and batch_size != 1:
            raise ValueError("interaction retargeting supports batch_size=1")
        self._build_solver(batch_size)
        limits = self.robot.spec.actuated_joint_limits.astype(np.float32, copy=False)
        self._target_joint_indices_wp = wp.from_numpy(self._targets.joint_indices, dtype=wp.int32, device=self.device)
        self._base_scale_wp = wp.from_numpy(self._targets.scales, dtype=wp.float32, device=self.device)
        self._rot_off_wp = wp.from_numpy(self._targets.rotation_offsets, dtype=wp.vec4, device=self.device)
        self._pos_off_wp = wp.from_numpy(self._targets.position_offsets, dtype=wp.vec3, device=self.device)
        self._joint_limits_wp = wp.from_numpy(limits, dtype=wp.float32, device=self.device)
        self._midrange_q_wp = wp.from_numpy(
            self.robot.spec.midrange_q.astype(np.float32, copy=False), dtype=wp.float32, device=self.device
        )
        self._max_dq_wp = wp.from_numpy(
            (
                np.array(self.config.velocity_limit_override, dtype=np.float32)
                if self.config.velocity_limit_override is not None
                else self.robot.spec.actuated_joint_velocity_limits.astype(np.float32, copy=False)
            )
            * self.config.velocity_limit_dt
            * self.config.velocity_clamp_scale,
            dtype=wp.float32,
            device=self.device,
        )

        perturbations = np.zeros((batch_size, self._num_seeds0, self._num_dofs), dtype=np.float32)
        joint_range = limits[:, 1] - limits[:, 0]
        for b in range(batch_size):
            rng = np.random.default_rng(self._seed if b == 0 else (self._seed, b))
            perturbations[b, 1:] = (
                rng.uniform(-0.1, 0.1, (self._num_seeds0 - 1, self._num_dofs)).astype(np.float32) * joint_range
            )
        self._perts_wp = wp.from_numpy(perturbations, dtype=wp.float32, device=self.device)
        self._n_local = (self._num_seeds0 - 1) // 2

        self._default_human_heights_wp = wp.from_numpy(
            np.full(batch_size, self.config.human_height_assumption, dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        # zero_q may sit outside asymmetric joint limits; clamp so frame-0 seeds start legal
        self._zero_q_wp = wp.from_numpy(
            np.tile(np.clip(self.robot.spec.zero_q, limits[:, 0], limits[:, 1]).astype(np.float32), (batch_size, 1)),
            dtype=wp.float32,
            device=self.device,
        )
        reset_base_np = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (batch_size, 1))
        self._reset_base_wp = wp.from_numpy(reset_base_np, dtype=wp_vec7, device=self.device)
        self._prev_q_wp = wp.clone(self._zero_q_wp)
        self._prev_state_base = wp.clone(self._reset_base_wp)
        self._prev_state = self.robot.state(q=self._prev_q_wp, T_world_base=self._prev_state_base)
        self._output_wp = wp.zeros((batch_size, 7 + self._num_dofs), dtype=wp.float32, device=self.device)

        self._base_limit_weights_wp = []
        if self.config.limit_warmup_frames > 0:
            self._base_limit_weights_wp = [
                wp.clone(cast(wp.array, task.residual_weight)) for task in self._position_limit_tasks
            ]

        self._corr_origin_indices_wp = wp.from_numpy(self._corr_origin_indices_np, dtype=wp.int32, device=self.device)
        self._corr_task_indices_wp = wp.from_numpy(self._corr_task_indices_np, dtype=wp.int32, device=self.device)
        self._corr_origin_scales_wp = wp.from_numpy(self._corr_origin_scales_np, dtype=wp.float32, device=self.device)
        self._corr_task_scales_wp = wp.from_numpy(self._corr_task_scales_np, dtype=wp.float32, device=self.device)
        self._target_vectors_wp = wp.zeros(
            (batch_size, max(self._num_vector_targets, 1), 3), dtype=wp.float32, device=self.device
        )
        self._base_target_wp = wp.from_numpy(
            np.empty((batch_size, 7), dtype=np.float32), dtype=wp_vec7, device=self.device
        )
        self._seed_base_wp = self._prev_state_base if self._interaction_task is not None else self._base_target_wp

        self._batch_size = batch_size
        self._frame_count = 0
        self.reset()

    def solve(
        self,
        T_world_human: wp.array,
        human_heights: Optional[wp.array] = None,
        *,
        out: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget one batched human frame from Warp inputs.

        Lifecycle:
            1. Validate the shape fixed by `warmup(...)`.
            2. Update temporal terms from the previous robot state.
            3. Compute targets and seed states.
            4. Solve once, retain the state, and write the requested output buffer.

        Args:
            T_world_human: Human joint transforms with shape `(batch, joints)`.
            human_heights: Optional human height in meters for each batch row.
            out: Optional caller-owned output with shape `(batch, 7 + dofs)`.

        Returns:
            ``out`` when provided; otherwise a reusable helper-owned buffer overwritten by the next call.
        """
        # --- validate inputs ---
        if self._batch_size is None:
            raise RuntimeError("call warmup() before solve()")
        expected_shape = (self._batch_size, len(self.human_joint_names))
        if T_world_human.shape != expected_shape:
            raise ValueError(f"expected T_world_human shape {expected_shape}, got {T_world_human.shape}")
        heights = self._default_human_heights_wp if human_heights is None else human_heights
        if heights.shape != (self._batch_size,):
            raise ValueError(f"expected human_heights shape ({self._batch_size},), got {heights.shape}")

        # --- update temporal terms ---
        for task in self._smoothness_tasks:
            if task is not None:
                task.set_prev_state(self._prev_state)
        for task in self._velocity_tasks:
            if task is not None:
                task.set_prev_state(self._prev_state)
        if self.config.limit_warmup_frames > 0:
            limit_scale = float(min(self._frame_count / self.config.limit_warmup_frames, 1.0))
            for base_weight, task in zip(self._base_limit_weights_wp, self._position_limit_tasks):
                wp.launch(
                    compute_scaled_weights_kernel,
                    dim=self._num_dofs,
                    inputs=[base_weight, limit_scale, task.residual_weight],
                    device=self.device,
                )

        # --- compute targets ---
        first_frame_task = self._frame_tasks[0]
        wp.launch(
            compute_targets_kernel,
            dim=(self._batch_size, self._num_targets + self._num_vector_targets + 1),
            inputs=[
                T_world_human,
                heights,
                self._target_joint_indices_wp,
                self._base_scale_wp,
                self._rot_off_wp,
                self._pos_off_wp,
                self._corr_origin_indices_wp,
                self._corr_task_indices_wp,
                self._corr_origin_scales_wp,
                self._corr_task_scales_wp,
                first_frame_task.T_world_target,
                self._base_target_wp,
                self._target_vectors_wp,
                self._num_targets,
                self._num_vector_targets,
                self._targets.root_joint_index,
                self._targets.root_target_index,
                self._targets.root_scale,
                float(self.config.human_height_assumption),
                1,
                float(self.config.ground_height),
            ],
            device=self.device,
        )
        for task in self._frame_tasks[1:]:
            wp.copy(task.T_world_target, first_frame_task.T_world_target)
        for task in self._correspondence_tasks:
            wp.copy(cast(wp.array, task.targets_wp), self._target_vectors_wp)

        # --- compute seed states ---
        wp.launch(
            compute_seed_states_kernel,
            dim=(self._batch_size, self._num_seeds0, self._num_dofs + 1),
            inputs=[
                self._prev_q_wp,
                self._midrange_q_wp,
                self._joint_limits_wp,
                self._perts_wp,
                self._seed_base_wp,
                self._initial_state.q,
                self._initial_state.T_world_base,
                self._n_local,
            ],
            device=self.device,
        )
        if self._interaction_task is not None:
            # Interaction runs one seed, warm-started from the previous frame verbatim: overwrite the
            # kernel's limit-clamped seed so the solve starts exactly where the last one ended.
            wp.copy(self._initial_state.q, self._prev_q_wp)

        # --- solve and finalize ---
        self._initial_state.invalidate()
        best_var, _ = self._solver.solve(self._initial_var)
        best_state = cast(RobotState, best_var.get("robot"))
        output = self._output_wp if out is None else out

        wp.launch(
            set_solution_state_kernel,
            dim=(self._batch_size, 7 + self._num_dofs),
            inputs=[
                best_state.q,
                best_state.T_world_base,
                self._joint_limits_wp,
                self._max_dq_wp,
                self._prev_q_wp,
                self._prev_state_base,
                output,
                1 if (self.config.velocity_clamp_scale > 0.0 and self._frame_count > 0) else 0,
            ],
            device=self.device,
        )
        self._frame_count += 1
        return output

    def solve_numpy(
        self,
        T_world_human: np.ndarray,
        human_heights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert NumPy inputs, call ``solve(...)``, and return a NumPy copy."""
        frames = np.ascontiguousarray(T_world_human, dtype=np.float32)
        if frames.ndim != 3 or frames.shape[-1] != 7:
            raise ValueError("T_world_human must have shape [B, J, 7]")
        batch_size, num_joints, _ = frames.shape
        if num_joints != len(self.human_joint_names):
            raise ValueError(f"expected J={len(self.human_joint_names)} in human_joint_names order, got {num_joints}")
        heights_wp = None
        if human_heights is not None:
            heights = np.asarray(human_heights, dtype=np.float32)
            if heights.shape != (batch_size,) or np.any(heights <= 0):
                raise ValueError(f"human_heights must contain {batch_size} positive values")
            heights_wp = wp.from_numpy(heights, dtype=wp.float32, device=self.device)
        return self.solve(wp.from_numpy(frames, dtype=wp_vec7, device=self.device), heights_wp).numpy().copy()

    def reset(self, T_world_base: Optional[wp.array] = None):
        """Reset temporal state, optionally from an initial floating-base transform."""
        self._frame_count = 0
        if self._batch_size is None:
            return
        wp.copy(self._prev_q_wp, self._zero_q_wp)
        wp.copy(self._prev_state_base, self._reset_base_wp if T_world_base is None else T_world_base)
        self._output_wp.zero_()


__all__ = ["HumanoidRetargetingOnline"]
