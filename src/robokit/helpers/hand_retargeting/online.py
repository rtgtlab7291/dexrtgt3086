# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportInvalidTypeForm=false
"""Stateful online hand retargeting from named point targets."""

from dataclasses import replace
from typing import Dict, List, Optional, cast

import numpy as np
import warp as wp

from robokit.helpers.hand_retargeting._kernels import (
    compute_direction_targets_kernel,
    compute_vector_targets_kernel,
    set_online_solution_kernel,
)
from robokit.helpers.hand_retargeting.config import HandRetargetingOnlineConfig, HandSpec
from robokit.lie.se3 import se3_identity
from robokit.opt.multi_seed_solver import MultiSeedSolver
from robokit.opt.var_values import VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.task import ResidualTask
from robokit.utils.sampling_utils import Sampler
from robokit.utils.warp_utils import wp_device_type


class HandRetargetingOnline:
    """Retarget independent point streams with a reusable dense Warp solver.

    The point order is declared by `HandSpec.target_names`. Named position, vector, direction, and pinch correspondences are resolved once during construction; `solve` performs only device work after `warmup`.

    Args:
        robot: Loaded robot model. Self-collision skip pairs come from its spec
            (`Robot.load(self_collision_ignore_path=...)`).
        spec: Ordered semantic point topology.
        config: Online solver, objective, filtering, and sampling settings.
        device: Warp device. The first CUDA device is used when available.
    """

    def __init__(
        self,
        robot: Robot,
        spec: HandSpec,
        config: HandRetargetingOnlineConfig,
        device: Optional[wp_device_type] = None,
    ) -> None:
        """Resolve named correspondences without allocating batch-shaped state."""
        self.robot = robot
        self.spec = spec
        self.config = config
        self.device = wp.get_device(device if device is not None else ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self._num_dofs = self.robot.spec.num_actuated_joints
        self._batch_size: Optional[int] = None
        self._output_ema_alpha = config.output_ema_alpha if 0.0 < config.output_ema_alpha < 1.0 else 1.0

        # --- compile named topology ---
        target_index = {name: index for index, name in enumerate(spec.target_names)}
        link_index = {name: index for index, name in enumerate(self.robot.spec.link_names)}
        position_names = tuple(config.position_weights)
        self._position_target_indices_np = np.asarray([target_index[name] for name in position_names], dtype=np.int32)
        self._position_link_indices = [link_index[spec.target_link_names[name]] for name in position_names]
        self._position_weights = [float(config.position_weights[name]) for name in position_names]

        chain_by_target: Dict[str, int] = {}
        for chain_index, targets in enumerate(spec.target_chains):
            for target in targets:
                chain_by_target[target] = chain_index
        next_chain = len(spec.target_chains)
        for target in spec.target_names:
            if target not in chain_by_target:
                chain_by_target[target] = next_chain
                next_chain += 1
        self._num_target_chains = next_chain

        vector_pairs = tuple(config.vector_weights)
        direction_pairs = tuple(config.direction_weights)
        if not position_names and not vector_pairs and not direction_pairs:
            raise ValueError("At least one position, vector, or direction target is required.")

        contact_targets = set(spec.contact_target_names)
        self._vector_origin_targets_np = np.asarray([target_index[pair[2]] for pair in vector_pairs], dtype=np.int32)
        self._vector_task_targets_np = np.asarray([target_index[pair[3]] for pair in vector_pairs], dtype=np.int32)
        self._vector_origin_links = [link_index[pair[0]] for pair in vector_pairs]
        self._vector_task_links = [link_index[pair[1]] for pair in vector_pairs]
        self._vector_weights = list(config.vector_weights.values())
        self._vector_soft_gate_mask = [
            pair[2] in contact_targets and pair[3] in contact_targets for pair in vector_pairs
        ]
        self._vector_pinch_mask_np = np.asarray(
            [pair in config.pinch_correspondences for pair in vector_pairs], dtype=np.bool_
        )
        self._vector_origin_chains_np = np.asarray([chain_by_target[pair[2]] for pair in vector_pairs], dtype=np.int32)
        self._vector_task_chains_np = np.asarray([chain_by_target[pair[3]] for pair in vector_pairs], dtype=np.int32)

        self._direction_origin_targets_np = np.asarray(
            [target_index[pair[2]] for pair in direction_pairs], dtype=np.int32
        )
        self._direction_task_targets_np = np.asarray(
            [target_index[pair[3]] for pair in direction_pairs], dtype=np.int32
        )
        self._direction_origin_links = [link_index[pair[0]] for pair in direction_pairs]
        self._direction_task_links = [link_index[pair[1]] for pair in direction_pairs]
        self._direction_chains_np = np.asarray([chain_by_target[pair[3]] for pair in direction_pairs], dtype=np.int32)
        self._direction_weights_np = np.asarray(tuple(config.direction_weights.values()), dtype=np.float32) * float(
            np.sqrt(1.0 / max(len(direction_pairs), 1))
        )
        self._num_positions = len(position_names)
        self._num_vectors = len(vector_pairs)
        self._num_directions = len(direction_pairs)
        self._gate_direction_on_pinch = bool(
            config.gate_direction_on_pinch and config.pinch_correspondences and direction_pairs
        )

        joint_index = {name: index for index, name in enumerate(self.robot.spec.actuated_joint_names)}
        self._reset_q_np = self.robot.spec.zero_q.astype(np.float32, copy=True)
        for name, value in spec.init_q_by_name.items():
            self._reset_q_np[joint_index[name]] = value
        self._rest_q_np = self.robot.spec.zero_q.astype(np.float32, copy=True)
        for name, value in spec.rest_q_by_name.items():
            self._rest_q_np[joint_index[name]] = value

    def warmup(self, batch_size: int) -> None:
        """Build all batch-shaped terms, buffers, and solver state.

        Args:
            batch_size: Number of independent streams advanced by each solve.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self._batch_size == batch_size:
            return
        self._batch_size = None

        # --- build target terms ---
        terms: List[ResidualTask] = []
        self._target_points_wp = wp.zeros(
            (batch_size, len(self.spec.target_names), 3), dtype=wp.float32, device=self.device
        )
        if self._num_positions:
            task = FrameVectorTask(
                robot=self.robot,
                origin_link_indices=[-1] * self._num_positions,
                task_link_indices=self._position_link_indices,
                targets=np.zeros((batch_size, len(self.spec.target_names), 3), dtype=np.float32),
                weight=np.asarray(self._position_weights, dtype=np.float32) * np.sqrt(1.0 / self._num_positions),
                target_indices=self._position_target_indices_np,
                huber_delta=self.config.position_huber_delta,
            )
            task.targets_wp = self._target_points_wp
            task.init_buffers(self.device)
            terms.append(task)

        self._vector_task: Optional[FrameVectorTask] = None
        if self._num_vectors:
            self._vector_task = FrameVectorTask(
                robot=self.robot,
                origin_link_indices=self._vector_origin_links,
                task_link_indices=self._vector_task_links,
                targets=np.zeros((batch_size, self._num_vectors, 3), dtype=np.float32),
                weight=np.sqrt(np.asarray(self._vector_weights, dtype=np.float32)),
                scale=self.config.vector_robot_scale,
                huber_delta=self.config.vector_huber_delta,
                huber_on_norm=True,
                soft_gate_start_distance=self.config.vector_soft_gate_start_distance,
                soft_gate_full_distance=self.config.vector_soft_gate_full_distance,
                soft_gate_pair_mask=self._vector_soft_gate_mask,
            )
            self._vector_task.init_buffers(self.device)
            terms.append(self._vector_task)

        self._direction_task: Optional[FrameVectorTask] = None
        if self._num_directions:
            self._direction_task = FrameVectorTask(
                robot=self.robot,
                origin_link_indices=self._direction_origin_links,
                task_link_indices=self._direction_task_links,
                targets=np.zeros((batch_size, self._num_directions, 3), dtype=np.float32),
                weight=np.zeros((batch_size, self._num_directions), dtype=np.float32),
                direction_only=True,
            )
            self._direction_task.init_buffers(self.device)
            terms.append(self._direction_task)

        reset_q = np.clip(
            self._reset_q_np,
            self.robot.spec.actuated_joint_limits[:, 0],
            self.robot.spec.actuated_joint_limits[:, 1],
        )
        self._reset_q_wp = wp.from_numpy(np.tile(reset_q, (batch_size, 1)), dtype=wp.float32, device=self.device)
        self._reset_base_wp = se3_identity(shape=(batch_size,), device=self.device)
        self._previous_state = self.robot.state(
            q=wp.clone(self._reset_q_wp),
            T_world_base=wp.clone(self._reset_base_wp) if self.spec.floating_base else None,
        )

        self._smoothness_task: Optional[SmoothnessTask] = None
        if self.config.q_smoothness_weight > 0:
            self._smoothness_task = SmoothnessTask(
                robot=self.robot,
                prev_var=self._previous_state,
                weight=self.config.q_smoothness_weight,
                base_weight=self.config.base_smoothness_weight,
            )
            self._smoothness_task.init_buffers(self.device)
            terms.append(self._smoothness_task)

        if self.config.regularization_weight > 0:
            rest_task = RestTask(
                robot=self.robot,
                rest_q=self._rest_q_np,
                weight=self.config.regularization_weight,
            )
            rest_task.init_buffers(self.device)
            terms.append(rest_task)
        if self.config.limit_weight > 0:
            limit_task = PositionLimit(
                robot=self.robot,
                weight=self.config.limit_weight,
                residual_mode=self.config.limit_residual_mode,
            )
            limit_task.init_buffers(self.device)
            terms.append(limit_task)
        if self.config.self_collision_weight > 0:
            terms.append(
                SelfCollisionTask(
                    robot=self.robot,
                    weight=self.config.self_collision_weight,
                    margin=self.config.self_collision_margin,
                    max_active_pairs=self.config.self_collision_max_active_pairs,
                )
            )

        # --- build static target buffers ---
        self._vector_origin_targets_wp = wp.from_numpy(
            self._vector_origin_targets_np, dtype=wp.int32, device=self.device
        )
        self._vector_task_targets_wp = wp.from_numpy(self._vector_task_targets_np, dtype=wp.int32, device=self.device)
        self._vector_pinch_mask_wp = wp.from_numpy(self._vector_pinch_mask_np, dtype=wp.bool, device=self.device)
        self._vector_origin_chains_wp = wp.from_numpy(self._vector_origin_chains_np, dtype=wp.int32, device=self.device)
        self._vector_task_chains_wp = wp.from_numpy(self._vector_task_chains_np, dtype=wp.int32, device=self.device)
        self._direction_origin_targets_wp = wp.from_numpy(
            self._direction_origin_targets_np, dtype=wp.int32, device=self.device
        )
        self._direction_task_targets_wp = wp.from_numpy(
            self._direction_task_targets_np, dtype=wp.int32, device=self.device
        )
        self._direction_chains_wp = wp.from_numpy(self._direction_chains_np, dtype=wp.int32, device=self.device)
        self._direction_weights_wp = wp.from_numpy(self._direction_weights_np, dtype=wp.float32, device=self.device)
        self._pinch_latched_wp = wp.zeros((batch_size, self._num_vectors), dtype=wp.bool, device=self.device)
        self._active_chains_wp = wp.zeros((batch_size, self._num_target_chains), dtype=wp.int32, device=self.device)

        # --- build solver and output state ---
        max_seeds = self.config.solver.stages[0].num_seeds
        self._sampler = Sampler(
            max_seeds,
            self._num_dofs,
            self.robot.spec.actuated_joint_limits,
            self.config.base_sampling_translation_mask,
            self.config.seed,
            True,
            self.device,
        )
        self._sampler.warmup(batch_size)
        initial_q = wp.empty((batch_size * max_seeds, self._num_dofs), dtype=wp.float32, device=self.device)
        initial_base = se3_identity(shape=(batch_size * max_seeds,), device=self.device)
        self._initial_state = self.robot.state(
            q=initial_q,
            T_world_base=initial_base if self.spec.floating_base else None,
        )
        self._initial_var = VarValues(robot=self._initial_state)
        self._sampler.sample_q(
            self._reset_q_wp if self.spec.init_q_by_name else None,
            self.config.sampling_distance,
            self._initial_state.q,
        )
        if self.spec.floating_base:
            self._sampler.sample_base(None, self.config.base_sampling_distance, self._initial_state.T_world_base)
        self._solver = MultiSeedSolver(
            terms=[terms] * len(self.config.solver.stages),
            config=replace(self.config.solver, use_early_stopping=False),
            device=self.device,
            score_terms=None,
        )
        self._solver.setup(self._initial_var)

        self._joint_limits_wp = self.robot.spec.get_tensors(str(self.device)).actuated_joint_limits
        self._output_q_history_wp = wp.zeros((batch_size, self._num_dofs), dtype=wp.float32, device=self.device)
        self._output_wp = wp.zeros((batch_size, 7 + self._num_dofs), dtype=wp.float32, device=self.device)
        self._batch_size = batch_size
        self.reset()
        self.solve(wp.zeros((batch_size, len(self.spec.target_names), 3), dtype=wp.float32, device=self.device))
        wp.synchronize_device(self.device)
        self.reset()

    def solve(
        self,
        target_points: wp.array,
        init_state: Optional[RobotState] = None,
        out: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget one batched frame using only Warp device operations.

        Lifecycle:
            1. Validate input and output buffers against the warmed shape and device.
            2. Update point targets and seed from `init_state` or the previous solution.
            3. Solve, clamp and filter joints, retain state, and write packed output.

        Args:
            target_points: Point targets with shape `(batch, targets, 3)` in
                `spec.target_names` order.
            init_state: Optional state overriding the streaming warm start.
            out: Optional packed output buffer with shape `(batch, 7 + dofs)`.

        Returns:
            Packed base transform and joint positions. A helper-owned buffer is
            reused when `out` is omitted.
        """
        # --- validate inputs ---
        if self._batch_size is None:
            raise RuntimeError("call warmup(batch_size) before solve().")
        expected_points = (self._batch_size, len(self.spec.target_names), 3)
        if target_points.shape != expected_points or target_points.dtype != wp.float32:
            raise ValueError(f"target_points must be float32 with shape {expected_points}.")
        if target_points.device != self.device:
            raise ValueError(f"target_points must be on {self.device}, got {target_points.device}.")
        output = self._output_wp if out is None else out
        expected_output = (self._batch_size, 7 + self._num_dofs)
        if out is not None and (out.shape != expected_output or out.dtype != wp.float32 or out.device != self.device):
            raise ValueError(f"out must be float32 on {self.device} with shape {expected_output}.")
        expected_state = (self._batch_size, self._num_dofs), (self._batch_size,), self.device
        if init_state is not None:
            if (init_state.q.shape, init_state.T_world_base.shape, init_state.device) != expected_state:
                raise ValueError("init_state must match the warmed batch, robot DOFs, and device.")

        # --- update targets ---
        wp.copy(self._target_points_wp, target_points)
        if self._gate_direction_on_pinch:
            self._active_chains_wp.zero_()
        if self._vector_task is not None:
            wp.launch(
                compute_vector_targets_kernel,
                dim=(self._batch_size, self._num_vectors),
                inputs=[
                    self._target_points_wp,
                    self._vector_origin_targets_wp,
                    self._vector_task_targets_wp,
                    self._vector_pinch_mask_wp,
                    self._vector_origin_chains_wp,
                    self._vector_task_chains_wp,
                    self.config.vector_target_scale,
                    self.config.pinch_threshold or 0.0,
                    self.config.pinch_release_threshold or 0.0,
                    self.config.pinch_target_norm,
                    int(self._gate_direction_on_pinch),
                    self.config.vector_target_ema_alpha,
                    1 if self._history_valid else 0,
                    self._pinch_latched_wp,
                    self._active_chains_wp,
                    self._vector_task.targets_wp,
                ],
                device=self.device,
            )
        if self._direction_task is not None:
            wp.launch(
                compute_direction_targets_kernel,
                dim=(self._batch_size, self._num_directions),
                inputs=[
                    self._target_points_wp,
                    self._direction_origin_targets_wp,
                    self._direction_task_targets_wp,
                    self._direction_chains_wp,
                    self._direction_weights_wp,
                    self._active_chains_wp,
                    self.config.direction_target_ema_alpha,
                    1 if self._history_valid else 0,
                    int(self._gate_direction_on_pinch),
                    self._direction_task.targets_wp,
                    self._direction_task.weights_wp,
                ],
                device=self.device,
            )

        # --- seed and solve ---
        seed_state = init_state
        if seed_state is None and (self._history_valid or self.spec.init_q_by_name):
            seed_state = self._previous_state
        self._sampler.sample_q(
            seed_state.q if seed_state is not None else None,
            self.config.sampling_distance,
            self._initial_state.q,
        )
        if self.spec.floating_base:
            seed_base = seed_state.T_world_base if seed_state is not None and seed_state.has_floating_base else None
            self._sampler.sample_base(seed_base, self.config.base_sampling_distance, self._initial_state.T_world_base)
        if self._smoothness_task is not None and seed_state is not None:
            self._smoothness_task.set_prev_state(seed_state)

        self._initial_state.invalidate()
        best_var, _ = self._solver.solve(self._initial_var)
        best_state = cast(RobotState, best_var.get("robot"))
        wp.launch(
            set_online_solution_kernel,
            dim=(self._batch_size, 7 + self._num_dofs),
            inputs=[
                best_state.q,
                best_state.T_world_base,
                self._joint_limits_wp,
                self._output_ema_alpha,
                1 if self._history_valid else 0,
                self._previous_state.q,
                self._previous_state.T_world_base,
                self._output_q_history_wp,
                output,
            ],
            device=self.device,
        )
        self._history_valid = True
        return output

    def solve_numpy(
        self,
        target_points: np.ndarray,
        init_state: Optional[RobotState] = None,
    ) -> np.ndarray:
        """NumPy wrapper for `solve()`."""
        points = np.asarray(target_points, dtype=np.float32)
        if points.ndim == 2:
            points = points[None]
        points_wp = wp.from_numpy(np.ascontiguousarray(points), dtype=wp.float32, device=self.device)
        return self.solve(points_wp, init_state=init_state).numpy().copy()

    def reset(self) -> None:
        """Clear streaming state and filters while retaining smoothness continuity."""
        self._history_valid = False
        if self._batch_size is None:
            return
        wp.copy(self._previous_state.q, self._reset_q_wp)
        wp.copy(self._previous_state.T_world_base, self._reset_base_wp)
        self._target_points_wp.zero_()
        if self._vector_task is not None:
            self._vector_task.targets_wp.zero_()
        if self._direction_task is not None:
            self._direction_task.targets_wp.zero_()
            self._direction_task.weights_wp.zero_()
        self._pinch_latched_wp.zero_()
        self._output_q_history_wp.zero_()
        self._output_wp.zero_()
        # The smoothness reference stays at the last solved frame for legacy reset continuity.

    def compute_costs(self, state: RobotState) -> Dict[str, wp.array]:
        """Evaluate the current targets for a caller-provided robot state.

        Args:
            state: Robot state matching the warmed batch and device.

        Returns:
            Per-term costs keyed by task cost name.
        """
        if self._batch_size is None:
            raise RuntimeError("call warmup(batch_size) before compute_costs().")
        expected_state = (self._batch_size, self._num_dofs), (self._batch_size,), self.device
        if (state.q.shape, state.T_world_base.shape, state.device) != expected_state:
            raise ValueError("state must match the warmed batch, robot DOFs, and device.")
        return self._solver.compute_costs(VarValues(robot=state))


__all__ = ["HandRetargetingOnline"]
