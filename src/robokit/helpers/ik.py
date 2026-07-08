from dataclasses import dataclass, field
from typing import Dict, List, Literal, NamedTuple, Optional, Sequence, Union, cast

import numpy as np
import torch
import warp as wp

from robokit.lie.warp_se3 import WarpSE3
from robokit.opt.sparse_warp_optimizer import SparseWarpOptimizer
from robokit.opt.warp_optimizer import WarpLMOptimizer, aggregate_residuals_to_costs
from robokit.opt.warp_solver import WarpSolver, WarpSolverConfig, WarpStageConfig
from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.terms.terms import WarpTask
from robokit.terms.warp.base_damping_task import WarpBaseDampingTask
from robokit.terms.warp.base_step_limit import WarpBaseStepLimit
from robokit.terms.warp.collision_task import WarpCollisionTask
from robokit.terms.warp.frame_task import WarpFrameTask
from robokit.terms.warp.position_limit import WarpPositionLimit
from robokit.terms.warp.position_score import WarpCompositeScoreTask
from robokit.terms.warp.position_task import WarpPositionTask
from robokit.terms.warp.rest_task import WarpRestTask
from robokit.terms.warp.rotation_task import WarpRotationTask
from robokit.terms.warp.smoothness_task import WarpSmoothnessTask
from robokit.terms.warp.velocity_limit_task import WarpVelocityLimitTask
from robokit.utils.warp_utils import repeat, tile, wp_device_type, wp_vec7


@wp.kernel
def _select_better_score_kernel(
    prev_q: wp.array2d(dtype=wp.float32),
    prev_base: wp.array1d(dtype=wp_vec7),
    result_q: wp.array2d(dtype=wp.float32),
    result_base: wp.array1d(dtype=wp_vec7),
    prev_score: wp.array1d(dtype=wp.float32),
    result_score: wp.array1d(dtype=wp.float32),
    num_joints: int,
):
    bid = wp.tid()
    if prev_score[bid] <= result_score[bid]:
        for j in range(num_joints):
            result_q[bid, j] = prev_q[bid, j]
        result_base[bid] = prev_base[bid]


class IKResultTorch(NamedTuple):
    q: torch.Tensor
    T_world_base: Optional[torch.Tensor] = None
    position_cost: Optional[torch.Tensor] = None
    orientation_cost: Optional[torch.Tensor] = None


def _default_stage_configs() -> List[WarpStageConfig]:
    return [
        WarpStageConfig(num_seeds=64, iters=6, lm_lambda=10.0),
        WarpStageConfig(num_seeds=4, iters=10, lm_lambda=1.0),
        WarpStageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
    ]


@dataclass
class IKHelperConfig:
    position_weight: Union[float, Sequence[float]] = 10.0
    orientation_weight: Union[float, Sequence[float]] = 5.0
    position_limit_weight: float = 50.0
    stages: Sequence[WarpStageConfig] = field(default_factory=_default_stage_configs)
    use_cuda_graph: bool = True
    seed: Optional[int] = None

    rest_weight: float = 0.0
    smoothness_weight: float = 0.0
    velocity_limit_weight: float = 0.0
    dt: float = 0.1
    # Mobile base parameters
    enable_T_world_base: bool = False
    base_damping_weight: float = 0.0
    base_step_limit_indices: Optional[Sequence[int]] = None
    base_step_limit_weight: float = 100.0
    base_weight_rest: float = 0.0
    base_weight_smoothness: float = 0.0
    # Sampling range as fraction of joint limits when init_state is provided
    init_sample_range: float = 0.2
    base_init_sample_range: float = 0.1
    score_position_weight: float = 0.0
    score_orientation_weight: float = 0.0
    score_smoothness_weight: float = 0.0
    acceptance_mode: Literal["strict_monotonic", "rho_only"] = "strict_monotonic"
    collision_weight: float = 0.0
    collision_margin: float = 0.0
    scene_meshes: Optional[List[wp.Mesh]] = None

    def __post_init__(self):
        if len(self.stages) == 0:
            raise ValueError("IKHelperConfig.stages must not be empty.")
        if self.stages[-1].num_seeds != 1:
            raise ValueError("IKHelperConfig.stages must end with num_seeds=1.")


def _compute_roberts_root(dim: int) -> float:
    exp = dim
    root = 1.0
    for _ in range(10000):
        f = root ** (exp + 1) - root - 1
        f_prime = (exp + 1) * root**exp - 1
        if abs(f_prime) < 1e-10:
            break
        root = root - f / f_prime
        if abs(f) < 1e-10:
            break
    return root


def _roberts_sequence_numpy(num_points: int, dim: int, root: float, offset: int = 0) -> np.ndarray:
    basis = 1 - (1 / root ** (1 + np.arange(dim)))
    n = np.arange(num_points) + offset
    x = n[:, None] * basis[None, :]
    x, _ = np.modf(x)
    return x


# Multi-seed layout: instance-major order (output_idx = instance_idx * num_seeds + seed_idx)
@wp.kernel
def _sample_q_around_init_kernel(
    init_q: wp.array2d(dtype=wp.float32),
    offsets: wp.array2d(dtype=wp.float32),
    joint_limits: wp.array2d(dtype=wp.float32),
    sample_range: wp.float32,
    num_seeds: wp.int32,
    out_q: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    instance_idx = tid // int(num_seeds)
    seed_idx = tid % int(num_seeds)
    num_joints = init_q.shape[1]

    for j in range(num_joints):
        joint_lower = joint_limits[j, 0]
        joint_upper = joint_limits[j, 1]
        joint_range = joint_upper - joint_lower
        offset = offsets[seed_idx, j] * sample_range * joint_range
        q_val = init_q[instance_idx, j] + offset
        out_q[tid, j] = wp.clamp(q_val, joint_lower, joint_upper)


# Multi-seed layout: instance-major order (output_idx = instance_idx * num_seeds + seed_idx)
@wp.kernel
def _sample_base_around_init_kernel(
    init_base: wp.array(dtype=wp_vec7),
    offsets: wp.array2d(dtype=wp.float32),
    sample_range: wp.float32,
    num_seeds: wp.int32,
    out_base: wp.array(dtype=wp_vec7),
):
    """Sample base poses around init_base with small xy perturbations.

    Only samples x and y (planar motion). z and orientation are preserved
    from init_base to keep the robot on the ground plane.
    """
    tid = wp.tid()
    instance_idx = tid // int(num_seeds)
    seed_idx = tid % int(num_seeds)

    base = init_base[instance_idx]
    out_base[tid] = wp_vec7(
        base[0] + offsets[seed_idx, 0] * sample_range,
        base[1] + offsets[seed_idx, 1] * sample_range,
        base[2],  # Keep z unchanged (ground plane)
        base[3],
        base[4],
        base[5],
        base[6],
    )


class IKHelper:
    def __init__(
        self,
        robot: WarpRobot,
        target_frames: Union[str, int, Sequence[Union[str, int]]],
        placeholder_targets: Union[WarpSE3, Sequence[WarpSE3]],
        config: Optional[IKHelperConfig] = None,
    ):
        # TODO reorganize this init func to group related parts together (but no small helper funcs)
        self.robot = robot
        self.config = config or IKHelperConfig()
        self.stage_configs = list(self.config.stages)
        # TODO: move this a property of config
        use_collision = self.config.collision_weight > 0 and bool(self.config.scene_meshes)

        # Mobile base detection and setup
        # TODO: move this a property of config
        self._has_mobile_base = self.config.enable_T_world_base

        # Normalize target_frames to a list
        if isinstance(target_frames, (str, int)):
            target_frames_list: List[Union[str, int]] = [target_frames]
        else:
            target_frames_list = list(target_frames)

        # Normalize placeholder_targets to a list
        if isinstance(placeholder_targets, WarpSE3):
            placeholder_targets_list: List[WarpSE3] = [placeholder_targets]
        else:
            placeholder_targets_list = list(placeholder_targets)

        if len(target_frames_list) != len(placeholder_targets_list):
            raise ValueError(
                f"Number of target frames ({len(target_frames_list)}) must match "
                f"number of placeholder targets ({len(placeholder_targets_list)})"
            )

        self.num_frames = len(target_frames_list)

        # Multi-frame IK (e.g. bimanual) needs extra iterations in the final stage
        # because the last stage with num_seeds=1 and iters=0 just selects the best
        # solution without refinement, which is insufficient for multi-frame problems.
        if self.num_frames > 1 and self.stage_configs[-1].iters == 0:
            fallback_iters = self.stage_configs[-2].iters if len(self.stage_configs) > 1 else 10
            self.stage_configs[-1] = WarpStageConfig(
                num_seeds=self.stage_configs[-1].num_seeds,
                iters=max(1, fallback_iters),
                lm_lambda=self.stage_configs[-1].lm_lambda,
            )

        self.target_link_indices: List[int] = []
        for frame in target_frames_list:
            if isinstance(frame, str):
                self.target_link_indices.append(self.robot.link_names.index(frame))
            else:
                self.target_link_indices.append(frame)

        wp_device = placeholder_targets_list[0].xyz_wxyz.device
        self.device = str(wp_device)

        # Create placeholder base for mobile robots
        if self._has_mobile_base:
            placeholder_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
            self._placeholder_base = WarpSE3(wp.from_numpy(placeholder_base_np, dtype=wp_vec7, device=wp_device))
        else:
            self._placeholder_base = None

        self.joint_limits_torch = self.robot.get_spec_tensors_torch(device=self.device).actuated_joint_limits
        self.num_joints = self.joint_limits_torch.shape[0]
        self.roberts_root = _compute_roberts_root(self.num_joints)
        self._base_seeds: Optional[wp.array] = None
        self._batch_size = placeholder_targets_list[0].batch_size

        # Apply seed for deterministic behavior
        self._roberts_offset = 0
        if self.config.seed is not None:
            self._roberts_offset = self.config.seed % 10000

        wp_device = wp.get_device(self.device)
        base_seeds = self._build_base_seeds(self.stage_configs[0].num_seeds)
        # tile: expand seed table for multiple instances [s0, s1] -> [s0, s1, s0, s1, ...]
        self._base_q_expanded = tile(base_seeds, self._batch_size)

        max_seeds = self.stage_configs[0].num_seeds
        # Seed 0 gets zero offset (preserves init_state exactly for warm-starting continuity).
        # Seeds 1..max_seeds-1 get Roberts quasi-random offsets centered at 0.
        roberts_samples_q = _roberts_sequence_numpy(
            max_seeds - 1, self.num_joints, self.roberts_root, offset=self._roberts_offset
        )
        q_offsets = np.vstack(
            [
                np.zeros((1, self.num_joints), dtype=np.float32),
                (roberts_samples_q - 0.5).astype(np.float32),
            ]
        )
        self._roberts_offsets_q = wp.from_numpy(q_offsets, dtype=wp.float32, device=wp_device)

        roberts_samples_base = _roberts_sequence_numpy(
            max_seeds - 1, 3, _compute_roberts_root(3), offset=self._roberts_offset
        )
        base_offsets = np.vstack(
            [
                np.zeros((1, 3), dtype=np.float32),
                (roberts_samples_base - 0.5).astype(np.float32),
            ]
        )
        self._roberts_offsets_base = wp.from_numpy(base_offsets, dtype=wp.float32, device=wp_device)

        if self._has_mobile_base:
            assert self._placeholder_base is not None
            first_stage_batch = self._batch_size * self.stage_configs[0].num_seeds
            # repeat: this is just for memory allocation, we do not care about the order
            self._expanded_base_buffer = repeat(self._placeholder_base.xyz_wxyz, first_stage_batch)
        else:
            self._expanded_base_buffer = None

        # Create base placeholder state (solver auto-creates per-stage vars via gather)
        placeholder_q = wp.empty(
            (self._batch_size, self.num_joints),
            dtype=wp.float32,  # type: ignore[reportArgumentType]
            device=wp_device,
        )
        if self._has_mobile_base:
            assert self._placeholder_base is not None
            base_se3 = WarpSE3(repeat(self._placeholder_base.xyz_wxyz, self._batch_size))
            placeholder_state = self.robot.state(q=placeholder_q, T_world_base=base_se3)
        else:
            placeholder_state = self.robot.state(q=placeholder_q)

        stage_position_tasks: List[WarpPositionTask] = []
        stage_rotation_tasks: List[WarpRotationTask] = []
        stage_position_limits: List[WarpPositionLimit] = []
        stage_rest_tasks: List[Optional[WarpRestTask]] = []
        stage_smoothness_tasks: List[Optional[WarpSmoothnessTask]] = []
        stage_velocity_tasks: List[Optional[WarpVelocityLimitTask]] = []
        terms: List[List[WarpTask]] = []
        score_terms: List[Optional[WarpTask]] = []
        has_score = (
            self.config.score_position_weight > 0
            or self.config.score_orientation_weight > 0
            or self.config.score_smoothness_weight > 0
        )

        for stage_config in self.stage_configs:
            total_batch = self._batch_size * stage_config.num_seeds

            # Create position and rotation tasks (split from frame task)
            stage_terms: List[WarpTask] = []
            if stage_config.num_seeds > 1:
                stage_placeholders = [
                    WarpSE3(repeat(t.xyz_wxyz, stage_config.num_seeds)) for t in placeholder_targets_list
                ]
            else:
                stage_placeholders = placeholder_targets_list

            position_task = WarpPositionTask(
                self.robot,
                self.target_link_indices,
                stage_placeholders,
                weight=self.config.position_weight,
            )
            stage_terms.append(position_task)

            rotation_task = WarpRotationTask(
                self.robot,
                self.target_link_indices,
                stage_placeholders,
                weight=self.config.orientation_weight,
            )
            stage_terms.append(rotation_task)

            position_limit = WarpPositionLimit(
                self.robot, weight=self.config.position_limit_weight, batch_size=total_batch
            )
            stage_terms.append(position_limit)

            # Mobile base tasks
            if self._has_mobile_base:
                if self.config.base_step_limit_indices is not None:
                    base_step_limit = WarpBaseStepLimit(
                        lock_indices=self.config.base_step_limit_indices,
                        T_world_base_ref=self._placeholder_base,
                        weight=self.config.base_step_limit_weight,
                        batch_size=total_batch,
                    )
                    stage_terms.append(base_step_limit)

                if self.config.base_damping_weight > 0:
                    base_damping = WarpBaseDampingTask(
                        robot=self.robot,
                        T_world_base_ref=self._placeholder_base,
                        weight=self.config.base_damping_weight,
                        batch_size=total_batch,
                    )
                    stage_terms.append(base_damping)

            rest_task: Optional[WarpRestTask] = None
            if self.config.rest_weight > 0 or (self._has_mobile_base and self.config.base_weight_rest > 0):
                rest_task = WarpRestTask(
                    robot=self.robot,
                    rest_q=self.robot.spec.zero_q,
                    T_world_base_rest=self._placeholder_base if self._has_mobile_base else None,
                    weight=self.config.rest_weight,
                    base_weight=self.config.base_weight_rest if self._has_mobile_base else 0.0,
                    batch_size=total_batch,
                )
                stage_terms.append(rest_task)

            # Create dummy prev_state for smoothness/velocity tasks if needed
            smoothness_task: Optional[WarpSmoothnessTask] = None
            if self.config.smoothness_weight > 0 or (self._has_mobile_base and self.config.base_weight_smoothness > 0):
                prev_state_placeholder = self._create_placeholder_prev_state(wp_device)
                smoothness_task = WarpSmoothnessTask(
                    robot=self.robot,
                    prev_var=prev_state_placeholder,
                    weight=self.config.smoothness_weight,
                    base_weight=self.config.base_weight_smoothness if self._has_mobile_base else None,
                    batch_size=total_batch,
                    num_seeds=stage_config.num_seeds,
                )
                stage_terms.append(smoothness_task)

            velocity_task: Optional[WarpVelocityLimitTask] = None
            if self.config.velocity_limit_weight > 0:
                prev_state_placeholder = self._create_placeholder_prev_state(wp_device)
                velocity_task = WarpVelocityLimitTask(
                    robot=self.robot,
                    dt=self.config.dt,
                    prev_state_var=prev_state_placeholder,
                    weight=self.config.velocity_limit_weight,
                    batch_size=total_batch,
                    num_seeds=stage_config.num_seeds,
                )
                stage_terms.append(velocity_task)

            # Create score task for beam selection and prev-state fallback
            score_task_for_stage: Optional[WarpTask] = None
            if has_score:
                score_parts: List[WarpTask] = []
                if self.config.score_position_weight > 0 or self.config.score_orientation_weight > 0:
                    score_parts.append(
                        WarpFrameTask(
                            self.robot,
                            self.target_link_indices,
                            placeholder_targets_list,
                            position_weight=self.config.score_position_weight,
                            orientation_weight=self.config.score_orientation_weight,
                            num_seeds=stage_config.num_seeds,
                        )
                    )
                if self.config.score_smoothness_weight > 0:
                    score_prev_state_placeholder = self._create_placeholder_prev_state(wp_device)
                    score_parts.append(
                        WarpSmoothnessTask(
                            robot=self.robot,
                            prev_var=score_prev_state_placeholder,
                            weight=self.config.score_smoothness_weight,
                            base_weight=self.config.score_smoothness_weight if self._has_mobile_base else None,
                            batch_size=total_batch,
                            num_seeds=stage_config.num_seeds,
                        )
                    )
                score_task_for_stage = score_parts[0] if len(score_parts) == 1 else WarpCompositeScoreTask(score_parts)

            if use_collision:
                assert self.config.scene_meshes is not None
                collision_task = WarpCollisionTask(
                    robot=self.robot,
                    scene_meshes=self.config.scene_meshes,
                    weight=self.config.collision_weight,
                    margin=self.config.collision_margin,
                    batch_size=total_batch,
                )
                stage_terms.append(collision_task)

            stage_position_tasks.append(position_task)
            stage_rotation_tasks.append(rotation_task)
            stage_position_limits.append(position_limit)
            stage_rest_tasks.append(rest_task)
            stage_smoothness_tasks.append(smoothness_task)
            stage_velocity_tasks.append(velocity_task)
            terms.append(stage_terms)
            score_terms.append(score_task_for_stage)

        final_score_terms = None if use_collision else score_terms
        solver_config = WarpSolverConfig(
            stages=self.stage_configs,
            use_cuda_graph=self.config.use_cuda_graph,
            acceptance_mode=self.config.acceptance_mode,
        )
        self._solver = WarpSolver(
            config=solver_config,
            placeholder_var=placeholder_state,
            terms=terms,
            score_terms=final_score_terms,  # type: ignore
        )
        self._validate_stage_optimizer_layouts()
        self._stage_position_tasks = stage_position_tasks
        self._stage_rotation_tasks = stage_rotation_tasks
        self._stage_position_limits = stage_position_limits
        self._stage_rest_tasks = stage_rest_tasks
        self._stage_smoothness_tasks = stage_smoothness_tasks
        self._stage_velocity_tasks = stage_velocity_tasks
        self._stage_score_tasks = score_terms
        self._placeholder_targets = placeholder_targets_list
        self._prev_score: Optional[wp.array] = None
        self._prev_q_save: Optional[wp.array] = None
        self._prev_base_save: Optional[wp.array] = None
        if self._stage_score_tasks[-1] is not None:
            self._prev_score = wp.empty((self._batch_size,), dtype=wp.float32, device=wp_device)  # type: ignore[reportArgumentType]
            self._prev_q_save = wp.empty(
                (self._batch_size, self.robot.num_actuated_joints),
                dtype=wp.float32,  # type: ignore[reportArgumentType]
                device=wp_device,
            )
            self._prev_base_save = wp.empty((self._batch_size,), dtype=wp_vec7, device=wp_device)  # type: ignore[reportArgumentType]

    def _validate_stage_optimizer_layouts(self) -> None:
        for stage_index, optimizer in enumerate(self._solver._optimizers):
            if isinstance(optimizer, WarpLMOptimizer):
                stage_terms = optimizer.terms
            elif isinstance(optimizer, SparseWarpOptimizer):
                stage_terms = optimizer.tasks
            else:
                raise TypeError(f"Unsupported optimizer type in IK helper: {type(optimizer).__name__}")

            expected_total = int(sum(term.residual_dim for term in stage_terms))
            actual_total = int(optimizer.total_residual_dim)
            if expected_total == actual_total:
                continue
            term_desc = ", ".join(f"{type(term).__name__}:{term.residual_dim}" for term in stage_terms)
            raise ValueError(
                "IK stage residual layout mismatch: "
                f"stage={stage_index}, expected_total={expected_total}, actual_total={actual_total}, "
                f"terms=[{term_desc}]"
            )

    @property
    def rest_q(self) -> np.ndarray:
        """Rest joint configuration used by the rest task."""
        rest_task = self._stage_rest_tasks[-1]
        assert rest_task is not None, "rest_q requires rest_weight > 0"
        return rest_task.rest_q_np

    def _build_base_seeds(self, num_seeds: int) -> wp.array:
        if self._base_seeds is not None:
            return self._base_seeds
        joint_limits = self.robot.spec.actuated_joint_limits
        roberts_samples = _roberts_sequence_numpy(
            num_seeds, joint_limits.shape[0], self.roberts_root, offset=self._roberts_offset
        )
        seed_qs_np = (joint_limits[:, 0] + roberts_samples * (joint_limits[:, 1] - joint_limits[:, 0])).astype(
            np.float32
        )
        self._base_seeds = wp.from_numpy(seed_qs_np, dtype=wp.float32, device=self.device)
        return self._base_seeds

    def _sample_q_around_init(self, init_q: wp.array, num_seeds: int, out_q: wp.array, sample_range: float) -> None:
        # Layout: instance-major order (output_idx = instance_idx * num_seeds + seed_idx)
        total_batch = self._batch_size * num_seeds
        spec_tensors = self.robot.get_spec_tensors(self.device)
        wp.launch(
            kernel=_sample_q_around_init_kernel,
            dim=total_batch,
            inputs=[
                init_q,
                self._roberts_offsets_q,
                spec_tensors.actuated_joint_limits,
                sample_range,
                num_seeds,
                out_q,
            ],
            device=self.device,
        )

    def _create_placeholder_prev_state(self, wp_device: wp_device_type) -> WarpRobotState:
        if self._has_mobile_base:
            assert self._placeholder_base is not None
            prev_state_batch = WarpSE3(repeat(self._placeholder_base.xyz_wxyz, self._batch_size))
            return self.robot.state(
                q=wp.zeros((self._batch_size, self.num_joints), dtype=wp.float32, device=wp_device),
                T_world_base=prev_state_batch,
            )
        else:
            return self.robot.state(q=wp.zeros((self._batch_size, self.num_joints), dtype=wp.float32, device=wp_device))

    def _sample_base_around_init(
        self, init_base: wp.array, num_seeds: int, out_base: wp.array, sample_range: float
    ) -> None:
        # Layout: instance-major order (output_idx = instance_idx * num_seeds + seed_idx)
        total_batch = self._batch_size * num_seeds
        wp.launch(
            kernel=_sample_base_around_init_kernel,
            dim=total_batch,
            inputs=[
                init_base,
                self._roberts_offsets_base,
                sample_range,
                num_seeds,
                out_base,
            ],
            device=self.device,
        )

    def solve(
        self,
        targets: Union[WarpSE3, Sequence[WarpSE3]],
        prev_state: Optional[WarpRobotState] = None,
        init_state: Optional[WarpRobotState] = None,
        rest_state: Optional[WarpRobotState] = None,
        out_state: Optional[WarpRobotState] = None,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[float] = None,
    ) -> WarpRobotState:
        if isinstance(targets, WarpSE3):
            targets_list: List[WarpSE3] = [targets]
        else:
            targets_list = list(targets)

        if len(targets_list) != self.num_frames:
            raise ValueError(f"Number of targets ({len(targets_list)}) must match number of frames ({self.num_frames})")

        batch_size = targets_list[0].batch_size
        if batch_size != self._batch_size:
            raise ValueError(f"Batch size mismatch: expected {self._batch_size}, got {batch_size}")

        # Resolve sample ranges (use provided values or fall back to config)
        q_sample_range = init_sample_range if init_sample_range is not None else self.config.init_sample_range
        base_sample_range = (
            base_init_sample_range if base_init_sample_range is not None else self.config.base_init_sample_range
        )

        # Set initial guess
        initial_var = self._solver.initial_var
        if init_state is not None:
            self._sample_q_around_init(init_state.q, self.stage_configs[0].num_seeds, initial_var.q, q_sample_range)
            if self._has_mobile_base and init_state.T_world_base is not None:
                self._sample_base_around_init(
                    init_state.T_world_base.xyz_wxyz,
                    self.stage_configs[0].num_seeds,
                    initial_var.T_world_base.xyz_wxyz,
                    base_sample_range,
                )
            elif self._has_mobile_base:
                assert self._expanded_base_buffer is not None
                wp.copy(initial_var.T_world_base.xyz_wxyz, self._expanded_base_buffer)
        else:
            wp.copy(initial_var.q, self._base_q_expanded)
            if self._has_mobile_base:
                assert self._expanded_base_buffer is not None
                wp.copy(initial_var.T_world_base.xyz_wxyz, self._expanded_base_buffer)

        # Update targets for all position/rotation tasks (expand for multi-seed stages)
        for stage_idx, stage_config in enumerate(self.stage_configs):
            if stage_config.num_seeds > 1:
                expanded = [WarpSE3(repeat(t.xyz_wxyz, stage_config.num_seeds)) for t in targets_list]
            else:
                expanded = targets_list
            self._stage_position_tasks[stage_idx].set_target(expanded)
            self._stage_rotation_tasks[stage_idx].set_target(expanded)

        # Update targets for all score tasks
        for score_task in self._stage_score_tasks:
            if score_task is not None:
                if isinstance(score_task, WarpCompositeScoreTask):
                    for inner_task in score_task.tasks:
                        if isinstance(inner_task, WarpFrameTask):
                            inner_task.set_target(targets_list)
                elif isinstance(score_task, WarpFrameTask):
                    score_task.set_target(targets_list)

        # Update smoothness/velocity tasks if prev_state provided
        if prev_state is not None:
            if prev_state.batch_size != self._batch_size:
                raise ValueError(
                    f"prev_state batch size mismatch: expected {self._batch_size}, got {prev_state.batch_size}"
                )
            for smoothness_task in self._stage_smoothness_tasks:
                if smoothness_task is not None:
                    smoothness_task.set_prev_state(prev_state)
            for velocity_task in self._stage_velocity_tasks:
                if velocity_task is not None:
                    velocity_task.set_prev_state(prev_state)
            # Update score smoothness tasks
            for score_task in self._stage_score_tasks:
                if isinstance(score_task, WarpCompositeScoreTask):
                    for inner_task in score_task.tasks:
                        if isinstance(inner_task, WarpSmoothnessTask):
                            inner_task.set_prev_state(prev_state)
                elif isinstance(score_task, WarpSmoothnessTask):
                    score_task.set_prev_state(prev_state)

        # Update rest tasks if rest_state provided
        if rest_state is not None:
            for rest_task in self._stage_rest_tasks:
                if rest_task is not None:
                    rest_task.set_rest_state(rest_state.q, rest_state.T_world_base if self._has_mobile_base else None)

        # Compute prev_score and save prev_state before solve (prev_state may alias _stage_vars[-1])
        last_score_task = self._stage_score_tasks[-1]
        if prev_state is not None and self._prev_score is not None and last_score_task is not None:
            rbuf = self._solver._score_residual_buffers[-1]
            last_score_task.compute_weighted_residual(prev_state, residual_buffer=rbuf, row_offset=0)
            wp.launch(
                aggregate_residuals_to_costs,
                dim=self._batch_size,
                inputs=[rbuf, self._prev_score],
                device=self._solver.device,
            )
            assert self._prev_q_save is not None and self._prev_base_save is not None
            wp.copy(self._prev_q_save, prev_state.q)
            wp.copy(self._prev_base_save, prev_state.T_world_base.xyz_wxyz)

        # Solve
        best_state, _ = self._solver.solve()

        # Keep prev_state if it has better position score (all GPU, no sync)
        if prev_state is not None and self._prev_score is not None and last_score_task is not None:
            wp.launch(
                _select_better_score_kernel,
                dim=self._batch_size,
                inputs=[
                    self._prev_q_save,
                    self._prev_base_save,
                    best_state.q,
                    best_state.T_world_base.xyz_wxyz,
                    self._prev_score,
                    self._solver.best_costs,
                    self.robot.num_actuated_joints,
                ],
                device=self._solver.device,
            )

        # Copy to out_state if provided (avoids allocation)
        if out_state is not None:
            wp.copy(out_state.q, best_state.q)
            if self._has_mobile_base and out_state.T_world_base is not None:
                wp.copy(out_state.T_world_base.xyz_wxyz, best_state.T_world_base.xyz_wxyz)
            return out_state

        return best_state

    def solve_torch(
        self,
        target_matrix: Union[torch.Tensor, Sequence[torch.Tensor]],
        prev_state: Optional[WarpRobotState] = None,
        init_q: Optional[torch.Tensor] = None,
        init_T_world_base: Optional[torch.Tensor] = None,
        rest_q: Optional[torch.Tensor] = None,
        rest_T_world_base: Optional[torch.Tensor] = None,
        return_costs: bool = False,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[float] = None,
    ) -> IKResultTorch:
        target_matrices = [target_matrix] if isinstance(target_matrix, torch.Tensor) else list(target_matrix)
        targets = [WarpSE3.from_matrix(wp.from_torch(mat.contiguous(), dtype=wp.mat44)) for mat in target_matrices]

        init_state = None
        if init_q is not None:
            init_q_wp = wp.from_torch(init_q.contiguous(), dtype=wp.float32)
            if init_T_world_base is not None and self._has_mobile_base:
                init_T_world_base_wp = WarpSE3.from_matrix(
                    wp.from_torch(init_T_world_base.contiguous(), dtype=wp.mat44)
                )
            else:
                init_T_world_base_wp = None
            init_state = self.robot.state(q=init_q_wp, T_world_base=init_T_world_base_wp)

        rest_state = None
        if rest_q is not None:
            rest_q_wp = wp.from_torch(rest_q.contiguous(), dtype=wp.float32)
            if rest_T_world_base is not None and self._has_mobile_base:
                rest_T_world_base_wp = WarpSE3.from_matrix(
                    wp.from_torch(rest_T_world_base.contiguous(), dtype=wp.mat44)
                )
            else:
                rest_T_world_base_wp = None
            rest_state = self.robot.state(q=rest_q_wp, T_world_base=rest_T_world_base_wp)

        state = self.solve(
            targets,
            prev_state=prev_state,
            init_state=init_state,
            rest_state=rest_state,
            init_sample_range=init_sample_range,
            base_init_sample_range=base_init_sample_range,
        )
        q_torch = wp.to_torch(state.q).clone()
        T_world_base_torch = None
        if self._has_mobile_base:
            T_world_base_torch = wp.to_torch(state.T_world_base.as_matrix()).clone()

        position_cost = None
        orientation_cost = None
        if return_costs:
            costs = self.compute_costs(state, targets)
            position_cost = wp.to_torch(costs["position_task"]).clone()
            orientation_cost = wp.to_torch(costs["orientation_task"]).clone()

        return IKResultTorch(
            q=q_torch,
            T_world_base=T_world_base_torch,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
        )

    def solve_torch_via_matrix(
        self,
        target_matrix: Union[torch.Tensor, Sequence[torch.Tensor]],
        prev_state: Optional[WarpRobotState] = None,
        init_q: Optional[torch.Tensor] = None,
        init_T_world_base: Optional[torch.Tensor] = None,
        rest_q: Optional[torch.Tensor] = None,
        rest_T_world_base: Optional[torch.Tensor] = None,
        return_costs: bool = False,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[float] = None,
    ) -> IKResultTorch:
        return self.solve_torch(
            target_matrix=target_matrix,
            prev_state=prev_state,
            init_q=init_q,
            init_T_world_base=init_T_world_base,
            rest_q=rest_q,
            rest_T_world_base=rest_T_world_base,
            return_costs=return_costs,
            init_sample_range=init_sample_range,
            base_init_sample_range=base_init_sample_range,
        )

    def solve_numpy(
        self,
        target_pos: Union[np.ndarray, List[np.ndarray]],
        target_quat_wxyz: Union[np.ndarray, List[np.ndarray]],
        prev_state: Optional[WarpRobotState] = None,
        rest_state: Optional[WarpRobotState] = None,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[float] = None,
    ) -> WarpRobotState:
        # Handle multi-frame: target_pos and target_quat_wxyz can be lists
        if isinstance(target_pos, list):
            targets_list: List[WarpSE3] = []
            quat_list = cast(List[np.ndarray], target_quat_wxyz)
            for pos, quat in zip(target_pos, quat_list):
                target_pose = np.atleast_2d(np.concatenate([pos, quat], axis=-1))
                targets_list.append(WarpSE3(wp.from_numpy(target_pose, dtype=wp_vec7, device=self.device)))
            return self.solve(
                targets_list,
                prev_state=prev_state,
                rest_state=rest_state,
                init_sample_range=init_sample_range,
                base_init_sample_range=base_init_sample_range,
            )
        else:
            quat_array = cast(np.ndarray, target_quat_wxyz)
            target_pose = np.atleast_2d(np.concatenate([target_pos, quat_array], axis=-1))
            target_se3 = WarpSE3(wp.from_numpy(target_pose, dtype=wp_vec7, device=self.device))
            return self.solve(
                target_se3,
                prev_state=prev_state,
                rest_state=rest_state,
                init_sample_range=init_sample_range,
                base_init_sample_range=base_init_sample_range,
            )

    def compute_costs(
        self,
        state: WarpRobotState,
        targets: Union[WarpSE3, Sequence[WarpSE3]],
        prev_state: Optional[WarpRobotState] = None,
    ) -> Dict[str, wp.array]:
        # Normalize targets to a list
        if isinstance(targets, WarpSE3):
            targets_list: List[WarpSE3] = [targets]
        else:
            targets_list = list(targets)

        self.robot.forward_kinematics(state)
        self.robot.compute_motion_subspace(state)

        final_stage_idx = len(self.stage_configs) - 1

        # Update all position/rotation tasks with targets
        self._stage_position_tasks[final_stage_idx].set_target(targets_list)
        self._stage_rotation_tasks[final_stage_idx].set_target(targets_list)

        # Update smoothness/velocity tasks if prev_state provided
        if prev_state is not None:
            smoothness_task = self._stage_smoothness_tasks[final_stage_idx]
            if smoothness_task is not None:
                smoothness_task.set_prev_state(prev_state)
            velocity_task = self._stage_velocity_tasks[final_stage_idx]
            if velocity_task is not None:
                velocity_task.set_prev_state(prev_state)

        optimizer = self._solver._optimizers[final_stage_idx]
        batch_size = state.batch_size
        device = wp.get_device(self.device)
        optimizer.compute_residuals(optimizer.terms, state, optimizer.residuals)

        costs: Dict[str, wp.array] = {}
        position_task_idx = 0
        rotation_task_idx = 0
        for term, offset in zip(optimizer.terms, optimizer.residual_offsets):
            term_residual = optimizer.residuals[:, offset : offset + term.residual_dim]

            if isinstance(term, WarpPositionTask):
                for frame_num in range(term.num_frames):
                    frame_offset = frame_num * 3
                    position_residual = term_residual[:, frame_offset : frame_offset + 3]  # type: ignore[index]
                    position_cost = wp.zeros((batch_size,), dtype=wp.float32, device=device)
                    wp.launch(
                        aggregate_residuals_to_costs,
                        dim=batch_size,
                        inputs=[position_residual, position_cost],
                        device=device,
                    )
                    suffix = f"_{position_task_idx}" if self.num_frames > 1 else ""
                    costs[f"position_task{suffix}"] = position_cost
                    position_task_idx += 1
            elif isinstance(term, WarpRotationTask):
                for frame_num in range(term.num_frames):
                    frame_offset = frame_num * 3
                    orientation_residual = term_residual[:, frame_offset : frame_offset + 3]  # type: ignore[index]
                    orientation_cost = wp.zeros((batch_size,), dtype=wp.float32, device=device)
                    wp.launch(
                        aggregate_residuals_to_costs,
                        dim=batch_size,
                        inputs=[orientation_residual, orientation_cost],
                        device=device,
                    )
                    suffix = f"_{rotation_task_idx}" if self.num_frames > 1 else ""
                    costs[f"orientation_task{suffix}"] = orientation_cost
                    rotation_task_idx += 1
            else:
                term_cost = wp.zeros((batch_size,), dtype=wp.float32, device=device)
                wp.launch(
                    aggregate_residuals_to_costs, dim=batch_size, inputs=[term_residual, term_cost], device=device
                )
                name = type(term).__name__
                if isinstance(term, WarpCollisionTask):
                    name = "collision"
                else:
                    if name.startswith("Warp"):
                        name = name[4:]
                    name = "".join("_" + c.lower() if c.isupper() else c for c in name).lstrip("_")
                costs[name] = term_cost
        return costs
