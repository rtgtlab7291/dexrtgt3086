# pyright: reportInvalidTypeForm=false

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.helpers.ik.config import IKConfig
from robokit.lie.se3 import se3_from_matrix, se3_identity, se3_to_matrix
from robokit.opt.multi_seed_solver import MultiSeedSolver, MultiSeedSolverConfig
from robokit.opt.optimizer import aggregate_residuals_to_costs_kernel
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot, RobotState
from robokit.terms.composite_score_task import CompositeScoreTask
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.terms.dense.manipulability_task import ManipulabilityTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.terms.task import ResidualTask
from robokit.utils.sampling_utils import Sampler
from robokit.utils.warp_utils import repeat, wp_device_type, wp_vec7


if TYPE_CHECKING:
    import torch


_FRAME_TASKS = (PositionTask, RotationTask)


@wp.kernel
def select_better_score_kernel(
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
    q: "torch.Tensor"
    T_world_base: Optional["torch.Tensor"] = None


class IK:
    """IK solver."""

    def __init__(
        self,
        config: IKConfig,
        robot: Robot,
        link: Union[str, int, Sequence[Union[str, int]]],
        device: wp_device_type = "cuda",
    ):
        self.config = config
        self.robot: Robot = robot
        self.device = str(device)
        self._wp_device = wp.get_device(device)

        if isinstance(link, (str, int)):
            self._link_names: List[Union[str, int]] = [link]
        else:
            self._link_names = list(link)

        self.target_link_indices: List[int] = []
        for frame in self._link_names:
            if isinstance(frame, str):
                if frame not in self.robot.link_names:
                    raise ValueError(f"link '{frame}' not in robot.link_names ({len(self.robot.link_names)} links)")
                self.target_link_indices.append(self.robot.link_names.index(frame))
            else:
                if frame < 0 or frame >= len(self.robot.link_names):
                    raise ValueError(f"link index {frame} out of range [0, {len(self.robot.link_names)})")
                self.target_link_indices.append(int(frame))
        self.num_joints = self.robot.spec.num_actuated_joints

        n_links = len(self._link_names)
        for term in [*config.terms, *config.score_terms]:
            if isinstance(term, _FRAME_TASKS) and not term.fixed_target:
                w = term.weight
                if isinstance(w, (list, tuple)) and len(w) != n_links:
                    raise ValueError(
                        f"weight sequence length {len(w)} != {n_links} (term '{term.name}' targets {n_links} link(s))"
                    )

        self._sampler = Sampler(
            config.solver.stages[0].num_seeds,
            self.num_joints,
            self.robot.spec.actuated_joint_limits,
            config.base_sample_translation_mask,
            config.seed,
            config.keep_init_seed,
            self._wp_device,
        )

        self._base_offset = 6 if config.enable_T_world_base else 0
        tangent_dim = self._base_offset + self.num_joints
        self._active_dof_mask = wp.ones(tangent_dim, dtype=wp.float32, device=self._wp_device)
        # views into _active_dof_mask: writing them updates the combined mask
        self._active_base_mask = cast(wp.array, self._active_dof_mask[: self._base_offset])
        self._active_joint_mask = cast(wp.array, self._active_dof_mask[self._base_offset :])
        self._joint_mask_staging_np = np.ones(self.num_joints, dtype=np.float32)
        self._base_mask_staging_np = np.asarray(config.base_lock_mask, dtype=np.float32).copy()
        if config.enable_T_world_base:
            wp.copy(self._active_base_mask, wp.from_numpy(self._base_mask_staging_np, dtype=wp.float32, device="cpu"))

        self._all_terms: List[ResidualTask] = [*config.terms, *config.score_terms]
        self._term_by_name: Dict[str, ResidualTask] = {t.name: t for t in self._all_terms}
        self._terms_by_name: Dict[str, List[ResidualTask]] = {n: [] for n in self._term_by_name}
        self._is_score_term: Dict[str, bool] = {
            **{t.name: False for t in config.terms},
            **{t.name: True for t in config.score_terms},
        }

        self._initialized = False

    # --- public mutation API ------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return len(self.target_link_indices)

    @property
    def num_solutions(self) -> int:
        """Number of solutions returned per target instance."""
        return self.config.solver.stages[-1].num_seeds

    def set_weight(self, name: str, value: Union[float, Sequence[float]]):
        """Update a term's weight in-place. Scalar for uniform; sequence for per-link.

        Only :class:`PositionTask` and :class:`RotationTask` accept per-link
        sequences. Other tasks must be a scalar.
        """
        if name not in self._term_by_name:
            raise KeyError(f"No term named '{name}'. Available: {list(self._term_by_name)}")
        term = self._term_by_name[name]

        if isinstance(value, (list, tuple, np.ndarray)):
            if not isinstance(term, _FRAME_TASKS):
                raise TypeError(
                    f"per-link weight sequence only supported on PositionTask / RotationTask, got {type(term).__name__}"
                )
            n_links = len(self._link_names)
            if len(value) != n_links:
                raise ValueError(
                    f"weight sequence length {len(value)} != {n_links} (term '{name}' targets {n_links} link(s))"
                )

        term.weight = value

        for w in self._terms_by_name[name]:
            w.set_weight(value)

    def set_active_joint_mask(self, joint_mask: Sequence[float]):
        """Set which actuated joints are active (1.0) or locked (0.0)."""
        np.copyto(self._joint_mask_staging_np, joint_mask)
        mask_wp = wp.from_numpy(self._joint_mask_staging_np, dtype=wp.float32, device="cpu")
        wp.copy(self._active_joint_mask, mask_wp)

    def set_active_base_mask(self, base_mask: Sequence[float]):
        """Set which floating-base DOFs are active (1.0) or locked (0.0).

        Indices follow the SE(3) twist convention: ``0..2`` are translation
        ``(x, y, z)``, ``3..5`` are rotation ``(roll, pitch, yaw)``. A masked
        DOF is a hard freeze — the underlying LM/GD solver zeroes its
        Jacobian column so the solver cannot move it. Requires
        ``enable_T_world_base=True``.
        """
        assert self._base_offset == 6, "set_active_base_mask requires enable_T_world_base=True"
        if len(base_mask) != 6:
            raise ValueError(f"base_mask length {len(base_mask)} != 6")
        np.copyto(self._base_mask_staging_np, base_mask)
        mask_wp = wp.from_numpy(self._base_mask_staging_np, dtype=wp.float32, device="cpu")
        wp.copy(self._active_base_mask, mask_wp)

    def warmup(self, batch_size: int):
        """Eagerly build the multi-seed solver for a given batch size."""
        self._build_solver(batch_size)

    # --- solve --------------------------------------------------------------------
    def solve(
        self,
        T_world_target: wp.array,
        prev_state: Optional[RobotState] = None,
        init_state: Optional[RobotState] = None,
        rest_state: Optional[RobotState] = None,
        out_state: Optional[RobotState] = None,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[float] = None,
        scene_indices: Optional[wp.array] = None,
    ) -> RobotState:
        """Solve IK and return the final-stage solutions in instance-major order.

        `init_state` initializes sampling; `rest_state` defines rest costs and inactive joints.
        The returned state has shape `(batch * num_solutions, dofs)`.
        """
        batch_size = self._validate_target(T_world_target)

        if not self._initialized:
            self._build_solver(batch_size)
        elif batch_size != self._batch_size:
            raise ValueError(f"Batch size mismatch: expected {self._batch_size}, got {batch_size}")
        if prev_state is not None and self.num_solutions != 1:
            raise ValueError("prev_state requires an IK final stage with num_seeds=1.")

        q_sample_range = init_sample_range if init_sample_range is not None else self.config.init_sample_range
        base_sample_range = (
            base_init_sample_range if base_init_sample_range is not None else self.config.base_init_sample_range
        )

        initial_state = cast(RobotState, self._initial_var.get("robot"))
        inactive_q = rest_state.q if rest_state is not None and np.any(self._joint_mask_staging_np == 0.0) else None
        self._sampler.sample_q(
            init_state.q if init_state is not None else None,
            q_sample_range,
            initial_state.q,
            joint_mask=self._active_joint_mask,
            inactive_q=inactive_q,
        )
        if self.config.enable_T_world_base:
            init_base = init_state.T_world_base if init_state is not None else None
            self._sampler.sample_base(
                init_base,
                base_sample_range,
                initial_state.T_world_base,
                base_mask=self._active_base_mask,
            )

        if prev_state is not None and prev_state.batch_size != self._batch_size:
            raise ValueError(
                f"prev_state batch size mismatch: expected {self._batch_size}, got {prev_state.batch_size}"
            )
        self._update_targets(T_world_target, prev_state, rest_state)
        self._update_scene_indices(scene_indices, batch_size)

        # Prev-state hysteresis: score prev with the same ruler the solver ranks by (explicit score
        # terms, else the cost), then keep prev iff it wins (select_better_score_kernel below). The
        # ruler is always set, so this runs whenever prev_state is passed.
        last_score_task = self._stage_score_tasks[-1]
        if prev_state is not None:
            rbuf = self._solver._score_residual_buffers[-1]
            last_score_task.compute_weighted_residual(VarValues(robot=prev_state), out_residual=rbuf, row_offset=0)
            wp.launch(
                aggregate_residuals_to_costs_kernel,
                dim=self._batch_size,
                inputs=[rbuf, self._prev_score],
                device=self._solver.device,
            )
            wp.copy(self._prev_q_save, prev_state.q)  # type: ignore[reportArgumentType]
            wp.copy(self._prev_base_save, prev_state.T_world_base)  # type: ignore[reportArgumentType]

        best_var, best_costs = self._solver.solve(self._initial_var)
        best_state = cast(RobotState, best_var.get("robot"))

        if prev_state is not None:
            wp.launch(
                select_better_score_kernel,
                dim=self._batch_size,
                inputs=[
                    self._prev_q_save,
                    self._prev_base_save,
                    best_state.q,
                    best_state.T_world_base,
                    self._prev_score,
                    best_costs,
                    self.robot.num_actuated_joints,
                ],
                device=self._solver.device,
            )

        state = cast(RobotState, self._solver.final_var.get("robot"))

        if out_state is not None:
            wp.copy(out_state.q, state.q)
            if self.config.enable_T_world_base and out_state.T_world_base is not None:
                wp.copy(out_state.T_world_base, state.T_world_base)
            return out_state

        return state

    def solve_torch(
        self,
        T_world_target: "torch.Tensor",
        prev_state: Optional[RobotState] = None,
        init_q: Optional["torch.Tensor"] = None,
        init_T_world_base: Optional["torch.Tensor"] = None,
        rest_q: Optional["torch.Tensor"] = None,
        rest_T_world_base: Optional["torch.Tensor"] = None,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[float] = None,
    ) -> IKResultTorch:
        if not 2 <= T_world_target.ndim <= 4 or T_world_target.shape[-2:] != (4, 4):
            raise ValueError(f"T_world_target must have shape (..., 4, 4), got {tuple(T_world_target.shape)}")
        if self.num_frames == 1:
            if T_world_target.ndim == 2:
                T_world_target = T_world_target[None, None]
            elif T_world_target.ndim == 3:
                T_world_target = T_world_target[:, None]
        elif T_world_target.ndim == 3:
            T_world_target = T_world_target[None]
        T_world_target_wp = se3_from_matrix(wp.from_torch(T_world_target.contiguous(), dtype=wp.mat44))

        def to_state(q: Optional["torch.Tensor"], T_world_base: Optional["torch.Tensor"]) -> Optional[RobotState]:
            if q is None:
                return None
            base = (
                se3_from_matrix(wp.from_torch(T_world_base.contiguous(), dtype=wp.mat44))
                if T_world_base is not None and self.config.enable_T_world_base
                else None
            )
            return self.robot.state(q=wp.from_torch(q.contiguous(), dtype=wp.float32), T_world_base=base)

        init_state = to_state(init_q, init_T_world_base)
        rest_state = to_state(rest_q, rest_T_world_base)

        state = self.solve(
            T_world_target_wp,
            prev_state=prev_state,
            init_state=init_state,
            rest_state=rest_state,
            init_sample_range=init_sample_range,
            base_init_sample_range=base_init_sample_range,
        )
        q_torch = wp.to_torch(state.q).clone()
        T_world_base_torch = None
        if self.config.enable_T_world_base:
            T_world_base_torch = wp.to_torch(se3_to_matrix(state.T_world_base)).clone()
        return IKResultTorch(q=q_torch, T_world_base=T_world_base_torch)

    def solve_numpy(
        self,
        T_world_target: np.ndarray,
        prev_state: Optional[RobotState] = None,
        init_state: Optional[RobotState] = None,
        rest_state: Optional[RobotState] = None,
        out_state: Optional[RobotState] = None,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[float] = None,
    ) -> RobotState:
        T_world_target = np.asarray(T_world_target, dtype=np.float32)
        if not 1 <= T_world_target.ndim <= 3 or T_world_target.shape[-1] != 7:
            raise ValueError(f"T_world_target must have shape (..., 7), got {T_world_target.shape}")
        if self.num_frames == 1:
            if T_world_target.ndim == 1:
                T_world_target = T_world_target[None, None]
            elif T_world_target.ndim == 2:
                T_world_target = T_world_target[:, None]
        elif T_world_target.ndim == 2:
            T_world_target = T_world_target[None]
        return self.solve(
            wp.from_numpy(T_world_target, dtype=wp_vec7, device=self.device),
            prev_state=prev_state,
            init_state=init_state,
            rest_state=rest_state,
            out_state=out_state,
            init_sample_range=init_sample_range,
            base_init_sample_range=base_init_sample_range,
        )

    def compute_costs(
        self,
        state: RobotState,
        T_world_target: wp.array,
        prev_state: Optional[RobotState] = None,
        scene_indices: Optional[wp.array] = None,
    ) -> Dict[str, wp.array]:
        batch_size = self._validate_target(T_world_target)
        if batch_size != state.batch_size:
            raise ValueError(f"Target batch size ({batch_size}) must match state batch size ({state.batch_size})")
        self._update_targets(T_world_target, prev_state, None)
        self._update_scene_indices(scene_indices, batch_size)
        return self._solver.compute_costs(VarValues(robot=state))

    def _validate_target(self, T_world_target: wp.array) -> int:
        if T_world_target.dtype != wp_vec7:
            raise TypeError(f"T_world_target must have dtype wp_vec7, got {T_world_target.dtype}")
        if T_world_target.device != self._wp_device:
            raise ValueError(f"T_world_target must be on {self._wp_device}, got {T_world_target.device}")
        if T_world_target.ndim != 2:
            raise ValueError(f"T_world_target must have logical shape [batch, frames], got {T_world_target.shape}")
        if T_world_target.shape[1] != self.num_frames:
            raise ValueError(
                f"Target frame count ({T_world_target.shape[1]}) must match IK frame count ({self.num_frames})"
            )
        return T_world_target.shape[0]

    # --- internal: solver build (lazy) -------------------------------------------
    def _build_solver(self, batch_size: int):
        if self._initialized:
            return

        self._batch_size = batch_size
        self._sampler.warmup(batch_size)

        if self.config.enable_T_world_base:
            # Fixed identity anchor for the RestTask base block when the user gives no rest pose.
            self._identity_base_rest = se3_identity(shape=(1,), device=self._wp_device)
        else:
            self._identity_base_rest = None

        placeholder_q = wp.empty((batch_size, self.num_joints), dtype=wp.float32, device=self._wp_device)  # type: ignore[reportArgumentType]
        if self.config.enable_T_world_base:
            T_world_base = se3_identity(shape=(batch_size,), device=self._wp_device)
            placeholder_state = self.robot.state(q=placeholder_q, T_world_base=T_world_base)
        else:
            placeholder_state = self.robot.state(q=placeholder_q)

        all_stage_terms: List[List[ResidualTask]] = []
        all_score_terms: List[Optional[ResidualTask]] = []
        self._stage_num_seeds: List[int] = []
        self._stage_frame_placeholders: List[wp.array] = []
        self._score_frame_placeholders: Optional[wp.array] = None

        for stage_config in self.config.solver.stages:
            total_batch = batch_size * stage_config.num_seeds
            stage_phs = se3_identity(shape=(total_batch, self.num_frames), device=self._wp_device)
            self._stage_frame_placeholders.append(stage_phs)
            self._stage_num_seeds.append(stage_config.num_seeds)

            prev_q = wp.zeros((batch_size, self.num_joints), dtype=wp.float32, device=self._wp_device)
            if self.config.enable_T_world_base:
                prev_base = se3_identity(shape=(batch_size,), device=self._wp_device)
                prev_placeholder = self.robot.state(q=prev_q, T_world_base=prev_base)
            else:
                prev_placeholder = self.robot.state(q=prev_q)

            stage_terms: List[ResidualTask] = []
            for term in self.config.terms:
                wt = self._build_term(
                    term, stage_phs=stage_phs, prev_placeholder=prev_placeholder, num_seeds=stage_config.num_seeds
                )
                stage_terms.append(wt)
            all_stage_terms.append(stage_terms)

            if self.config.score_terms:
                if self._score_frame_placeholders is None:
                    self._score_frame_placeholders = se3_identity(
                        shape=(batch_size, self.num_frames), device=self._wp_device
                    )
                score_warp_list: List[ResidualTask] = []
                for term in self.config.score_terms:
                    swt = self._build_term(
                        term, stage_phs=self._score_frame_placeholders, prev_placeholder=prev_placeholder
                    )
                    score_warp_list.append(swt)
                all_score_terms.append(
                    score_warp_list[0] if len(score_warp_list) == 1 else CompositeScoreTask(score_warp_list)
                )
            else:
                # No score terms: rank by the cost itself. Reusing the cost as the ruler lets
                # prev-state hysteresis hold the previous solution on ties, killing null-space jitter.
                all_score_terms.append(CompositeScoreTask(list(stage_terms)))

        self._stage_score_tasks: List[Optional[ResidualTask]] = all_score_terms

        supports_cuda_graph = all(
            term.SUPPORTS_CUDA_GRAPH for stage_terms in all_stage_terms for term in stage_terms
        ) and all(score_term is None or score_term.SUPPORTS_CUDA_GRAPH for score_term in all_score_terms)

        solver_config = MultiSeedSolverConfig(
            stages=self.config.solver.stages,
            cuda_graph_mode=self.config.solver.cuda_graph_mode if supports_cuda_graph else "none",
            optimizer_type=self.config.solver.optimizer_type,
            gain_ratio_epsilon=self.config.solver.gain_ratio_epsilon,
            active_dof_mask=self._active_dof_mask,
            lambda_factor=self.config.solver.lambda_factor,
            rho_min=self.config.solver.rho_min,
        )
        self._solver = MultiSeedSolver(
            terms=all_stage_terms,
            config=solver_config,
            device=self._wp_device,
            score_terms=all_score_terms if any(s is not None for s in all_score_terms) else None,
        )
        first_stage_batch = batch_size * self.config.solver.stages[0].num_seeds
        indices = wp.zeros((first_stage_batch,), dtype=wp.int32, device=self._wp_device)
        self._initial_var = VarValues(robot=placeholder_state.gather(indices))
        self._solver.setup(self._initial_var)

        self._prev_score: Optional[wp.array] = None
        self._prev_q_save: Optional[wp.array] = None
        self._prev_base_save: Optional[wp.array] = None
        if self._stage_score_tasks and self._stage_score_tasks[-1] is not None:
            self._prev_score = wp.empty((batch_size,), dtype=wp.float32, device=self._wp_device)  # type: ignore[reportArgumentType]
            self._prev_q_save = wp.empty(
                (batch_size, self.robot.num_actuated_joints),
                dtype=wp.float32,  # type: ignore[reportArgumentType]
                device=self._wp_device,
            )
            self._prev_base_save = wp.empty((batch_size,), dtype=wp_vec7, device=self._wp_device)  # type: ignore[reportArgumentType]

        self._initialized = True

    # --- internal: term construction (per-stage Warp* clone) ---------------------
    def _build_term(
        self,
        term: ResidualTask,
        *,
        stage_phs: wp.array,
        prev_placeholder: RobotState,
        num_seeds: int = 1,
    ) -> ResidualTask:
        robot = self.robot
        indices = self.target_link_indices
        if isinstance(term, _FRAME_TASKS):
            # type(term), not the base class, so autodiff-Jacobian subclasses survive the rebuild.
            wt: ResidualTask = term if term.fixed_target else type(term)(robot, indices, stage_phs, weight=term.weight)
        elif isinstance(term, PositionLimit):
            wt = type(term)(
                robot,
                weight=term.weight,
                residual_mode=term.residual_mode,  # type: ignore[arg-type]
                residual_eps=term.residual_eps,
                mask=term.mask,
                base_axis=term.base_axis,
                base_bounds=term.base_bounds,
                base_weight=term.base_weight,
                include_joints=term.include_joints,
            )
        elif isinstance(term, SmoothnessTask):
            wt = SmoothnessTask(
                robot,
                prev_var=prev_placeholder,
                weight=term.weight,
                base_weight=term._base_weight,
            )
        elif isinstance(term, RestTask):
            if term._T_world_base_rest_arg is not None:
                base_rest = term._T_world_base_rest_arg
            elif self.config.enable_T_world_base:
                # Use a fixed identity reference (not the moving _placeholder_base)
                # so set_rest_state overwrites against a stable anchor.
                base_rest = self._identity_base_rest
            else:
                base_rest = None
            wt = RestTask(
                robot,
                rest_q=term.rest_q_arg,
                T_world_base_rest=base_rest,
                weight=term.weight,
                base_weight=term.base_weight,
            )
        elif isinstance(term, VelocityLimitTask):
            wt = VelocityLimitTask(
                robot,
                dt=term.dt,
                prev_state_var=prev_placeholder,
                velocity_limits=term._velocity_limits_arg,
                weight=term.weight,
            )
        elif isinstance(term, SceneCollisionTask):
            scene_indices = (
                repeat(term.scene_indices, num_seeds)
                if term.scene_indices is not None and num_seeds > 1
                else term.scene_indices
            )
            wt = SceneCollisionTask(
                robot=robot,
                scene=term.scene,
                geometry=term.geometry,
                scene_indices=scene_indices,
                use_link_bounding_sphere_filter=term.use_link_bounding_sphere_filter,
                weight=term.weight,
                margin=term.margin,
                sphere_indices=term.sphere_indices,
                residual_mode=term.residual_mode,  # type: ignore[arg-type]
                residual_eps=term.residual_eps,
            )
        elif isinstance(term, SelfCollisionTask):
            wt = SelfCollisionTask(
                robot=robot,
                weight=term.weight,
                margin=term.margin,
                representation=term.representation,  # type: ignore[arg-type]
                residual_mode=term.residual_mode,  # type: ignore[arg-type]
                filter_adjacent_links=term.filter_adjacent_links,
                max_active_pairs=term.max_active_pairs,
            )
        elif isinstance(term, ComPositionTask):
            wt = ComPositionTask(
                robot,
                target_com_position=term.target_com_position,
                weight=term.weight,
            )
        elif isinstance(term, ManipulabilityTask):
            wt = type(term)(robot, indices[0], weight=term.weight, epsilon=term.epsilon)
        else:
            raise ValueError(f"Unsupported term type: {type(term).__name__}")

        self._terms_by_name[term.name].append(wt)
        return wt

    # --- internal: per-solve target updates --------------------------------------
    def _update_targets(
        self,
        T_world_target: wp.array,
        prev_state: Optional[RobotState],
        rest_state: Optional[RobotState],
    ):
        expanded_cache: Dict[int, wp.array] = {}

        for term in self._all_terms:
            warps = self._terms_by_name[term.name]
            if not warps:
                continue
            is_score = self._is_score_term[term.name]

            if isinstance(term, _FRAME_TASKS) and not term.fixed_target:
                if is_score:
                    for w in warps:
                        w.set_target(T_world_target)
                else:
                    for w, num_seeds in zip(warps, self._stage_num_seeds):
                        if num_seeds not in expanded_cache:
                            expanded_cache[num_seeds] = (
                                repeat(T_world_target, num_seeds) if num_seeds > 1 else T_world_target
                            )
                        w.set_target(expanded_cache[num_seeds])

            elif isinstance(term, (SmoothnessTask, VelocityLimitTask)):
                if prev_state is not None:
                    for w in warps:
                        w.set_prev_state(prev_state)

            elif isinstance(term, RestTask):
                if rest_state is not None:
                    for w in warps:
                        w.set_rest_state(
                            rest_state.q,
                            rest_state.T_world_base if self.config.enable_T_world_base else None,
                        )

    def _update_scene_indices(self, scene_indices: Optional[wp.array], batch_size: int):
        if scene_indices is None:
            return
        if scene_indices.dtype != wp.int32 or scene_indices.ndim != 1:
            raise TypeError("scene_indices must be a 1D wp.int32 array.")
        if scene_indices.shape != (batch_size,):
            raise ValueError("scene_indices must have shape (batch_size,).")
        if scene_indices.device != self._wp_device:
            raise ValueError(f"scene_indices must be on {self._wp_device}, got {scene_indices.device}.")
        expanded: Dict[int, wp.array] = {1: scene_indices}
        for term in self._all_terms:
            if not isinstance(term, SceneCollisionTask):
                continue
            warps = self._terms_by_name[term.name]
            if self._is_score_term[term.name]:
                for w in warps:
                    w.set_scene_indices(scene_indices)
            else:
                for w, num_seeds in zip(warps, self._stage_num_seeds):
                    if num_seeds not in expanded:
                        expanded[num_seeds] = repeat(scene_indices, num_seeds)
                    w.set_scene_indices(expanded[num_seeds])


__all__ = ["IK", "IKResultTorch"]
