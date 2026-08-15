# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportIndexIssue=false
"""Multi-seed Warp-LM grasp optimization (graduated from graspkit ``stages/grasp_optim/lm_helper.py``).

``GraspOptHelper`` optimizes hand joint values (+ optionally a floating wrist pose) so that selected
contact points reach the target object surface with force closure, while avoiding scene collision and
joint-limit violation. The signature algorithm is ``solve_with_contact_resampling``: MCMC re-draws of
the discrete contact-point indices interleaved with LM refinement of the continuous variables.

Changes vs the graspkit original: ``SelfPenetrationTask`` (removed from robokit) is replaced by
``SelfCollisionTask(representation="link_sphere", link_sphere_radius=0.0, link_sphere_center="origin")``
(its documented equivalent), seeding uses ``sample_base_translation_kernel``, and contact candidates
load from the ``contact_points.json`` asset format via ``load_contact_candidates``.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union, cast, overload

import numpy as np
import torch
import warp as wp

from robokit.geom import WarpScene
from robokit.lie.se3 import se3_identity
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.multi_seed_solver import MultiSeedSolver, MultiSeedSolverConfig, StageConfig
from robokit.opt.optimizer import aggregate_residuals_to_costs_kernel
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot, RobotState
from robokit.terms.dense.force_closure_task import ForceClosureTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.task import Task
from robokit.utils.sampling_utils import sample_q_around_init_kernel
from robokit.utils.warp_utils import repeat, wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_multiply_func
from robokit.xform.warp.torch_wrappers import quaternion_to_matrix


def compute_roberts_root(dim: int) -> float:
    """Root of x^(dim+1) = x + 1 — the low-discrepancy Roberts-sequence base."""
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


def roberts_sequence_numpy(num_points: int, dim: int, root: float, offset: int = 0) -> np.ndarray:
    basis = 1 - (1 / root ** (1 + np.arange(dim)))
    n = np.arange(num_points) + offset
    x = n[:, None] * basis[None, :]
    x, _ = np.modf(x)
    return x


def load_contact_candidates(json_path: Union[str, Path], robot: Robot, device: str) -> Tuple[wp.array, wp.array]:
    """Load ``{link_name: [[x,y,z], ...]}`` link-local contact candidates as flat warp arrays.

    Returns ``(points (N,) vec3, link_indices (N,) int32)`` resolved against ``robot.spec.link_names``.
    """
    with open(json_path) as f:
        points_by_link: Dict[str, List[List[float]]] = json.load(f)
    points: List[List[float]] = []
    link_indices: List[int] = []
    for link_name, link_points in points_by_link.items():
        points.extend(link_points)
        link_indices.extend([robot.spec.link_names.index(link_name)] * len(link_points))
    points_wp = wp.from_numpy(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device)
    link_indices_wp = wp.from_numpy(np.asarray(link_indices, dtype=np.int32), dtype=wp.int32, device=device)
    return points_wp, link_indices_wp


@wp.kernel
def _sample_random_contact_indices_kernel(
    rand_states: wp.array1d(dtype=wp.uint32),
    num_candidates: wp.int32,
    out_contact_indices: wp.array2d(dtype=wp.int32),
):
    tid = wp.tid()
    num_contacts = out_contact_indices.shape[1]
    state = rand_states[tid]

    for c in range(num_contacts):
        selected = wp.int32(0)
        unique = wp.int32(0)
        for _attempt in range(8):
            candidate = wp.randi(state, wp.int32(0), num_candidates)
            unique = wp.int32(1)
            for prev in range(c):
                if candidate == out_contact_indices[tid, prev]:
                    unique = wp.int32(0)
            if unique == wp.int32(1):
                selected = candidate
                break
            selected = candidate
        if unique == 0:
            for shift in range(num_candidates):
                candidate = (selected + wp.int32(shift)) % num_candidates
                unique = wp.int32(1)
                for prev in range(c):
                    if candidate == out_contact_indices[tid, prev]:
                        unique = wp.int32(0)
                if unique == 1:
                    selected = candidate
                    break
        out_contact_indices[tid, c] = selected

    rand_states[tid] = state


@wp.kernel
def _merge_by_energy_kernel(
    current_energy: wp.array1d(dtype=wp.float32),
    new_energy: wp.array1d(dtype=wp.float32),
    current_q: wp.array2d(dtype=wp.float32),
    new_q: wp.array2d(dtype=wp.float32),
    current_base: wp.array1d(dtype=wp_vec7),
    new_base: wp.array1d(dtype=wp_vec7),
    current_contact_indices: wp.array2d(dtype=wp.int32),
    new_contact_indices: wp.array2d(dtype=wp.int32),
):
    tid = wp.tid()
    use_new = new_energy[tid] < current_energy[tid]

    if use_new:
        num_joints = current_q.shape[1]
        for j in range(num_joints):
            current_q[tid, j] = new_q[tid, j]
        current_base[tid] = new_base[tid]

        num_contacts = current_contact_indices.shape[1]
        for c in range(num_contacts):
            current_contact_indices[tid, c] = new_contact_indices[tid, c]


# Instance-major layout: flat_idx = instance_idx * num_seeds + seed_idx
@wp.kernel
def _select_best_seed_kernel(
    energies: wp.array1d(dtype=wp.float32),
    num_seeds: wp.int32,
    src_q: wp.array2d(dtype=wp.float32),
    src_base: wp.array1d(dtype=wp_vec7),
    src_contact_indices: wp.array2d(dtype=wp.int32),
    out_q: wp.array2d(dtype=wp.float32),
    out_base: wp.array1d(dtype=wp_vec7),
    out_contact_indices: wp.array2d(dtype=wp.int32),
):
    instance_idx = wp.tid()
    start = instance_idx * num_seeds

    best_seed = wp.int32(0)
    best_energy = energies[start]
    for seed_idx in range(1, num_seeds):
        flat_idx = start + seed_idx
        energy = energies[flat_idx]
        if energy < best_energy:
            best_energy = energy
            best_seed = wp.int32(seed_idx)

    best_flat_idx = start + best_seed
    num_joints = src_q.shape[1]
    for j in range(num_joints):
        out_q[instance_idx, j] = src_q[best_flat_idx, j]
    out_base[instance_idx] = src_base[best_flat_idx]

    num_contacts = src_contact_indices.shape[1]
    for c in range(num_contacts):
        out_contact_indices[instance_idx, c] = src_contact_indices[best_flat_idx, c]


@wp.kernel
def _gather_contact_points_kernel(
    contact_indices: wp.array2d(dtype=wp.int32),
    contact_candidates: wp.array1d(dtype=wp.vec3),
    contact_candidate_link_indices: wp.array1d(dtype=wp.int32),
    out_contact_points: wp.array2d(dtype=wp.vec3),
    out_link_indices: wp.array2d(dtype=wp.int32),
):
    tid = wp.tid()
    num_contacts = contact_indices.shape[1]
    for c in range(num_contacts):
        idx = contact_indices[tid, c]
        out_contact_points[tid, c] = contact_candidates[idx]
        out_link_indices[tid, c] = contact_candidate_link_indices[idx]


@wp.kernel
def _init_rand_states_kernel(
    seed: wp.int32,
    out_rand_states: wp.array1d(dtype=wp.uint32),
):
    tid = wp.tid()
    out_rand_states[tid] = wp.uint32(seed + tid * 1099087573)


@wp.kernel
def _add_score_to_energy_kernel(
    energy: wp.array1d(dtype=wp.float32),
    score: wp.array1d(dtype=wp.float32),
):
    tid = wp.tid()
    energy[tid] = energy[tid] + score[tid]


@wp.kernel
def _sample_base_pose_kernel(  # noqa: PLR0917
    init_base: wp.array(dtype=wp_vec7),
    offsets: wp.array2d(dtype=wp.float32),
    translation_range: wp.float32,
    rotation_range: wp.float32,
    num_seeds: wp.int32,
    mask: wp.array1d(dtype=wp.float32),
    out_base: wp.array(dtype=wp_vec7),
):
    tid = wp.tid()
    instance_idx = tid // int(num_seeds)
    seed_idx = tid % int(num_seeds)
    base = init_base[instance_idx]
    rotvec = wp.vec3(
        offsets[seed_idx, 0] * rotation_range,
        offsets[seed_idx, 1] * rotation_range,
        offsets[seed_idx, 2] * rotation_range,
    )
    angle = wp.length(rotvec)
    scale = wp.where(angle > 1.0e-6, wp.sin(0.5 * angle) / angle, 0.5)
    delta = wp.vec4(wp.cos(0.5 * angle), rotvec[0] * scale, rotvec[1] * scale, rotvec[2] * scale)
    quat = quaternion_multiply_func(wp.vec4(base[3], base[4], base[5], base[6]), delta)
    out_base[tid] = wp_vec7(
        base[0] + offsets[seed_idx, 0] * translation_range * mask[0],
        base[1] + offsets[seed_idx, 1] * translation_range * mask[1],
        base[2] + offsets[seed_idx, 2] * translation_range * mask[2],
        quat[0],
        quat[1],
        quat[2],
        quat[3],
    )


def _default_grasp_opt_stages() -> Tuple[StageConfig, ...]:
    return (
        StageConfig(num_seeds=8, iters=32, lm_lambda=10.0),
        StageConfig(num_seeds=2, iters=48, lm_lambda=5.0),
        StageConfig(num_seeds=1, iters=96, lm_lambda=1.0),
    )


@dataclass
class GraspOptHelperConfig:
    stages: Tuple[StageConfig, ...] = field(default_factory=_default_grasp_opt_stages)
    distance_weight: float = 100.0
    contact_patch_distance_weight: float = 0.0
    collision_weight: Union[float, Sequence[float]] = 150.0
    force_closure_weight: float = 1.0
    force_closure_contact_weights: Optional[Tuple[float, ...]] = None
    position_limit_weight: float = 10.0
    self_collision_weight: float = 10.0
    self_collision_margin: float = 0.02
    collision_margin: float = 0.005
    residual_eps: float = 1e-6
    base_step_scale: float = 1e-3
    init_sample_range_q: float = 0.2
    init_sample_range_base: float = 0.05
    init_sample_range_base_rotation: float = 0.0
    contact_resample_interval: int = 12
    contact_resample_refine_iters: Optional[int] = None
    lock_base: bool = False
    self_collision: bool = False
    use_link_bounding_sphere_filter: bool = False
    anchor_q_weight: float = 0.0
    anchor_base_translation_weight: float = 0.0
    anchor_base_rotation_weight: float = 0.0


class GraspOptHelper:
    def __init__(
        self,
        robot: Robot,
        batch_size: int,
        num_contact_points: int,
        target_meshes: WarpScene,
        scene_meshes: WarpScene,
        contact_candidates: wp.array,
        contact_candidate_link_indices: wp.array,
        config: Optional[GraspOptHelperConfig] = None,
        score_term: Optional[Task] = None,
        num_contact_patch_points: int = 0,
    ):
        self.robot = robot
        self._score_term = score_term
        self.num_contact_points = num_contact_points
        self.num_contact_patch_points = num_contact_patch_points
        self.batch_size = batch_size
        self.device = target_meshes.device
        self.config = config or GraspOptHelperConfig()
        self.stage_configs = list(self.config.stages)

        self._num_candidates = contact_candidates.shape[0]
        self._contact_candidates_wp = contact_candidates
        self._contact_candidate_link_indices_wp = contact_candidate_link_indices
        # References for `compute_energy` (graspkit-style metric reporting).
        self._target_geom = target_meshes
        self._scene_geom = scene_meshes

        num_joints = robot.spec.num_actuated_joints
        self._joint_mask_wp = wp.ones(num_joints, dtype=wp.float32, device=str(self.device))
        self._base_mask_wp = wp.ones(3, dtype=wp.float32, device=str(self.device))
        spec_tensors = robot.spec.get_tensors(str(self.device))
        self._joint_limits_wp = spec_tensors.actuated_joint_limits

        self._roberts_root_q = compute_roberts_root(num_joints)
        self._roberts_root_base = compute_roberts_root(3)

        max_seeds = self.stage_configs[0].num_seeds
        roberts_q = roberts_sequence_numpy(max_seeds, num_joints, self._roberts_root_q)
        roberts_q[0] = 0.5
        self._roberts_offsets_q = wp.from_numpy(
            (roberts_q - 0.5).astype(np.float32), dtype=wp.float32, device=self.device
        )

        roberts_base = roberts_sequence_numpy(max_seeds, 3, self._roberts_root_base)
        roberts_base[0] = 0.5
        self._roberts_offsets_base = wp.from_numpy(
            (roberts_base - 0.5).astype(np.float32), dtype=wp.float32, device=self.device
        )

        # Create base placeholder state (solver auto-creates per-stage vars via gather)
        placeholder_q = wp.zeros((batch_size, num_joints), dtype=wp.float32, device=self.device)
        placeholder_base = wp.zeros((batch_size,), dtype=wp_vec7, device=self.device)
        placeholder_state = self.robot.state(q=placeholder_q, T_world_base=placeholder_base)
        placeholder_state.base_step_scale = self.config.base_step_scale
        if self.config.lock_base:
            placeholder_state.has_floating_base = False

        (
            self._stage_terms,
            self._stage_distance_tasks,
            self._stage_patch_distance_tasks,
            self._stage_force_closure_tasks,
            self._stage_rest_tasks,
        ) = self._create_stage_terms(target_meshes=target_meshes, scene_meshes=scene_meshes)

        solver_config = MultiSeedSolverConfig(
            stages=list(self.stage_configs),
            cuda_graph_mode="none",
        )
        self._solver = MultiSeedSolver(
            terms=self._stage_terms,
            config=solver_config,
            device=self.device,
            score_terms=None,
        )
        first_stage_batch = batch_size * self.stage_configs[0].num_seeds
        indices = wp.zeros((first_stage_batch,), dtype=wp.int32, device=self.device)
        self._initial_var = placeholder_state.gather(indices)
        self._initial_var_values = VarValues(robot=self._initial_var)

        self._chunk_lm_optimizer = LMOptimizer(
            terms=self._stage_terms[0],
            config=LMOptimizerConfig(
                num_dofs=placeholder_state.tangent_dim,
                max_iter=self.config.contact_resample_interval,
                lm_lambda=self.stage_configs[0].lm_lambda,
            ),
            device=self.device,
        )

        self._current_contact_indices_wp = wp.zeros(
            (first_stage_batch, num_contact_points), dtype=wp.int32, device=self.device
        )
        self._new_contact_indices_wp = wp.zeros(
            (first_stage_batch, num_contact_points), dtype=wp.int32, device=self.device
        )
        self._rand_states = wp.zeros((first_stage_batch,), dtype=wp.uint32, device=self.device)
        self._energy_buffer = wp.zeros((first_stage_batch,), dtype=wp.float32, device=self.device)
        self._new_energy_buffer = wp.zeros((first_stage_batch,), dtype=wp.float32, device=self.device)
        if self._score_term is not None:
            self._score_residual_buffer = wp.zeros(
                (first_stage_batch, self._score_term.residual_dim), dtype=wp.float32, device=self.device
            )
            self._score_cost_buffer = wp.zeros((first_stage_batch,), dtype=wp.float32, device=self.device)
        self._out_q = wp.zeros((batch_size, num_joints), dtype=wp.float32, device=self.device)
        self._out_base = wp.zeros((batch_size,), dtype=wp_vec7, device=self.device)
        self._out_contact_indices = wp.zeros((batch_size, num_contact_points), dtype=wp.int32, device=self.device)
        self._contact_points_buffer = wp.zeros(
            (first_stage_batch, num_contact_points), dtype=wp.vec3, device=self.device
        )
        self._link_indices_buffer = wp.zeros(
            (first_stage_batch, num_contact_points), dtype=wp.int32, device=self.device
        )
        stage0_residual_dim = sum(term.residual_dim for term in self._stage_terms[0])
        self._stage0_residual_buffer = wp.zeros(
            (first_stage_batch, stage0_residual_dim), dtype=wp.float32, device=self.device
        )
        self._proposal_q = wp.zeros((first_stage_batch, num_joints), dtype=wp.float32, device=self.device)
        self._proposal_base = wp.zeros((first_stage_batch,), dtype=wp_vec7, device=self.device)
        self._proposal_state = robot.state(
            q=self._proposal_q,
            T_world_base=self._proposal_base,
        )
        self._proposal_state.base_step_scale = self.config.base_step_scale
        if self.config.lock_base:
            self._proposal_state.has_floating_base = False
        proposal_iters = self.config.contact_resample_refine_iters
        if proposal_iters is None:
            proposal_iters = max(1, self.config.contact_resample_interval // 2)
        proposal_iters = min(proposal_iters, self.config.contact_resample_interval)
        self._proposal_lm_optimizer = LMOptimizer(
            terms=self._stage_terms[0],
            config=LMOptimizerConfig(
                num_dofs=self._proposal_state.tangent_dim,
                max_iter=proposal_iters,
                lm_lambda=self.stage_configs[0].lm_lambda,
            ),
            device=self.device,
        )

    def _create_stage_terms(
        self,
        target_meshes: WarpScene,
        scene_meshes: WarpScene,
    ) -> Tuple[
        List[List], List[SceneDistanceTask], List[Optional[SceneDistanceTask]], List[ForceClosureTask], List[RestTask]
    ]:
        stage_terms: List[List] = []
        stage_distance_tasks: List[SceneDistanceTask] = []
        stage_patch_distance_tasks: List[Optional[SceneDistanceTask]] = []
        stage_force_closure_tasks: List[ForceClosureTask] = []
        stage_rest_tasks: List[RestTask] = []

        for _ in self.stage_configs:
            terms = []

            distance_task = SceneDistanceTask(
                robot=self.robot,
                warp_meshes=target_meshes,
                num_contact_points=self.num_contact_points,
                weight=self.config.distance_weight,
                residual_eps=self.config.residual_eps,
            )
            distance_task.local_contact_points = None
            distance_task.contact_points_link_indices = None
            terms.append(distance_task)
            stage_distance_tasks.append(distance_task)

            patch_distance_task = None
            if self.num_contact_patch_points:
                patch_distance_task = SceneDistanceTask(
                    robot=self.robot,
                    warp_meshes=target_meshes,
                    num_contact_points=self.num_contact_patch_points,
                    weight=self.config.contact_patch_distance_weight,
                    residual_eps=self.config.residual_eps,
                )
                terms.append(patch_distance_task)
            stage_patch_distance_tasks.append(patch_distance_task)

            force_closure_task = ForceClosureTask(
                robot=self.robot,
                num_contact_points=self.num_contact_points,
                weight=self.config.force_closure_weight,
                contact_force_weights=self.config.force_closure_contact_weights,
                warp_meshes=target_meshes,
            )
            force_closure_task.local_contact_points = None
            force_closure_task.contact_points_link_indices = None
            terms.append(force_closure_task)
            stage_force_closure_tasks.append(force_closure_task)

            position_limit_task = PositionLimit(
                robot=self.robot,
                weight=self.config.position_limit_weight,
                residual_eps=self.config.residual_eps,
            )
            terms.append(position_limit_task)

            collision_task = SceneCollisionTask(
                robot=self.robot,
                scene=scene_meshes,
                weight=self.config.collision_weight,
                margin=self.config.collision_margin,
                use_link_bounding_sphere_filter=self.config.use_link_bounding_sphere_filter,
            )
            terms.append(collision_task)

            if self.config.self_collision:
                self_collision_task = SelfCollisionTask(
                    robot=self.robot,
                    representation="link_sphere",
                    link_sphere_radius=0.0,
                    link_sphere_center="origin",
                    weight=self.config.self_collision_weight,
                    margin=self.config.self_collision_margin,
                    residual_eps=self.config.residual_eps,
                )
                terms.append(self_collision_task)

            if (
                self.config.anchor_q_weight > 0.0
                or self.config.anchor_base_translation_weight > 0.0
                or self.config.anchor_base_rotation_weight > 0.0
            ):
                base_rest = se3_identity(shape=(1,), device=target_meshes.device)
                rest_task = RestTask(
                    robot=self.robot,
                    T_world_base_rest=base_rest,
                    weight=self.config.anchor_q_weight,
                    base_weight=(
                        *([self.config.anchor_base_translation_weight] * 3),
                        *([self.config.anchor_base_rotation_weight] * 3),
                    ),
                )
                terms.append(rest_task)
                stage_rest_tasks.append(rest_task)

            stage_terms.append(terms)

        return (
            stage_terms,
            stage_distance_tasks,
            stage_patch_distance_tasks,
            stage_force_closure_tasks,
            stage_rest_tasks,
        )

    def _sample_seeds_from_init(self, init_state: RobotState) -> None:
        first_stage_seeds = self.stage_configs[0].num_seeds
        first_stage_batch = self.batch_size * first_stage_seeds

        wp.launch(
            kernel=sample_q_around_init_kernel,
            dim=first_stage_batch,
            inputs=[
                init_state.q,
                self._roberts_offsets_q,
                self._joint_limits_wp,
                self.config.init_sample_range_q,
                first_stage_seeds,
                self._joint_mask_wp,
                self._initial_var.q,
            ],
            device=self.device,
        )
        for rest_task in self._stage_rest_tasks:
            rest_task.set_rest_state(init_state.q, init_state.T_world_base)

        base_sample_range = 0.0 if self.config.lock_base else self.config.init_sample_range_base
        wp.launch(
            kernel=_sample_base_pose_kernel,
            dim=first_stage_batch,
            inputs=[
                init_state.T_world_base,
                self._roberts_offsets_base,
                base_sample_range,
                self.config.init_sample_range_base_rotation,
                first_stage_seeds,
                self._base_mask_wp,
                self._initial_var.T_world_base,
            ],
            device=self.device,
        )

    def solve(self, init_state: RobotState) -> RobotState:
        self._sample_seeds_from_init(init_state)
        best_var, _ = self._solver.solve(self._initial_var_values)
        return cast(RobotState, best_var.get("robot"))

    def solve_all_seeds(self, init_state: RobotState) -> RobotState:
        """Optimize and return every first-stage seed without soft-cost pruning."""
        self._sample_seeds_from_init(init_state)
        self._initial_var.invalidate()
        self._chunk_lm_optimizer.solve(self._initial_var_values)
        return self._initial_var

    def set_contact_points(
        self,
        local_contact_points: wp.array,
        contact_points_link_indices: wp.array,
        local_contact_patch_points: Optional[wp.array] = None,
        contact_patch_link_indices: Optional[wp.array] = None,
    ) -> None:
        for stage_idx, stage_config in enumerate(self.stage_configs):
            num_seeds = stage_config.num_seeds
            repeated_points = repeat(local_contact_points, num_seeds)
            repeated_link_indices = repeat(contact_points_link_indices, num_seeds)

            self._stage_distance_tasks[stage_idx].local_contact_points = repeated_points
            self._stage_distance_tasks[stage_idx].contact_points_link_indices = repeated_link_indices
            self._stage_force_closure_tasks[stage_idx].local_contact_points = repeated_points
            self._stage_force_closure_tasks[stage_idx].contact_points_link_indices = repeated_link_indices
            patch_task = self._stage_patch_distance_tasks[stage_idx]
            if patch_task is not None:
                assert local_contact_patch_points is not None and contact_patch_link_indices is not None
                patch_task.local_contact_points = repeat(local_contact_patch_points, num_seeds)
                patch_task.contact_points_link_indices = repeat(contact_patch_link_indices, num_seeds)

    @overload
    def compute_energy(
        self,
        state: RobotState,
        contact_indices: Optional[wp.array] = ...,
        return_per_term: Literal[False] = ...,
        w_fc: float = ...,
        w_dis: float = ...,
        w_pen: float = ...,
        w_spen: float = ...,
        w_joints: float = ...,
        spen_threshold: float = ...,
    ) -> wp.array: ...

    @overload
    def compute_energy(
        self,
        state: RobotState,
        contact_indices: Optional[wp.array] = ...,
        *,
        return_per_term: Literal[True],
        w_fc: float = ...,
        w_dis: float = ...,
        w_pen: float = ...,
        w_spen: float = ...,
        w_joints: float = ...,
        spen_threshold: float = ...,
    ) -> Tuple[wp.array, Dict[str, wp.array]]: ...

    def compute_energy(
        self,
        state: RobotState,
        contact_indices: Optional[wp.array] = None,
        return_per_term: bool = False,
        w_fc: float = 1.0,
        w_dis: float = 100.0,
        w_pen: float = 100.0,
        w_spen: float = 10.0,
        w_joints: float = 1.0,
        spen_threshold: float = 0.02,
    ) -> Union[wp.array, Tuple[wp.array, Dict[str, wp.array]]]:
        """Graspkit-style first-order penalty energy per grasp.

        Returns a flat `(batch_size,)` warp array of total energy. With
        `return_per_term=True`, also returns a dict keyed
        `E_fc / E_dis / E_pen / E_spen / E_joints` (each `(batch_size,)`).

        `contact_indices`: `(batch_size, num_contact_points)` int32; if omitted,
        the most recently bound indices (`self._current_contact_indices_wp`)
        are used.

        Joint values are clamped to `actuated_joint_limits` before FK so
        E_spen evaluates physical link origins when the optimizer hasn't
        converged. `E_joints` is computed on the *unclamped* values
        (matches graspkit's reference).
        """

        if contact_indices is None:
            contact_indices = self._current_contact_indices_wp
        device = wp.device_to_torch(self.device)
        n_scene = self._target_geom.num_scenes
        b = state.batch_size
        assert b % n_scene == 0, f"batch_size {b} must be divisible by num_scenes {n_scene}"
        n_samples = b // n_scene
        n_contact = self.num_contact_points

        q = wp.to_torch(state.q)
        limits = wp.to_torch(self._joint_limits_wp)
        q_clamped = torch.clamp(q, min=limits[:, 0], max=limits[:, 1])
        base_xyz_wxyz = wp.to_torch(state.T_world_base)
        base_mat = torch.zeros(b, 4, 4, device=device, dtype=torch.float32)
        base_mat[..., :3, :3] = quaternion_to_matrix(base_xyz_wxyz[..., 3:7])
        base_mat[..., :3, 3] = base_xyz_wxyz[..., :3]
        base_mat[..., 3, 3] = 1.0
        link_poses = self.robot.forward_kinematics_via_matrix_torch(q_clamped.contiguous(), base_mat)

        # Contact points world: gather local pts and link transforms by indices.
        candidates = wp.to_torch(self._contact_candidates_wp)
        cand_link_idx = wp.to_torch(self._contact_candidate_link_indices_wp).long()
        c_idx = wp.to_torch(contact_indices).long()
        local_pts = candidates[c_idx]  # (b, n_contact, 3)
        contact_link_idx = cand_link_idx[c_idx]  # (b, n_contact)
        T_per_contact = torch.gather(
            link_poses, dim=-3, index=contact_link_idx.unsqueeze(-1).unsqueeze(-1).expand(b, n_contact, 4, 4)
        )
        ones = torch.ones(b, n_contact, 1, device=device, dtype=torch.float32)
        contact_pts = (T_per_contact @ torch.cat([local_pts, ones], dim=-1).unsqueeze(-1)).squeeze(-1)[..., :3]

        contact_pts_flat = contact_pts.reshape(b * n_contact, 3).contiguous()
        target_first_idx = torch.arange(
            0, (b + n_samples) * n_contact, n_samples * n_contact, dtype=torch.int32, device=device
        )
        sdf, _, closest = self._target_geom.query_sdf_torch(contact_pts_flat, target_first_idx)
        sdf = sdf.reshape(b, n_contact)
        closest = closest.reshape(b, n_contact, 3)
        contact_normal = torch.nn.functional.normalize(contact_pts - closest, p=2, dim=-1)
        contact_normal = contact_normal * torch.sign(sdf).unsqueeze(-1).detach()
        if self.config.force_closure_contact_weights is not None:
            force_weights = torch.tensor(
                self.config.force_closure_contact_weights,
                device=device,
                dtype=torch.float32,
            )
            contact_normal = contact_normal * force_weights[None, :, None]
        E_dis = (-sdf).abs().sum(dim=-1)

        cross_M = torch.tensor(
            [
                [0, 0, 0, 0, 0, -1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, -1, 0, 0],
                [0, -1, 0, 1, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
            device=device,
        )
        eye_part = torch.eye(3, device=device).expand(b, n_contact, 3, 3).reshape(b, 3 * n_contact, 3)
        cross_part = (contact_pts @ cross_M).view(b, 3 * n_contact, 3)
        g_mat = torch.cat([eye_part, cross_part], dim=2)
        fc_norm = torch.norm(contact_normal.reshape(b, 1, 3 * n_contact) @ g_mat, dim=(1, 2))
        E_fc = fc_norm * fc_norm

        sphere_centers_local = torch.from_numpy(self.robot.spec.local_collision_sphere_centers).to(device).float()
        sphere_radii = torch.from_numpy(self.robot.spec.collision_sphere_radii).to(device).float()
        sphere_link_idx = torch.from_numpy(self.robot.spec.collision_spheres_link_indices).to(device).long()
        sphere_centers_world = self.robot.transform_link_points_torch(link_poses, sphere_centers_local, sphere_link_idx)
        n_spheres = sphere_centers_world.shape[-2]
        sphere_first_idx = torch.arange(
            0, (b + n_samples) * n_spheres, n_samples * n_spheres, dtype=torch.int32, device=device
        )
        sphere_sdf, _, _ = self._scene_geom.query_sdf_torch(
            sphere_centers_world.reshape(b * n_spheres, 3).contiguous(), sphere_first_idx
        )
        sphere_sdf = sphere_sdf.reshape(b, n_spheres)
        pen = -(sphere_sdf - sphere_radii.expand(b, n_spheres))
        pen = torch.where(pen <= 0, torch.zeros_like(pen), pen)
        E_pen = pen.sum(dim=-1)

        over = (q > limits[:, 1]).float() * (q - limits[:, 1])
        under = (q < limits[:, 0]).float() * (limits[:, 0] - q)
        E_joints = over.sum(dim=-1) + under.sum(dim=-1)

        link_origins = link_poses[..., :3, 3]
        dis = ((link_origins.unsqueeze(1) - link_origins.unsqueeze(2)).square().sum(-1) + 1e-13).sqrt()
        dis = torch.where(dis < 1e-6, torch.full_like(dis, 1e6), dis)
        E_spen = (spen_threshold - dis).clamp(min=0).sum(dim=(1, 2))

        E_fc_w = w_fc * E_fc
        E_dis_w = w_dis * E_dis
        E_pen_w = w_pen * E_pen
        E_spen_w = w_spen * E_spen
        E_joints_w = w_joints * E_joints
        total = E_fc_w + E_dis_w + E_pen_w + E_spen_w + E_joints_w

        total_wp = wp.from_torch(total.contiguous(), dtype=wp.float32)
        if not return_per_term:
            return total_wp
        per_term = {
            "E_fc": wp.from_torch(E_fc_w.contiguous(), dtype=wp.float32),
            "E_dis": wp.from_torch(E_dis_w.contiguous(), dtype=wp.float32),
            "E_pen": wp.from_torch(E_pen_w.contiguous(), dtype=wp.float32),
            "E_spen": wp.from_torch(E_spen_w.contiguous(), dtype=wp.float32),
            "E_joints": wp.from_torch(E_joints_w.contiguous(), dtype=wp.float32),
        }
        return total_wp, per_term

    def compute_penetration(self, state: RobotState) -> Tuple[float, float]:
        """Returns `(mean_max_pen_per_grasp, global_max_pen)` in meters using
        `scene_meshes`. Clamps q to joint limits before FK (same reason as
        `compute_energy`)."""

        device = wp.device_to_torch(self.device)
        n_scene = self._scene_geom.num_scenes
        b = state.batch_size
        assert b % n_scene == 0
        n_samples = b // n_scene

        joint_values = wp.to_torch(state.q)
        limits = wp.to_torch(self._joint_limits_wp)
        q_clamped = torch.clamp(joint_values, min=limits[:, 0], max=limits[:, 1])
        base_xyz_wxyz = wp.to_torch(state.T_world_base)
        base_R = quaternion_to_matrix(base_xyz_wxyz[..., 3:7])
        base_mat = torch.zeros(b, 4, 4, device=device, dtype=torch.float32)
        base_mat[..., :3, :3] = base_R
        base_mat[..., :3, 3] = base_xyz_wxyz[..., :3]
        base_mat[..., 3, 3] = 1.0
        link_poses = self.robot.forward_kinematics_via_matrix_torch(q_clamped.contiguous(), base_mat)

        sphere_centers_local = torch.from_numpy(self.robot.spec.local_collision_sphere_centers).to(device).float()
        sphere_radii = torch.from_numpy(self.robot.spec.collision_sphere_radii).to(device).float()
        sphere_link_idx = torch.from_numpy(self.robot.spec.collision_spheres_link_indices).to(device).long()
        centers_world = self.robot.transform_link_points_torch(link_poses, sphere_centers_local, sphere_link_idx)
        n_spheres = centers_world.shape[-2]
        centers_flat = centers_world.reshape(n_scene * n_samples * n_spheres, 3).contiguous()
        first_idx = torch.arange(
            0, (n_scene + 1) * n_samples * n_spheres, n_samples * n_spheres, dtype=torch.int32, device=device
        )
        sphere_sdf, _, _ = self._scene_geom.query_sdf_torch(centers_flat, first_idx)
        sphere_sdf = sphere_sdf.reshape(b, n_spheres)
        pen = -(sphere_sdf - sphere_radii.expand(b, n_spheres))
        pen = torch.where(pen <= 0, torch.zeros_like(pen), pen)
        per_grasp_max = pen.max(dim=-1).values
        return per_grasp_max.mean().item(), per_grasp_max.max().item()

    def _set_contact_points_for_stage(self, contact_indices_wp: wp.array, batch_size: int, stage_idx: int) -> None:
        wp.launch(
            kernel=_gather_contact_points_kernel,
            dim=batch_size,
            inputs=[
                contact_indices_wp,
                self._contact_candidates_wp,
                self._contact_candidate_link_indices_wp,
                self._contact_points_buffer,
                self._link_indices_buffer,
            ],
            device=self.device,
        )
        self._stage_distance_tasks[stage_idx].local_contact_points = self._contact_points_buffer
        self._stage_distance_tasks[stage_idx].contact_points_link_indices = self._link_indices_buffer
        self._stage_force_closure_tasks[stage_idx].local_contact_points = self._contact_points_buffer
        self._stage_force_closure_tasks[stage_idx].contact_points_link_indices = self._link_indices_buffer

    def _compute_energy_for_state(
        self,
        state: RobotState,
        batch_size: int,
        stage_idx: int,
        out_energy: wp.array,
        residual_buffer: Optional[wp.array] = None,
    ) -> None:
        terms = self._stage_terms[stage_idx]
        if residual_buffer is None:
            total_residual_dim = sum(term.residual_dim for term in terms)
            residual_buffer = wp.zeros((batch_size, total_residual_dim), dtype=wp.float32, device=self.device)
        var_values = VarValues(robot=state)
        for term in terms:
            term.precompute(var_values, need_gradient=False)
        offset = 0
        for term in terms:
            term.compute_weighted_residual(var_values, out_residual=residual_buffer, row_offset=offset)
            offset += term.residual_dim
        wp.launch(
            kernel=aggregate_residuals_to_costs_kernel,
            dim=batch_size,
            inputs=[residual_buffer, out_energy],
            device=self.device,
        )

    def solve_with_contact_resampling(
        self, init_state: RobotState, init_contact_indices: wp.array
    ) -> Tuple[RobotState, wp.array]:
        first_stage_seeds = self.stage_configs[0].num_seeds
        first_stage_batch = self.batch_size * first_stage_seeds
        total_iters = self.stage_configs[0].iters
        chunk_size = self.config.contact_resample_interval
        num_chunks = total_iters // chunk_size

        self._sample_seeds_from_init(init_state)

        repeat(init_contact_indices, first_stage_seeds, out=self._current_contact_indices_wp)

        self._set_contact_points_for_stage(self._current_contact_indices_wp, first_stage_batch, stage_idx=0)

        current_state = self._initial_var
        seed = np.random.randint(0, 2147483647)
        wp.launch(
            kernel=_init_rand_states_kernel,
            dim=first_stage_batch,
            inputs=[seed, self._rand_states],
            device=self.device,
        )

        proposal_var = VarValues(robot=self._proposal_state)
        for _chunk in range(num_chunks):
            current_state.invalidate()
            self._chunk_lm_optimizer.solve(self._initial_var_values)
            wp.copy(self._energy_buffer, self._chunk_lm_optimizer.costs)

            wp.copy(self._proposal_state.q, current_state.q)
            wp.copy(self._proposal_state.T_world_base, current_state.T_world_base)

            wp.launch(
                kernel=_sample_random_contact_indices_kernel,
                dim=first_stage_batch,
                inputs=[self._rand_states, self._num_candidates, self._new_contact_indices_wp],
                device=self.device,
            )

            self._set_contact_points_for_stage(self._new_contact_indices_wp, first_stage_batch, stage_idx=0)
            self._proposal_state.invalidate()
            self._proposal_lm_optimizer.solve(proposal_var)
            wp.copy(self._new_energy_buffer, self._proposal_lm_optimizer.costs)

            wp.launch(
                kernel=_merge_by_energy_kernel,
                dim=first_stage_batch,
                inputs=[
                    self._energy_buffer,
                    self._new_energy_buffer,
                    current_state.q,
                    self._proposal_state.q,
                    current_state.T_world_base,
                    self._proposal_state.T_world_base,
                    self._current_contact_indices_wp,
                    self._new_contact_indices_wp,
                ],
                device=self.device,
            )

            self._set_contact_points_for_stage(self._current_contact_indices_wp, first_stage_batch, stage_idx=0)

        current_state.invalidate()
        self._compute_energy_for_state(
            current_state,
            first_stage_batch,
            stage_idx=0,
            out_energy=self._energy_buffer,
            residual_buffer=self._stage0_residual_buffer,
        )
        if self._score_term is not None:
            score_var = VarValues(robot=current_state)
            self._score_term.precompute(score_var, need_gradient=False)
            self._score_term.compute_weighted_residual(
                score_var, out_residual=self._score_residual_buffer, row_offset=0
            )
            wp.launch(
                kernel=aggregate_residuals_to_costs_kernel,
                dim=first_stage_batch,
                inputs=[self._score_residual_buffer, self._score_cost_buffer],
                device=self.device,
            )
            wp.launch(
                kernel=_add_score_to_energy_kernel,
                dim=first_stage_batch,
                inputs=[self._energy_buffer, self._score_cost_buffer],
                device=self.device,
            )

        wp.launch(
            kernel=_select_best_seed_kernel,
            dim=self.batch_size,
            inputs=[
                self._energy_buffer,
                first_stage_seeds,
                current_state.q,
                current_state.T_world_base,
                self._current_contact_indices_wp,
                self._out_q,
                self._out_base,
                self._out_contact_indices,
            ],
            device=self.device,
        )

        best_state = self.robot.state(q=self._out_q, T_world_base=self._out_base)
        best_state.base_step_scale = self.config.base_step_scale
        if self.config.lock_base:
            best_state.has_floating_base = False

        return best_state, self._out_contact_indices

    def close(self) -> None:
        """Release solver/buffer references so warp memory can be reclaimed between scenes."""
        for attr in [
            "_solver",
            "_chunk_lm_optimizer",
            "_proposal_lm_optimizer",
            "_stage_terms",
            "_stage_distance_tasks",
            "_stage_patch_distance_tasks",
            "_stage_force_closure_tasks",
            "_stage_rest_tasks",
            "_roberts_offsets_q",
            "_roberts_offsets_base",
            "_joint_limits_wp",
            "_contact_candidates_wp",
            "_contact_candidate_link_indices_wp",
            "_current_contact_indices_wp",
            "_new_contact_indices_wp",
            "_rand_states",
            "_energy_buffer",
            "_new_energy_buffer",
            "_out_q",
            "_out_base",
            "_out_contact_indices",
            "_contact_points_buffer",
            "_link_indices_buffer",
            "_stage0_residual_buffer",
            "_proposal_q",
            "_proposal_base",
            "_proposal_state",
        ]:
            delattr(self, attr)
        wp.synchronize()
