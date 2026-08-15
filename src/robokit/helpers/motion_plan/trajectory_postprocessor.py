# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
from typing import TYPE_CHECKING, Callable, Optional, cast


if TYPE_CHECKING:
    from warp.context import Graph

import numpy as np
import warp as wp

from robokit.geom import WarpScene
from robokit.helpers.ik import IK, IKConfig
from robokit.helpers.motion_plan.trajectory_retimer import TrajectoryRetimer, _compute_retimed_kinematics
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.utils import warp_utils
from robokit.utils.warp_utils import wp_vec7


_SHORTCUT_SWEEPS = 4
_SHORTCUT_RADII = (1,) + (8, 4, 2, 1) * _SHORTCUT_SWEEPS
_SHORTCUT_COLLISION_WEIGHT = 1.0e6
_SNAP_POSITION_THRESHOLD_M = 0.005


# --- laplacian shortcut kernels --------------------------------------------
@wp.kernel
def _compute_laplacian_proposal_kernel(
    q: wp.array3d(dtype=wp.float32),
    num_dofs: int,
    num_frames: int,
    radius: int,
    init: int,
    q_prop: wp.array2d(dtype=wp.float32),
    lens: wp.array2d(dtype=wp.float32),
):
    bi, f = wp.tid()
    row = bi * (num_frames - 2) + f
    lo = wp.max(f + 1 - radius, 0)
    hi = wp.min(f + 1 + radius, num_frames - 1)
    len_old_a = float(0.0)
    len_old_b = float(0.0)
    len_new_a = float(0.0)
    len_new_b = float(0.0)
    for d in range(num_dofs):
        a = q[bi, f, d]
        b = q[bi, f + 1, d]
        c = q[bi, f + 2, d]
        p = 0.5 * (q[bi, lo, d] + q[bi, hi, d])
        if init != 0:
            p = b
        q_prop[row, d] = p
        len_old_a += (b - a) * (b - a)
        len_old_b += (c - b) * (c - b)
        len_new_a += (p - a) * (p - a)
        len_new_b += (c - p) * (c - p)
    lens[0, row] = wp.sqrt(len_old_a) + wp.sqrt(len_old_b)
    lens[1, row] = wp.sqrt(len_new_a) + wp.sqrt(len_new_b)


@wp.kernel
def _compute_laplacian_query_mask_kernel(
    lens: wp.array2d(dtype=wp.float32),
    clearance: wp.array1d(dtype=wp.float32),
    num_spheres: int,
    init: int,
    mask: wp.array1d(dtype=wp.int32),
):
    pt = wp.tid()
    row = pt // num_spheres
    mask[pt] = int(init != 0 or lens[1, row] < lens[0, row] or clearance[row] < 0.0)


@wp.kernel
def _set_accepted_laplacian_proposal_kernel(
    q_prop: wp.array2d(dtype=wp.float32),
    sdf: wp.array1d(dtype=wp.float32),
    eff_radii: wp.array1d(dtype=wp.float32),
    lens: wp.array2d(dtype=wp.float32),
    num_spheres: int,
    num_dofs: int,
    num_internal_frames: int,
    init: int,
    q: wp.array3d(dtype=wp.float32),
    clearance: wp.array1d(dtype=wp.float32),
):
    bi, f = wp.tid()
    row = bi * num_internal_frames + f
    c = float(1.0e9)
    for s in range(num_spheres):
        c = wp.min(c, sdf[row * num_spheres + s] - eff_radii[s])
    ok = c >= 0.0 or c >= clearance[row]
    shorter = lens[1, row] < lens[0, row]
    escape = clearance[row] < 0.0 and c > clearance[row]
    if init == 0 and not ((shorter and ok) or escape):
        return
    if init == 0:
        for d in range(num_dofs):
            q[bi, f + 1, d] = q_prop[row, d]
    clearance[row] = c


@wp.kernel
def _compute_laplacian_acceptance_kernel(
    q: wp.array3d(dtype=wp.float32),
    candidate_q: wp.array3d(dtype=wp.float32),
    clearance: wp.array1d(dtype=wp.float32),
    candidate_clearance: wp.array1d(dtype=wp.float32),
    velocity_limits: wp.array1d(dtype=wp.float32),
    num_frames: int,
    num_dofs: int,
    dt: float,
    dt_scale: float,
    minimum_dt: float,
    max_dt: float,
    accel_limit: float,
    jerk_limit: float,
    retime: int,
    collision_weight: float,
    accept_mask: wp.array1d(dtype=wp.int32),
):
    batch = wp.tid()
    kinematics = _compute_retimed_kinematics(
        q,
        batch,
        velocity_limits,
        num_frames,
        num_dofs,
        dt,
        dt_scale,
        minimum_dt,
        max_dt,
        accel_limit,
        jerk_limit,
        retime,
    )
    candidate_kinematics = _compute_retimed_kinematics(
        candidate_q,
        batch,
        velocity_limits,
        num_frames,
        num_dofs,
        dt,
        dt_scale,
        minimum_dt,
        max_dt,
        accel_limit,
        jerk_limit,
        retime,
    )
    jerk = kinematics[1] / (jerk_limit + 1.0e-8)
    candidate_jerk = candidate_kinematics[1] / (jerk_limit + 1.0e-8)
    dt_ratio = kinematics[0] / dt
    candidate_dt_ratio = candidate_kinematics[0] / dt
    cost = jerk * jerk * dt_ratio * dt_ratio * dt_ratio
    candidate_cost = candidate_jerk * candidate_jerk * candidate_dt_ratio * candidate_dt_ratio * candidate_dt_ratio
    penetration = float(0.0)
    candidate_penetration = float(0.0)
    for frame in range(num_frames - 2):
        row = batch * (num_frames - 2) + frame
        penetration = wp.max(penetration, -clearance[row])
        candidate_penetration = wp.max(candidate_penetration, -candidate_clearance[row])
    cost += collision_weight * penetration * penetration
    candidate_cost += collision_weight * candidate_penetration * candidate_penetration
    accept_mask[batch] = int(candidate_penetration <= penetration + 1.0e-6 and candidate_cost < cost)


# --- endpoint snap kernels -------------------------------------------------
@wp.kernel
def _set_endpoint_snap_seed_kernel(q: wp.array3d(dtype=wp.float32), row: int, out: wp.array2d(dtype=wp.float32)):
    bi, d = wp.tid()
    out[bi, d] = q[bi, row, d]


@wp.kernel
def _set_blended_endpoint_kernel(
    polished: wp.array2d(dtype=wp.float32),
    accept_mask: wp.array1d(dtype=wp.int32),
    num_frames: int,
    nb: int,
    num_dofs: int,
    q: wp.array3d(dtype=wp.float32),
):
    bi = wp.tid()
    if accept_mask[bi] == 0:
        return
    n2 = float(0.0)
    for d in range(num_dofs):
        dd = polished[bi, d] - q[bi, num_frames - 1, d]
        n2 += dd * dd
    s = wp.min(1.0, 0.05 / wp.max(wp.sqrt(n2), 1.0e-12))
    for j in range(nb - 1):
        t = wp.float32(j + 1) / wp.float32(nb)
        w = (3.0 * t * t - 2.0 * t * t * t) * s
        for d in range(num_dofs):
            q[bi, num_frames - nb + j, d] += w * (polished[bi, d] - q[bi, num_frames - 1, d])
    for d in range(num_dofs):
        q[bi, num_frames - 1, d] = polished[bi, d]


@wp.kernel
def _compute_endpoint_snap_acceptance_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_target: wp.array1d(dtype=wp_vec7),
    ee_link_index: int,
    position_threshold_m: float,
    accept_mask: wp.array1d(dtype=wp.int32),
):
    batch = wp.tid()
    ee = T_world_link[batch, ee_link_index]
    target = T_world_target[batch]
    dx = ee[0] - target[0]
    dy = ee[1] - target[1]
    dz = ee[2] - target[2]
    accept_mask[batch] = int(wp.sqrt(dx * dx + dy * dy + dz * dz) < position_threshold_m)


# --- base ------------------------------------------------------------------
class TrajectoryPostprocessor:
    """Base class for ordered stages that mutate trajectories in place.

    Args:
        robot: Robot model used for kinematics and collision geometry.
        ee_link_index: End-effector link index.
        num_frames: Number of trajectory frames.
        scene: Collision scene.
        retimer: Shared trajectory timing evaluator.
        use_cuda_graph: Whether to capture supported work in CUDA graphs.
        device: Warp device.
    """

    def __init__(
        self,
        robot: Robot,
        ee_link_index: int,
        num_frames: int,
        scene: WarpScene,
        retimer: TrajectoryRetimer,
        use_cuda_graph: bool,
        device: str,
    ):
        self.robot = robot
        self.ee_link_index = ee_link_index
        self.num_frames = num_frames
        self.scene = scene
        self.retimer = retimer
        self.use_cuda_graph = use_cuda_graph
        self.device = wp.get_device(device)
        self._bufs: Optional[tuple] = None
        self._graph: Optional["Graph"] = None
        self._graph_scene_revision: Optional[int] = None
        self._graph_body: Optional[Callable[[], None]] = None
        self._has_inactive_joints = False
        scene._register_graph_change_callback(self._clear_cache_on_scene_update)

    def warmup(self, batch_size: int, joint_mask: np.ndarray):
        """Set batch-independent joint state before the first solve."""
        self._has_inactive_joints = bool(np.any(joint_mask == 0.0))

    def set_active_joint_mask(self, joint_mask: np.ndarray):
        """Set which joints the stage may modify."""
        self._has_inactive_joints = bool(np.any(joint_mask == 0.0))

    def clear_cache(self):
        """Clear captured execution state."""
        self._graph = None
        self._graph_scene_revision = None

    def _clear_cache_on_scene_update(self) -> None:
        """Clear and recapture execution state after a scene update."""
        self.clear_cache()
        if not self.use_cuda_graph or self._graph_body is None:
            return
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self._graph_body()
        self._graph = cast("Graph", capture.graph)
        self._graph_scene_revision = self.scene.graph_revision

    def solve(
        self,
        q: wp.array,
        T_world_target: Optional[wp.array],  # None for q-goal plans (EndpointSnap skips those)
        target_q: Optional[wp.array],
        scene_indices: wp.array,
    ):
        """Mutate `q` in place.

        Args:
            q: Batched trajectories.
            T_world_target: Optional batched pose goals.
            target_q: Optional batched joint-space goals.
            scene_indices: Scene index for each query.
        """
        raise NotImplementedError


# --- laplacian shortcut ----------------------------------------------------
class LaplacianShortcut(TrajectoryPostprocessor):
    """Shorten trajectories without increasing scene penetration."""

    def solve(
        self,
        q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
    ):
        if self._has_inactive_joints:
            return
        batch_size = q.shape[0]
        num_dofs = self.robot.num_actuated_joints
        n_int = self.num_frames - 2
        n_rows = batch_size * n_int
        num_spheres = len(self.robot.spec.collision_sphere_radii)
        if self._bufs is None:
            eff_radii = np.where(
                self.robot.spec.collision_spheres_link_indices > 0,
                self.robot.spec.collision_sphere_radii,
                -1.0e9,
            ).astype(np.float32)
            self._bufs = (
                wp.zeros((batch_size, self.num_frames, num_dofs), dtype=wp.float32, device=self.device),
                self.robot.state(q=wp.zeros((n_rows, num_dofs), dtype=wp.float32, device=self.device)),
                wp.zeros((2, n_rows), dtype=wp.float32, device=self.device),
                wp.zeros(n_rows, dtype=wp.float32, device=self.device),
                wp.zeros(n_rows, dtype=wp.float32, device=self.device),
                wp.zeros(n_rows * num_spheres, dtype=wp.float32, device=self.device),
                wp.zeros(n_rows * num_spheres, dtype=wp.int32, device=self.device),
                wp.zeros(n_rows * num_spheres, dtype=wp.int32, device=self.device),
                wp.from_numpy(eff_radii, dtype=wp.float32, device=self.device),
                wp.zeros(batch_size, dtype=wp.int32, device=self.device),
            )
        (
            q3d,
            prop_state,
            lens,
            clearance,
            clearance0,
            sdf,
            scene_pp,
            query_mask,
            eff_radii_wp,
            accept_mask,
        ) = self._bufs
        wp.copy(q3d, q)
        warp_utils.repeat(scene_indices, n_int * num_spheres, out=scene_pp)
        centers = prop_state.collision_sphere_centers_world.reshape((n_rows * num_spheres,))

        def _solve_laplacian_shortcuts():
            for it, radius in enumerate(_SHORTCUT_RADII):
                init = int(it == 0)
                wp.launch(
                    _compute_laplacian_proposal_kernel,
                    dim=[batch_size, n_int],
                    inputs=[q3d, num_dofs, self.num_frames, radius, init, prop_state.q, lens],
                    device=self.device,
                )
                self.robot.forward_kinematics(prop_state)
                self.robot.transform_collision_spheres(prop_state)
                wp.launch(
                    _compute_laplacian_query_mask_kernel,
                    dim=[n_rows * num_spheres],
                    inputs=[lens, clearance, num_spheres, init, query_mask],
                    device=self.device,
                )
                self.scene.query_sdf(
                    centers,
                    scene_indices=scene_pp,
                    query_mask=query_mask,
                    distance_only=True,
                    out_signed_dists=sdf,
                )
                wp.launch(
                    _set_accepted_laplacian_proposal_kernel,
                    dim=[batch_size, n_int],
                    inputs=[
                        prop_state.q,
                        sdf,
                        eff_radii_wp,
                        lens,
                        num_spheres,
                        num_dofs,
                        n_int,
                        init,
                        q3d,
                        clearance,
                    ],
                    device=self.device,
                )
                if init:
                    wp.copy(clearance0, clearance)

        self._graph_body = _solve_laplacian_shortcuts
        if self.use_cuda_graph:
            revision = self.scene.graph_revision
            graph = self._graph if self._graph_scene_revision == revision else None
            if graph is None:
                _solve_laplacian_shortcuts()
                wp.synchronize()
                wp.copy(q3d, q)
                with wp.ScopedCapture(device=self.device) as capture:
                    _solve_laplacian_shortcuts()
                graph = cast("Graph", capture.graph)
                self._graph = graph
                self._graph_scene_revision = revision
            wp.capture_launch(graph)
        else:
            _solve_laplacian_shortcuts()
        retimer = self.retimer
        config = retimer.config
        wp.launch(
            _compute_laplacian_acceptance_kernel,
            dim=batch_size,
            inputs=[
                q,
                q3d,
                clearance0,
                clearance,
                retimer._velocity_limits,
                self.num_frames,
                num_dofs,
                retimer.dt,
                config.dt_scale if config is not None else 1.0,
                config.minimum_dt if config is not None else retimer.dt,
                config.max_dt if config is not None else 0.0,
                config.accel_limit if config is not None else 1.0,
                config.jerk_limit if config is not None else 1.0,
                int(config is not None),
                _SHORTCUT_COLLISION_WEIGHT,
                accept_mask,
            ],
            device=self.device,
        )
        warp_utils.masked_copy(accept_mask, q3d, q)


# --- endpoint snap ---------------------------------------------------------
class EndpointSnap(TrajectoryPostprocessor):
    """Polish and blend the final configurations toward the target poses."""

    def warmup(self, batch_size: int, joint_mask: np.ndarray):
        super().warmup(batch_size, joint_mask)
        if self._bufs is not None:
            return
        num_dofs = self.robot.num_actuated_joints
        config = (
            IKConfig(
                solver=MultiSeedSolverConfig(
                    stages=[StageConfig(num_seeds=1, iters=5, lm_lambda=1.0)],
                    cuda_graph_mode=("full" if batch_size == 1 else "iter") if self.use_cuda_graph else "none",
                )
            )
            .add(PositionTask(weight=20.0))
            .add(RotationTask(weight=10.0))
        )
        ik = IK(config, robot=self.robot, link=self.ee_link_index, device=self.device)
        ik.set_active_joint_mask(joint_mask.tolist())
        ik.warmup(batch_size=batch_size)
        q = wp.zeros((batch_size, self.num_frames, num_dofs), dtype=wp.float32, device=self.device)
        targets = [t.T_world_target.reshape((batch_size,)) for t in ik._solver.terms[0]]
        self._bufs = (
            ik,
            q,
            targets,
            wp.zeros(batch_size, dtype=wp.int32, device=self.device),
        )

    def set_active_joint_mask(self, joint_mask: np.ndarray):
        super().set_active_joint_mask(joint_mask)
        if self._bufs is not None:
            cast(IK, self._bufs[0]).set_active_joint_mask(joint_mask.tolist())

    def solve(
        self,
        q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
    ):
        if target_q is not None:
            return
        batch_size = q.shape[0]
        num_dofs = self.robot.num_actuated_joints
        ik, q_buf, targets, accept_mask = self._bufs
        wp.copy(q_buf, q)
        for target in targets:
            wp.copy(target, cast(wp.array, T_world_target))
        ik_initial_state = cast(RobotState, ik._initial_var.get("robot"))
        wp.launch(
            _set_endpoint_snap_seed_kernel,
            dim=(batch_size, num_dofs),
            inputs=[q_buf, self.num_frames - 1, ik_initial_state.q],
            device=self.device,
        )
        ik._solver.solve(ik._initial_var)
        self.robot.forward_kinematics(ik_initial_state)
        wp.launch(
            _compute_endpoint_snap_acceptance_kernel,
            dim=batch_size,
            inputs=[
                ik_initial_state.T_world_link,
                T_world_target,
                self.ee_link_index,
                _SNAP_POSITION_THRESHOLD_M,
                accept_mask,
            ],
            device=self.device,
        )
        wp.launch(
            _set_blended_endpoint_kernel,
            dim=batch_size,
            inputs=[
                ik_initial_state.q,
                accept_mask,
                self.num_frames,
                min(6, self.num_frames - 1),
                num_dofs,
                q_buf,
            ],
            device=self.device,
        )
        wp.copy(q, q_buf)


__all__ = ["TrajectoryPostprocessor", "LaplacianShortcut", "EndpointSnap"]
