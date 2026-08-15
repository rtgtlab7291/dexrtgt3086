import inspect
import sys as _sys
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from pathlib import Path as _Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
import warp as wp
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.geom import BoxGeom, MeshGeom, VolumeGeom, WarpScene
from robokit.helpers.motion_plan import (
    EndpointSnap,
    LaplacianShortcut,
    MotionPlanEvaluator,
    MotionPlanner,
    MotionPlannerConfig,
    MotionPlanResult,
    MppiTrajectoryOptimizerConfig,
    OnlineState,
)
from robokit.helpers.motion_plan.gradient_trajectory_optimizer import (
    GradientTrajectoryOptimizer,
    GradientTrajectoryOptimizerConfig,
    _compute_collision_margins_kernel,
)
from robokit.helpers.motion_plan.motion_planner import (
    _build_seed_trajectory_kernel,
    _set_best_retry_result_kernel,
)
from robokit.helpers.motion_plan.mppi_trajectory_optimizer import AccelerationParticleSolver
from robokit.helpers.motion_plan.trajectory_postprocessor import (
    _compute_endpoint_snap_acceptance_kernel,
    _compute_laplacian_acceptance_kernel,
)
from robokit.opt.var_values import VarValues
from robokit.robo import Robot, RobotState
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.warp_utils import wp_vec7


_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))  # repo root, for the shared scene fixture

from tests.helpers.hetero_scenes import (  # noqa: E402
    MP_SCENE_IDS,
    MP_SCENE_SPECS,
    SCENE_IDS,
    build_mp_scene,
    build_multi_scene,
    build_single_scene,
    min_clearance,
    sample_mp_problems,
    sample_targets,
)


wp.init()


def _make_config(q_goal: bool = False) -> MotionPlannerConfig:
    from robokit.helpers.motion_plan.motion_planner import build_default_motion_plan_ik_config
    from robokit.opt.multi_seed_solver import MultiSeedSolverConfig

    ik_config = build_default_motion_plan_ik_config()
    ik_config.solver = MultiSeedSolverConfig(
        stages=ik_config.solver.stages,
        cuda_graph_mode="none",
        optimizer_type=ik_config.solver.optimizer_type,
        gain_ratio_epsilon=ik_config.solver.gain_ratio_epsilon,
        lambda_factor=ik_config.solver.lambda_factor,
        rho_min=ik_config.solver.rho_min,
    )

    config = MotionPlannerConfig(
        num_timesteps=4,
        dt=0.25,
        use_cuda_graph=False,
        q_goal=q_goal,
        ik_config=ik_config,
    )
    config.trajectory_optimizer.max_iter = 50
    if q_goal:  # mirrors presets._q_goal_traj: the locked endpoint carries the goal, not the pose tasks
        config.trajectory_optimizer.lock_endpoints = True
        config.trajectory_optimizer.position_weight = 0.0
        config.trajectory_optimizer.orientation_weight = 0.0
    return config


def _load_panda_motion_plan(
    config: MotionPlannerConfig, device: str, scene: Optional[WarpScene] = None
) -> MotionPlanner:
    urdf: yourdfpy.URDF = load_robot_description("panda_description")
    xml_tree = urdf.write_xml()
    for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
    xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
    fixed_urdf = yourdfpy.URDF.load(StringIO(xml_str))
    assert fixed_urdf.validate()
    robot = Robot.load(
        fixed_urdf,
        load_collision_spheres=True,
        collision_spheres_path=str(Path(__file__).parents[2] / "fixtures/franka_collision_spheres.yaml"),
    )
    return MotionPlanner(config=config, robot=robot, ee_link_name_or_index="panda_hand", scene=scene, device=device)


def _goal_from_q(planner: MotionPlanner, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_wp = wp.from_numpy(q[None, None].astype(np.float32), dtype=wp.float32, device=planner.device)
    state = planner.robot.state(q=q_wp)
    state = planner.robot.forward_kinematics(state)
    xyz_wxyz_world_ee = state.T_world_link.numpy()[0, 0, planner.ee_link_index]
    return xyz_wxyz_world_ee[:3].copy(), xyz_wxyz_world_ee[3:].copy()


def _target(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([position, quaternion_wxyz], axis=-1)


def _make_box_mesh(device: str, center: np.ndarray, half_extent: float) -> wp.Mesh:
    x_coord, y_coord, z_coord = center.astype(np.float32)
    half_extent_float = float(half_extent)
    vertices = np.array(
        [
            [x_coord - half_extent_float, y_coord - half_extent_float, z_coord - half_extent_float],
            [x_coord + half_extent_float, y_coord - half_extent_float, z_coord - half_extent_float],
            [x_coord + half_extent_float, y_coord + half_extent_float, z_coord - half_extent_float],
            [x_coord - half_extent_float, y_coord + half_extent_float, z_coord - half_extent_float],
            [x_coord - half_extent_float, y_coord - half_extent_float, z_coord + half_extent_float],
            [x_coord + half_extent_float, y_coord - half_extent_float, z_coord + half_extent_float],
            [x_coord + half_extent_float, y_coord + half_extent_float, z_coord + half_extent_float],
            [x_coord - half_extent_float, y_coord + half_extent_float, z_coord + half_extent_float],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 5, 1],
            [0, 4, 5],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int32,
    )
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.reshape(-1), dtype=int, device=device),
    )


def _mesh_scene(meshes: List[wp.Mesh]) -> WarpScene:
    return WarpScene(1, "cpu").add(MeshGeom(meshes, np.array([0, len(meshes)], dtype=np.int32)))


def _get_sparse_jacobian(task, var) -> np.ndarray:
    pattern = task.compute_sparse_jacobian_pattern(var)
    values = task.compute_weighted_sparse_jacobian_values(var)
    jacobian = np.zeros((var.batch_size, task.residual_dim, var.tangent_dim), dtype=np.float32)
    row_indices = pattern.row_indices.numpy()
    col_indices = pattern.col_indices.numpy()
    values_np = values.numpy()

    for batch_idx in range(var.batch_size):
        for i in range(len(row_indices)):
            jacobian[batch_idx, row_indices[i], col_indices[i]] = values_np[batch_idx, i]

    return jacobian


class TestMotionPlanner:
    def test_collision_margin_tapers_near_goal(self):
        start_q = wp.zeros((3, 2), dtype=wp.float32, device="cpu")
        goal_q = wp.array([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]], dtype=wp.float32, device="cpu")
        margin = wp.empty(3, dtype=wp.float32, device="cpu")

        wp.launch(
            _compute_collision_margins_kernel,
            dim=3,
            inputs=[start_q, goal_q, 0.02, 1.0],
            outputs=[margin],
            device="cpu",
        )

        np.testing.assert_allclose(margin.numpy(), [0.0, 0.01, 0.02], atol=1e-7)

    def test_retry_seed_rng_varies_by_round(self):
        num_seeds = 2
        num_frames = 5
        num_dofs = 2
        start_q = wp.array(np.arange(6, dtype=np.float32).reshape(3, 2), dtype=wp.float32, device="cpu")
        target_q = wp.array(np.arange(12, dtype=np.float32).reshape(6, 2) * 0.1, dtype=wp.float32, device="cpu")
        active = wp.ones(num_dofs, dtype=wp.float32, device="cpu")
        limits = wp.array([[-1.0, 1.0]] * num_dofs, dtype=wp.float32, device="cpu")

        def build(rng_seed: int) -> np.ndarray:
            out = wp.empty((6, num_frames, num_dofs), dtype=wp.float32, device="cpu")
            wp.launch(
                _build_seed_trajectory_kernel,
                dim=out.size,
                inputs=[
                    start_q,
                    target_q,
                    rng_seed,
                    active,
                    limits,
                    0.2,
                    num_seeds,
                    num_frames,
                    num_dofs,
                    1,
                    out,
                ],
                device="cpu",
            )
            return out.numpy()

        round0 = build(0)
        np.testing.assert_array_equal(round0, build(0))  # deterministic per rng seed
        round1 = build(1)
        np.testing.assert_array_equal(round0[0::2], round1[0::2])  # direct seeds ignore the rng
        assert not np.array_equal(round0[1::2], round1[1::2])  # via-point seeds vary by round

    def test_retry_keep_best_and_early_stop(self):
        batch_size = 4
        best_score = wp.full(batch_size, 1.0e30, dtype=wp.float32, device="cpu")
        best_pen = wp.zeros(batch_size, dtype=wp.float32, device="cpu")
        condition = wp.zeros(1, dtype=wp.int32, device="cpu")
        pens = wp.zeros(batch_size, dtype=wp.float32, device="cpu")
        final_q = wp.zeros((batch_size, 1, 1), dtype=wp.float32, device="cpu")

        def select(scores: List[float], q_value: float, always_retry: int = 0):
            condition.fill_(0)
            scores_wp = wp.array(np.array(scores, dtype=np.float32), dtype=wp.float32, device="cpu")
            q = wp.full((batch_size, 1, 1), q_value, dtype=wp.float32, device="cpu")
            wp.launch(
                _set_best_retry_result_kernel,
                dim=batch_size,
                inputs=[scores_wp, pens, q, always_retry, best_score, best_pen, condition, final_q],
                device="cpu",
            )

        select([0.0, 2.0e6, 5.0, 3.0e6], 1.0)  # round 0: everything beats the 1e30 init
        np.testing.assert_array_equal(final_q.numpy()[:, 0, 0], np.full(batch_size, 1.0, dtype=np.float32))
        assert condition.numpy()[0] == 1  # rows 1 and 3 failed to reach

        rounds = 0

        def retry_round():
            nonlocal rounds
            rounds += 1
            select([1.0, 3.0, 9.0, 4.0], 2.0)

        for _ in range(3):
            wp.capture_if(condition, retry_round)

        assert rounds == 1  # all queries pass after one round -> remaining rounds are no-ops
        np.testing.assert_array_equal(final_q.numpy()[:, 0, 0], np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(best_score.numpy(), np.array([0.0, 3.0, 5.0, 4.0], dtype=np.float32))

        # shallow penetrations retry; deep or negligible ones do not; always_retry forces a round
        best_pen.assign(np.array([0.001, 0.0, 0.0, 0.0], dtype=np.float32))
        select([9.0, 9.0, 9.0, 9.0], 3.0)  # worse everywhere: keep-best rejects
        np.testing.assert_array_equal(final_q.numpy()[:, 0, 0], np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32))
        assert condition.numpy()[0] == 1
        best_pen.assign(np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32))
        select([9.0, 9.0, 9.0, 9.0], 3.0)
        assert condition.numpy()[0] == 0
        select([9.0, 9.0, 9.0, 9.0], 3.0, always_retry=1)
        assert condition.numpy()[0] == 1

    def test_laplacian_shortcut_never_increases_penetration(self):
        q = wp.array([[[0.0], [0.0], [1.0], [0.0], [0.0]]], dtype=wp.float32, device="cpu")
        candidate_q = wp.array([[[0.0], [0.25], [0.5], [0.75], [1.0]]], dtype=wp.float32, device="cpu")
        accept_mask = wp.zeros(1, dtype=wp.int32, device="cpu")
        wp.launch(
            _compute_laplacian_acceptance_kernel,
            dim=1,
            inputs=[
                q,
                candidate_q,
                wp.zeros(3, dtype=wp.float32, device="cpu"),
                wp.array([-1.0e-3] * 3, dtype=wp.float32, device="cpu"),
                wp.ones(1, dtype=wp.float32, device="cpu"),
                5,
                1,
                0.25,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0,
                1.0e6,
                accept_mask,
            ],
            device="cpu",
        )

        assert accept_mask.numpy()[0] == 0

    def test_endpoint_snap_rejects_inaccurate_ik(self):
        T_world_link = wp.array(
            [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
            dtype=wp_vec7,
            device="cpu",
        )
        targets = wp.array(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=wp_vec7,
            device="cpu",
        )
        accept_mask = wp.zeros(2, dtype=wp.int32, device="cpu")

        wp.launch(
            _compute_endpoint_snap_acceptance_kernel,
            dim=2,
            inputs=[T_world_link, targets, 0, 0.005, accept_mask],
            device="cpu",
        )

        np.testing.assert_array_equal(accept_mask.numpy(), [1, 0])

    def test_plan_and_update_signatures(self):
        plan_parameters = inspect.signature(MotionPlanner.solve_offline).parameters
        update_parameters = inspect.signature(MotionPlanner.solve_online).parameters
        target_parameters = inspect.signature(MotionPlanner._build_target_q_per_seed).parameters
        build_parameters = inspect.signature(MotionPlanner._build_initial_trajectories).parameters

        assert tuple(update_parameters) == (
            "self",
            "start_q",
            "T_world_target",
            "target_q",
            "scene_indices",
            "online_state",
        )
        assert tuple(target_parameters) == (
            "self",
            "T_world_target",
            "target_q",
            "scene_indices",
            "target_q_init",
            "rest_q",
        )
        assert tuple(build_parameters) == ("self", "start_q", "target_q_per_seed")
        gradient_source = inspect.getsource(MotionPlanner._solve_online_gradient)
        mppi_source = inspect.getsource(MotionPlanner._solve_online_mppi)
        assert gradient_source.count("_build_target_q_per_seed(") == 1
        assert "_ik_helper.solve(" not in gradient_source
        assert "if " not in mppi_source and "else" not in mppi_source
        assert "online_state" not in plan_parameters
        assert "_solve" not in MotionPlanner.__dict__
        for parameters in (plan_parameters, update_parameters):
            assert "num_seeds" not in parameters
            assert "max_iter" not in parameters

    def test_postprocessors_are_configured_as_an_ordered_tuple(self):
        assert MotionPlannerConfig().postprocessors == (LaplacianShortcut, EndpointSnap)
        for postprocessors in ((LaplacianShortcut,), (EndpointSnap,), (), (EndpointSnap, LaplacianShortcut)):
            config = _make_config()
            config.postprocessors = postprocessors
            planner = _load_panda_motion_plan(config, device="cpu")
            assert tuple(type(processor) for processor in planner._postprocessors) == postprocessors

    def test_warmup_fixes_batch_size(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")

        planner.warmup(2)
        planner.warmup(2)

        assert planner._batch_size == 2
        assert planner._default_scene_indices.shape == (2,)
        with pytest.raises(ValueError, match="Batch size mismatch: expected 2, got 1"):
            planner.warmup(1)

    def test_plan_and_update_fix_batch_size_lazily(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        plan_planner = _load_panda_motion_plan(config, device="cpu")
        q = plan_planner.robot.spec.zero_q.astype(np.float32)

        plan_planner.solve_offline_numpy(np.stack([q, q]), target_q=np.stack([q, q]))
        assert plan_planner._batch_size == 2
        q_wp = wp.from_numpy(q[None], dtype=wp.float32, device="cpu")
        with pytest.raises(ValueError, match="Batch size mismatch: expected 2, got 1"):
            plan_planner.solve_online(q_wp, target_q=q_wp)

        update_planner = _load_panda_motion_plan(config, device="cpu")
        update_planner.solve_online(q_wp, target_q=q_wp)
        assert update_planner._batch_size == 1
        with pytest.raises(ValueError, match="Batch size mismatch: expected 1, got 2"):
            update_planner.solve_offline_numpy(np.stack([q, q]), target_q=np.stack([q, q]))

    def test_solve_goal_ik_supports_single_and_batched_targets(self):
        single_planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = single_planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(single_planner, start_q)
        target = _target(goal_position, goal_quaternion)

        single_q = single_planner.solve_goal_ik_numpy(target)
        batch_planner = _load_panda_motion_plan(_make_config(), device="cpu")
        batch_target = np.stack([target, target])
        batch_q = batch_planner.solve_goal_ik_numpy(batch_target)
        batch_q_wp = batch_planner.solve_goal_ik(wp.from_numpy(batch_target, dtype=wp_vec7, device="cpu"))

        assert single_q.shape == (single_planner.robot.num_actuated_joints,)
        assert batch_q.shape == (2, batch_planner.robot.num_actuated_joints)
        assert batch_q_wp.shape == (2, batch_planner.robot.num_actuated_joints)
        with pytest.raises(ValueError, match="Batch size mismatch: expected 2, got 1"):
            batch_planner.solve_goal_ik_numpy(target)

    def test_plan_without_scene_meshes(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=_target(goal_position, goal_quaternion),
            target_q_init=start_q,
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)

    def test_plan_with_scene_meshes(self):
        scene = _mesh_scene(
            [_make_box_mesh("cpu", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), half_extent=0.1)]
        )
        planner = _load_panda_motion_plan(_make_config(), device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=_target(goal_position, goal_quaternion),
            target_q_init=start_q,
        )

        assert isinstance(result, MotionPlanResult)

    def test_plan_without_target_q_init_uses_ik(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=_target(goal_position, goal_quaternion),
        )

        assert isinstance(result, MotionPlanResult)

    def test_pose_goal_seed_expansion_is_explicit(self):
        config = _make_config()
        config.enable_retry = False
        config.trajectory_optimizer.num_seeds = 4
        config.ik_config.solver.stages[-1].num_seeds = 2
        with pytest.raises(ValueError, match="expand_ik_goal_seeds=True"):
            _load_panda_motion_plan(config, device="cpu")

        config.expand_ik_goal_seeds = True
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)
        start_q_wp = wp.from_numpy(start_q[None], dtype=wp.float32, device="cpu")
        target_wp = wp.from_numpy(_target(goal_position, goal_quaternion)[None], dtype=wp_vec7, device="cpu")
        planner.warmup(1)

        expanded = planner._build_target_q_per_seed(
            target_wp,
            None,
            planner._default_scene_indices,
            rest_q=start_q_wp,
        ).numpy()
        goals = planner._ik_goal_q.numpy()

        np.testing.assert_allclose(expanded, np.repeat(goals, 2, axis=0), atol=1e-6)

    def test_plan_q_goal_pins_endpoints(self):
        planner = _load_panda_motion_plan(_make_config(q_goal=True), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q = planner.robot.spec.midrange_q.astype(np.float32)

        result = planner.solve_offline_numpy(start_q=start_q, target_q=goal_q)

        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)
        np.testing.assert_allclose(result.q_traj[0, -1], goal_q, atol=1e-4)  # endpoint hard-pinned to goal_q
        np.testing.assert_allclose(result.q_traj[0, 0], start_q, atol=1e-4)

    def test_plan_q_goal_with_scene(self):
        scene = _mesh_scene(
            [_make_box_mesh("cpu", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), half_extent=0.1)]
        )
        planner = _load_panda_motion_plan(_make_config(q_goal=True), device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q = planner.robot.spec.midrange_q.astype(np.float32)

        result = planner.solve_offline_numpy(start_q=start_q, target_q=goal_q)

        np.testing.assert_allclose(result.q_traj[0, -1], goal_q, atol=1e-4)

    def test_plan_q_goal_batched_defaults_scene_indices(self):
        planner = _load_panda_motion_plan(_make_config(q_goal=True), device="cpu")
        start_q = np.stack([planner.robot.spec.zero_q, planner.robot.spec.midrange_q]).astype(np.float32)
        goal_q = np.stack([planner.robot.spec.midrange_q, planner.robot.spec.zero_q]).astype(np.float32)

        result = planner.solve_offline_numpy(start_q=start_q, target_q=goal_q)

        assert result.q_traj.shape[0] == 2
        np.testing.assert_allclose(result.q_traj[:, -1], goal_q, atol=1e-4)

    def test_plan_q_goal_rejects_pose_args(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q = planner.robot.spec.midrange_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, goal_q)

        with pytest.raises(ValueError):
            planner.solve_offline_numpy(
                start_q=start_q, T_world_target=_target(goal_position, goal_quaternion), target_q=goal_q
            )
        with pytest.raises(ValueError, match="q_goal"):  # q-goals need a q_goal=True planner
            planner.solve_offline_numpy(start_q=start_q, target_q=goal_q)

    def test_plan_returns_failure_trajectory(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        goal_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        result = planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=_target(goal_position, goal_quaternion),
            target_q_init=start_q,
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)

    def test_end_frame_pose_task_sparse_jacobian_matches_autodiff(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        q_traj = np.stack([start_q] * 5, axis=0)[None]
        q_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=planner.device, requires_grad=True)
        robot_state = planner.robot.state(q=q_wp)
        planner.robot.forward_kinematics(robot_state)
        xyz_wxyz_world_goal = robot_state.T_world_link.numpy()[:, -1, planner.ee_link_index]
        var = VarValues(robot=robot_state)
        dense = FrameTask(
            robot=planner.robot,
            frame_index=planner.ee_link_index,
            T_world_target=wp.from_numpy(xyz_wxyz_world_goal.astype(np.float32), dtype=wp_vec7, device=planner.device),
            position_weight=50.0,
            orientation_weight=20.0,
        )
        task = TrajectoryTask(dense, num_frames=5, frame_indices=[-1])
        task.init_buffers(planner.device)

        jacobian_sparse = _get_sparse_jacobian(task, var)
        jacobian_autodiff = np.asarray(autodiff_weighted_jacobian(task, var), dtype=np.float32)
        np.testing.assert_allclose(jacobian_sparse, jacobian_autodiff, atol=5e-5, rtol=1e-5)

    @pytest.mark.parametrize("penalty", ["smooth", "plain", "surface_distance"])
    def test_box_collision_task_sparse_jacobian_matches_fd(self, penalty):
        """The collision Jacobian's gradient is the analytical SDF closest-point direction, which a
        Warp tape can't see — so finite differences is the ground truth. A thin slab (huge in xy,
        thin in z) keeps the nearest box face a z-face, so the gradient is smooth (no edge kinks)."""
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 3
        q_traj = np.stack([robot.spec.midrange_q.astype(np.float32)] * num_frames)[None]
        state = robot.state(q=wp.from_numpy(q_traj, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        z_level = float(state.collision_sphere_centers_world.numpy().reshape(num_frames, -1, 3)[1, :, 2].mean())
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = z_level
        scene = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[5.0, 5.0, 0.15]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cpu"),
            )
        )
        task = TrajectoryCollisionTask(
            robot=robot,
            scene=scene,
            num_frames=num_frames,
            weight=3.0,
            margin=0.05,
            penalty=penalty,
        )
        assert task._geom == "primitive"  # box scene → the primitive collision path
        var = VarValues(robot=state)
        jac = _get_sparse_jacobian(task, var)[0]

        tangent_dim = var.tangent_dim
        eps = 1e-4
        zero = np.zeros((1, tangent_dim), dtype=np.float32)

        def residual(vel: np.ndarray) -> np.ndarray:
            v = var.integrate(wp.from_numpy(vel, dtype=wp.float32, device="cpu"))
            return task.compute_weighted_residual(v).numpy()[0]

        active = residual(zero) > 1e-6
        jac_fd = np.zeros_like(jac)
        for k in range(tangent_dim):
            plus, minus = zero.copy(), zero.copy()
            plus[0, k], minus[0, k] = eps, -eps
            jac_fd[:, k] = (residual(plus) - residual(minus)) / (2.0 * eps)
        assert active.sum() > 0
        np.testing.assert_allclose(jac[active], jac_fd[active], atol=2e-2)

    @pytest.mark.parametrize("penalty", ["smooth", "plain", "surface_distance"])
    def test_dense_and_trajectory_collision_residuals_match(self, penalty):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        q = robot.spec.midrange_q.astype(np.float32)[None]
        dense_state = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(dense_state)
        robot.transform_collision_spheres(dense_state)

        z_level = float(dense_state.collision_sphere_centers_world.numpy()[0, :, 2].mean())
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = z_level
        scene = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[5.0, 5.0, 0.15]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cpu"),
            )
        )
        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        dense_task = SceneCollisionTask(
            robot=robot, scene=scene, sphere_indices=sphere_indices, weight=3.0, margin=0.05, penalty=penalty
        )
        trajectory_task = TrajectoryCollisionTask(
            robot=robot,
            scene=scene,
            num_frames=1,
            sphere_indices=sphere_indices,
            weight=3.0,
            margin=0.05,
            penalty=penalty,
        )
        trajectory_state = robot.state(q=wp.from_numpy(q[:, None], dtype=wp.float32, device="cpu"))

        dense_residual = dense_task.compute_weighted_residual(VarValues(robot=dense_state)).numpy()
        trajectory_residual = trajectory_task.compute_weighted_residual(VarValues(robot=trajectory_state)).numpy()
        np.testing.assert_allclose(trajectory_residual, dense_residual, atol=1e-6)

    @pytest.mark.parametrize("penalty", ["smooth", "plain", "surface_distance"])
    def test_swept_collision_matches_static_frames(self, penalty):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 3
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, num_frames, 1))
        state = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)

        z_level = float(state.collision_sphere_centers_world.numpy().reshape(num_frames, -1, 3)[:, :, 2].mean())
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = z_level
        scene = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[5.0, 5.0, 0.15]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cpu"),
            )
        )
        task_args = dict(
            robot=robot,
            scene=scene,
            num_frames=num_frames,
            weight=3.0,
            margin=0.05,
            penalty=penalty,
        )
        plain_task = TrajectoryCollisionTask(**task_args)
        swept_task = TrajectoryCollisionTask(**task_args, sweep_steps=2)
        var_values = VarValues(robot=state)

        plain_residual = plain_task.compute_weighted_residual(var_values).numpy()
        swept_residual = swept_task.compute_weighted_residual(var_values).numpy()
        np.testing.assert_allclose(swept_residual, plain_residual, atol=1e-6)

    def test_empty_collision_task_residual_and_sparse_jacobian_are_zero(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 3
        q_traj = np.stack([robot.spec.midrange_q.astype(np.float32)] * num_frames)[None]
        state = robot.state(q=wp.from_numpy(q_traj, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        task = TrajectoryCollisionTask(
            robot=robot,
            scene=WarpScene(1, "cpu"),
            num_frames=num_frames,
            weight=3.0,
            margin=0.05,
        )
        assert task._geom == "empty"
        var = VarValues(robot=state)

        np.testing.assert_allclose(task.compute_weighted_residual(var).numpy(), 0.0, atol=1e-7)
        np.testing.assert_allclose(_get_sparse_jacobian(task, var), 0.0, atol=1e-7)

    def test_start_and_end_config_tasks_target_expected_timesteps(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        num_frames = 5
        q_traj = np.stack([start_q] * num_frames, axis=0)[None]
        q_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=planner.device)
        var = VarValues(robot=planner.robot.state(q=q_wp))
        target_q = start_q.copy()
        target_q[0] = 0.2

        start_task = TrajectoryTask(
            RestTask(robot=planner.robot, rest_q=target_q, weight=2.0), num_frames, frame_indices=[0]
        )
        end_task = TrajectoryTask(
            RestTask(robot=planner.robot, rest_q=target_q, weight=3.0), num_frames, frame_indices=[-1]
        )
        for task in (start_task, end_task):
            task.init_buffers(planner.device)

        start_residual = start_task.compute_weighted_residual(var).numpy()
        end_residual = end_task.compute_weighted_residual(var).numpy()
        np.testing.assert_allclose(start_residual[0], 2.0 * (start_q - target_q), atol=1e-6)
        np.testing.assert_allclose(end_residual[0], 3.0 * (start_q - target_q), atol=1e-6)

        start_pattern = start_task.compute_sparse_jacobian_pattern(var)
        end_pattern = end_task.compute_sparse_jacobian_pattern(var)
        num_dofs = planner.robot.num_actuated_joints
        assert start_pattern.col_indices.numpy().max() < num_dofs
        assert end_pattern.col_indices.numpy().min() >= (num_frames - 1) * num_dofs
        for task, weight in ((start_task, 2.0), (end_task, 3.0)):
            values = task.compute_weighted_sparse_jacobian_values(var).numpy().reshape(1, num_dofs, num_dofs)
            np.testing.assert_allclose(values[0], weight * np.eye(num_dofs), atol=1e-6)

    def test_reuses_single_query_optimizer(self):
        scene = _mesh_scene(
            [_make_box_mesh("cpu", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), half_extent=0.1)]
        )
        config = _make_config()
        config.ik_config.solver.cuda_graph_mode = "full"
        planner = _load_panda_motion_plan(config, device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=_target(goal_position, goal_quaternion),
        )
        trajectory_optimizer = planner._trajectory_optimizer
        ik_helper = planner._ik_helper

        mesh = scene.geoms[0]
        assert isinstance(mesh, MeshGeom)
        mesh.update([_make_box_mesh("cpu", center=np.array([1.5, 0.0, 0.0], dtype=np.float32), half_extent=0.1)])

        planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=_target(goal_position, goal_quaternion),
        )

        assert planner._scene is scene
        assert planner._ik_helper is ik_helper
        assert planner._trajectory_optimizer is trajectory_optimizer
        assert planner._ik_helper.config.solver.cuda_graph_mode == "full"

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_shortcut_recaptures_after_scene_update(self):
        config = _make_config(q_goal=True)
        config.use_cuda_graph = True
        config.enable_retry = False
        config.postprocessors = (LaplacianShortcut,)
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        scene = WarpScene(1, "cuda:0").add(
            MeshGeom(
                [_make_box_mesh("cuda:0", np.array([2.0, 0.0, 0.0], dtype=np.float32), 0.1)],
                np.array([0, 1], dtype=np.int32),
            )
        )
        planner = _load_panda_motion_plan(config, device="cuda:0", scene=scene)
        q = planner.robot.spec.zero_q.astype(np.float32)

        planner.solve_offline_numpy(start_q=q, target_q=q)
        shortcut = planner._postprocessors[0]
        mesh = scene.geoms[0]
        assert isinstance(mesh, MeshGeom)
        for x in np.linspace(1.5, 1.8, 8):
            old_graph = shortcut._graph
            mesh.update([_make_box_mesh("cuda:0", np.array([x, 0.0, 0.0], dtype=np.float32), 0.1)])
            assert shortcut._graph is not old_graph
            updated_graph = shortcut._graph
            planner.solve_offline_numpy(start_q=q, target_q=q)
            assert shortcut._graph is updated_graph

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_shortcut_reuses_graph_after_in_place_scene_update(self):
        config = _make_config(q_goal=True)
        config.use_cuda_graph = True
        config.enable_retry = False
        config.postprocessors = (LaplacianShortcut,)
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        box_pose = np.eye(4, dtype=np.float32)[None]
        box = BoxGeom(
            np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            poses=wp.from_numpy(box_pose, dtype=wp.mat44, device="cuda:0"),
        )
        planner = _load_panda_motion_plan(config, device="cuda:0", scene=WarpScene(1, "cuda:0").add(box))
        q = planner.robot.spec.zero_q.astype(np.float32)

        planner.solve_offline_numpy(start_q=q, target_q=q)
        shortcut = planner._postprocessors[0]
        old_graph = shortcut._graph
        box_pose[0, 0, 3] = 0.2
        box.update(poses=wp.from_numpy(box_pose, dtype=wp.mat44, device="cuda:0"))
        planner.solve_offline_numpy(start_q=q, target_q=q)

        assert shortcut._graph is old_graph

    def test_scene_bound_at_construction_is_reused(self):
        config = _make_config(q_goal=True)
        config.postprocessors = ()
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        scene = WarpScene(2, "cpu")
        planner = _load_panda_motion_plan(config, device="cpu", scene=scene)
        q = planner.robot.spec.zero_q.astype(np.float32)

        planner.solve_offline_numpy(start_q=q, target_q=q, scene_indices=np.array([0], dtype=np.int32))
        trajectory_optimizer = planner._trajectory_optimizer
        planner.solve_offline_numpy(start_q=q, target_q=q, scene_indices=np.array([1], dtype=np.int32))
        np.testing.assert_array_equal(trajectory_optimizer._collision_tasks[0]._scene_indices_wp.numpy(), [1])
        assert planner._scene is scene

        empty_planner = _load_panda_motion_plan(config, device="cpu")
        empty_planner.solve_offline_numpy(start_q=q, target_q=q)
        empty_scene = empty_planner._scene
        empty_planner.solve_offline_numpy(start_q=q, target_q=q)
        assert empty_planner._scene is empty_scene

    def test_fixed_solver_config_builds_and_reuses_one_optimizer(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 2
        config.trajectory_optimizer.max_iter = 2
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target_q = start_q.copy()
        target_q[:2] = (0.1, -0.1)

        planner.solve_offline_numpy(start_q=start_q, target_q=target_q)
        trajectory_optimizer = planner._trajectory_optimizer

        assert trajectory_optimizer._num_seeds == 2
        assert trajectory_optimizer._solver is not None
        assert trajectory_optimizer._solver._updaters[0].config.max_iter == 2

        planner.solve_offline_numpy(start_q=start_q, target_q=target_q)

        assert planner._trajectory_optimizer is trajectory_optimizer

    def test_plan_retries_but_update_runs_once(self, monkeypatch: pytest.MonkeyPatch):
        config = _make_config()
        config.always_finetune_retry = True
        config.postprocessors = ()
        config.trajectory_optimizer.solver = "lbfgs"
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 2
        config.ik_config.solver.stages[-1].num_seeds = 2
        config.trajectory_optimizer.max_iter = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)
        target = _target(goal_position, goal_quaternion)
        solve_calls = 0
        original_solve = GradientTrajectoryOptimizer.solve

        def solve(
            optimizer: GradientTrajectoryOptimizer,
            start_q: wp.array,
            T_world_target: wp.array,
            target_q: Optional[wp.array],
            traj_q_init: wp.array,
            target_q_per_seed: wp.array,
            scene_indices: wp.array,
            q_history: Optional[wp.array] = None,
            history_count: int = 0,
        ) -> Tuple[RobotState, wp.array]:
            nonlocal solve_calls
            solve_calls += 1
            return original_solve(
                optimizer,
                start_q,
                T_world_target,
                target_q,
                traj_q_init,
                target_q_per_seed,
                scene_indices,
                q_history,
                history_count,
            )

        monkeypatch.setattr(GradientTrajectoryOptimizer, "solve", solve)

        planner.solve_offline_numpy(start_q, target, target_q_init=start_q)
        assert solve_calls == 2

        start_q_wp = wp.from_numpy(start_q[None], dtype=wp.float32, device="cpu")
        target_wp = wp.from_numpy(target[None], dtype=wp_vec7, device="cpu")
        planner.solve_online(start_q_wp, target_wp)
        assert solve_calls == 3

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_cuda_retry_replays_without_error(self):
        config = _make_config()
        config.use_cuda_graph = True
        config.always_finetune_retry = True
        config.trajectory_optimizer.solver = "lbfgs"
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 2
        config.ik_config.solver.stages[-1].num_seeds = 2
        config.trajectory_optimizer.max_iter = 1
        planner = _load_panda_motion_plan(config, device="cuda:0")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)
        target = _target(goal_position, goal_quaternion)

        for _ in range(40):
            planner.solve_offline_numpy(start_q, target, target_q_init=start_q)

        assert planner._retry_condition is not None

    def test_plan_result_does_not_compute_metrics(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q = start_q.copy()
        goal_q[0] = 0.2

        result = planner.solve_offline_numpy(start_q=start_q, target_q=goal_q)

        assert not hasattr(result, "metrics")
        np.testing.assert_allclose(result.q_traj[0, -1], goal_q, atol=1e-4)

    def test_motion_plan_evaluator_is_opt_in(self):
        planner = _load_panda_motion_plan(_make_config(q_goal=True), device="cpu")
        q = planner.robot.spec.zero_q.astype(np.float32)
        q_traj = wp.from_numpy(np.tile(q, (1, 2, 1)), dtype=wp.float32, device="cpu")
        target_q = wp.from_numpy(q[None], dtype=wp.float32, device="cpu")
        evaluator = MotionPlanEvaluator(planner.robot, planner.ee_link_index, 2, 0.005, 0.05, 0.01, "cpu")

        metrics = evaluator.evaluate(q_traj, target_q=target_q)

        assert metrics.success.numpy().tolist() == [True]
        np.testing.assert_array_equal(metrics.position_error_m.numpy(), [0.0])

    def test_plan_always_returns_owned_numpy_arrays(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)

        result = planner.solve_offline_numpy(start_q=start_q, target_q=start_q)
        first_q = result.q_traj.copy()
        goal_q = start_q.copy()
        goal_q[0] = 0.2
        second = planner.solve_offline_numpy(start_q=start_q, target_q=goal_q)

        assert isinstance(result.q_traj, np.ndarray)
        assert result.motion_time.shape == (1,)
        np.testing.assert_array_equal(result.q_traj, first_q)
        np.testing.assert_allclose(second.q_traj[0, -1], goal_q, atol=1e-4)

    def test_plan_returns_warp_arrays(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = wp.from_numpy(planner.robot.spec.zero_q.astype(np.float32)[None], dtype=wp.float32, device="cpu")

        result = planner.solve_offline(start_q=start_q, target_q=start_q)

        for value in (result.q_traj, result.motion_time):
            assert isinstance(value, wp.array)

    def test_plan_torch_returns_owned_tensors(self):
        torch = pytest.importorskip("torch")
        from robokit.xform.numpy import quaternion_to_matrix

        def make_planner(q_goal: bool) -> MotionPlanner:
            config = _make_config(q_goal)
            config.enable_retry = False
            config.postprocessors = ()
            config.trajectory_optimizer.collision_weight = 0.0
            config.trajectory_optimizer.self_collision_weight = 0.0
            config.trajectory_optimizer.num_seeds = 1
            config.trajectory_optimizer.max_iter = 1
            return _load_panda_motion_plan(config, device="cpu")

        planner = make_planner(q_goal=True)
        start_q = torch.from_numpy(planner.robot.spec.zero_q.astype(np.float32))

        result = planner.solve_offline_torch(start_q=start_q, target_q=start_q)
        first_q = result.q_traj.clone()
        goal_q = start_q.clone()
        goal_q[0] = 0.2
        second = planner.solve_offline_torch(start_q=start_q, target_q=goal_q)

        assert isinstance(result.q_traj, torch.Tensor)
        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)
        torch.testing.assert_close(result.q_traj, first_q)
        torch.testing.assert_close(second.q_traj[0, -1], goal_q)

        position, wxyz = _goal_from_q(planner, goal_q.numpy())
        T_world_target = torch.eye(4, dtype=torch.float32)
        T_world_target[:3, :3] = torch.from_numpy(quaternion_to_matrix(wxyz).astype(np.float32))
        T_world_target[:3, 3] = torch.from_numpy(position)
        pose_result = make_planner(q_goal=False).solve_offline_torch(
            start_q=start_q,
            T_world_target=T_world_target,
            target_q_init=goal_q,
        )
        assert pose_result.q_traj.shape == second.q_traj.shape

    def test_update_returns_device_trajectory_without_metrics(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = wp.from_numpy(planner.robot.spec.zero_q.astype(np.float32)[None], dtype=wp.float32, device="cpu")

        result = planner.solve_online(start_q=start_q, target_q=start_q)

        assert isinstance(result.q_traj, wp.array)
        assert result.q_traj.ptr == planner._final_q_buf.ptr
        assert result.online_state is not None

    def test_solve_online_numpy_returns_owned_trajectory(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 1
        config.trajectory_optimizer.max_iter = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target_q = start_q.copy()
        target_q[0] = 0.2

        result = planner.solve_online_numpy(start_q=start_q, target_q=target_q)

        assert isinstance(result.q_traj, np.ndarray)
        assert result.q_traj.flags.owndata
        assert result.q_traj.shape == (1, config.num_timesteps + 1, planner.robot.num_actuated_joints)
        assert result.online_state is not None

    def test_gradient_online_state_keeps_history_without_trajectory(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 3
        config.trajectory_optimizer.max_iter = 0
        config.trajectory_optimizer.accel_smoothness_weight = 1.0
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target_q = start_q.copy()
        target_q[0] = 0.2
        first = planner.solve_online_numpy(start_q, target_q=target_q)
        state = first.online_state
        assert state is not None
        assert state.prev_q_traj is None
        assert state.prev_goal_q is not None
        assert state.history_count == 1

        current_q = start_q.copy()
        current_q[0] = 0.03
        planner.solve_online_numpy(current_q, target_q=target_q, online_state=state)
        next_q = current_q.copy()
        next_q[0] = 0.05
        third = planner.solve_online_numpy(next_q, target_q=target_q, online_state=state)

        expected_seed = np.linspace(next_q, target_q, config.num_timesteps + 1, dtype=np.float32)
        np.testing.assert_allclose(planner._traj_q_init_buf.numpy()[0], expected_seed, atol=1e-6)
        np.testing.assert_allclose(state.q_history.numpy()[0, :, 0], [0.0, 0.03, 0.05], atol=1e-6)
        assert state.history_count == 3
        assert all(np.all(task.history_valid.numpy() == 1.0) for task in planner._trajectory_optimizer._history_tasks)
        assert third.online_state is state

    def test_update_q_goal_reaches_target(self):
        config = _make_config(q_goal=True)
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer.lock_endpoints = False
        config.trajectory_optimizer.start_config_weight = 100.0
        config.trajectory_optimizer.goal_config_weight = 100.0
        config.trajectory_optimizer.smoothness_weight = 0.0
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.velocity_limit_weight = 0.0
        config.trajectory_optimizer.position_limit_weight = 0.0
        config.trajectory_optimizer.rest_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.num_seeds = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q_np = planner.robot.spec.zero_q.astype(np.float32)
        target_q_np = start_q_np.copy()
        target_q_np[0] = 0.4
        start_q = wp.from_numpy(start_q_np[None], dtype=wp.float32, device="cpu")
        target_q = wp.from_numpy(target_q_np[None], dtype=wp.float32, device="cpu")

        result = planner.solve_online(start_q=start_q, target_q=target_q)

        np.testing.assert_allclose(result.q_traj.numpy()[0, -1], target_q_np, atol=1e-3)

    def test_plan_batch_with_shared_scene(self):
        shared_scene = _mesh_scene(
            [_make_box_mesh("cpu", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), half_extent=0.1)]
        )
        config = _make_config()
        config.ik_config.solver.cuda_graph_mode = "full"
        planner = _load_panda_motion_plan(config, device="cpu", scene=shared_scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        offset_q = start_q.copy()
        offset_q[3] = 0.2
        goal_position_a, goal_quaternion_a = _goal_from_q(planner, start_q)
        goal_position_b, goal_quaternion_b = _goal_from_q(planner, offset_q)

        result = planner.solve_offline_numpy(
            start_q=np.stack([start_q, offset_q], axis=0),
            T_world_target=_target(
                np.stack([goal_position_a, goal_position_b]), np.stack([goal_quaternion_a, goal_quaternion_b])
            ),
            scene_indices=np.array([0, 0], dtype=np.int32),
            target_q_init=np.stack([start_q, offset_q], axis=0),
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape == (2, 5, planner.robot.num_actuated_joints)
        assert planner._ik_helper.config.solver.cuda_graph_mode == "iter"

    def test_plan_batch_with_per_query_scenes(self):
        scene = WarpScene(2, "cpu").add(
            MeshGeom(
                [_make_box_mesh("cpu", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), half_extent=0.1)],
                np.array([0, 0, 1], dtype=np.int32),
            )
        )
        planner = _load_panda_motion_plan(_make_config(), device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            start_q=np.stack([start_q, start_q], axis=0),
            T_world_target=_target(
                np.stack([goal_position, goal_position]), np.stack([goal_quaternion, goal_quaternion])
            ),
            scene_indices=np.array([0, 1], dtype=np.int32),
            target_q_init=np.stack([start_q, start_q], axis=0),
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape == (2, 5, planner.robot.num_actuated_joints)
        assert result.motion_time.shape == (2,)

    @pytest.mark.slow
    def test_plan_batch_active_joint_mask_applies_to_goal_ik_and_trajectory(self):
        config = _make_config()
        config.trajectory_optimizer.position_weight = 0.0
        config.trajectory_optimizer.orientation_weight = 0.0
        config.trajectory_optimizer.smoothness_weight = 0.0
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.velocity_limit_weight = 0.0
        config.trajectory_optimizer.position_limit_weight = 0.0
        config.trajectory_optimizer.rest_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        config.trajectory_optimizer.goal_config_weight = 100.0
        scene = WarpScene(2, "cpu").add(
            MeshGeom(
                [_make_box_mesh("cpu", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), half_extent=0.1)],
                np.array([0, 0, 1], dtype=np.int32),
            )
        )
        planner = _load_panda_motion_plan(config, device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q_a = start_q.copy()
        goal_q_b = start_q.copy()
        active_joint_indices = [0, 1]
        joint_mask = np.zeros(planner.robot.num_actuated_joints, dtype=np.float32)
        joint_mask[active_joint_indices] = 1.0
        planner.set_active_joint_mask(joint_mask)
        goal_q_a[0] = 0.2
        goal_q_a[1] = -0.1
        goal_q_b[0] = 0.15
        goal_q_b[1] = 0.2
        goal_position_a, goal_quaternion_a = _goal_from_q(planner, goal_q_a)
        goal_position_b, goal_quaternion_b = _goal_from_q(planner, goal_q_b)

        result = planner.solve_offline_numpy(
            start_q=np.stack([start_q, start_q], axis=0),
            T_world_target=_target(
                np.stack([goal_position_a, goal_position_b]), np.stack([goal_quaternion_a, goal_quaternion_b])
            ),
            scene_indices=np.array([0, 1], dtype=np.int32),
        )

        assert isinstance(result, MotionPlanResult)
        assert np.all(np.linalg.norm(result.q_traj[:, -1, active_joint_indices], axis=1) > 0.1)
        inactive_q = np.broadcast_to(start_q[None, None, 2:], result.q_traj[:, :, 2:].shape)
        np.testing.assert_allclose(result.q_traj[:, :, 2:], inactive_q, atol=1e-6)
        np.testing.assert_array_equal(planner._ik_helper._active_joint_mask.numpy(), joint_mask)

    def test_default_goal_config_is_opt_in_and_disabled_collision_tasks_are_skipped(self):
        config = _make_config()
        config.trajectory_optimizer.collision_weight = 0.0
        config.trajectory_optimizer.self_collision_weight = 0.0
        scene = _mesh_scene(
            [_make_box_mesh("cpu", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), half_extent=0.1)]
        )
        planner = _load_panda_motion_plan(config, device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=_target(goal_position, goal_quaternion),
            target_q_init=start_q,
        )
        trajectory_optimizer = planner._trajectory_optimizer

        assert trajectory_optimizer._goal_task is None
        assert len(trajectory_optimizer._collision_tasks) == 0
        assert "TrajectorySelfCollisionTask" not in [type(task).__name__ for task in trajectory_optimizer._tasks]

    @pytest.mark.slow
    def test_set_active_joint_mask_updates_ik_and_reuses_optimizer(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)
        T_world_target = _target(goal_position, goal_quaternion)

        planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=T_world_target,
        )
        trajectory_optimizer = planner._trajectory_optimizer
        joint_mask = np.zeros(planner.robot.num_actuated_joints, dtype=np.float32)
        joint_mask[:2] = 1.0
        planner.set_active_joint_mask(joint_mask.tolist())
        np.testing.assert_array_equal(planner._ik_helper._active_joint_mask.numpy(), joint_mask)
        snap = next(processor for processor in planner._postprocessors if isinstance(processor, EndpointSnap))
        np.testing.assert_array_equal(snap._bufs[0]._active_joint_mask.numpy(), joint_mask)
        planner.solve_offline_numpy(
            start_q=start_q,
            T_world_target=T_world_target,
            target_q_init=start_q,
        )
        planner.set_active_joint_mask(joint_mask)
        planner.solve_offline_numpy(start_q=start_q, T_world_target=T_world_target, target_q_init=start_q)

        assert planner._trajectory_optimizer is trajectory_optimizer  # mask changes never rebuild

    def test_plan_batch_matches_single_query_planner_on_seeded_queries(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        offset_q = start_q.copy()
        offset_q[3] = 0.2
        goal_position_a, goal_quaternion_a = _goal_from_q(planner, start_q)
        goal_position_b, goal_quaternion_b = _goal_from_q(planner, offset_q)

        batched_result = planner.solve_offline_numpy(
            start_q=np.stack([start_q, offset_q], axis=0),
            T_world_target=_target(
                np.stack([goal_position_a, goal_position_b]), np.stack([goal_quaternion_a, goal_quaternion_b])
            ),
            scene_indices=np.array([0, 0], dtype=np.int32),
            target_q_init=np.stack([start_q, offset_q], axis=0),
        )
        planner_a = _load_panda_motion_plan(_make_config(), device="cpu")
        result_a = planner_a.solve_offline_numpy(
            start_q=start_q,
            T_world_target=_target(goal_position_a, goal_quaternion_a),
            target_q_init=start_q,
        )
        planner_b = _load_panda_motion_plan(_make_config(), device="cpu")
        result_b = planner_b.solve_offline_numpy(
            start_q=offset_q,
            T_world_target=_target(goal_position_b, goal_quaternion_b),
            target_q_init=offset_q,
        )

        assert isinstance(batched_result, MotionPlanResult)
        # Batched planning now runs the same shortcut/snap postprocessing as single-query.
        np.testing.assert_allclose(batched_result.q_traj[0], result_a.q_traj[0], atol=5e-1)
        np.testing.assert_allclose(batched_result.q_traj[1], result_b.q_traj[0], atol=5e-1)

    def test_plan_batch_without_target_q_init_uses_ik(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        offset_q = start_q.copy()
        offset_q[3] = 0.2
        goal_position_a, goal_quaternion_a = _goal_from_q(planner, start_q)
        goal_position_b, goal_quaternion_b = _goal_from_q(planner, offset_q)

        result = planner.solve_offline_numpy(
            start_q=np.stack([start_q, offset_q], axis=0),
            T_world_target=_target(
                np.stack([goal_position_a, goal_position_b]), np.stack([goal_quaternion_a, goal_quaternion_b])
            ),
            scene_indices=np.array([0, 0], dtype=np.int32),
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape[0] == 2

    def test_treats_same_mesh_shape_at_different_poses_as_distinct_scenes(self):
        scene = WarpScene(2, "cpu").add(
            MeshGeom(
                [
                    _make_box_mesh("cpu", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), half_extent=0.1),
                    _make_box_mesh("cpu", center=np.array([2.5, 0.0, 0.0], dtype=np.float32), half_extent=0.1),
                ],
                np.array([0, 1, 2], dtype=np.int32),
            )
        )
        planner = _load_panda_motion_plan(_make_config(), device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        planner.solve_offline_numpy(
            start_q=np.stack([start_q, start_q], axis=0),
            T_world_target=_target(
                np.stack([goal_position, goal_position]), np.stack([goal_quaternion, goal_quaternion])
            ),
            scene_indices=np.array([0, 1], dtype=np.int32),
            target_q_init=np.stack([start_q, start_q], axis=0),
        )

        assert planner._trajectory_optimizer._collision_tasks  # one union optimizer served both scenes

    def test_plan_batch_defaults_scene_indices(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            start_q=np.stack([start_q, start_q], axis=0),
            T_world_target=_target(
                np.stack([goal_position, goal_position]), np.stack([goal_quaternion, goal_quaternion])
            ),
            target_q_init=np.stack([start_q, start_q], axis=0),
        )

        assert result.q_traj.shape[0] == 2

    def test_plan_batch_rejects_out_of_range_scene_index(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_position, goal_quaternion = _goal_from_q(planner, start_q)

        with pytest.raises(ValueError, match="out-of-range"):
            planner.solve_offline_numpy(
                start_q=np.stack([start_q, start_q], axis=0),
                T_world_target=_target(
                    np.stack([goal_position, goal_position]), np.stack([goal_quaternion, goal_quaternion])
                ),
                scene_indices=np.array([0, 1], dtype=np.int32),
                target_q_init=np.stack([start_q, start_q], axis=0),
            )

    def test_static_mixed_scene_expands_collision_tasks(self):
        """A heterogeneous scene creates one collision task per geometry and remains unchanged by solve()."""
        torch = pytest.importorskip("torch")

        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 7
        dofs = robot.num_actuated_joints
        mid = robot.spec.midrange_q.astype(np.float32)

        state = robot.state(q=wp.from_numpy(mid[None], dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        centers = state.collision_sphere_centers_world.numpy().reshape(-1, 3)
        mesh_center = centers[len(centers) // 2].astype(np.float32)
        box_center = centers[len(centers) // 3].astype(np.float32)

        scene = WarpScene(1, "cpu").add(
            MeshGeom([_make_box_mesh("cpu", mesh_center, 0.08)], np.array([0, 1], dtype=np.int32))
        )
        box_pose = np.eye(4, dtype=np.float32)
        box_pose[:3, 3] = box_center
        scene.add(
            BoxGeom(
                np.array([[0.08, 0.08, 0.08]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(box_pose[None], dtype=wp.mat44, device="cpu"),
            )
        )

        config = GradientTrajectoryOptimizerConfig(
            position_weight=0.0,
            orientation_weight=0.0,
            smoothness_weight=1.0,
            collision_weight=50.0,
            lm_lambda=1.0,
            start_config_weight=100.0,
            goal_config_weight=100.0,
            velocity_limit_weight=0.0,
            position_limit_weight=0.0,
            rest_weight=0.0,
            self_collision_weight=0.0,
            collision_margin=0.02,
            max_iter=80,
            use_cuda_graph=False,
        )
        optimizer = GradientTrajectoryOptimizer(
            config=config,
            robot=robot,
            ee_link_name_or_index=planner.ee_link_index,
            scene=scene,
            num_frames=num_frames,
            dt=0.25,
            device="cpu",
        )
        optimizer.warmup(1)
        assert [t._geom for t in optimizer._collision_tasks] == ["mesh", "primitive"]

        def probe(scene: WarpScene, center: np.ndarray) -> np.ndarray:
            pts = torch.from_numpy(center[None].astype(np.float32))
            sdf, _, _ = scene.query_sdf_torch(pts, torch.tensor([0, pts.shape[0]], dtype=torch.int32))
            return sdf.detach().cpu().numpy().copy()

        mesh_sdf_before = probe(scene, mesh_center + 0.2)
        box_sdf_before = probe(scene, box_center + 0.2)

        def collision_l2(q_traj: np.ndarray) -> float:
            s = robot.state(q=wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device="cpu"))
            robot.forward_kinematics(s)
            var = VarValues(robot=s)
            total = 0.0
            for task in optimizer._collision_tasks:
                total += float(np.sum(task.compute_weighted_residual(var).numpy() ** 2))
            return total

        traj_q_init = np.broadcast_to(mid[None, None], (1, num_frames, dofs)).astype(np.float32).copy()
        cost_init = collision_l2(traj_q_init)

        result, _ = optimizer.solve(
            start_q=wp.from_numpy(mid[None], dtype=wp.float32, device="cpu"),
            T_world_target=wp.from_numpy(
                np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                dtype=wp_vec7,
                device="cpu",
            ),
            target_q=None,
            traj_q_init=wp.from_numpy(traj_q_init, dtype=wp.float32, device="cpu"),
            target_q_per_seed=wp.from_numpy(mid[None], dtype=wp.float32, device="cpu"),
            scene_indices=wp.array([0], dtype=wp.int32, device="cpu"),
        )
        q_opt = result.q.numpy()

        assert q_opt.shape == (1, num_frames, dofs)
        # endpoints are softly anchored (collision on the penetrating endpoint perturbs them slightly)
        np.testing.assert_allclose(q_opt[0, 0], mid, atol=5e-2)  # start pinned
        np.testing.assert_allclose(q_opt[0, -1], mid, atol=5e-2)  # goal pinned
        # static scenes were not clobbered by solve()
        np.testing.assert_array_equal(probe(scene, mesh_center + 0.2), mesh_sdf_before)
        np.testing.assert_array_equal(probe(scene, box_center + 0.2), box_sdf_before)
        # both collision geometries acted: the optimized interior escapes the obstacles
        assert collision_l2(q_opt) < cost_init

    def test_volume_collision_residual_matches_mesh(self):
        """The SDF-volume collision path (closest_on_volumes) must agree with the trusted mesh-BVH path
        on the same obstacle: a NanoVDB box volume vs a box triangle mesh give the same sphere
        penetrations (up to voxel resolution), so the optimizer can use the cheaper volume lookup."""
        import trimesh

        if not wp.is_cuda_available():
            pytest.skip("SDF volumes require CUDA (NanoVDB)")
        from robokit.geom.sdf_volume import mesh_to_sdf_volume

        planner = _load_panda_motion_plan(_make_config(), device="cuda:0")
        robot = planner.robot
        num_frames = 3
        q_traj = np.stack([robot.spec.midrange_q.astype(np.float32)] * num_frames)[None]
        state = robot.state(q=wp.from_numpy(q_traj, dtype=wp.float32, device="cuda:0"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        center = state.collision_sphere_centers_world.numpy().reshape(num_frames, -1, 3)[1].mean(axis=0)

        mesh_scene = WarpScene(1, "cuda:0").add(
            MeshGeom([_make_box_mesh("cuda:0", center.astype(np.float32), 0.12)], np.array([0, 1], dtype=np.int32))
        )
        box = trimesh.creation.box(extents=(0.24, 0.24, 0.24))
        box.apply_translation(center)
        volume = mesh_to_sdf_volume(box, voxel_size=0.01, padding=0.1, device="cuda:0")
        volume_scene = WarpScene(1, "cuda:0").add(VolumeGeom([volume], np.array([0, 1], dtype=np.int32)))

        var = VarValues(robot=state)
        mesh_task = TrajectoryCollisionTask(
            robot=robot, scene=mesh_scene, num_frames=num_frames, weight=1.0, margin=0.05
        )
        vol_task = TrajectoryCollisionTask(
            robot=robot, scene=volume_scene, num_frames=num_frames, weight=1.0, margin=0.05
        )
        assert mesh_task._geom == "mesh"
        assert vol_task._geom == "volume"

        mesh_res = mesh_task.compute_weighted_residual(var).numpy()[0]
        vol_res = vol_task.compute_weighted_residual(var).numpy()[0]

        assert (mesh_res > 0).sum() > 0  # the box is in collision with some spheres
        np.testing.assert_allclose(vol_res, mesh_res, atol=0.02)

    def test_lbfgs_solver_skips_dense_jacobian_and_reduces_collision(self):
        """LBFGS (sparse analytic_jacobian) stores only sparse J values, so the dense (B,R,D)
        Jacobian must not be allocated; with normalize_direction=False + a small h0_scale it converges
        on a penetrating box (the first step can't overshoot before the curvature history forms)."""
        if not wp.is_cuda_available():
            pytest.skip("LBFGS path exercised on CUDA")
        planner = _load_panda_motion_plan(_make_config(), device="cuda:0")
        robot = planner.robot
        num_frames = 5
        dofs = robot.num_actuated_joints
        mid = robot.spec.midrange_q.astype(np.float32)
        state = robot.state(q=wp.from_numpy(mid[None], dtype=wp.float32, device="cuda:0"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        center = state.collision_sphere_centers_world.numpy().reshape(-1, 3).mean(axis=0)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = center
        scene = WarpScene(1, "cuda:0").add(
            BoxGeom(
                np.array([[0.12, 0.12, 0.12]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cuda:0"),
            )
        )
        cfg = GradientTrajectoryOptimizerConfig(
            position_weight=0.0,
            orientation_weight=0.0,
            smoothness_weight=100.0,
            collision_weight=150.0,
            lm_lambda=1.0,
            start_config_weight=100.0,
            goal_config_weight=100.0,
            self_collision_weight=0.0,
            collision_margin=0.025,
            smooth_boundary=True,
            solver="lbfgs",
            max_iter=60,
            lbfgs_normalize_direction=False,
            lbfgs_h0_scale=1e-4,
            lbfgs_line_search_alphas=(0.1, 0.3, 0.5, 1.0),
            use_cuda_graph=False,
        )
        optimizer = GradientTrajectoryOptimizer(
            config=cfg,
            robot=robot,
            ee_link_name_or_index=planner.ee_link_index,
            scene=scene,
            num_frames=num_frames,
            dt=0.25,
            device="cuda:0",
        )
        optimizer.warmup(1)

        traj_q_init = np.broadcast_to(mid[None, None], (1, num_frames, dofs)).astype(np.float32).copy()
        collision = optimizer._collision_tasks[0]

        def collision_cost(q_traj: np.ndarray) -> float:
            s = robot.state(q=wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device="cuda:0"))
            robot.forward_kinematics(s)
            var = VarValues(robot=s)
            return float(np.sum(collision.compute_weighted_residual(var).numpy() ** 2))

        result, _ = optimizer.solve(
            start_q=wp.from_numpy(mid[None], dtype=wp.float32, device="cuda:0"),
            T_world_target=wp.from_numpy(
                np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                dtype=wp_vec7,
                device="cuda:0",
            ),
            target_q=None,
            traj_q_init=wp.from_numpy(traj_q_init, dtype=wp.float32, device="cuda:0"),
            target_q_per_seed=wp.from_numpy(mid[None], dtype=wp.float32, device="cuda:0"),
            scene_indices=wp.array([0], dtype=wp.int32, device="cuda:0"),
        )
        q_opt = result.q.numpy()
        assert optimizer._solver._updaters[0].jacobians is None  # dense (B,R,D) Jacobian skipped
        assert np.isfinite(q_opt).all()  # no NaN blow-up
        assert collision_cost(q_opt) < collision_cost(traj_q_init)

    def test_smooth_boundary_penalizes_first_and_last_transitions(self):
        """By default the first/last velocity transitions carry zero smoothness weight; smooth_boundary
        penalizes every transition uniformly (used when endpoints are soft joint-config anchors)."""
        from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask

        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 6
        dofs = robot.num_actuated_joints

        def smoothness_weight(smooth_boundary: bool) -> np.ndarray:
            cfg = GradientTrajectoryOptimizerConfig(
                position_weight=0.0,
                orientation_weight=0.0,
                smoothness_weight=7.0,
                collision_weight=0.0,
                lm_lambda=1.0,
                start_config_weight=1.0,
                velocity_limit_weight=0.0,
                position_limit_weight=0.0,
                rest_weight=0.0,
                self_collision_weight=0.0,
                smooth_boundary=smooth_boundary,
                use_cuda_graph=False,
            )
            optimizer = GradientTrajectoryOptimizer(
                config=cfg,
                robot=robot,
                ee_link_name_or_index=planner.ee_link_index,
                scene=None,
                num_frames=num_frames,
                dt=0.25,
                device="cpu",
            )
            optimizer.warmup(1)
            task = next(t for t in optimizer._tasks if isinstance(t, TrajectorySmoothnessTask))
            return np.asarray(task.weight).reshape(num_frames - 1, dofs)

        w_default = smoothness_weight(False)
        assert np.allclose(w_default[0], 0.0) and np.allclose(w_default[-1], 0.0)
        assert np.allclose(w_default[1:-1], 7.0)
        assert np.allclose(smoothness_weight(True), 7.0)

    @pytest.mark.slow
    def test_reactive_position_online_state_only_uses_trajectory(self):
        config = _make_config()
        config.enable_retry = False
        config.postprocessors = ()
        config.trajectory_optimizer = MppiTrajectoryOptimizerConfig(
            position_weight=400.0,
            orientation_weight=200.0,
            smoothness_weight=5.0,
            collision_weight=0.0,
            self_collision_weight=0.0,
            max_iter=2,
            num_particles=16,
            control_space="position",
            lock_dim=7,
        )
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target_q = start_q.copy()
        target_q[:2] = (0.2, -0.1)
        position, quaternion = _goal_from_q(planner, target_q)
        target = _target(position, quaternion)

        first = planner.solve_online_numpy(start_q, target, target_q=target_q)
        state = first.online_state
        assert state is not None and state.prev_start_q is None and state.mean_action is None
        exec_q = first.q_traj[0, 1].copy()
        second = planner.solve_online_numpy(exec_q, target, target_q=target_q, online_state=state)

        assert isinstance(second.q_traj, np.ndarray)
        assert second.online_state is state
        assert state.prev_q_traj.ptr != planner._trajectory_optimizer._state.q.ptr

    @pytest.mark.slow
    def test_reactive_mppi_online_state_is_explicit_and_interleavable(self):
        if not wp.is_cuda_available():
            pytest.skip("reactive ParticleSolver runs on CUDA")
        from robokit.helpers.motion_plan import presets

        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = (5.0, 5.0, 5.0)
        scene = WarpScene(1, "cuda:0").add(
            BoxGeom(
                np.array([[0.05, 0.05, 0.05]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(far[None], dtype=wp.mat44, device="cuda:0"),
            )
        )
        planner = _load_panda_motion_plan(presets.online_mppi, device="cuda:0", scene=scene)
        start_q = planner.robot.spec.midrange_q.astype(np.float32)
        target_q_a, target_q_b = start_q.copy(), start_q.copy()
        target_q_a[:2] += (0.4, -0.3)
        target_q_b[:2] += (-0.5, 0.4)
        pos_a, quat_a = _goal_from_q(planner, target_q_a)
        pos_b, quat_b = _goal_from_q(planner, target_q_b)
        target_a, target_b = _target(pos_a, quat_a), _target(pos_b, quat_b)
        start_q_wp = wp.from_numpy(start_q[None], dtype=wp.float32, device="cuda:0")
        target_a_wp = wp.from_numpy(target_a[None], dtype=wp_vec7, device="cuda:0")
        target_b_wp = wp.from_numpy(target_b[None], dtype=wp_vec7, device="cuda:0")
        target_q_a_wp = wp.from_numpy(target_q_a[None], dtype=wp.float32, device="cuda:0")
        target_q_b_wp = wp.from_numpy(target_q_b[None], dtype=wp.float32, device="cuda:0")

        def cold(target: wp.array, goal_q: wp.array):
            return planner.solve_online(start_q_wp, target, target_q=goal_q)

        def clone_array(array):
            if array is None:
                return None
            out = wp.empty_like(array)
            wp.copy(out, array)
            return out

        def clone_state(state):
            return OnlineState(
                prev_q_traj=clone_array(state.prev_q_traj),
                prev_goal_q=clone_array(state.prev_goal_q),
                prev_start_q=clone_array(state.prev_start_q),
                mean_action=clone_array(state.mean_action),
            )

        cold_a = cold(target_a_wp, target_q_a_wp)
        state_a = cold_a.online_state
        assert state_a is not None and state_a.prev_start_q is not None and state_a.mean_action is not None
        cold_q = cold_a.q_traj.numpy().copy()
        cold_a_repeat = cold(target_a_wp, target_q_a_wp)
        np.testing.assert_allclose(cold_a_repeat.q_traj.numpy(), cold_q, atol=1e-6)
        state_baseline, state_interleaved = clone_state(state_a), clone_state(state_a)
        exec_q = cold_q[0, 1]
        exec_q_wp = wp.from_numpy(exec_q[None], dtype=wp.float32, device="cuda:0")

        baseline = planner.solve_online(exec_q_wp, target_a_wp, target_q=target_q_a_wp, online_state=state_baseline)
        baseline_q = baseline.q_traj.numpy().copy()
        cold(target_b_wp, target_q_b_wp)
        interleaved = planner.solve_online(
            exec_q_wp, target_a_wp, target_q=target_q_a_wp, online_state=state_interleaved
        )

        assert interleaved.online_state is state_interleaved
        np.testing.assert_allclose(interleaved.q_traj.numpy(), baseline_q, atol=1e-6)
        np.testing.assert_allclose(
            state_interleaved.prev_start_q.numpy(), state_baseline.prev_start_q.numpy(), atol=1e-6
        )
        np.testing.assert_allclose(state_interleaved.mean_action.numpy(), state_baseline.mean_action.numpy(), atol=1e-6)
        optimizer = planner._trajectory_optimizer
        assert isinstance(optimizer._solver, AccelerationParticleSolver)
        assert state_interleaved.prev_q_traj.ptr != optimizer._state.q.ptr
        assert state_interleaved.prev_start_q.ptr != optimizer._solver.prev_q.ptr
        assert state_interleaved.mean_action.ptr != optimizer._solver.mean_action.ptr

    @pytest.mark.slow
    def test_reactive_mppi_tracks_a_moved_target(self):
        """Reactive MPPI tracks a moved pose when the caller updates its solved joint goal."""
        if not wp.is_cuda_available():
            pytest.skip("reactive ParticleSolver runs on CUDA")
        from robokit.helpers.motion_plan import presets

        # online_mppi sweeps -> it needs a primitive scene; a far box is an obstacle-free stand-in
        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = (5.0, 5.0, 5.0)
        scene = WarpScene(1, "cuda:0").add(
            BoxGeom(
                np.array([[0.05, 0.05, 0.05]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(far[None], dtype=wp.mat44, device="cuda:0"),
            )
        )
        planner = _load_panda_motion_plan(presets.online_mppi, device="cuda:0", scene=scene)
        start_q = planner.robot.spec.midrange_q.astype(np.float32)
        q_a, q_b = start_q.copy(), start_q.copy()
        q_a[0] += 0.4
        q_a[1] -= 0.3
        q_b[0] -= 0.6
        q_b[1] += 0.4
        q_b[3] -= 0.3
        pos_a, quat_a = _goal_from_q(planner, q_a)
        pos_b, quat_b = _goal_from_q(planner, q_b)
        assert np.linalg.norm(pos_a - pos_b) > 0.05  # a genuine move, well past the reactive target step

        # Reactive MPPI = update() looped: execute frame 1 and feed it back when the target moves.
        num_dofs = planner.robot.num_actuated_joints
        start_q_wp = wp.from_numpy(np.empty((1, num_dofs), np.float32), dtype=wp.float32, device="cuda:0")
        target_wp = wp.from_numpy(np.empty((1, 7), np.float32), dtype=wp_vec7, device="cuda:0")
        target_q_wp = wp.from_numpy(np.empty((1, num_dofs), np.float32), dtype=wp.float32, device="cuda:0")

        def tick(exec_q, T_world_target, target_q, online_state):
            start_q_wp.assign(exec_q[None])
            target_wp.assign(T_world_target[None])
            target_q_wp.assign(target_q[None])
            res = planner.solve_online(start_q_wp, target_wp, target_q=target_q_wp, online_state=online_state)
            return res.q_traj.numpy()[0][1].copy(), res.online_state

        target_a = _target(pos_a, quat_a)
        target_b = _target(pos_b, quat_b)
        exec_q, online_state = tick(start_q, target_a, q_a, None)
        for _ in range(60):
            exec_q, online_state = tick(exec_q, target_a, q_a, online_state)
        for _ in range(200):
            exec_q, online_state = tick(exec_q, target_b, q_b, online_state)

        ee_pos, _ = _goal_from_q(planner, exec_q)
        assert np.linalg.norm(ee_pos - pos_b) < 0.01  # tracked the move (stale goal_q stalls ~0.03)

    def test_motion_plan_path_has_no_batched_helper(self):
        root = Path(__file__).parents[3]
        motion_plan_dir = root / "src/robokit/helpers/motion_plan"
        trajectory_collision_task = (root / "src/robokit/terms/sparse/trajectory_collision_task.py").read_text(
            encoding="utf-8"
        )
        trajectory_optimizer_source = "\n".join(
            (motion_plan_dir / name).read_text(encoding="utf-8")
            for name in ("gradient_trajectory_optimizer.py", "mppi_trajectory_optimizer.py")
        )
        assert not (motion_plan_dir / "batched_traj_opt_helper.py").exists()
        assert not (root / "src/robokit/terms/sparse/batched_trajectory_collision_task.py").exists()
        assert "BatchedTrajectoryCollisionTask" not in trajectory_optimizer_source
        assert "BatchedTrajectoryCollisionTask" not in trajectory_collision_task

    # --- Heterogeneous batched planning (slow; local only): each query solved against its own scene. ---

    @pytest.mark.slow
    def test_config_goal_heterogeneous_matches_single_scene(self):
        """Batched q-goal planning over 3 mixed box/sphere/plane/capsule scenes reaches each goal_q collision-free in its own scene."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        scene = build_mp_scene(MP_SCENE_SPECS, device)
        planner = _load_panda_motion_plan(_make_config(q_goal=True), device=device, scene=scene)
        robot = planner.robot
        goal_q, _, _ = sample_mp_problems(robot, device)
        start_q = np.tile(robot.spec.midrange_q.astype(np.float32), (len(MP_SCENE_IDS), 1))
        margin = planner.config.trajectory_optimizer.collision_margin

        result = planner.solve_offline_numpy(
            start_q=start_q,
            target_q=goal_q,
            scene_indices=np.asarray(MP_SCENE_IDS, np.int32),
        )
        np.testing.assert_allclose(result.q_traj[:, -1], goal_q, atol=1e-4)  # config goal hard-pins endpoints
        for i, sid in enumerate(MP_SCENE_IDS):
            single = build_mp_scene([MP_SCENE_SPECS[sid]], device)
            for f in range(result.q_traj.shape[1]):
                clearance = min_clearance(robot, single, result.q_traj[i, f], device)
                assert clearance >= -margin, f"query {i} frame {f} collides in scene {sid}: {clearance * 1000:.1f}mm"

    @pytest.mark.slow
    def test_pose_goal_heterogeneous_routes_goal_ik(self):
        """Batched pose-goal planning routes goal-IK to each mixed scene; unequal 3/1/4 crashes without the fix."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        scene = build_mp_scene(MP_SCENE_SPECS, device)
        planner = _load_panda_motion_plan(_make_config(), device=device, scene=scene)
        robot = planner.robot
        _, target_pos, target_wxyz = sample_mp_problems(robot, device)
        start_q = np.tile(robot.spec.midrange_q.astype(np.float32), (len(MP_SCENE_IDS), 1))
        margin = planner.config.trajectory_optimizer.collision_margin

        result = planner.solve_offline_numpy(  # no target_q_init -> the goal-IK path runs (this is what the fix routes)
            start_q=start_q,
            T_world_target=_target(target_pos, target_wxyz),
            scene_indices=np.asarray(MP_SCENE_IDS, np.int32),
        )
        # Every goal-IK stage gets the per-query scene mapping repeated for its seeds.
        helper = planner._ik_helper
        assert helper is not None
        seen = 0
        for stage_terms, ns in zip(helper._solver.terms, helper._stage_num_seeds):
            for term in stage_terms:
                if isinstance(term, SceneCollisionTask):
                    seen += 1
                    np.testing.assert_array_equal(term.scene_indices.numpy(), np.repeat(MP_SCENE_IDS, ns))
        assert seen == len(scene.geoms) * len(helper._stage_num_seeds)
        # behavioral: each query reaches its target collision-free in its own scene
        for i, sid in enumerate(MP_SCENE_IDS):
            single = build_mp_scene([MP_SCENE_SPECS[sid]], device)
            clearance = min_clearance(robot, single, result.q_traj[i, -1], device)
            assert clearance >= -margin, f"query {i} collides in scene {sid}: {clearance * 1000:.1f}mm"

    @pytest.mark.slow
    def test_plan_unions_coexisting_geometries(self):
        """Every geometry attached to a scene must be avoided as one union."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        scene = build_multi_scene(device)
        assert tuple(type(geometry).__name__ for geometry in scene.geoms) == ("BoxGeom", "PlaneGeom", "MeshGeom")
        planner = _load_panda_motion_plan(_make_config(), device=device, scene=scene)
        robot = planner.robot
        start_q, target_pos, target_wxyz = sample_targets(robot, device)
        margin = planner.config.trajectory_optimizer.collision_margin

        result = planner.solve_offline_numpy(  # no target_q_init -> goal IK also unions every geometry
            start_q=start_q,
            T_world_target=_target(target_pos, target_wxyz),
            scene_indices=np.asarray(SCENE_IDS, np.int32),
        )
        # white-box: the sparse optimizer got one collision task per geometry
        opt = planner._trajectory_optimizer
        assert [t._geom for t in opt._collision_tasks] == ["primitive", "primitive", "mesh"]
        # behavioral: each query ends collision-free against every geometry in its own scene
        for i, sid in enumerate(SCENE_IDS):
            single = build_single_scene(sid, device)
            clearance = min_clearance(robot, single, result.q_traj[i, -1], device)
            assert clearance >= -margin, f"query {i} collides in scene {sid}: {clearance * 1000:.1f}mm"

    @pytest.mark.slow
    def test_plan_refine_pair_coexists_with_primitive(self):
        """A hybrid mesh/SDF geometry and an independent box are both avoided."""
        import trimesh

        if not wp.is_cuda_available():
            pytest.skip("SDF volumes require CUDA (NanoVDB)")
        config = _make_config()
        robot = _load_panda_motion_plan(config, device="cuda:0").robot
        mid = robot.spec.midrange_q.astype(np.float32)
        state = robot.state(q=wp.from_numpy(mid[None], dtype=wp.float32, device="cuda:0"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        centers = state.collision_sphere_centers_world.numpy().reshape(-1, 3)
        refine_center = (centers[len(centers) // 2] + np.array([0.0, 0.25, 0.0], np.float32)).astype(np.float32)
        box_center = (centers[len(centers) // 3] + np.array([0.0, -0.25, 0.0], np.float32)).astype(np.float32)

        refine_mesh = trimesh.creation.box(extents=(0.16, 0.16, 0.16)).apply_translation(refine_center)
        box_pose = np.eye(4, dtype=np.float32)
        box_pose[:3, 3] = box_center
        scene = (
            WarpScene(1, "cuda:0")
            .add(
                BoxGeom(
                    np.array([[0.08, 0.08, 0.08]], np.float32),
                    np.array([0, 1], np.int32),
                    poses=wp.from_numpy(box_pose[None], dtype=wp.mat44, device="cuda:0"),
                )
            )
            .add(
                MeshGeom(
                    [refine_mesh],
                    np.array([0, 1], np.int32),
                    enable_sdf=True,
                    sdf_voxel_size=0.01,
                    sdf_padding=0.1,
                )
            )
        )
        assert tuple(type(geometry).__name__ for geometry in scene.geoms) == ("BoxGeom", "MeshGeom")

        planner = MotionPlanner(
            robot=robot, ee_link_name_or_index="panda_hand", config=config, device="cuda:0", scene=scene
        )
        # white-box: one collision task per geometry
        planner.warmup(1)
        opt = planner._trajectory_optimizer
        assert [t._geom for t in opt._collision_tasks] == ["primitive", "hybrid"]

        # behavioral: plan a small motion; the trajectory stays clear of both the box and the refine object's mesh
        goal_position, goal_quaternion = _goal_from_q(planner, mid)
        margin = planner.config.trajectory_optimizer.collision_margin
        result = planner.solve_offline_numpy(
            start_q=mid,
            T_world_target=_target(goal_position, goal_quaternion),
            target_q_init=mid,
        )
        ref = WarpScene(1, "cuda:0").add(
            MeshGeom([_make_box_mesh("cuda:0", refine_center, 0.08)], np.array([0, 1], np.int32))
        )
        for f in range(result.q_traj.shape[1]):
            assert min_clearance(robot, ref, result.q_traj[0, f], "cuda:0") >= -margin
