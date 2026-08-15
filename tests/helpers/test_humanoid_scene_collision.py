"""Caller-owned scene collision in online humanoid retargeting."""

import numpy as np
import pytest
import warp as wp

from robokit.geom import BoxGeom, PlaneGeom, WarpScene
from robokit.helpers.humanoid_retarget import (
    HumanoidRetargetingOnline,
    HumanoidRetargetingOnlineConfig,
    LinkMapping,
)
from robokit.opt.multi_seed_solver import StageConfig
from robokit.opt.var_values import VarValues
from robokit.utils.warp_utils import wp_vec7


_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_ZERO3 = np.zeros(3, dtype=np.float32)


def _config(weight: float) -> HumanoidRetargetingOnlineConfig:
    return HumanoidRetargetingOnlineConfig(
        urdf_path="<robot-passed-in>",
        human_root_name="root",
        link_mapping={"panda_link0": LinkMapping("root", 100.0, 100.0, _ZERO3, _IDENTITY)},
        scene_collision_weight=weight,
        scene_collision_margin=0.0,
        position_limit_weight=0.0,
        smoothness_weight=0.0,
        base_smoothness_weight=0.0,
        velocity_limit_weight=0.0,
        velocity_clamp_scale=0.0,
        cuda_graph_mode="none",
        stages=[StageConfig(num_seeds=1, iters=1, lm_lambda=1.0)],
    )


class TestHumanoidSceneCollision:
    def test_weight_zero_needs_no_scene(self, panda_robot_with_collision):
        helper = HumanoidRetargetingOnline(_config(0.0), robot=panda_robot_with_collision, device=_DEVICE)
        helper.warmup(1)
        assert helper._scene_collision_tasks == []

    def test_positive_weight_requires_scene(self, panda_robot_with_collision):
        with pytest.raises(ValueError, match="requires scene"):
            HumanoidRetargetingOnline(_config(1.0), robot=panda_robot_with_collision, device=_DEVICE)

    def test_caller_update_reaches_built_task(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = [10.0, 0.0, 0.0]
        geom = BoxGeom(
            np.array([[0.02, 0.02, 0.02]], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            poses=wp.from_numpy(far[None], dtype=wp.mat44, device=_DEVICE),
        )
        scene = WarpScene(1, _DEVICE).add(geom)
        helper = HumanoidRetargetingOnline(_config(10.0), robot=robot, scene=scene, device=_DEVICE)
        helper.warmup(1)
        task = helper._scene_collision_tasks[0]
        assert task.scene is scene

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q[None], dtype=wp.float32, device=_DEVICE))
        far_residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        near = np.eye(4, dtype=np.float32)
        near[:3, 3] = state.collision_sphere_centers_world.numpy()[0, 0]
        geom.update(poses=wp.from_numpy(near[None], dtype=wp.mat44, device=_DEVICE))
        near_residual = task.compute_weighted_residual(VarValues(robot=state)).numpy()

        np.testing.assert_array_equal(far_residual, np.zeros_like(far_residual))
        assert np.max(near_residual) > 0.0

    def test_floor_term_reduces_penetration(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        frame = np.array([[[0.0, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        helper = HumanoidRetargetingOnline(_config(0.0), robot=robot, device=_DEVICE)
        helper.warmup(1)
        without_collision = helper.solve_numpy(frame)[0]
        scene = WarpScene(1, _DEVICE).add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
        helper = HumanoidRetargetingOnline(_config(1000.0), robot=robot, scene=scene, device=_DEVICE)
        helper.warmup(1)
        with_collision = helper.solve_numpy(frame)[0]

        clearances = []
        for qpos in (without_collision, with_collision):
            state = robot.state(
                q=wp.from_numpy(qpos[None, 7:], dtype=wp.float32, device=_DEVICE),
                T_world_base=wp.from_numpy(qpos[None, :7], dtype=wp_vec7, device=_DEVICE),
            )
            robot.forward_kinematics(state)
            robot.transform_collision_spheres(state)
            num_spheres = len(robot.spec.collision_sphere_radii)
            offsets = wp.from_numpy(np.array([0, num_spheres], dtype=np.int32), dtype=wp.int32, device=_DEVICE)
            sdf = scene.query_sdf(
                state.collision_sphere_centers_world.reshape((num_spheres,)), offsets, distance_only=True
            )
            clearances.append(float(np.min(sdf.numpy() - robot.spec.collision_sphere_radii)))

        assert clearances[1] > clearances[0]
