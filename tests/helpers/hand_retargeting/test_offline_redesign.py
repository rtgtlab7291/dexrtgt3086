"""Regression tests for the named offline hand-retargeting pipeline."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.hand_retargeting.config import HandRetargetingOfflineConfig, HandSpec
from robokit.helpers.hand_retargeting.offline import HandRetargetingOffline
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizerConfig
from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.utils.hand_coord_utils import MANOPTH_HAND_COORD_SPEC
from robokit.utils.warp_utils import wp_vec7


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_URDF = _FIXTURES / "shadow_hand_right.urdf"
_TARGET_NAMES = ("index_tip", "root", "thumb_tip", "index_middle", "unused")
_SPEC = HandSpec(
    floating_base=True,
    target_names=_TARGET_NAMES,
    target_link_names={
        "root": "palm",
        "thumb_tip": "thtip",
        "index_middle": "ffmiddle",
        "index_tip": "fftip",
    },
    target_chains=(("thumb_tip",), ("index_middle", "index_tip"), ("unused",)),
    root_target_name="root",
    contact_target_names=("thumb_tip", "index_tip"),
    target_coord_spec=MANOPTH_HAND_COORD_SPEC,
    root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
)
_CONFIG = HandRetargetingOfflineConfig(
    target_scale=1.0,
    global_position_weight=2.0,
    root_position_weight=1.0,
    root_orientation_weight=0.2,
    vector_weight=0.5,
    direction_weight=1.0,
    velocity_weight=0.2,
    acceleration_weight=0.1,
    root_position_velocity_weight=0.2,
    root_orientation_velocity_weight=0.1,
    limit_weight=1.0,
    rest_weight=0.01,
    contact_weight=2.0,
    anchor_q_weight=1.0,
    anchor_base_weight=1.0,
    active_joint_names=("FFJ3", "FFJ2", "FFJ1"),
    locked_prefix_frames=1,
    solver=SparseLMOptimizerConfig(max_iter=1, lm_lambda=0.5, use_cuda_graph=False),
)


class TestHandRetargetingOfflineRedesign:
    def test_single_chunk_uses_init_state(self) -> None:
        """Preserve a supplied state in one non-optimizing public NumPy solve."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = replace(
            _CONFIG,
            root_orientation_weight=0.0,
            solver=SparseLMOptimizerConfig(max_iter=0, lm_lambda=0.5, use_cuda_graph=False),
        )
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, 3, 1))
        base = np.zeros((1, 3, 7), dtype=np.float32)
        base[:, :, :3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        base[:, :, 3] = 1.0
        init_state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )
        points = np.zeros((3, len(_TARGET_NAMES), 3), dtype=np.float32)

        result = HandRetargetingOffline(robot, _SPEC, config, device="cpu").solve_numpy(points, init_state=init_state)

        np.testing.assert_array_equal(result[:, :7], base[0])
        np.testing.assert_array_equal(result[:, 7:], q[0])

    def test_non_contact_spec_discovers_hand_joints(self) -> None:
        """Derive hand DOFs from mapped links even when contacts are disabled."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        retargeter = HandRetargetingOffline(
            robot,
            replace(_SPEC, contact_target_names=()),
            replace(
                _CONFIG,
                root_orientation_weight=0.0,
                velocity_weight=0.0,
                acceleration_weight=0.0,
                root_position_velocity_weight=0.0,
                root_orientation_velocity_weight=0.0,
                limit_weight=0.0,
                rest_weight=0.0,
                solver=SparseLMOptimizerConfig(max_iter=0, lm_lambda=0.5, use_cuda_graph=False),
            ),
            device="cpu",
        )
        result = retargeter.solve_numpy(np.zeros((1, 3, len(_TARGET_NAMES), 3), dtype=np.float32))
        indices = [robot.spec.actuated_joint_names.index(name) for name in ("FFJ1", "FFJ2", "FFJ3")]
        expected = np.mean(robot.spec.actuated_joint_limits[indices], axis=1)

        np.testing.assert_allclose(result[0, :, 7:][:, indices], np.tile(expected, (3, 1)))
        assert np.isfinite(result).all()

    def test_named_normal_contact_and_refinement(self) -> None:
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, 3, 1))
        T_world_base = np.zeros((1, 3, 7), dtype=np.float32)
        T_world_base[:, :, 3] = 1.0
        state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(T_world_base, dtype=wp_vec7, device="cpu"),
        )
        robot.forward_kinematics(state)
        T_world_link = state.T_world_link.numpy()
        link_index = {name: index for index, name in enumerate(robot.spec.link_names)}
        points = np.zeros((1, 3, len(_TARGET_NAMES), 3), dtype=np.float32)
        for target, link in _SPEC.target_link_names.items():
            points[:, :, _TARGET_NAMES.index(target)] = T_world_link[:, :, link_index[link], :3]
        root_quat = T_world_link[:, :, link_index["palm"], 3:].copy()
        contact_points = np.stack(
            [
                T_world_link[:, :, link_index["thtip"], :3],
                T_world_link[:, :, link_index["fftip"], :3],
            ],
            axis=2,
        ).astype(np.float32)
        contact_points += np.array([0.002, -0.001, 0.001], dtype=np.float32)
        contact_mask = np.ones((1, 3, 2), dtype=np.bool_)

        retargeter = HandRetargetingOffline(robot, _SPEC, _CONFIG, device="cpu")
        retargeter.warmup(1, 3)
        normal = retargeter.solve_numpy(points, root_quat)
        contact = retargeter.solve_with_contact_numpy(points, contact_points, contact_mask)
        refined = retargeter.solve_with_contact_numpy(
            points,
            contact_points,
            contact_mask,
            init_state=state,
        )

        assert normal.shape == contact.shape == refined.shape == (1, 3, 7 + robot.spec.num_actuated_joints)
        assert np.isfinite(normal).all()
        assert np.isfinite(contact).all()
        assert np.isfinite(refined).all()

    def test_fixed_base_keeps_init_state_base(self) -> None:
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        spec = replace(_SPEC, floating_base=False)
        config = replace(_CONFIG, solver=SparseLMOptimizerConfig(max_iter=0, lm_lambda=0.5))
        points = np.zeros((1, 3, len(_TARGET_NAMES), 3), dtype=np.float32)
        root_quat = np.zeros((1, 3, 4), dtype=np.float32)
        root_quat[:, :, 0] = 1.0
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, 3, 1))
        base = np.zeros((1, 3, 7), dtype=np.float32)
        base[:, :, :3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        base[:, :, 3] = 1.0
        init_state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )

        retargeter = HandRetargetingOffline(robot, spec, config, device="cpu")
        result = retargeter.solve_numpy(points, root_quat, init_state=init_state)

        np.testing.assert_array_equal(result[:, :, :7], base)

    def test_caller_owned_chunks(self) -> None:
        """Split, stitch, and adjust the global locked prefix in caller code."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        rng = np.random.default_rng(7)
        points = rng.normal(0.0, 0.04, (1, 5, len(_TARGET_NAMES), 3)).astype(np.float32)
        chunk_config = replace(
            _CONFIG,
            global_position_weight=0.0,
            root_position_weight=0.0,
            root_orientation_weight=0.0,
            velocity_weight=0.0,
            acceleration_weight=0.0,
            root_position_velocity_weight=0.0,
            root_orientation_velocity_weight=0.0,
            rest_weight=0.0,
        )
        chunks = []
        for start, end, keep_start in ((0, 3, 0), (2, 5, 1)):
            helper = HandRetargetingOffline(robot, _SPEC, chunk_config, device="cpu")
            chunk = helper.solve_numpy(points[:, start:end])
            chunks.append(chunk[:, keep_start:])
        result = np.concatenate(chunks, axis=1)
        assert result.shape == (1, 5, 7 + robot.spec.num_actuated_joints)
        assert np.isfinite(result).all()

        refine_config = replace(chunk_config, optimize_base=False)
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, 5, 1))
        base = np.zeros((1, 5, 7), dtype=np.float32)
        base[:, :, 3] = 1.0
        init_state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )
        contact_points = rng.normal(0.0, 0.02, (1, 5, 2, 3)).astype(np.float32)
        contact_mask = np.ones((1, 5, 2), dtype=np.bool_)
        local_points = np.full((1, 5, 2, 3), 0.001, dtype=np.float32)
        chunks = []
        for start, end, locked_prefix, keep_start in ((0, 3, 1, 0), (2, 5, 0, 1)):
            chunk_state = robot.state(
                q=init_state.q[:, start:end].contiguous(),
                T_world_base=init_state.T_world_base[:, start:end].contiguous(),
            )
            chunk_helper = HandRetargetingOffline(
                robot,
                _SPEC,
                replace(refine_config, locked_prefix_frames=locked_prefix),
                device="cpu",
            )
            chunk = chunk_helper.solve_with_contact_numpy(
                points[:, start:end],
                contact_points[:, start:end],
                contact_mask[:, start:end],
                init_state=chunk_state,
                local_contact_points=local_points[:, start:end],
            )
            chunks.append(chunk[:, keep_start:])
        result = np.concatenate(chunks, axis=1)
        assert result.shape == (1, 5, 7 + robot.spec.num_actuated_joints)
        np.testing.assert_array_equal(result[:, 0, 7:], q[:, 0])


class TestHandRetargetingOfflineRefinement:
    def test_caller_owned_scene_collision_is_active(self, panda_robot_with_collision: Robot) -> None:
        """Propagate caller scene updates into anchored collision refinement."""
        num_frames = 3
        spec = HandSpec(
            floating_base=False,
            target_names=("root", "contact"),
            target_link_names={"root": "panda_link7", "contact": "panda_hand"},
            target_chains=(("root", "contact"),),
            root_target_name="root",
            contact_target_names=("contact",),
            target_coord_spec=MANOPTH_HAND_COORD_SPEC,
            root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
        )
        config = HandRetargetingOfflineConfig(
            global_position_weight=0.0,
            root_position_weight=0.0,
            root_orientation_weight=0.0,
            vector_weight=0.0,
            direction_weight=0.0,
            velocity_weight=0.0,
            acceleration_weight=0.0,
            root_position_velocity_weight=0.0,
            root_orientation_velocity_weight=0.0,
            limit_weight=0.0,
            rest_weight=0.0,
            contact_weight=0.0,
            collision_weight=10.0,
            anchor_q_weight=0.0,
            anchor_base_weight=0.0,
            optimize_base=False,
            solver=SparseLMOptimizerConfig(max_iter=0, lm_lambda=0.5, use_cuda_graph=False),
        )
        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = (10.0, 0.0, 0.0)
        box = BoxGeom(
            np.array([[0.01, 0.01, 0.01]], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            poses=wp.from_numpy(far[None], dtype=wp.mat44, device="cpu"),
        )
        scene = WarpScene(1, "cpu").add(box)
        helper = HandRetargetingOffline(panda_robot_with_collision, spec, config, scene=scene, device="cpu")
        q = np.tile(panda_robot_with_collision.spec.midrange_q.astype(np.float32), (1, num_frames, 1))
        base = np.zeros((1, num_frames, 7), dtype=np.float32)
        base[:, :, 3] = 1.0
        init_state = panda_robot_with_collision.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )
        points = np.zeros((num_frames, 2, 3), dtype=np.float32)
        contacts = np.zeros((num_frames, 1, 3), dtype=np.float32)
        mask = np.zeros((num_frames, 1), dtype=np.bool_)

        helper.solve_with_contact_numpy(points, contacts, mask, init_state=init_state)
        far_cost = helper._refinement_optimizer.costs.numpy()[0]
        panda_robot_with_collision.forward_kinematics(init_state)
        panda_robot_with_collision.transform_collision_spheres(init_state)
        sphere_index = helper._collision_sphere_indices[0]
        near = np.eye(4, dtype=np.float32)
        near[:3, 3] = init_state.collision_sphere_centers_world.numpy()[0, 0, sphere_index]
        box.update(poses=wp.from_numpy(near[None], dtype=wp.mat44, device="cpu"))
        helper.solve_with_contact_numpy(points, contacts, mask, init_state=init_state)
        near_cost = helper._refinement_optimizer.costs.numpy()[0]

        assert far_cost == 0.0
        assert near_cost > 0.0

    def test_fixed_base_and_start_do_not_move(self, panda_robot: Robot) -> None:
        """Keep inactive bases and locked trajectory prefixes unchanged."""
        num_frames = 3
        q = np.tile(panda_robot.spec.midrange_q.astype(np.float32), (num_frames, 1))
        q[1, 0] += 0.2
        T_world_base = np.array(
            [
                [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.1, 0.3, 0.9238795, 0.0, 0.0, 0.3826834],
                [0.3, 0.2, 0.1, 0.9659258, 0.0, 0.2588190, 0.0],
            ],
            dtype=np.float32,
        )
        points = np.zeros((num_frames, 2, 3), dtype=np.float32)
        contact_points = np.zeros((num_frames, 1, 3), dtype=np.float32)
        contact_mask = np.array([[False], [False], [True]])
        spec = HandSpec(
            floating_base=False,
            target_names=("root", "contact"),
            target_link_names={"root": "panda_link7", "contact": "panda_hand"},
            target_chains=(("root", "contact"),),
            root_target_name="root",
            contact_target_names=("contact",),
            target_coord_spec=MANOPTH_HAND_COORD_SPEC,
            root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
        )
        config = HandRetargetingOfflineConfig(
            optimize_base=False,
            global_position_weight=0.0,
            root_position_weight=0.0,
            root_orientation_weight=0.0,
            vector_weight=0.0,
            direction_weight=0.0,
            velocity_weight=0.0,
            acceleration_weight=0.0,
            limit_weight=0.0,
            locked_prefix_frames=2,
            solver=SparseLMOptimizerConfig(max_iter=2, lm_lambda=0.5, use_cuda_graph=False),
        )
        init_state = panda_robot.state(
            q=wp.from_numpy(q[None], dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(T_world_base[None], dtype=wp_vec7, device="cpu"),
        )

        refined = HandRetargetingOffline(panda_robot, spec, config, device="cpu").solve_with_contact_numpy(
            points,
            contact_points,
            contact_mask,
            init_state=init_state,
        )

        np.testing.assert_array_equal(refined[:2, 7:], q[:2])
        np.testing.assert_array_equal(refined[:, :7], T_world_base)
