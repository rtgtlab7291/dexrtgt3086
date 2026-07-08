"""G1 Humanoid Multi-frame IK using IKHelper

Tracks five end-effectors: pelvis, left foot, right foot, left wrist, and right wrist.
Uses Warp backend with IKHelper for efficient solving.
Mobile base enabled with planar motion constraints.
"""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKHelper, IKHelperConfig
from robokit.lie.warp_se3 import WarpSE3
from robokit.opt.warp_solver import WarpStageConfig
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def main():
    """Main function for G1 humanoid multi-frame IK."""

    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("g1_description")
    target_link_names = [
        "pelvis_contour_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ]

    robot = Robot.load(urdf, backend="warp")

    numpy_robot = Robot.load(urdf, backend="numpy")
    numpy_state = numpy_robot.state(q=numpy_robot.zero_q)
    numpy_state = numpy_robot.forward_kinematics(numpy_state)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3)
    base_frame = server.scene.add_frame("/pelvis", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/pelvis")
    urdf_vis.update_cfg(numpy_robot.zero_q)

    ik_targets = []
    placeholder_targets = []

    for link_name in target_link_names:
        link_idx = numpy_robot.link_names.index(link_name)
        link_pose = numpy_state.get_T_world_link(link_idx)

        scale = 0.15 if "pelvis" in link_name else 0.12

        ik_target = server.scene.add_transform_controls(
            f"/ik_targets/{link_name}",
            scale=scale,
            position=link_pose.xyz,
            wxyz=link_pose.quat_wxyz,
        )
        ik_targets.append(ik_target)

        placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        placeholder_targets.append(WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device)))

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    config = IKHelperConfig(
        stages=[
            WarpStageConfig(num_seeds=64, iters=10, lm_lambda=10.0),
            WarpStageConfig(num_seeds=4, iters=15, lm_lambda=1.0),
            WarpStageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
        ],
        position_weight=1.5,
        orientation_weight=0.3,
        rest_weight=0.1,
        smoothness_weight=0.02,
        velocity_limit_weight=0.0,
        enable_T_world_base=True,
        base_damping_weight=0.5,
        base_step_limit_indices=[2, 3, 4],
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,
        base_weight_smoothness=0.0,
        init_sample_range=0.1,
    )

    ik_helper = IKHelper(
        robot,
        target_link_names,
        placeholder_targets,
        config,
    )
    zero_state = robot.state(
        q=wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device),
        T_world_base=WarpSE3.identity(shape=(1,), device=device),
    )
    prev_state = None

    while True:
        start_time = time.time()

        targets = []
        for ik_target in ik_targets:
            wxyz = np.asarray(ik_target.wxyz, dtype=np.float32)
            wxyz /= np.linalg.norm(wxyz) + 1e-12
            target_pose = np.concatenate([ik_target.position, wxyz], axis=-1).reshape(1, 7)
            targets.append(WarpSE3(wp.from_numpy(target_pose, dtype=wp_vec7, device=device)))

        solved_state = ik_helper.solve(targets, prev_state=prev_state, init_state=zero_state)
        solved_q_np = solved_state.q.numpy()[0]

        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        urdf_vis.update_cfg(solved_q_np)

        base_np = solved_state.T_world_base.xyz_wxyz.numpy()[0]
        base_frame.position = base_np[:3]
        base_frame.wxyz = base_np[3:]

        prev_state = solved_state

        time.sleep(0.01)


if __name__ == "__main__":
    main()
