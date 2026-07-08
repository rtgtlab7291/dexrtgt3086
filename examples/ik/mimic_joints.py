import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKHelper, IKHelperConfig
from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def main():
    """Main function for multi-frame IK using ability hand with 5 finger targets."""

    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("ability_hand_description")
    target_link_names = ["thumb_anchor", "index_anchor", "middle_anchor", "ring_anchor", "pinky_anchor"]

    robot = Robot.load(urdf, backend="warp")

    numpy_robot = Robot.load(urdf, backend="numpy")
    numpy_state = numpy_robot.state(q=numpy_robot.zero_q)
    numpy_state = numpy_robot.forward_kinematics(numpy_state)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.zero_q)

    ik_targets = []
    placeholder_targets = []
    for link_name in target_link_names:
        link_idx = numpy_robot.link_names.index(link_name)
        link_pose = numpy_state.get_T_world_link(link_idx)
        ik_target = server.scene.add_transform_controls(
            f"/ik_targets/{link_name}",
            scale=0.05,
            position=link_pose.xyz,
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        ik_targets.append(ik_target)
        placeholder_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        placeholder_targets.append(WarpSE3(wp.from_numpy(placeholder_np, dtype=wp_vec7, device=device)))

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    config = IKHelperConfig(
        position_weight=1.0,
        orientation_weight=0.0,
        rest_weight=0.0,
        smoothness_weight=1.0,
        velocity_limit_weight=1.0,
        dt=0.1,
    )
    ik_helper = IKHelper(robot, target_link_names, placeholder_targets, config)

    prev_q = robot.spec.zero_q
    prev_state = robot.state(q=wp.from_numpy(prev_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device))

    while True:
        start_time = time.time()

        targets = []
        for ik_target in ik_targets:
            wxyz = np.asarray(ik_target.wxyz, dtype=np.float32)
            wxyz /= np.linalg.norm(wxyz) + 1e-12
            target_pose = np.concatenate([ik_target.position, wxyz], axis=-1).reshape(1, 7)
            targets.append(WarpSE3(wp.from_numpy(target_pose, dtype=wp_vec7, device=device)))

        state = ik_helper.solve(targets, prev_state=prev_state)
        solved_q_np = state.q.numpy()[0]

        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        urdf_vis.update_cfg(solved_q_np)

        prev_state = state
        prev_q = solved_q_np

        time.sleep(0.01)


if __name__ == "__main__":
    main()
