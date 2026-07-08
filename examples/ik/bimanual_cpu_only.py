"""Bimanual IK (CPU-only)

Same as bimanual.py, but uses CPUIKHelper with the numpy backend (no GPU required).
"""

import time

import numpy as np
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik_cpu import CPUIKHelper
from robokit.robo import Robot


def main():
    """Main function for bimanual IK using CPU-only backend."""

    urdf = load_robot_description("yumi_description")
    target_link_names = ["yumi_link_7_r", "yumi_link_7_l"]

    robot = Robot.load(urdf, backend="numpy")
    state = robot.state(q=robot.spec.zero_q)
    state = robot.forward_kinematics(state)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.zero_q)

    ik_targets = []
    for link_name in target_link_names:
        link_idx = robot.link_names.index(link_name)
        link_pose = state.get_T_world_link(link_idx)
        ik_target = server.scene.add_transform_controls(
            f"/ik_targets/{link_name}",
            scale=0.2,
            position=link_pose.xyz,
            wxyz=link_pose.quat_wxyz,
        )
        ik_targets.append(ik_target)

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    ik_helper = CPUIKHelper(robot, target_link_names)

    while True:
        start_time = time.time()

        target_positions = []
        target_quats = []
        for ik_target in ik_targets:
            wxyz = np.asarray(ik_target.wxyz, dtype=np.float32)
            wxyz /= np.linalg.norm(wxyz) + 1e-12
            target_positions.append(np.asarray(ik_target.position, dtype=np.float32))
            target_quats.append(wxyz)

        solved_q = ik_helper.solve_numpy(target_positions, target_quats)

        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        urdf_vis.update_cfg(solved_q)

        time.sleep(0.01)


if __name__ == "__main__":
    main()
