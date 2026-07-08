import time

import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik_cpu import CPUIKHelper
from robokit.robo import Robot


def main():
    """Main function for CPU-only IK using Pinocchio backend."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot with numpy backend (CPU-only).
    robot = Robot.load(urdf, backend="numpy")

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.midrange_q)

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.2, 0.5), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Create CPU IK helper (no placeholder target needed).
    ik_helper = CPUIKHelper(robot, target_link_name)

    while True:
        # Solve IK.
        start_time = time.time()

        target_pos = ik_target.position
        target_quat_wxyz = ik_target.wxyz
        solved_q = ik_helper.solve_numpy(target_pos, target_quat_wxyz)

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solved_q)

        time.sleep(0.01)  # Sleep to limit update rate


if __name__ == "__main__":
    main()
