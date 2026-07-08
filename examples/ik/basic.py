import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKHelper
from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def main():
    """Main function for basic IK."""

    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    robot = Robot.load(urdf, backend="warp")

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

    # Create placeholder target for IKHelper initialization
    placeholder_target_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder_target = WarpSE3(wp.from_numpy(placeholder_target_np, dtype=wp_vec7, device=device))
    ik_helper = IKHelper(robot, target_link_name, placeholder_target)

    while True:
        # Solve IK.
        start_time = time.time()

        target_pos = ik_target.position
        target_quat_wxyz = ik_target.wxyz
        state = ik_helper.solve_numpy(target_pos, target_quat_wxyz)

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(state.q.numpy()[0])

        time.sleep(0.01)  # Sleep to limit update rate


if __name__ == "__main__":
    main()
