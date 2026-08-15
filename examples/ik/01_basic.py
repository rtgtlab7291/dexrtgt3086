"""Basic IK example."""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IK, presets
from robokit.robo import Robot


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")

    # --- IK ---
    robot = Robot.load(urdf)
    ik = IK(presets.basic, robot=robot, link="panda_hand", device=device)
    ik.warmup(batch_size=1)  # optional

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.midrange_q)
    target = server.scene.add_transform_controls("/ik_target", scale=0.2, position=(0.5, 0.2, 0.5), wxyz=(0, 0, 1, 0))
    timing = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # --- IK: one solve per frame ---
        t0 = time.time()
        state = ik.solve_numpy(np.concatenate([target.position, target.wxyz]))
        wp.synchronize()
        timing.value = 0.99 * timing.value + 0.01 * (time.time() - t0) * 1000

        # --- viewer: visualize the result ---
        urdf_vis.update_cfg(state.q.numpy()[0])
        time.sleep(0.001)


if __name__ == "__main__":
    main()
