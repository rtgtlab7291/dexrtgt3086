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
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    robot = Robot.load(urdf, backend="warp")

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.midrange_q)

    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.2, 0.5), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    placeholder_target_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder_target = WarpSE3(wp.from_numpy(placeholder_target_np, dtype=wp_vec7, device=device))

    config = IKHelperConfig(
        rest_weight=0.1,
        smoothness_weight=0.3,
        velocity_limit_weight=1.0,
        dt=0.1,
    )
    ik_helper = IKHelper(robot, target_link_name, placeholder_target, config)

    prev_state = None
    while True:
        start_time = time.time()

        wxyz = np.asarray(ik_target.wxyz, dtype=np.float32)
        wxyz /= np.linalg.norm(wxyz) + 1e-12
        target_pose = np.concatenate([ik_target.position, wxyz], axis=-1).reshape(1, 7)
        target_se3 = WarpSE3(wp.from_numpy(target_pose, dtype=wp_vec7, device=device))

        state = ik_helper.solve(target_se3, prev_state=prev_state)

        elapsed_time = (time.time() - start_time) * 1000.0
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * elapsed_time

        urdf_vis.update_cfg(state.q.numpy()[0])
        prev_state = state

        time.sleep(0.01)


if __name__ == "__main__":
    main()
