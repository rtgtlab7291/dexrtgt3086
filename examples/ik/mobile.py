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

    urdf = load_robot_description("fetch_description")
    target_link_name = "gripper_link"

    robot = Robot.load(urdf, backend="warp")

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.midrange_q)

    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.6, 0.0, 0.55), wxyz=(0, 0.707, 0, -0.707)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Create placeholder target
    placeholder_target_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
    placeholder_target = WarpSE3(wp.from_numpy(placeholder_target_np, dtype=wp_vec7, device=device))

    # Configure IKHelper with mobile base enabled
    config = IKHelperConfig(
        position_weight=10.0,
        orientation_weight=5.0,
        rest_weight=0.01,  # Very light rest bias
        dt=0.1,
        # Mobile base parameters
        enable_T_world_base=True,  # Enable mobile base
        base_damping_weight=0.5,  # Moderate base damping
        base_step_limit_indices=[2, 3, 4],  # Lock z, roll, pitch for planar motion
        base_step_limit_weight=100.0,
        base_weight_rest=0.01,  # Very light base rest bias
        base_weight_smoothness=0.0,  # No base smoothness constraint
        init_sample_range=0.1,  # Tight sampling for smooth tracking
        seed=42,  # Deterministic behavior
    )

    ik_helper = IKHelper(
        robot,
        target_link_name,
        placeholder_target,
        config,
    )

    while True:
        start_time = time.time()

        wxyz = np.asarray(ik_target.wxyz, dtype=np.float32)
        wxyz /= np.linalg.norm(wxyz) + 1e-12
        target_pose = np.concatenate([ik_target.position, wxyz], axis=-1).reshape(1, 7)
        target_se3 = WarpSE3(wp.from_numpy(target_pose, dtype=wp_vec7, device=device))

        state = ik_helper.solve(target_se3)

        elapsed_time = (time.time() - start_time) * 1000.0
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * elapsed_time

        # Update visualization
        urdf_vis.update_cfg(state.q.numpy()[0])

        # Update base frame visualization
        base_np = state.T_world_base.xyz_wxyz.numpy()[0]
        base_frame.position = base_np[:3]
        base_frame.wxyz = base_np[3:]

        time.sleep(0.01)


if __name__ == "__main__":
    main()
