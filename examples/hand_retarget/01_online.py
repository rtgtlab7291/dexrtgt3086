"""Online hand retargeting: track Manus keypoints with a Sharpa hand."""

import pickle
import time

import numpy as np
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets import fetch
from robokit.assets.robots.hands import sharpa_hand
from robokit.helpers.hand_retargeting import HandRetargetingOnline
from robokit.helpers.hand_retargeting.presets import sharpa
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserHandSkeleton


def main() -> None:
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    # --- retargeting ---
    robot = Robot.load(
        str(sharpa_hand.URDF_PATH),
        load_collision_spheres=True,
        collision_spheres_path=sharpa_hand.COLLISION_SPHERE_PATH,
        self_collision_ignore_path=sharpa_hand.SELF_COLLISION_IGNORE_PATH,
    )
    retarget = HandRetargetingOnline(robot, sharpa.spec, sharpa.online, device=device)
    retarget.warmup(1)
    data_dir = fetch(["benchmarks/hand_retargeting_vector/**"]) / "benchmarks/hand_retargeting_vector"
    with open(data_dir / "manus_data_right.pkl", "rb") as file:
        keypoints = np.asarray(pickle.load(file, encoding="latin1")["keypoints"], dtype=np.float32)
    keypoints -= keypoints[:, :1]

    # --- viewer: show the human hand and robot side by side ---
    fk_state = robot.forward_kinematics(robot.state(q=robot.zero_q))
    t_base_wrist = fk_state.T_world_link.numpy().reshape(-1, 7)[robot.spec.link_names.index("right_hand_C_MC"), :3]
    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0)
    human = ViserHandSkeleton(server, "/human", position=(0.0, -0.1, 0.0))
    server.scene.add_frame("/robot", show_axes=False, position=-t_base_wrist + (0.0, 0.1, 0.0))
    urdf_vis = ViserUrdf(server, sharpa_hand.URDF_PATH, root_node_name="/robot")

    frame = 0
    while True:
        f = frame % len(keypoints)
        frame += 1

        # --- retarget: solve one frame (the helper streams: it warm-chains and filters across frames) ---
        packed = retarget.solve_numpy(keypoints[f])

        # --- viewer: update both hands ---
        with server.atomic():
            urdf_vis.update_cfg(packed[0, 7:])
            human.update(keypoints[f])
        time.sleep(0.01)


if __name__ == "__main__":
    main()
