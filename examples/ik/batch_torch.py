"""
Batch IK example using PyTorch - 10 robots solving IK for the same target simultaneously.
"""

import time

import numpy as np
import torch
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKHelper
from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.torch import quaternion_to_matrix, rot_tl_to_tf_mat


BATCH_SIZE = 10


def main():
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp")
    device = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=10, height=10, cell_size=0.5)

    urdf_vis_list = []
    for idx in range(BATCH_SIZE):
        row, col = idx // 5, idx % 5
        base_frame = server.scene.add_frame(f"/robot_{idx}/base", show_axes=False)
        base_frame.position = (col * 1.5 - 3.0, row * 1.5 - 0.75, 0.0)
        urdf_vis = ViserUrdf(server, urdf, root_node_name=f"/robot_{idx}/base")
        urdf_vis.update_cfg(robot.spec.midrange_q)
        urdf_vis_list.append(urdf_vis)

    ik_target = server.scene.add_transform_controls("/target", scale=0.15, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0))
    timing_handle = server.gui.add_number("Solve Time (ms)", 0.0, disabled=True)

    placeholder_target = WarpSE3(wp.zeros((BATCH_SIZE,), dtype=wp_vec7, device=device))
    ik_helper = IKHelper(robot, target_frames="panda_hand", placeholder_targets=placeholder_target)

    while True:
        wxyz = np.asarray(ik_target.wxyz, dtype=np.float32)
        wxyz /= np.linalg.norm(wxyz) + 1e-12
        position = np.asarray(ik_target.position, dtype=np.float32)

        wxyz_torch = torch.from_numpy(wxyz).to(device)
        position_torch = torch.from_numpy(position).to(device)
        rot_mat = quaternion_to_matrix(wxyz_torch)
        target_matrix = rot_tl_to_tf_mat(rot_mat=rot_mat, tl=position_torch)
        target_matrices = target_matrix.unsqueeze(0).repeat(BATCH_SIZE, 1, 1)

        start = time.perf_counter()
        state = ik_helper.solve_torch(target_matrices)
        if device == "cuda":
            torch.cuda.synchronize()
        timing_handle.value = (time.perf_counter() - start) * 1000.0

        q_batch = state.q.cpu().numpy()
        for idx in range(BATCH_SIZE):
            urdf_vis_list[idx].update_cfg(q_batch[idx])

        time.sleep(0.01)


if __name__ == "__main__":
    main()
