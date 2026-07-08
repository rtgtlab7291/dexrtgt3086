"""
Batch IK example - robots solving IK for per-robot targets with a shared offset control.
"""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IKHelper, IKHelperConfig
from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.utils.warp_utils import wp_vec7


BATCH_SIZE = 100
NUM_COLS = 10
GRID_SPACING = 1.5


def main():
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, backend="warp", load_meshes=True)
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=40, height=40, cell_size=1.0)

    num_rows = BATCH_SIZE // NUM_COLS
    T_world_base = np.zeros((BATCH_SIZE, 7), dtype=np.float32)
    T_world_base[:, 3] = 1.0  # identity quaternion w=1
    for index in range(BATCH_SIZE):
        row, col = index // NUM_COLS, index % NUM_COLS
        T_world_base[index, 0] = -col * GRID_SPACING
        T_world_base[index, 1] = (row - (num_rows - 1) / 2) * GRID_SPACING

    batch_urdf = ViserBatchUrdf(
        target=server,
        robot=robot,
        batch_size=BATCH_SIZE,
        root_node_name="/robots",
    )

    target_axes = server.scene.add_batched_axes(
        "/targets",
        batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (BATCH_SIZE, 1)).astype(np.float32),
        batched_positions=np.zeros((BATCH_SIZE, 3), dtype=np.float32),
        axes_length=0.1,
        axes_radius=0.005,
    )

    main_control = server.scene.add_transform_controls(
        "/offset_control", scale=0.15, position=(0.0, 0.0, 0.0), wxyz=(1, 0, 0, 0)
    )

    rng = np.random.default_rng(42)
    per_robot_positions = np.stack(
        [
            rng.uniform(0.3, 0.6, BATCH_SIZE),
            rng.uniform(-0.15, 0.15, BATCH_SIZE),
            rng.uniform(0.3, 0.6, BATCH_SIZE),
        ],
        axis=-1,
    ).astype(np.float32)

    # Orientations: small axis-angle perturbations composed with downward-facing base (wxyz=0,0,1,0)
    axis_angles = rng.uniform(-0.3, 0.3, (BATCH_SIZE, 3)).astype(np.float32)
    angles = np.linalg.norm(axis_angles, axis=-1, keepdims=True)
    half = angles / 2
    axes = axis_angles / (angles + 1e-8)
    pw = np.cos(half)
    pxyz = np.sin(half) * axes
    px, py, pz = pxyz[:, 0:1], pxyz[:, 1:2], pxyz[:, 2:3]
    per_robot_wxyz = np.concatenate([-py, -pz, pw, px], axis=-1).astype(np.float32)
    per_robot_wxyz /= np.linalg.norm(per_robot_wxyz, axis=-1, keepdims=True) + 1e-8

    per_robot_data = np.concatenate([per_robot_positions, per_robot_wxyz], axis=-1)
    per_robot_targets = WarpSE3(wp.from_numpy(per_robot_data, dtype=wp_vec7, device=device))

    placeholder_target = WarpSE3(wp.zeros((BATCH_SIZE,), dtype=wp_vec7, device=device))
    ik_config = IKHelperConfig(smoothness_weight=1.0)
    ik_helper = IKHelper(robot, target_frames="panda_hand", placeholder_targets=placeholder_target, config=ik_config)
    out_state = robot.state(q=wp.empty((BATCH_SIZE, ik_helper.num_joints), dtype=wp.float32, device=device))  # type: ignore[arg-type]

    prev_state = None
    while True:
        wxyz = np.asarray(main_control.wxyz, dtype=np.float32)
        wxyz /= np.linalg.norm(wxyz) + 1e-12
        main_pose = np.concatenate([main_control.position, wxyz]).astype(np.float32)
        main_se3 = WarpSE3(wp.from_numpy(np.tile(main_pose, (BATCH_SIZE, 1)), dtype=wp_vec7, device=device))

        final_targets = main_se3.multiply(per_robot_targets)

        ik_helper.solve(final_targets, prev_state=prev_state, out_state=out_state)
        wp.synchronize()
        prev_state = robot.state(q=wp.clone(out_state.q))

        target_xyz = final_targets.xyz.numpy()
        target_wxyz = final_targets.quat_wxyz.numpy()
        q_batch = out_state.q.numpy()

        batch_urdf.update(q_batch, T_world_base=T_world_base)
        base_positions = T_world_base[:, :3]
        target_axes.batched_positions = target_xyz + base_positions
        target_axes.batched_wxyzs = target_wxyz

        time.sleep(0.01)


if __name__ == "__main__":
    main()
