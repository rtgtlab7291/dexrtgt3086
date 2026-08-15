"""Batched online retargeting: track several DexYCB captures with xArm7 and XHand."""

import time

import numpy as np
import viser
import warp as wp

from robokit.assets.robots.arms import xarm7_xhand as xarm7_xhand_asset
from robokit.helpers.hand_retargeting import HandRetargetingOnline
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets import xarm7_xhand
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.xform.numpy import quaternion_apply, quaternion_invert


N = 4  # captures tracked together, one arm each
CELL = 1.5  # x spacing between display cells

# Use the same arm base pose for every capture.
BASE_POSE = np.array([0.9, 0.9, 0.0, 0.70710678, 0.0, 0.0, -0.70710678], np.float32)


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    robot = Robot.load(
        str(xarm7_xhand_asset.URDF_PATH),
        load_meshes=True,
        load_collision_spheres=True,
        collision_spheres_path=xarm7_xhand_asset.COLLISION_SPHERE_PATH,
    )
    target_index = {name: index for index, name in enumerate(xarm7_xhand.spec.target_names)}
    anchor_target_idx = [target_index[name] for name in xarm7_xhand.spec.contact_target_names]
    anchor_links = [xarm7_xhand.spec.target_link_names[name] for name in xarm7_xhand.spec.contact_target_names]
    anchor_link_idx = [robot.spec.link_names.index(n) for n in anchor_links]

    retarget = HandRetargetingOnline(robot, xarm7_xhand.spec, xarm7_xhand.online, device=device)
    retarget.warmup(N)

    datas = [load_dexycb_grasp(i, device=device) for i in range(N)]
    length = min(len(data["keypoints"]) for data in datas)  # shared length so the captures stack into one batch
    keypoints = np.stack([data["keypoints"][:length] for data in datas])
    q_wxyz_base_world = quaternion_invert(BASE_POSE[3:])
    keypoints_base = quaternion_apply(q_wxyz_base_world, keypoints - BASE_POSE[:3]).astype(np.float32)

    # --- viewer: place each capture in its own cell ---
    offset = np.zeros((N, 3), np.float32)
    offset[:, 0] = np.arange(N) * CELL
    base_grid = np.tile(BASE_POSE, (N, 1))
    base_grid[:, :3] += offset

    server = viser.ViserServer()
    server.scene.add_grid("/grid", 2.0 + CELL * N, 2.0, position=((N - 1) * CELL / 2 + 0.5, 0.5, 0.0))
    batch_urdf = ViserBatchUrdf(server, robot, batch_size=N, root_node_name="/arms")
    object_frames = [server.scene.add_frame("/object_%d" % i, show_axes=False) for i in range(N)]
    for i, data in enumerate(datas):
        server.scene.add_mesh_simple(
            "/object_%d/mesh" % i,
            data["object_mesh"].vertices,
            data["object_mesh"].faces,
            color=(180, 180, 190),
            opacity=0.6,
        )
    manos = [
        server.scene.add_mesh_simple(
            "/mano_%d" % i,
            datas[i]["mano_vertices"][0] + offset[i],
            datas[i]["mano_faces"],
            color=(120, 200, 130),
            opacity=0.5,
        )
        for i in range(N)
    ]
    frame = 0
    while True:
        f = frame % length
        frame += 1

        # --- retarget: solve all captures from their previous results ---
        packed = retarget.solve_numpy(keypoints_base[:, f])

        if frame % 60 == 1:
            state = robot.state(q=wp.from_numpy(packed[:, 7:], dtype=wp.float32, device=device))
            robot.forward_kinematics(state)
            pos = state.T_world_link.numpy()[:, anchor_link_idx, :3]  # (N, 5, 3)
            err = np.linalg.norm(pos - keypoints_base[:, f, anchor_target_idx], axis=2).mean(axis=1) * 1000
            print("frame %d batch %d tips err mm %s" % (frame, N, np.round(err, 1)))

        # --- viewer: update every robot, MANO hand, and object ---
        with server.atomic():
            batch_urdf.update_cfg(packed[:, 7:], T_world_base=base_grid)
            for i in range(N):
                object_frames[i].position = datas[i]["object_pos"][f] + offset[i]
                object_frames[i].wxyz = datas[i]["object_quat_wxyz"][f]
                manos[i].vertices = datas[i]["mano_vertices"][f] + offset[i]
        time.sleep(0.03)


if __name__ == "__main__":
    main()
