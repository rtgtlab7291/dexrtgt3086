"""Batched offline retargeting: solve several DexYCB trajectories together."""

import time

import numpy as np
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets.robots.hands import shadow_hand
from robokit.helpers.hand_retargeting import HandRetargetingOffline
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets.shadow import offline, spec
from robokit.robo import Robot


N = 4  # captures solved together


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    datas = [load_dexycb_grasp(i, device=device) for i in range(N)]
    length = min(len(d["keypoints"]) for d in datas)  # shared length so the captures stack into one batch

    # --- retargeting: solve all captures in one batch ---
    robot = Robot.load(
        str(shadow_hand.URDF_PATH),
        load_collision_spheres=True,
        collision_spheres_path=shadow_hand.COLLISION_SPHERE_PATH,
    )
    retarget = HandRetargetingOffline(robot, spec, offline, device=device)
    packed = retarget.solve_numpy(
        np.stack([d["keypoints"][:length] for d in datas]),
        np.stack([d["wrist_quat_wxyz"][:length] for d in datas]),
    )
    print("solved %d trajectories x %d frames" % (N, length))
    T_world_base, joints = packed[..., :7], packed[..., 7:]

    # --- viewer: place each capture in its own cell ---
    cell = np.zeros((N, 3), np.float32)
    cell[:, 0] = np.arange(N) * 0.3
    offset = cell - T_world_base[:, 0, :3]  # shift so each capture's first base lands on its cell

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", 1.0, 1.0, position=(0.5, -0.5, 0.0))
    robot_frames = [server.scene.add_frame("/hands/robot_%d" % i, show_axes=False) for i in range(N)]
    urdf_visualizers = [
        ViserUrdf(server, shadow_hand.URDF_PATH, root_node_name="/hands/robot_%d" % i) for i in range(N)
    ]
    object_frames = [server.scene.add_frame("/hands/object_%d" % i, show_axes=False) for i in range(N)]
    for i, data in enumerate(datas):
        server.scene.add_mesh_simple(
            "/hands/object_%d/mesh" % i,
            data["object_mesh"].vertices,
            data["object_mesh"].faces,
            color=(180, 180, 190),
            opacity=0.6,
        )
    manos = [
        server.scene.add_mesh_simple(
            "/hands/mano_%d" % i,
            datas[i]["mano_vertices"][0] + offset[i],
            datas[i]["mano_faces"],
            color=(120, 200, 130),
            opacity=0.5,
        )
        for i in range(N)
    ]
    playing = server.gui.add_checkbox("Play", True)
    frame_slider = server.gui.add_slider("Frame", 0, length - 1, 1, 0)

    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % length
        f = frame_slider.value
        bp = T_world_base[:, f].copy()
        bp[:, :3] += offset
        with server.atomic():
            for i in range(N):
                robot_frames[i].position, robot_frames[i].wxyz = bp[i, :3], bp[i, 3:]
                urdf_visualizers[i].update_cfg(joints[i, f])
                object_frames[i].position = datas[i]["object_pos"][f] + offset[i]
                object_frames[i].wxyz = datas[i]["object_quat_wxyz"][f]
                manos[i].vertices = datas[i]["mano_vertices"][f] + offset[i]
        time.sleep(0.03)


if __name__ == "__main__":
    main()
