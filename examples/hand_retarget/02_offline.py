"""Offline hand retargeting: solve a complete DexYCB grasp trajectory."""

import time

import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets.robots.hands import shadow_hand
from robokit.helpers.hand_retargeting import HandRetargetingOffline
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets.shadow import offline, spec
from robokit.robo import Robot


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    data = load_dexycb_grasp(0, device=device)
    num_frames = len(data["keypoints"])

    # --- retargeting ---
    robot = Robot.load(
        str(shadow_hand.URDF_PATH),
        load_collision_spheres=True,
        collision_spheres_path=shadow_hand.COLLISION_SPHERE_PATH,
    )
    retarget = HandRetargetingOffline(robot, spec, offline, device=device)
    packed = retarget.solve_numpy(data["keypoints"], data["wrist_quat_wxyz"])
    print("solved %d frames" % num_frames)
    T_world_base, joints = packed[:, :7], packed[:, 7:]

    # --- viewer: play the robot, MANO hand, and object together ---
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.85, 0.02, 0.50)
        client.camera.look_at = (0.58, 0.36, 0.16)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    server.scene.add_grid("/grid", 0.3, 0.3, cell_size=0.05, position=(0.58, 0.36, 0.0))
    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, shadow_hand.URDF_PATH, root_node_name="/robot")
    object_frame = server.scene.add_frame("/object", show_axes=False)
    server.scene.add_mesh_simple(
        "/object/mesh",
        data["object_mesh"].vertices,
        data["object_mesh"].faces,
        color=(180, 180, 190),
        opacity=0.6,
    )
    mano = server.scene.add_mesh_simple(
        "/mano", data["mano_vertices"][0], data["mano_faces"], color=(120, 200, 130), opacity=0.5
    )
    playing = server.gui.add_checkbox("Play", True)
    frame_slider = server.gui.add_slider("Frame", 0, num_frames - 1, 1, 0)

    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % num_frames
        f = frame_slider.value
        with server.atomic():
            robot_frame.position, robot_frame.wxyz = T_world_base[f, :3], T_world_base[f, 3:]
            urdf_vis.update_cfg(joints[f])
            object_frame.position, object_frame.wxyz = data["object_pos"][f], data["object_quat_wxyz"][f]
            mano.vertices = data["mano_vertices"][f]
        time.sleep(0.05)


if __name__ == "__main__":
    main()
