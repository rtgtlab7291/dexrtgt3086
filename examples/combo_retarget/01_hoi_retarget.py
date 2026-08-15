"""Retarget one SAGA human-object interaction to G1 + Inspire: body, captured fingers, and object."""

import argparse
import time

import trimesh
import viser
import warp as wp

from robokit.assets.motions import saga as saga_motion
from robokit.helpers.combo_retarget import PRESETS, ComboRetargeter
from robokit.helpers.combo_retarget.loaders import load_saga_clip
from robokit.utils.visualize_utils import ViserBatchUrdf, ViserHandSkeleton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None, help="SAGA FullGraspPose npz (default: fetch s1 airplane_fly_1 from HF)")
    args = parser.parse_args()
    npz = args.npz or str(saga_motion.data_root() / "train/s1/airplane_fly_1.npz")

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    preset = PRESETS["g1_inspire_right"]
    retargeter = ComboRetargeter(preset, device=device, load_meshes=True)
    saga = load_saga_clip(npz, robot_height=preset.body.robot_height, device=device)

    t0 = time.time()
    qpos = retargeter.solve(saga)
    print(f"solved {qpos.shape[0]} frames in {time.time() - t0:.1f}s")
    ndof = retargeter.body_robot.spec.num_actuated_joints

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=6, height=6)
    robot_viz = ViserBatchUrdf(server, retargeter.body_robot, batch_size=1, root_node_name="/robot")
    track = saga.hands["right"]
    mesh = track.object_mesh
    assert isinstance(mesh, trimesh.Trimesh)
    mesh_handle = server.scene.add_mesh_simple("/object", mesh.vertices, mesh.faces, color=(200, 140, 60), opacity=0.9)
    hand_viz = ViserHandSkeleton(server, "/human_hand", color=(220, 40, 40))
    frame_slider = server.gui.add_slider("frame", min=0, max=qpos.shape[0] - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % qpos.shape[0]
        i = frame_slider.value
        with server.atomic():
            robot_viz.update_cfg(qpos[i : i + 1, 7 : 7 + ndof], T_world_base=qpos[i : i + 1, :7])
            mesh_handle.position = qpos[i, 7 + ndof : 7 + ndof + 3]
            mesh_handle.wxyz = qpos[i, 7 + ndof + 3 :]
            hand_viz.update(track.mano_overlay[i])
        time.sleep(1.0 / saga.fps)


if __name__ == "__main__":
    main()
