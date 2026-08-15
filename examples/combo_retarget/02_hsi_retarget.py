"""Retarget one mined ParaHome human-scene interaction to G1 + Inspire: body, fingers, and the room.

Unlike the object-only SAGA path (``01_hoi_retarget.py``), the clip carries a furnished room whose
parts move, so the solve can also eject the body from furniture it penetrates
(``--scene_clearance``).
"""

import argparse
import time

import viser
import warp as wp

from robokit.assets.motions import parahome as parahome_motion
from robokit.helpers.combo_retarget import PRESETS, ComboRetargeter
from robokit.helpers.combo_retarget.loaders import find_parahome_intervals, load_parahome_clip
from robokit.utils.visualize_utils import ViserBatchUrdf, ViserHandSkeleton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None, help="ParaHome data/ dir (default: fetch from HF)")
    parser.add_argument("--sequence", default="s1", help="sequence id, s1..s207")
    parser.add_argument("--index", type=int, default=7, help="which mined interval to retarget")
    parser.add_argument("--list", action="store_true", help="print the sequence's intervals and exit")
    parser.add_argument("--no_scene", action="store_true", help="hide the furnished room, show only the target object")
    parser.add_argument("--scene_clearance", action="store_true", help="eject the body from penetrated furniture")
    args = parser.parse_args()
    data_root = args.data_root or str(parahome_motion.data_root())

    intervals = find_parahome_intervals(data_root, args.sequence)
    assert intervals, f"no hand-engaged intervals in {args.sequence}"
    if args.list:
        for i, it in enumerate(intervals):
            print(f"[{i:2d}] {it.start:5d}-{it.end:5d} {it.label}")
        return
    interval = intervals[args.index]
    print(f"retargeting [{args.index}] {interval.label}")

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    preset = PRESETS["g1_inspire_bimanual"]
    retargeter = ComboRetargeter(preset, device=device, load_meshes=True, scene_clearance=args.scene_clearance)
    combo = load_parahome_clip(
        data_root,
        interval,
        robot_height=preset.body.robot_height,
        device=device,
        load_scene=not args.no_scene,
    )

    t0 = time.time()
    qpos = retargeter.solve(combo)
    print(f"solved {qpos.shape[0]} frames in {time.time() - t0:.1f}s")
    ndof = retargeter.body_robot.spec.num_actuated_joints
    fps = combo.fps

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=6, height=6)
    robot_viz = ViserBatchUrdf(server, retargeter.body_robot, batch_size=1, root_node_name="/robot")
    object_handles = []
    for obj_name in combo.object_names:
        track = next(t for t in combo.hands.values() if t.object_name == obj_name)
        object_handles.append(
            server.scene.add_mesh_simple(
                f"/object/{obj_name}",
                track.object_mesh.vertices,
                track.object_mesh.faces,
                color=(200, 140, 60),
                opacity=0.9,
            )
        )
    hand_viz = {
        side: ViserHandSkeleton(
            server, f"/human_hand/{side}", color=(220, 40, 40) if side == "right" else (40, 90, 220)
        )
        for side in combo.sides
    }
    print(f"hands: {combo.sides} | objects: {combo.object_names}", flush=True)

    scene_handles = []
    scene = combo.scene
    if scene is not None:
        for idx, (name, part_mesh) in enumerate(zip(scene.names, scene.meshes)):
            shade = 120 + (hash(name) % 80)
            handle = server.scene.add_mesh_simple(
                f"/scene/{name}", part_mesh.vertices, part_mesh.faces, color=(shade, shade - 20, shade - 40)
            )
            handle.position = scene.poses[idx, 0, :3]
            handle.wxyz = scene.poses[idx, 0, 3:]
            if not scene.static[idx]:
                scene_handles.append((idx, handle))
        print(f"scene: {len(scene.names)} parts ({len(scene_handles)} moving)", flush=True)

    frame_slider = server.gui.add_slider("frame", min=0, max=qpos.shape[0] - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % qpos.shape[0]
        i = frame_slider.value
        with server.atomic():
            robot_viz.update_cfg(qpos[i : i + 1, 7 : 7 + ndof], T_world_base=qpos[i : i + 1, :7])
            for obj_i, handle in enumerate(object_handles):
                base = 7 + ndof + 7 * obj_i
                handle.position = qpos[i, base : base + 3]
                handle.wxyz = qpos[i, base + 3 : base + 7]
            for side, skeleton in hand_viz.items():
                skeleton.update(combo.hands[side].mano_overlay[i])
            for idx, handle in scene_handles:
                handle.position = scene.poses[idx, i, :3]
                handle.wxyz = scene.poses[idx, i, 3:]
        time.sleep(1.0 / fps)


if __name__ == "__main__":
    main()
