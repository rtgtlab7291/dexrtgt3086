"""Browse SAGA human-object interaction: pick a subject and a clip, solve once, then scrub it.

Data-driven by default — captured MANO fingers plus the labelled-contact refine — and the green
markers are the dataset's own contact labels. ``--grasp`` swaps the finger stage for force-closure
grasp synthesis (``SAGA_GRASP``): one canonical grasp held fixed in the object frame across the
carry interval, drawn as the pads (purple) chasing their object-surface targets (green).
"""

import argparse
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets.motions import saga as saga_motion
from robokit.helpers.combo_retarget import PRESETS, ComboClip, ComboRetargeter
from robokit.helpers.combo_retarget.clip import FINGERS
from robokit.helpers.combo_retarget.grasp import SAGA_GRASP, solve_combo_grasp
from robokit.helpers.combo_retarget.loaders import load_saga_clip
from robokit.utils.visualize_utils import ViserBatchUrdf, ViserHandSkeleton


MARKER_RADIUS = 0.004
# one solved clip: qpos, its clip, and the contact markers to draw (targets, pads or None, mask)
Solved = Tuple[np.ndarray, ComboClip, np.ndarray, Optional[np.ndarray], np.ndarray]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="subject, for example test/s10")
    parser.add_argument("--clip_index", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0, help="0 keeps all frames")
    parser.add_argument("--grasp", action="store_true", help="synthesize a force-closure grasp instead of refining")
    parser.add_argument("--optim_iters", type=int, default=SAGA_GRASP.iters, help="LM iterations per stage")
    parser.add_argument("--num_seeds", type=int, default=SAGA_GRASP.num_seeds, help="seeds per contact topology")
    # Pins the sampler only: the GPU solve itself is not reproducible run to run (a fixed seed still
    # swings this clip's safe-candidate count 12-16/16 and its wrist shift by ~10 mm).
    parser.add_argument("--seed", type=int, default=2, help="grasp sampling seed")
    parser.add_argument("--headless", action="store_true", help="solve the first clip and exit")
    args = parser.parse_args()

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    preset = PRESETS["g1_inspire_right"]  # SAGA captures the right hand only
    config = replace(SAGA_GRASP, iters=args.optim_iters, num_seeds=args.num_seeds)
    # synthesis owns the fingers when it runs, so the data-driven contact refine steps aside
    retargeter = ComboRetargeter(preset, device=device, load_meshes=True, contact_refine=not args.grasp)
    root = saga_motion.data_root()
    sources = sorted(str(path.relative_to(root)) for path in root.glob("*/*") if path.is_dir())

    def clips_of(source: str) -> Tuple[str, ...]:
        return tuple(sorted(path.stem for path in (root / source).glob("*.npz")))

    def solve(source: str, clip_name: str) -> Solved:
        print(f"retargeting saga {source}/{clip_name} (grasp={args.grasp})")
        combo = load_saga_clip(
            str(root / source / f"{clip_name}.npz"), robot_height=preset.body.robot_height, device=device
        )
        track = combo.hands["right"]
        labels = np.asarray(track.contact_mask.reshape(len(track.contact_mask), len(FINGERS), 2).any(axis=2))
        print(
            "SAGA labeled contact frames: "
            + ", ".join(f"{finger}={int(labels[:, i].sum())}" for i, finger in enumerate(FINGERS))
        )
        start = time.time()
        if args.grasp:
            np.random.seed(args.seed)  # the grasp sampler draws its Warp seed from the global RNG
            result = solve_combo_grasp(
                combo, preset, device, config=config, max_frames=args.max_frames, body_retargeter=retargeter
            )
            assert result.qpos is not None
            qpos = result.qpos
            targets, pads, mask = (
                result.targets_world["right"],
                result.contacts_world["right"],
                result.contact_masks["right"],
            )
        else:
            qpos = retargeter.solve(combo, max_frames=args.max_frames)
            frames = qpos.shape[0]
            # labels live in the native world; bridge them onto the scaled body world the qpos is in
            targets = (
                track.contact_points[:frames]
                - track.object_pose_native[:frames, None, :3]
                + track.object_pose_world[:frames, None, :3]
            )
            pads, mask = None, track.contact_mask[:frames]
        print(f"solved {qpos.shape[0]} frames in {time.time() - start:.1f}s")
        return qpos, combo, targets, pads, mask

    initial_source = args.source or sources[0]
    assert initial_source in sources, f"unknown source: {initial_source}"
    initial_clips = clips_of(initial_source)
    assert 0 <= args.clip_index < len(initial_clips), f"clip index must be in [0, {len(initial_clips) - 1}]"
    initial_clip = initial_clips[args.clip_index]
    if args.headless:
        solve(initial_source, initial_clip)
        return

    # --- viewer: source panel + playback controls (solves are cached per selection) ---
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=6, height=6)
    with server.gui.add_folder("Source"):
        source_dd = server.gui.add_dropdown("Subject", sources, initial_value=initial_source)
        clip_dd = server.gui.add_dropdown("Clip", initial_clips, initial_value=initial_clip)
    frame_slider = server.gui.add_slider("frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    show_hand = server.gui.add_checkbox("human hand", True)
    show_contacts = server.gui.add_checkbox("grasp contacts" if args.grasp else "SAGA contacts", True)
    status = server.gui.add_text("status", initial_value="solving...", disabled=True)
    dirty = [True]

    def on_source(_: object) -> None:
        clip_dd.options = clips_of(source_dd.value)
        clip_dd.value = clip_dd.options[0]
        dirty[0] = True

    source_dd.on_update(on_source)
    clip_dd.on_update(lambda _: dirty.__setitem__(0, True))

    robot_viz = ViserBatchUrdf(server, retargeter.body_robot, batch_size=1, root_node_name="/robot")
    ndof = retargeter.body_robot.spec.num_actuated_joints
    solved: Dict[Tuple[str, str], Solved] = {}
    current: Optional[Solved] = None
    object_handle: Optional[viser.MeshHandle] = None
    hand_viz: Optional[ViserHandSkeleton] = None
    markers: List[Tuple[viser.IcosphereHandle, Optional[viser.IcosphereHandle]]] = []
    built = ("", "")
    while True:
        if dirty[0]:
            # Snapshot the dropdowns ONCE (callbacks fire on GUI worker threads at any time).
            dirty[0] = False
            selection = (source_dd.value, clip_dd.value)
            if selection != built and clip_dd.value in clips_of(source_dd.value):
                if selection not in solved:
                    status.value = f"solving {clip_dd.value}..."
                    solved[selection] = solve(*selection)
                current = solved[selection]
                qpos, combo, targets, pads, mask = current
                track = combo.hands["right"]
                assert isinstance(track.object_mesh, trimesh.Trimesh)

                for node in ("/object", "/human_hand", "/contacts"):
                    server.scene.remove_by_name(node)
                object_handle = server.scene.add_mesh_simple(
                    "/object", track.object_mesh.vertices, track.object_mesh.faces, color=(200, 140, 60), opacity=0.9
                )
                hand_viz = ViserHandSkeleton(server, "/human_hand", color=(220, 40, 40))
                markers = [
                    (
                        server.scene.add_icosphere(
                            f"/contacts/target_{slot}", radius=MARKER_RADIUS, color=(40, 220, 80)
                        ),
                        None
                        if pads is None
                        else server.scene.add_icosphere(
                            f"/contacts/pad_{slot}", radius=MARKER_RADIUS * 0.7, color=(190, 50, 230)
                        ),
                    )
                    for slot in range(mask.shape[1])
                ]
                frame_slider.value = 0
                frame_slider.max = qpos.shape[0] - 1
                status.value = f"{selection[0]}/{selection[1]}: {qpos.shape[0]} frames"
                built = selection

        # --- viewer: playback (the object pose is the qpos tail) ---
        if current is not None and object_handle is not None and hand_viz is not None:
            qpos, combo, targets, pads, mask = current
            track = combo.hands["right"]
            frame = min(frame_slider.value, qpos.shape[0] - 1)
            if playing.value:
                frame = (frame + 1) % qpos.shape[0]
                frame_slider.value = frame
            with server.atomic():
                robot_viz.update_cfg(qpos[frame : frame + 1, 7 : 7 + ndof], T_world_base=qpos[frame : frame + 1, :7])
                object_handle.position = qpos[frame, 7 + ndof : 10 + ndof]
                object_handle.wxyz = qpos[frame, 10 + ndof : 14 + ndof]
                hand_viz.update(track.mano_overlay[frame])
                hand_viz.visible = show_hand.value
                for slot, (target, pad) in enumerate(markers):
                    visible = bool(show_contacts.value and mask[frame, slot])
                    target.visible = visible
                    target.position = targets[frame, slot]
                    if pad is not None and pads is not None:
                        pad.visible = visible
                        pad.position = pads[frame, slot]
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
