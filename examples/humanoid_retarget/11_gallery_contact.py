"""HOI contact-retargeting gallery: pick a robot and an OMOMO clip; solve once, then scrub it."""

import importlib
import time
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
import viser
import warp as wp

from robokit.assets.motions import omomo
from robokit.assets.objects import omomo as omomo_objects
from robokit.geom import MeshGeom, PlaneGeom, WarpScene
from robokit.helpers.humanoid_retarget import (
    HumanoidRetargetingOnline,
    HumanoidRetargetingOnlineConfig,
    build_interaction_mesh_frame,
)
from robokit.helpers.humanoid_retarget.loaders import load_omomo_clip
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_to_matrix


ROBOTS = ["g1", "gr3", "h1_2", "bhl"]
FPS = 30.0


def load_preset(robot_name: str) -> HumanoidRetargetingOnlineConfig:
    module = importlib.import_module(f"robokit.helpers.humanoid_retarget.presets.{robot_name}_smplh")
    return getattr(module, f"{robot_name}_smplh")


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    clips = omomo.clip_names()  # listed over the HF API; each clip downloads only when selected

    # --- viewer: source panel + playback controls (solves are cached per selection) ---
    server = viser.ViserServer()
    with server.gui.add_folder("Source"):
        robot_dd = server.gui.add_dropdown("Robot", tuple(ROBOTS), initial_value=ROBOTS[0])
        clip_dd = server.gui.add_dropdown("Clip", tuple(clips), initial_value=clips[0])
    frame_slider = server.gui.add_slider("frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    status = server.gui.add_text("status", initial_value="solving...", disabled=True)

    dirty = [True]

    def mark_dirty(_) -> None:
        dirty[0] = True

    robot_dd.on_update(mark_dirty)
    clip_dd.on_update(mark_dirty)

    server.scene.add_grid("/ground", width=6, height=6)
    box_handle: Optional[viser.GlbHandle] = None

    robots: Dict[str, Robot] = {}
    solved: Dict[Tuple[str, str], np.ndarray] = {}
    robot_viz: Optional[ViserBatchUrdf] = None
    qpos = np.zeros((1, 43), dtype=np.float32)
    ndof = 29
    built = ("", "")
    built_obj = ""
    while True:
        if dirty[0]:
            # Snapshot the dropdowns ONCE (callbacks fire on GUI worker threads at any time).
            dirty[0] = False
            selection = (robot_dd.value, clip_dd.value)
            if selection != built:
                robot_name, clip_name = selection
                preset = load_preset(robot_name)
                if robot_name not in robots:
                    robots[robot_name] = preset.load_robot(load_meshes=True)
                robot = robots[robot_name]
                obj = clip_name.split("_")[1]
                box_path = str(omomo_objects.mesh_path(obj))

                # --- retarget: interaction-mesh solve over the whole clip, cached per selection ---
                if selection not in solved:
                    status.value = f"solving {clip_name} on {robot_name}... (~20 s on GPU)"
                    print(f"retargeting {clip_name} -> {robot_name}...")
                    assert preset.robot_height is not None
                    T_world_human, object_poses, _, interaction_scale = load_omomo_clip(
                        str(omomo.clip_path(clip_name)),
                        str(omomo.height_dict_path()),
                        clip_name.split("_", maxsplit=1)[0],
                        preset.human_joint_names,
                        robot_height=preset.robot_height,
                    )
                    box = trimesh.load(box_path, force="mesh")
                    assert isinstance(box, trimesh.Trimesh)
                    box.apply_scale(interaction_scale)
                    assert preset.interaction is not None
                    interaction_points, _ = trimesh.sample.sample_surface_even(
                        box, preset.interaction.num_object_points, seed=42
                    )
                    T_world_object = np.tile(np.eye(4, dtype=np.float32), (len(object_poses), 1, 1))
                    T_world_object[:, :3, :3] = quaternion_to_matrix(object_poses[:, 3:])
                    T_world_object[:, :3, 3] = object_poses[:, :3]
                    identity = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
                    object_geom = MeshGeom([box], np.array([0, 1], dtype=np.int32), poses=identity)
                    scene = WarpScene(1, device).add(object_geom).add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
                    retargeter = HumanoidRetargetingOnline(preset, robot=robot, scene=scene, device=device)
                    key_indices = [
                        retargeter.human_joint_names.index(entry.human_joint) for entry in preset.link_mapping.values()
                    ]
                    object_geom.update(poses=wp.from_numpy(T_world_object[:1], dtype=wp.mat44, device=device))
                    retargeter.interaction_task.set_frame(
                        *build_interaction_mesh_frame(
                            T_world_human[0, key_indices, :3], interaction_points, object_poses[0]
                        )
                    )
                    retargeter.warmup(1)
                    root = T_world_human[0, retargeter.human_joint_names.index(preset.human_root_name), :3]
                    yaw = np.arctan2(object_poses[0, 1] - root[1], object_poses[0, 0] - root[0])
                    T_world_base = np.array([*root, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)
                    retargeter.reset(wp.from_numpy(T_world_base[None], dtype=wp_vec7, device=device))
                    t0 = time.time()
                    qpos = []
                    for i in range(len(T_world_human)):
                        object_geom.update(
                            poses=wp.from_numpy(T_world_object[i : i + 1], dtype=wp.mat44, device=device)
                        )
                        retargeter.interaction_task.set_frame(
                            *build_interaction_mesh_frame(
                                T_world_human[i, key_indices, :3], interaction_points, object_poses[i]
                            )
                        )
                        qpos.append(retargeter.solve_numpy(T_world_human[i : i + 1])[0])
                    solved[selection] = np.concatenate([np.stack(qpos), object_poses], axis=1)
                    print(f"solved {solved[selection].shape[0]} frames in {time.time() - t0:.1f}s")
                qpos = solved[selection]
                ndof = robot.spec.num_actuated_joints

                # --- viewer: swap in the selected robot and object ---
                if robot_viz is not None:
                    robot_viz.remove()
                robot_viz = ViserBatchUrdf(server, robot, batch_size=1, root_node_name="/robot")
                if obj != built_obj:
                    box = trimesh.load(box_path, force="mesh")
                    assert isinstance(box, trimesh.Trimesh)
                    box.apply_scale(interaction_scale)
                    box_handle = server.scene.add_mesh_trimesh("/object", box)
                    built_obj = obj
                frame_slider.value = 0
                frame_slider.max = qpos.shape[0] - 1
                status.value = f"{clip_name} on {robot_name}: {qpos.shape[0]} frames"
                built = selection

        # --- viewer: playback (box pose lives in qpos[:, -7:]) ---
        idx = min(frame_slider.value, qpos.shape[0] - 1)
        if playing.value:
            idx = (idx + 1) % qpos.shape[0]
            frame_slider.value = idx
        if robot_viz is not None and box_handle is not None:
            with server.atomic():
                robot_viz.update_cfg(qpos[idx : idx + 1, 7 : 7 + ndof], T_world_base=qpos[idx : idx + 1, :7])
                box_handle.position = qpos[idx, 7 + ndof : 7 + ndof + 3]
                box_handle.wxyz = qpos[idx, 7 + ndof + 3 : 7 + ndof + 7]
        time.sleep(1.0 / FPS)


if __name__ == "__main__":
    main()
