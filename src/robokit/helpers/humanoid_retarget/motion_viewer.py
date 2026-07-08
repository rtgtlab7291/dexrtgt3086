"""Motion viewer for retargeted robot motions using Viser."""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf


RENDER_PRESETS = {
    "paper": {},
    "clean_white": {},
    "dark": {},
    "sunset": {},
    "simple": {},
}


class SmplxMeshData:
    def __init__(
        self, vertices: np.ndarray, faces: np.ndarray, fps: float, pelvis_positions: Optional[np.ndarray] = None
    ):
        self.vertices = vertices
        self.faces = faces
        self.fps = fps
        self.pelvis_positions = pelvis_positions


def load_smplx_mesh_data(
    smplx_file: str,
    smplx_body_model_path: str,
    target_fps: float = 30.0,
    robot_height: float = 1.2,
) -> SmplxMeshData:
    from robokit.helpers.humanoid_retarget.utils.smplx_loader import load_smplx_file

    smplx_data, body_model, smplx_output, _human_height = load_smplx_file(smplx_file, smplx_body_model_path)

    vertices = smplx_output.vertices.detach().cpu().numpy()  # [T, 10475, 3]
    faces = body_model.faces.astype(np.int32)  # [F, 3]
    joints = smplx_output.joints.detach().cpu().numpy()  # [T, J, 3]
    pelvis = joints[:, 0, :]  # [T, 3] — joint 0 is pelvis

    src_fps = smplx_data["mocap_frame_rate"].item() if "mocap_frame_rate" in smplx_data else 60.0

    if target_fps < src_fps and src_fps % target_fps == 0:
        step = int(src_fps // target_fps)
        vertices = vertices[::step]
        pelvis = pelvis[::step]
        aligned_fps = target_fps
    else:
        aligned_fps = src_fps

    first_frame = vertices[0]
    human_mesh_height = first_frame[:, 2].max() - first_frame[:, 2].min()
    if human_mesh_height > 0:
        scale = robot_height / human_mesh_height
        vertices = vertices * scale
        pelvis = pelvis * scale

    vertices = vertices - pelvis[:, None, :]

    return SmplxMeshData(vertices=vertices, faces=faces, fps=aligned_fps, pelvis_positions=pelvis)


class MotionViewer:
    def __init__(
        self,
        urdf_path: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        render_preset: str = "simple",
        robot_color: Optional[Tuple[int, int, int]] = None,
        ground_z: float = -0.67,
    ):
        self.urdf_path = urdf_path

        self.server = viser.ViserServer(host=host, port=port)

        self.base_frame = self.server.scene.add_frame("/robot_base", show_axes=False)

        urdf = yourdfpy.URDF.load(urdf_path)
        self.robot_urdf = ViserUrdf(self.server, urdf, root_node_name="/robot_base")

        self._paused = False
        self._speed = 1.0
        self._current_frame = 0
        self._current_motion_idx = 0
        self._loop = True

        self._motion_files: List[str] = []
        self._motion_data_cache: Dict[int, dict] = {}

        self._smplx_data: Optional[SmplxMeshData] = None
        self._human_mesh_handle = None
        self._human_offset = np.array([0.0, 0.0, 0.0])
        self._show_human_mesh = True

        self.server.scene.add_grid("ground", width=10, height=10, cell_size=0.5, position=(0.0, 0.0, ground_z))

        print(f"\n{'=' * 60}")
        print("Motion Viewer started!")
        print(f"Open in browser: http://localhost:{port}")
        print(f"{'=' * 60}\n")

    def _update_robot(self, qpos: np.ndarray):
        root_pos = qpos[:3]
        root_quat_wxyz = qpos[3:7]
        dof_pos = qpos[7:]

        self.base_frame.position = root_pos
        self.base_frame.wxyz = root_quat_wxyz

        self.robot_urdf.update_cfg(dof_pos)

    def _update_robot_from_motion_data(self, motion_data: dict, frame_idx: int):
        root_pos = motion_data["root_pos"][frame_idx]
        root_rot_xyzw = motion_data["root_rot"][frame_idx]
        root_rot_wxyz = root_rot_xyzw[[3, 0, 1, 2]]
        dof_pos = motion_data["dof_pos"][frame_idx]

        self.base_frame.position = root_pos
        self.base_frame.wxyz = root_rot_wxyz

        self.robot_urdf.update_cfg(dof_pos)

    def _setup_human_mesh(
        self, smplx_data: SmplxMeshData, offset: np.ndarray, robot_root_pos: Optional[np.ndarray] = None
    ):
        self._smplx_data = smplx_data
        self._human_offset = offset
        pos = offset.copy()
        if robot_root_pos is not None:
            pos = pos + robot_root_pos
        initial_vertices = smplx_data.vertices[0] + pos
        self._human_mesh_handle = self.server.scene.add_mesh_simple(
            "/human_mesh",
            vertices=initial_vertices.astype(np.float32),
            faces=smplx_data.faces,
            opacity=0.5,
            color=(150, 200, 255),
        )

    def _update_human_mesh(self, frame_idx: int, robot_root_pos: Optional[np.ndarray] = None):
        if self._human_mesh_handle is None or self._smplx_data is None:
            return
        if not self._show_human_mesh:
            return
        idx = min(frame_idx, len(self._smplx_data.vertices) - 1)
        offset = self._human_offset.copy()
        if robot_root_pos is not None:
            offset = offset + robot_root_pos
        self._human_mesh_handle.vertices = (self._smplx_data.vertices[idx] + offset).astype(np.float32)

    def _add_human_mesh_controls(self):
        with self.server.gui.add_folder("Human Mesh"):
            show_checkbox = self.server.gui.add_checkbox("Show Human Mesh", initial_value=True)
            opacity_slider = self.server.gui.add_slider("Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.5)
            offset_y = self.server.gui.add_slider(
                "Y Offset", min=-3.0, max=3.0, step=0.1, initial_value=self._human_offset[1]
            )

            @show_checkbox.on_update
            def _(event):
                self._show_human_mesh = event.target.value
                if self._human_mesh_handle is not None:
                    self._human_mesh_handle.visible = event.target.value

            @opacity_slider.on_update
            def _(event):
                pass

            @offset_y.on_update
            def _(event):
                self._human_offset[1] = event.target.value

    def _setup_keypoints(
        self,
        link_names: List[str],
        radius: float = 0.02,
        color: Tuple[int, int, int] = (255, 50, 50),
    ):
        """Create sphere handles for keypoint visualization.

        Args:
            link_names: Link names to visualize.
            radius: Sphere radius in meters.
            color: RGB color tuple (0-255).
        """
        self._keypoint_handles: Dict[str, object] = {}

        for name in link_names:
            handle = self.server.scene.add_icosphere(
                f"/keypoints/{name}",
                radius=radius,
                color=color,
                position=(0.0, 0.0, 0.0),
            )
            self._keypoint_handles[name] = handle

    def _update_keypoints(self, keypoint_positions: Dict[str, np.ndarray]):
        """Update keypoint sphere positions for the current frame."""
        for name, pos in keypoint_positions.items():
            if name in self._keypoint_handles:
                self._keypoint_handles[name].position = pos

    def _setup_robot_keypoints(
        self,
        link_names: List[str],
        radius: float = 0.02,
        color: Tuple[int, int, int] = (50, 120, 255),
    ):
        """Create sphere handles for robot link keypoint visualization.

        Args:
            link_names: Link label names to visualize.
            radius: Sphere radius in meters.
            color: RGB color tuple (0-255).
        """
        self._robot_keypoint_handles: Dict[str, object] = {}

        for name in link_names:
            handle = self.server.scene.add_icosphere(
                f"/robot_keypoints/{name}",
                radius=radius,
                color=color,
                position=(0.0, 0.0, 0.0),
            )
            self._robot_keypoint_handles[name] = handle

    def _update_robot_keypoints(self, keypoint_positions: Dict[str, np.ndarray]):
        """Update robot keypoint sphere positions for the current frame."""
        for name, pos in keypoint_positions.items():
            if name in self._robot_keypoint_handles:
                self._robot_keypoint_handles[name].position = pos

    def _draw_human_skeleton(
        self,
        human_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ):
        for joint_name, (pos, quat_wxyz) in human_data.items():
            self.server.scene.add_frame(
                f"human/{joint_name}",
                wxyz=quat_wxyz,
                position=pos,
                axes_length=0.05,
                axes_radius=0.005,
            )

    def _clear_human_skeleton(self):
        pass

    def _add_playback_controls(self, num_frames: int):
        with self.server.gui.add_folder("Playback"):
            play_button = self.server.gui.add_button("Play/Pause")
            speed_slider = self.server.gui.add_slider("Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0)
            frame_slider = self.server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
            loop_checkbox = self.server.gui.add_checkbox("Loop", initial_value=True)

            @play_button.on_click
            def _(_):
                self._paused = not self._paused

            @speed_slider.on_update
            def _(event):
                self._speed = event.target.value

            @frame_slider.on_update
            def _(event):
                self._current_frame = int(event.target.value)

            @loop_checkbox.on_update
            def _(event):
                self._loop = event.target.value

            self._frame_slider = frame_slider

    def _add_dataset_controls(self):
        with self.server.gui.add_folder("Dataset"):
            self._motion_label = self.server.gui.add_text("Motion", initial_value="0 / 0", disabled=True)
            prev_button = self.server.gui.add_button("Previous")
            next_button = self.server.gui.add_button("Next")

            @prev_button.on_click
            def _(_):
                if len(self._motion_files) > 0:
                    self._current_motion_idx = (self._current_motion_idx - 1) % len(self._motion_files)
                    self._current_frame = 0
                    self._update_motion_label()

            @next_button.on_click
            def _(_):
                if len(self._motion_files) > 0:
                    self._current_motion_idx = (self._current_motion_idx + 1) % len(self._motion_files)
                    self._current_frame = 0
                    self._update_motion_label()

    def _update_motion_label(self):
        if len(self._motion_files) > 0:
            name = Path(self._motion_files[self._current_motion_idx]).stem
            self._motion_label.value = f"{self._current_motion_idx + 1}/{len(self._motion_files)}: {name}"

    def play_motion(
        self,
        qpos_array: np.ndarray,
        human_data: Optional[List[Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
        fps: float = 30.0,
        loop: bool = True,
        smplx_data: Optional[SmplxMeshData] = None,
        human_offset: Optional[np.ndarray] = None,
        keypoint_positions: Optional[List[Dict[str, np.ndarray]]] = None,
    ):
        num_frames = len(qpos_array)
        frame_time = 1.0 / fps
        self._current_frame = 0
        self._paused = False
        self._loop = loop

        if smplx_data is not None:
            offset = human_offset if human_offset is not None else np.array([0.0, 0.0, 0.0])
            self._setup_human_mesh(smplx_data, offset, robot_root_pos=qpos_array[0][:3])
            self._add_human_mesh_controls()

        self._add_playback_controls(num_frames)

        print(f"Playing {num_frames} frames @ {fps} fps")
        print("Controls in browser GUI")

        last_time = time.time()

        try:
            while True:
                current_time = time.time()
                dt = current_time - last_time

                if not self._paused and dt >= frame_time / self._speed:
                    self._update_robot(qpos_array[self._current_frame])

                    if human_data is not None and self._current_frame < len(human_data):
                        self._draw_human_skeleton(human_data[self._current_frame])

                    robot_root = qpos_array[self._current_frame][:3]
                    self._update_human_mesh(self._current_frame, robot_root_pos=robot_root)

                    if keypoint_positions is not None:
                        self._update_keypoints(keypoint_positions[self._current_frame])

                    if hasattr(self, "_frame_slider"):
                        self._frame_slider.value = self._current_frame

                    # Advance frame
                    self._current_frame += 1
                    if self._current_frame >= num_frames:
                        if self._loop:
                            self._current_frame = 0
                        else:
                            break

                    last_time = current_time

                time.sleep(0.001)  # Small sleep to prevent busy loop

        except KeyboardInterrupt:
            print("\nStopped by user")

    def _load_motion_data(self, idx: int) -> Optional[dict]:
        """Load motion data from cache or file."""
        if idx in self._motion_data_cache:
            return self._motion_data_cache[idx]

        try:
            with open(self._motion_files[idx], "rb") as f:
                data = pickle.load(f)
            # Cache last N motions
            if len(self._motion_data_cache) > 10:
                oldest = min(self._motion_data_cache.keys())
                del self._motion_data_cache[oldest]
            self._motion_data_cache[idx] = data
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load {self._motion_files[idx]}: {e}")
            return None

    def browse_dataset(
        self,
        dataset_folder: str,
    ):
        self._motion_files = []
        for root, _, files in os.walk(dataset_folder):
            for f in sorted(files):
                if f.endswith(".pkl"):
                    self._motion_files.append(os.path.join(root, f))

        if len(self._motion_files) == 0:
            print(f"No .pkl files found in {dataset_folder}")
            return

        print(f"Found {len(self._motion_files)} motions")

        self._current_motion_idx = 0
        self._current_frame = 0
        self._paused = False

        first_motion = self._load_motion_data(0)
        if first_motion is None:
            return

        fps = first_motion.get("fps", 30.0)
        num_frames = len(first_motion["root_pos"])

        self._add_playback_controls(num_frames)
        self._add_dataset_controls()
        self._update_motion_label()

        print("Use browser GUI to navigate")

        frame_time = 1.0 / fps
        last_time = time.time()
        last_motion_idx = -1

        while True:
            if self._current_motion_idx != last_motion_idx:
                motion_data = self._load_motion_data(self._current_motion_idx)
                if motion_data is None:
                    self._current_motion_idx = (self._current_motion_idx + 1) % len(self._motion_files)
                    continue
                fps = motion_data.get("fps", 30.0)
                frame_time = 1.0 / fps
                num_frames = len(motion_data["root_pos"])
                if hasattr(self, "_frame_slider"):
                    self._frame_slider.max = num_frames - 1
                last_motion_idx = self._current_motion_idx
            else:
                motion_data = self._load_motion_data(self._current_motion_idx)

            if motion_data is None:
                continue

            num_frames = len(motion_data["root_pos"])

            current_time = time.time()
            dt = current_time - last_time

            if not self._paused and dt >= frame_time / self._speed:
                self._update_robot_from_motion_data(motion_data, self._current_frame)

                if hasattr(self, "_frame_slider"):
                    self._frame_slider.value = self._current_frame

                self._current_frame += 1
                if self._current_frame >= num_frames:
                    if self._loop:
                        self._current_frame = 0
                    else:
                        self._current_motion_idx = (self._current_motion_idx + 1) % len(self._motion_files)
                        self._current_frame = 0
                        self._update_motion_label()

                last_time = current_time

            time.sleep(0.001)

    def close(self):
        pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="View retargeted robot motions")
    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Path to robot URDF",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default=None,
        help="Path to single motion .pkl file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset folder for browsing",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS (default: from motion file or 30)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    args = parser.parse_args()

    if args.motion is None and args.dataset is None:
        parser.error("Either --motion or --dataset must be specified")

    viewer = MotionViewer(urdf_path=args.urdf, port=args.port)

    if args.motion is not None:
        with open(args.motion, "rb") as f:
            motion_data = pickle.load(f)

        root_pos = motion_data["root_pos"]
        root_rot_xyzw = motion_data["root_rot"]
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
        dof_pos = motion_data["dof_pos"]

        qpos_array = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)
        fps = args.fps or motion_data.get("fps", 30.0)

        viewer.play_motion(qpos_array, fps=fps, loop=True)
    else:
        viewer.browse_dataset(args.dataset)

    viewer.close()


if __name__ == "__main__":
    main()
