#!/usr/bin/env python3
"""
Interactive SMPL-X to Robot Retargeting Browser.

Browse raw SMPL-X datasets, select a motion, retarget on-the-fly, and visualize.

Usage:
    uv run python examples/humanoid_retarget/interactive_retarget.py \
        --dataset motion_data/raw/ACCAD --robot gr3
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import viser
import warp as wp
import yourdfpy
from natsort import natsorted
from viser.extras import ViserUrdf

from robokit.helpers.humanoid_retarget.humanoid_retargeting_online import (
    HumanoidRetargetingHelper,
    RetargetConfig,
)
from robokit.helpers.humanoid_retarget.utils.smplx_loader import (
    get_smplx_frames_offline_fast,
    load_smplx_file,
)
from robokit.lie.warp_se3 import WarpSE3
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


def collect_smplx_files(folder: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for filename in natsorted(filenames):
            if filename.endswith(".npz") and not filename.endswith("_stagei.npz"):
                files.append(os.path.join(root, filename))
    return files


TARGET_COLOR = (255, 80, 80)  # red - what IK aims for
ROBOT_COLOR = (80, 200, 80)  # green - what robot actually achieves
FRAME_AXES_LENGTH = 0.08
FRAME_AXES_RADIUS = 0.004


def _get_robot_link_poses(
    robot: Robot,
    robot_pose: np.ndarray,
    link_indices: Dict[str, int],
    device: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    root_xyz_quat = robot_pose[:7].reshape(1, 7).astype(np.float32)
    q = robot_pose[7:].reshape(1, -1).astype(np.float32)

    wp_device = wp.get_device(device)
    base_se3 = WarpSE3(wp.from_numpy(root_xyz_quat, dtype=wp_vec7, device=wp_device))
    q_wp = wp.from_numpy(q, dtype=wp.float32, device=wp_device)
    state = robot.state(q=q_wp, T_world_base=base_se3)
    state = robot.forward_kinematics(state)

    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, idx in link_indices.items():
        T = state.get_T_world_link(idx)
        pos = T.xyz.numpy()[0].astype(np.float32)
        q_wxyz = T.quat_wxyz.numpy()[0].astype(np.float32)
        poses[label] = (pos, q_wxyz)
    return poses


class InteractiveRetargetViewer:
    def __init__(
        self,
        dataset_folder: str,
        urdf_path: str,
        config_path: str,
        smplx_body_model_path: str,
        device: str = "cuda:0",
        target_fps: int = 30,
        port: int = 8080,
    ):
        self.dataset_folder = dataset_folder
        self.smplx_body_model_path = smplx_body_model_path
        self.target_fps = target_fps
        self.device = device

        self.smplx_files = collect_smplx_files(dataset_folder)
        print(f"Found {len(self.smplx_files)} SMPL-X files")

        self.file_display_names = []
        for f in self.smplx_files:
            rel_path = os.path.relpath(f, dataset_folder)
            self.file_display_names.append(rel_path)

        wp.init()

        print(f"Loading robot from {urdf_path}...")
        self.robot = Robot.load(urdf_path, backend="warp")

        print(f"Loading config from {config_path}...")
        self.base_config = RetargetConfig.from_yaml(config_path)

        self.helper: Optional[HumanoidRetargetingHelper] = None

        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        self.base_frame = self.server.scene.add_frame("/robot_base", show_axes=False)
        urdf = yourdfpy.URDF.load(urdf_path)
        self.robot_urdf = ViserUrdf(self.server, urdf, root_node_name="/robot_base")

        self.server.scene.add_grid("ground", width=10, height=10, cell_size=0.5, position=(0.0, 0.0, 0.0))

        self._current_file_idx = 0
        self._current_frame = 0
        self._paused = True
        self._speed = 1.0
        self._loop = True

        self._qpos_array: Optional[np.ndarray] = None
        self._target_frames: Optional[List[Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None
        self._robot_link_indices: Dict[str, int] = {}
        self._link_mapping_keys: List[str] = []
        self._fps = 30.0
        self._num_frames = 0
        self._is_retargeting = False
        self._page_number = None
        self._show_frames = False
        self._frame_handles: Dict[str, viser.FrameHandle] = {}

        self._setup_gui()

        print(f"\n{'=' * 60}")
        print("Interactive Retarget Viewer started!")
        print(f"Open in browser: http://localhost:{port}")
        print(f"{'=' * 60}\n")

    def _setup_gui(self):
        with self.server.gui.add_folder("Dataset Browser", expand_by_default=True):
            self._file_dropdown = self.server.gui.add_dropdown(
                "Motion File",
                options=self.file_display_names[:100]
                if len(self.file_display_names) > 100
                else self.file_display_names,
                initial_value=self.file_display_names[0] if self.file_display_names else "",
            )
            self._status_text = self.server.gui.add_text(
                "Status", initial_value="Select a motion and click Retarget", disabled=True
            )
            self._retarget_button = self.server.gui.add_button("Retarget & Play")

            if len(self.file_display_names) > 100:
                self._page_number = self.server.gui.add_number(
                    "Page (100 per page)", initial_value=0, min=0, max=len(self.file_display_names) // 100, step=1
                )

                @self._page_number.on_update
                def _(event):
                    page = int(event.target.value)
                    start = page * 100
                    end = min(start + 100, len(self.file_display_names))
                    self._file_dropdown.options = self.file_display_names[start:end]

            @self._retarget_button.on_click
            def _(_):
                self._start_retargeting()

        with self.server.gui.add_folder("Playback", expand_by_default=True):
            self._play_button = self.server.gui.add_button("Play/Pause")
            self._speed_slider = self.server.gui.add_slider("Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0)
            self._frame_slider = self.server.gui.add_slider("Frame", min=0, max=100, step=1, initial_value=0)
            self._loop_checkbox = self.server.gui.add_checkbox("Loop", initial_value=True)

            @self._play_button.on_click
            def _(_):
                if self._qpos_array is not None:
                    self._paused = not self._paused

            @self._speed_slider.on_update
            def _(event):
                self._speed = event.target.value

            @self._frame_slider.on_update
            def _(event):
                self._current_frame = int(event.target.value)

            @self._loop_checkbox.on_update
            def _(event):
                self._loop = event.target.value

        with self.server.gui.add_folder("Frame Comparison", expand_by_default=True):
            self._frame_checkbox = self.server.gui.add_checkbox("Show Target/Robot Frames", initial_value=False)
            self._axes_length_slider = self.server.gui.add_slider(
                "Axes Length", min=0.02, max=0.20, step=0.01, initial_value=FRAME_AXES_LENGTH
            )

            @self._frame_checkbox.on_update
            def _(event):
                self._show_frames = event.target.value
                self._set_frame_visibility(self._show_frames)

            @self._axes_length_slider.on_update
            def _(_):
                self._rebuild_comparison_frames()

        with self.server.gui.add_folder("Info", expand_by_default=False):
            self._info_frames = self.server.gui.add_text("Frames", initial_value="-", disabled=True)
            self._info_fps = self.server.gui.add_text("FPS", initial_value="-", disabled=True)
            self._info_height = self.server.gui.add_text("Human Height", initial_value="-", disabled=True)

    def _start_retargeting(self):
        if self._is_retargeting:
            return

        selected_name = self._file_dropdown.value
        if not selected_name:
            return

        try:
            file_idx = self.file_display_names.index(selected_name)
        except ValueError:
            page_start = int(self._page_number.value) * 100 if self._page_number is not None else 0
            file_idx = page_start + self._file_dropdown.options.index(selected_name)

        smplx_file = self.smplx_files[file_idx]

        self._is_retargeting = True
        self._status_text.value = "Loading SMPL-X..."
        self._paused = True

        smplx_data, body_model, smplx_output, human_height = load_smplx_file(smplx_file, self.smplx_body_model_path)

        frames, fps, _ = get_smplx_frames_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=self.target_fps)

        self._info_frames.value = str(len(frames))
        self._info_fps.value = f"{fps:.1f}"
        self._info_height.value = f"{human_height:.2f}m"

        config = self.base_config.with_actual_height(human_height)

        self._status_text.value = "Initializing retargeter..."
        self.helper = HumanoidRetargetingHelper(self.robot, config, device=self.device)

        link_name_to_idx = {name: i for i, name in enumerate(self.robot.link_names)}
        self._robot_link_indices = {}
        self._link_mapping_keys = []
        for robot_link in config.link_mapping:
            if robot_link in link_name_to_idx:
                self._robot_link_indices[robot_link] = link_name_to_idx[robot_link]
                self._link_mapping_keys.append(robot_link)

        self._status_text.value = f"Retargeting {len(frames)} frames..."

        qpos_list = []
        target_frame_list: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
        for i, frame in enumerate(frames):
            qpos, targets = self.helper.retarget(frame, offset_to_ground=False)
            qpos_list.append(qpos)
            target_poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for robot_link, mapping in config.link_mapping.items():
                if mapping.human_joint in targets:
                    pos, q_wxyz = targets[mapping.human_joint]
                    target_poses[robot_link] = (pos.copy(), q_wxyz.copy())
            target_frame_list.append(target_poses)
            if i % 30 == 0:
                self._status_text.value = f"Retargeting... {i}/{len(frames)}"

        wp.synchronize()

        self._qpos_array = np.array(qpos_list)
        self._target_frames = target_frame_list
        self._fps = fps
        self._num_frames = len(qpos_list)
        self._current_frame = 0

        self._frame_slider.max = self._num_frames - 1
        self._frame_slider.value = 0

        self._rebuild_comparison_frames()

        self._status_text.value = f"Ready! {self._num_frames} frames @ {fps:.0f} fps"
        self._is_retargeting = False
        self._paused = False

    def _rebuild_comparison_frames(self):
        for handle in self._frame_handles.values():
            handle.remove()
        self._frame_handles.clear()

        if not self._link_mapping_keys:
            return

        axes_length = self._axes_length_slider.value
        axes_radius = axes_length * 0.05
        for link_name in self._link_mapping_keys:
            self._frame_handles[f"target/{link_name}"] = self.server.scene.add_frame(
                f"/comparison/target/{link_name}",
                axes_length=axes_length,
                axes_radius=axes_radius,
                origin_radius=axes_radius * 3,
                origin_color=TARGET_COLOR,
            )
            self._frame_handles[f"robot/{link_name}"] = self.server.scene.add_frame(
                f"/comparison/robot/{link_name}",
                axes_length=axes_length,
                axes_radius=axes_radius,
                origin_radius=axes_radius * 3,
                origin_color=ROBOT_COLOR,
            )

        self._set_frame_visibility(self._show_frames)

    def _set_frame_visibility(self, visible: bool):
        for handle in self._frame_handles.values():
            handle.visible = visible

    def _update_comparison_frames(self, frame_idx: int, qpos: np.ndarray):
        if not self._show_frames or self._target_frames is None:
            return

        target_poses = self._target_frames[frame_idx]
        robot_poses = _get_robot_link_poses(self.robot, qpos, self._robot_link_indices, self.device)

        for link_name in self._link_mapping_keys:
            target_handle = self._frame_handles.get(f"target/{link_name}")
            robot_handle = self._frame_handles.get(f"robot/{link_name}")

            if target_handle is not None and link_name in target_poses:
                pos, q_wxyz = target_poses[link_name]
                target_handle.position = pos
                target_handle.wxyz = q_wxyz

            if robot_handle is not None and link_name in robot_poses:
                pos, q_wxyz = robot_poses[link_name]
                robot_handle.position = pos
                robot_handle.wxyz = q_wxyz

    def _update_robot(self, qpos: np.ndarray):
        root_pos = qpos[:3]
        root_quat_wxyz = qpos[3:7]
        dof_pos = qpos[7:]

        self.base_frame.position = root_pos
        self.base_frame.wxyz = root_quat_wxyz
        self.robot_urdf.update_cfg(dof_pos)

    def run(self):
        frame_time = 1.0 / self._fps
        last_time = time.time()

        while True:
            if self._qpos_array is not None and not self._paused:
                current_time = time.time()
                dt = current_time - last_time

                if dt >= frame_time / self._speed:
                    qpos = self._qpos_array[self._current_frame]
                    self._update_robot(qpos)
                    self._update_comparison_frames(self._current_frame, qpos)
                    self._frame_slider.value = self._current_frame

                    self._current_frame += 1
                    if self._current_frame >= self._num_frames:
                        if self._loop:
                            self._current_frame = 0
                        else:
                            self._paused = True
                            self._current_frame = self._num_frames - 1

                    last_time = current_time
                    frame_time = 1.0 / self._fps

            time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser(description="Interactive SMPL-X to Robot Retargeting Browser")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to raw SMPL-X dataset folder (can be nested)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot shorthand (e.g. g1, h1_2, bhl); resolves to ik_config/smplx_{robot}.yaml",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to robot URDF (overrides config urdf_path)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to retarget config YAML (overrides --robot)",
    )
    parser.add_argument(
        "--smplx_body_model",
        type=str,
        default=None,
        help="Path to SMPL-X body model folder",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="Target FPS for retargeting",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for retargeting",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    if args.config is not None:
        config_path = args.config
    elif args.robot is not None:
        config_path = str(script_dir / "ik_config" / f"smplx_{args.robot}.yaml")
    else:
        config_path = str(script_dir / "ik_config" / "smplx_g1.yaml")

    config = RetargetConfig.from_yaml(config_path)

    if args.urdf is not None:
        urdf_path = args.urdf
    elif config.urdf_path is not None:
        urdf_path = config.urdf_path
    else:
        urdf_path = str(project_root / "assets" / "robot_description" / "unitree_g1" / "g1_custom_collision_29dof.urdf")

    smplx_body_model_path = args.smplx_body_model
    if smplx_body_model_path is None:
        smplx_body_model_path = str(project_root / "assets" / "body_models")

    viewer = InteractiveRetargetViewer(
        dataset_folder=args.dataset,
        urdf_path=urdf_path,
        config_path=config_path,
        smplx_body_model_path=smplx_body_model_path,
        device=args.device,
        target_fps=args.target_fps,
        port=args.port,
    )

    viewer.run()


if __name__ == "__main__":
    main()
