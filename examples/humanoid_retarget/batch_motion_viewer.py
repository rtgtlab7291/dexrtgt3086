#!/usr/bin/env python3
"""Batch motion viewer: visualize many robots playing the same retargeted motion.

Uses GPU-instanced mesh rendering (ViserBatchUrdf) for efficient rendering of
50+ robots simultaneously. Designed for paper videos.

Usage:
    # Interactive viewer
    uv run python examples/humanoid_retarget/batch_motion_viewer.py \
        --motion motion_data/retargeted/ACCAD/walk.pkl

    # Record to video (open browser, then recording starts automatically)
    uv run python examples/humanoid_retarget/batch_motion_viewer.py \
        --motion motion_data/retargeted/ACCAD/walk.pkl \
        --record output.mp4 --resolution 1920x1080

    # Custom grid with color
    uv run python examples/humanoid_retarget/batch_motion_viewer.py \
        --motion motion_data/retargeted/ACCAD/walk.pkl \
        --num_robots 50 --grid 10x5 --spacing 2.0 --mesh_color 0.3 0.5 0.8
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import viser

from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf


def load_motion(motion_path: str) -> dict:
    with open(motion_path, "rb") as f:
        data = pickle.load(f)

    root_pos = data["root_pos"]
    root_rot_xyzw = data["root_rot"]
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    dof_pos = data["dof_pos"]
    fps = data.get("fps", 30.0)

    return {
        "root_pos": root_pos,
        "root_rot_wxyz": root_rot_wxyz,
        "dof_pos": dof_pos,
        "fps": fps,
        "num_frames": len(root_pos),
    }


def compute_grid(num_robots: int, grid_str: str | None) -> Tuple[int, int]:
    if grid_str is not None:
        parts = grid_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    cols = int(math.ceil(math.sqrt(num_robots)))
    rows = int(math.ceil(num_robots / cols))
    return rows, cols


def compute_grid_offsets(num_robots: int, rows: int, cols: int, spacing: float) -> np.ndarray:
    offsets = np.zeros((num_robots, 3), dtype=np.float32)
    for i in range(num_robots):
        row = i // cols
        col = i % cols
        offsets[i, 0] = (col - (cols - 1) / 2) * spacing
        offsets[i, 1] = (row - (rows - 1) / 2) * spacing
    return offsets


def tile_motion(motion: dict, num_robots: int, grid_offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-tile motion data for all robots.

    Returns:
        dof_pos_tiled: (T, N, J) joint positions
        T_world_base_tiled: (T, N, 7) as [x, y, z, qw, qx, qy, qz]
    """
    dof_pos = motion["dof_pos"]
    root_pos = motion["root_pos"]
    root_rot_wxyz = motion["root_rot_wxyz"]

    dof_pos_tiled = np.tile(dof_pos[:, None, :], (1, num_robots, 1))

    root_pos_tiled = np.tile(root_pos[:, None, :], (1, num_robots, 1))
    root_pos_tiled = root_pos_tiled + grid_offsets[None, :, :]

    root_rot_tiled = np.tile(root_rot_wxyz[:, None, :], (1, num_robots, 1))

    T_world_base_tiled = np.concatenate([root_pos_tiled, root_rot_tiled], axis=-1).astype(np.float32)

    return dof_pos_tiled.astype(np.float32), T_world_base_tiled


def compute_camera_pose(rows: int, cols: int, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a good camera position and look-at point for the robot grid.

    Returns a 3/4 elevated view that frames all robots.

    Returns:
        cam_pos: (3,) camera position
        look_at: (3,) look-at point
    """
    grid_width = (cols - 1) * spacing
    grid_depth = (rows - 1) * spacing
    extent = max(grid_width, grid_depth, 2.0)

    cam_dist = extent * 0.8 + 3.0
    elevation = math.radians(30)
    azimuth = math.radians(-50)

    look_at = np.array([0.0, 0.0, 0.6])
    cam_pos = look_at + np.array(
        [
            cam_dist * math.cos(elevation) * math.cos(azimuth),
            cam_dist * math.cos(elevation) * math.sin(azimuth),
            cam_dist * math.sin(elevation),
        ]
    )
    return cam_pos, look_at


def record_video(
    server: viser.ViserServer,
    batch_urdf: ViserBatchUrdf,
    dof_pos_tiled: np.ndarray,
    T_world_base_tiled: np.ndarray,
    num_frames: int,
    fps: float,
    output_path: str,
    resolution: Tuple[int, int],
    cam_pos: np.ndarray,
    look_at: np.ndarray,
    num_loops: int,
) -> None:
    import imageio.v3 as iio

    width, height = resolution

    print(f"\nWaiting for browser client to connect at http://localhost:{server._port}...")
    print("Open the URL in your browser to start recording.")

    while len(server.get_clients()) == 0:
        time.sleep(0.1)

    client = list(server.get_clients().values())[0]
    print("Client connected! Setting camera...")

    client.camera.position = cam_pos
    client.camera.look_at = look_at
    client.camera.up_direction = (0.0, 0.0, 1.0)
    time.sleep(0.5)

    total_frames = num_frames * num_loops
    print(f"Recording {total_frames} frames ({num_loops} loops) at {width}x{height}...")

    frames = []
    for i in range(total_frames):
        frame_idx = i % num_frames
        batch_urdf.update(
            joint_positions=dof_pos_tiled[frame_idx],
            T_world_base=T_world_base_tiled[frame_idx],
        )
        time.sleep(0.03)

        image = client.get_render(height=height, width=width, transport_format="jpeg")
        frames.append(image)

        if (i + 1) % 30 == 0 or i == total_frames - 1:
            print(f"  Frame {i + 1}/{total_frames}")

    print(f"Writing video to {output_path}...")
    iio.imwrite(output_path, np.stack(frames), fps=fps)
    print(f"Done! Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch motion viewer: many robots playing the same motion")
    parser.add_argument("--motion", type=str, required=True, help="Path to retargeted motion .pkl file")
    parser.add_argument("--urdf", type=str, default=None, help="Path to robot URDF (default: G1 robot)")
    parser.add_argument("--num_robots", type=int, default=50, help="Number of robots (default: 50)")
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Grid layout e.g. 10x5 (auto-calculated if not specified)",
    )
    parser.add_argument("--spacing", type=float, default=1.5, help="Spacing between robots (default: 1.5)")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (default: from motion file)")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    parser.add_argument(
        "--mesh_color",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Mesh color as floats 0-1, e.g. --mesh_color 0.3 0.5 0.8",
    )
    parser.add_argument("--opacity", type=float, default=None, help="Mesh opacity 0-1")
    parser.add_argument("--scale", type=float, default=1.0, help="Robot scale factor (default: 1.0)")
    parser.add_argument("--ground_offset", type=float, default=-0.67, help="Ground plane Z position (default: -0.67)")
    parser.add_argument("--record", type=str, default=None, help="Record to video file (e.g. output.mp4)")
    parser.add_argument(
        "--resolution",
        type=str,
        default="1920x1080",
        help="Recording resolution (default: 1920x1080)",
    )
    parser.add_argument("--num_loops", type=int, default=1, help="Number of motion loops to record (default: 1)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    urdf_path = args.urdf
    if urdf_path is None:
        urdf_path = str(project_root / "assets" / "robot_description" / "unitree_g1" / "g1_custom_collision_29dof.urdf")

    print("=" * 60)
    print("Batch Motion Viewer")
    print("=" * 60)

    motion = load_motion(args.motion)
    fps = args.fps or motion["fps"]
    num_frames = motion["num_frames"]

    print(f"Motion: {args.motion}")
    print(f"Frames: {num_frames}, FPS: {fps}, Duration: {num_frames / fps:.1f}s")
    print(f"DOF: {motion['dof_pos'].shape[1]}")

    rows, cols = compute_grid(args.num_robots, args.grid)
    grid_offsets = compute_grid_offsets(args.num_robots, rows, cols, args.spacing)
    print(f"Grid: {rows}x{cols} = {args.num_robots} robots, spacing={args.spacing}m")

    dof_pos_tiled, T_world_base_tiled = tile_motion(motion, args.num_robots, grid_offsets)
    print(f"Tiled shapes: dof_pos={dof_pos_tiled.shape}, T_world_base={T_world_base_tiled.shape}")

    print("Loading robot...")
    robot = Robot.load(urdf_path, backend="warp", load_meshes=True)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("ground", width=20, height=20, cell_size=0.5, position=(0.0, 0.0, args.ground_offset))

    mesh_color = tuple(args.mesh_color) if args.mesh_color else None

    batch_urdf = ViserBatchUrdf(
        target=server,
        robot=robot,
        batch_size=args.num_robots,
        scale=args.scale,
        mesh_color_override=mesh_color,
        opacity=args.opacity,
    )

    cam_pos, look_at = compute_camera_pose(rows, cols, args.spacing)

    if args.record is not None:
        res_w, res_h = map(int, args.resolution.split("x"))

        batch_urdf.update(
            joint_positions=dof_pos_tiled[0],
            T_world_base=T_world_base_tiled[0],
        )

        record_video(
            server=server,
            batch_urdf=batch_urdf,
            dof_pos_tiled=dof_pos_tiled,
            T_world_base_tiled=T_world_base_tiled,
            num_frames=num_frames,
            fps=fps,
            output_path=args.record,
            resolution=(res_w, res_h),
            cam_pos=cam_pos,
            look_at=look_at,
            num_loops=args.num_loops,
        )
        return

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = cam_pos
        client.camera.look_at = look_at
        client.camera.up_direction = (0.0, 0.0, 1.0)

    paused = False
    speed = 1.0
    current_frame = 0
    loop = True

    with server.gui.add_folder("Playback"):
        play_button = server.gui.add_button("Play/Pause")
        speed_slider = server.gui.add_slider("Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0)
        frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

    @play_button.on_click
    def _(_):
        nonlocal paused
        paused = not paused

    @speed_slider.on_update
    def _(event):
        nonlocal speed
        speed = event.target.value

    @frame_slider.on_update
    def _(event):
        nonlocal current_frame
        current_frame = int(event.target.value)

    @loop_checkbox.on_update
    def _(event):
        nonlocal loop
        loop = event.target.value

    print(f"\nOpen in browser: http://localhost:{args.port}")
    print("=" * 60)

    frame_time = 1.0 / fps
    last_time = time.time()

    batch_urdf.update(
        joint_positions=dof_pos_tiled[0],
        T_world_base=T_world_base_tiled[0],
    )

    while True:
        current_time = time.time()
        dt = current_time - last_time

        if not paused and dt >= frame_time / speed:
            batch_urdf.update(
                joint_positions=dof_pos_tiled[current_frame],
                T_world_base=T_world_base_tiled[current_frame],
            )
            frame_slider.value = current_frame

            current_frame += 1
            if current_frame >= num_frames:
                if loop:
                    current_frame = 0
                else:
                    paused = True
                    current_frame = num_frames - 1

            last_time = current_time

        time.sleep(0.001)


if __name__ == "__main__":
    main()
