from __future__ import annotations

from typing import List

import numpy as np
import viser
import warp as wp

from robokit.lie.warp_se3 import WarpSE3
from robokit.robo.warp_robot import WarpRobot
from robokit.utils.warp_utils import wp_vec7


class ViserBatchUrdf:
    """Batched URDF visualizer using viser's instanced mesh rendering.

    Renders N copies of a robot using one batched mesh handle per link,
    with GPU-batched forward kinematics via WarpRobot.
    """

    def __init__(
        self,
        target: viser.ViserServer | viser.ClientHandle,
        robot: WarpRobot,
        batch_size: int,
        root_node_name: str = "/",
        scale: float = 1.0,
        mesh_color_override: tuple[float, float, float] | None = None,
        opacity: float | None = None,
    ):
        self._robot = robot
        self._batch_size = batch_size
        self._scale = scale
        self._target = target

        self._link_mesh_handles: List[viser.BatchedMeshHandle] = []
        self._link_indices: List[int] = []

        root = root_node_name.rstrip("/")
        identity_wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (batch_size, 1))
        zero_positions = np.zeros((batch_size, 3), dtype=np.float32)

        color_kwargs = {}
        if mesh_color_override is not None:
            color_kwargs["batched_colors"] = np.array([int(c * 255) for c in mesh_color_override], dtype=np.uint8)

        for link_index, link_name in enumerate(robot.spec.link_names):
            scene = robot.spec.link_visual_geometries[link_name]
            if len(scene.geometry) == 0:
                continue
            mesh = scene.to_mesh()
            if len(mesh.vertices) == 0:
                continue

            vertices = np.array(mesh.vertices, dtype=np.float32) * scale
            faces = np.array(mesh.faces, dtype=np.uint32)

            node_name = f"{root}/{link_name}" if root else f"/{link_name}"
            handle = target.scene.add_batched_meshes_simple(
                node_name,
                vertices=vertices,
                faces=faces,
                batched_wxyzs=identity_wxyzs,
                batched_positions=zero_positions,
                opacity=opacity,
                **color_kwargs,
            )
            self._link_mesh_handles.append(handle)
            self._link_indices.append(link_index)

    def update(
        self,
        joint_positions: np.ndarray,
        T_world_base: np.ndarray | None = None,
    ):
        warp_T_world_base: WarpSE3 | None = None
        if T_world_base is not None:
            warp_T_world_base = WarpSE3(wp.from_numpy(T_world_base.astype(np.float32), dtype=wp_vec7))

        state = self._robot.state(q=joint_positions, T_world_base=warp_T_world_base)
        state = self._robot.forward_kinematics(state)

        all_transforms = state.T_world_link.xyz_wxyz.numpy()

        for handle, link_index in zip(self._link_mesh_handles, self._link_indices):
            positions = all_transforms[:, link_index, :3] * self._scale
            wxyzs = all_transforms[:, link_index, 3:]
            handle.batched_positions = positions
            handle.batched_wxyzs = wxyzs

    def remove(self):
        for handle in self._link_mesh_handles:
            handle.remove()
        self._link_mesh_handles.clear()
        self._link_indices.clear()
