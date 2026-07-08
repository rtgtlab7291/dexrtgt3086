# pyright: reportArgumentType=false
from typing import List, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.warp_se3 import WarpSE3
from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_vec7


@wp.kernel
def compute_position_score_residual_kernel(
    achieved_xyz: wp.array1d(dtype=wp.vec3f),
    target_pose: wp.array2d(dtype=wp_vec7),  # (batch_size, num_frames)
    num_seeds: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
    row_offset: int,
):
    tid = wp.tid()
    target_batch_idx = tid // num_seeds
    achieved = achieved_xyz[tid]
    target_vec7 = target_pose[target_batch_idx, 0]

    target_x = target_vec7[0]
    target_y = target_vec7[1]
    target_z = target_vec7[2]

    dx = achieved[0] - target_x
    dy = achieved[1] - target_y
    dz = achieved[2] - target_z
    residual_buffer[tid, row_offset] = wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.kernel
def compute_multi_position_score_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_indices: wp.array1d(dtype=wp.int32),  # (num_frames,)
    target_poses: wp.array2d(dtype=wp_vec7),  # (batch_size, num_frames)
    num_frames: int,
    num_seeds: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
    row_offset: int,
):
    tid = wp.tid()
    target_batch_idx = tid // num_seeds

    for frame_num in range(num_frames):
        frame_index = frame_indices[frame_num]

        achieved_pose = T_world_link[tid, frame_index]
        target_vec7 = target_poses[target_batch_idx, frame_num]

        dx = achieved_pose[0] - target_vec7[0]
        dy = achieved_pose[1] - target_vec7[1]
        dz = achieved_pose[2] - target_vec7[2]
        residual_buffer[tid, row_offset + frame_num] = wp.sqrt(dx * dx + dy * dy + dz * dz)


class WarpCompositeScoreTask(WarpTask):
    """Combines multiple score tasks into one by concatenating their residuals."""

    def __init__(self, tasks: List[WarpTask]):
        self.tasks = tasks

    @property
    def residual_dim(self) -> int:
        return sum(task.residual_dim for task in self.tasks)

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if residual_buffer is None:
            raise ValueError("residual_buffer is required.")
        offset = row_offset
        for task in self.tasks:
            task.compute_weighted_residual(var, residual_buffer=residual_buffer, row_offset=offset)
            offset += task.residual_dim
        return residual_buffer


class WarpPositionScoreTask(WarpTask):
    def __init__(
        self,
        robot: WarpRobot,
        frame_index: Union[int, Sequence[int]],
        target_se3: Union[WarpSE3, Sequence[WarpSE3]],
        num_seeds: int,
        batch_size: int,
    ):
        self.robot = robot
        self.num_seeds = num_seeds
        self._batch_size = batch_size

        if isinstance(frame_index, int):
            frame_indices_list = [frame_index]
        else:
            frame_indices_list = list(frame_index)

        if isinstance(target_se3, WarpSE3):
            targets_list = [target_se3]
        else:
            targets_list = list(target_se3)

        if len(frame_indices_list) != len(targets_list):
            raise ValueError(
                f"Number of frame indices ({len(frame_indices_list)}) must match "
                f"number of targets ({len(targets_list)})"
            )

        self.num_frames = len(frame_indices_list)
        self._frame_indices_np = np.array(frame_indices_list, dtype=np.int32)

        first_target = targets_list[0]
        self._device = first_target.xyz_wxyz.device
        self._target_batch_size = first_target.batch_size

        self.frame_indices = wp.from_numpy(self._frame_indices_np, dtype=wp.int32, device=self._device)

        self.T_world_target = WarpSE3.stack(targets_list, axis=1)

        if self.num_frames == 1:
            self.frame_index = frame_indices_list[0]
        else:
            self.frame_index = -1

    @property
    def residual_dim(self) -> int:
        return self.num_frames

    def set_target(self, target: Union[WarpSE3, Sequence[WarpSE3]]) -> None:
        """Update target poses in-place. Batch size must match the original."""
        targets = [target] if isinstance(target, WarpSE3) else list(target)
        WarpSE3.stack(targets, axis=1, dest=self.T_world_target)

    def compute_weighted_residual(
        self,
        var: WarpRobotState,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if residual_buffer is None:
            raise ValueError("residual_buffer is required.")
        state = self.robot.forward_kinematics(var)

        if self.num_frames == 1:
            achieved = state.get_T_world_link(self.frame_index)
            wp.launch(
                compute_position_score_residual_kernel,
                dim=self._batch_size,
                inputs=[achieved.xyz, self.T_world_target, self.num_seeds, residual_buffer, row_offset],
                device=self._device,
            )
        else:
            wp.launch(
                compute_multi_position_score_residual_kernel,
                dim=self._batch_size,
                inputs=[
                    state.T_world_link.xyz_wxyz,
                    self.frame_indices,
                    self.T_world_target,
                    self.num_frames,
                    self.num_seeds,
                    residual_buffer,
                    row_offset,
                ],
                device=self._device,
            )
        return residual_buffer
