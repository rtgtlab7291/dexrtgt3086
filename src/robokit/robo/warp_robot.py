# pyright: reportArgumentType=false
import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import trimesh
import warp as wp
from jaxtyping import Float
from typing_extensions import Self

from robokit.lie.warp_se3 import WarpSE3
from robokit.opt.variables import WarpVar
from robokit.robo.robot import Robot, RobotState
from robokit.robo.robot_spec import RobotSpec
from robokit.robo.warp_robot_kernels import (
    compute_link_jacobian_kernel,
    compute_motion_subspace_kernel,
    forward_kinematics_kernel,
    integrate_floating_base_kernel,
    integrate_joint_positions_kernel,
    transform_link_points_kernel,
)
from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather as warp_gather
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


if TYPE_CHECKING:
    from robokit.opt.warp_optimizer import WarpLMOptimizer, WarpLMOptimizerConfig
    from robokit.robo.robot_spec import RobotSpec
    from robokit.robo.torch_robot import TorchSpecTensors
    from robokit.terms import Term


@dataclass
class WarpSpecTensors:
    spec: RobotSpec
    device: str

    @cached_property
    def joint_axes(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_axes.astype(np.float32), device=self.device, dtype=wp.vec3)

    @cached_property
    def joint_types(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_types.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def joint_twists(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_twists.astype(np.float32), device=self.device, dtype=wp_vec6)

    @cached_property
    def actuated_joint_indices(self) -> wp.array:
        return wp.from_numpy(self.spec.actuated_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def parent_joint_indices(self) -> wp.array:
        return wp.from_numpy(self.spec.parent_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def parent_joint_transforms(self) -> wp.array:
        return wp.from_numpy(self.spec.parent_joint_transforms.astype(np.float32), device=self.device, dtype=wp_vec7)

    @cached_property
    def parent_joint_transforms_matrix(self) -> wp.array:
        return wp.from_numpy(
            self.spec.parent_joint_transforms_matrix.astype(np.float32), device=self.device, dtype=wp.mat44
        )

    @cached_property
    def mimic_actuated_joint_indices(self) -> wp.array:
        return wp.from_numpy(
            self.spec.mimic_actuated_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32
        )

    @cached_property
    def mimic_multipliers(self) -> wp.array:
        return wp.from_numpy(self.spec.mimic_multipliers.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def mimic_offsets(self) -> wp.array:
        return wp.from_numpy(self.spec.mimic_offsets.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def joint_limits(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_limits.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def joint_velocity_limits(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_velocity_limits.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def actuated_joint_limits(self) -> wp.array:
        return wp.from_numpy(self.spec.actuated_joint_limits.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def actuated_joint_velocity_limits(self) -> wp.array:
        return wp.from_numpy(
            self.spec.actuated_joint_velocity_limits.astype(np.float32), device=self.device, dtype=wp.float32
        )

    @cached_property
    def link_parent_joint_indices(self) -> wp.array:
        return wp.from_numpy(self.spec.link_parent_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def link_ancestor_joints_mask(self) -> wp.array:
        return wp.from_numpy(self.spec.link_ancestor_joints_mask.astype(np.bool_), device=self.device, dtype=wp.bool)

    @cached_property
    def joints_to_actuated_mapping(self) -> wp.array:
        return wp.from_numpy(
            self.spec.joints_to_actuated_mapping.astype(np.float32), device=self.device, dtype=wp.float32
        )

    @cached_property
    def topological_order_joint_indices(self) -> wp.array:
        return wp.from_numpy(
            self.spec.topological_order_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32
        )

    @cached_property
    def zero_q(self) -> wp.array:
        return wp.from_numpy(self.spec.zero_q.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def midrange_q(self) -> wp.array:
        return wp.from_numpy(self.spec.midrange_q.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def local_collision_sphere_centers(self) -> wp.array:
        return wp.from_numpy(
            self.spec.local_collision_sphere_centers.astype(np.float32), device=self.device, dtype=wp.vec3
        )

    @cached_property
    def local_collision_sphere_radii(self) -> wp.array:
        return wp.from_numpy(
            self.spec.local_collision_sphere_radii.astype(np.float32), device=self.device, dtype=wp.float32
        )

    @cached_property
    def collision_spheres_link_indices(self) -> wp.array:
        return wp.from_numpy(
            self.spec.collision_spheres_link_indices.astype(np.int32), device=self.device, dtype=wp.int32
        )

    @cached_property
    def local_contact_point_centers(self) -> wp.array:
        return wp.from_numpy(
            self.spec.local_contact_point_centers.astype(np.float32), device=self.device, dtype=wp.vec3
        )

    @cached_property
    def contact_points_link_indices(self) -> wp.array:
        return wp.from_numpy(self.spec.contact_points_link_indices.astype(np.int32), device=self.device, dtype=wp.int32)


class WarpRobotState(RobotState, WarpVar):
    spec: RobotSpec
    q: wp.array  # [batch_size, num_dofs] or [batch_size, num_frames, num_dofs]
    T_world_base: WarpSE3

    T_world_joint: WarpSE3
    T_world_link: WarpSE3
    S_world: wp.array
    # TODO world_collision_sphere_centers may be confusing, it's world frame/coordinate not world scene
    world_collision_sphere_centers: wp.array
    world_contact_point_centers: wp.array

    has_floating_base: bool
    is_fk_computed: bool
    is_motion_subspace_computed: bool
    is_collision_spheres_computed: bool
    is_contact_points_computed: bool

    def __init__(
        self,
        spec: RobotSpec,
        q: wp.array,
        T_world_base: Optional[WarpSE3] = None,
    ):
        self.spec = spec
        self.q = self._validate_q_shape(q).contiguous()
        requires_grad = self.q.requires_grad
        self.base_step_scale = 1e-4

        if T_world_base is None:
            self.T_world_base = WarpSE3.identity(shape=self.shape, device=self.device, requires_grad=requires_grad)
            self.has_floating_base = False
        else:
            self.T_world_base = T_world_base
            self.has_floating_base = True

        self._allocate_buffers(requires_grad)
        self.invalidate()

    def set_configuration(self, q: wp.array, T_world_base: Optional[WarpSE3] = None) -> "WarpRobotState":
        old_shape = self.shape
        q = self._validate_q_shape(q).contiguous()
        requires_grad = q.requires_grad

        self.q = q
        if T_world_base is None:
            self.T_world_base = WarpSE3.identity(shape=self.shape, device=self.device, requires_grad=requires_grad)
            self.has_floating_base = False
        else:
            self.T_world_base = T_world_base
            self.has_floating_base = True

        if self.shape != old_shape:
            self._allocate_buffers(requires_grad)

        self.invalidate()
        return self

    def _allocate_buffers(self, requires_grad: bool):
        self.T_world_joint = WarpSE3.identity(
            shape=self.shape + (self.spec.num_joints,), device=self.device, requires_grad=requires_grad
        )
        self.T_world_link = WarpSE3.identity(
            shape=self.shape + (self.spec.num_links,), device=self.device, requires_grad=requires_grad
        )
        self.S_world = wp.zeros(
            shape=self.shape + (6, self.spec.num_joints),
            dtype=wp.float32,
            device=self.device,
            requires_grad=requires_grad,
        )
        num_spheres = len(self.spec.local_collision_sphere_centers) if self.spec.has_collision_spheres else 0
        self.world_collision_sphere_centers = wp.zeros(
            shape=self.shape + (num_spheres,), dtype=wp.vec3, device=self.device, requires_grad=requires_grad
        )
        num_contact_points = len(self.spec.local_contact_point_centers) if self.spec.has_contact_points else 0
        self.world_contact_point_centers = wp.zeros(
            shape=self.shape + (num_contact_points,), dtype=wp.vec3, device=self.device, requires_grad=requires_grad
        )

    def invalidate(self) -> None:
        self.is_fk_computed = False
        self.is_motion_subspace_computed = False
        self.is_collision_spheres_computed = False
        self.is_contact_points_computed = False

    def _validate_q_shape(self, q: wp.array) -> wp.array:
        if q.ndim == 1:
            q = q.reshape((1, q.shape[0]))
        elif q.ndim in {2, 3}:
            if q.shape[-1] != self.spec.num_dofs:
                raise ValueError(f"Expected q to have {self.spec.num_dofs} DOFs, but got {q.shape[1]}")
        else:
            raise ValueError(f"Expected q to have 1, 2, or 3 dimensions, but got {q.ndim}")
        return q

    @property
    def is_trajectory(self) -> bool:
        return self.q.ndim == 3  # [batch_size, num_frames, num_dofs]

    @property
    def spec_tensors(self) -> WarpSpecTensors:
        device = str(self.q.device) if self.q is not None else None
        return WarpRobot._get_spec_tensors(self.spec, device)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.q.shape[:-1]

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def device(self) -> wp_device_type:
        return cast(wp_device_type, self.q.device)

    @property
    def tangent_dim(self) -> int:
        single_tangent_dim = (6 if self.has_floating_base else 0) + self.spec.num_actuated_joints
        return single_tangent_dim * self.shape[1] if self.is_trajectory else single_tangent_dim

    def get_T_world_link(self, link_index: int) -> WarpSE3:
        if not self.is_fk_computed:
            raise RuntimeError("Forward kinematics has not been computed. Call robot.forward_kinematics() first.")
        xyz_wxyz = self.T_world_link.xyz_wxyz
        index = tuple([slice(None)] * (xyz_wxyz.ndim - 1) + [link_index])
        return WarpSE3(xyz_wxyz[index])

    def get_motion_subspace(self) -> wp.array:
        if not self.is_motion_subspace_computed:
            raise RuntimeError("Motion subspace has not been computed. Call robot.compute_motion_subspace() first.")
        return self.S_world

    def get_link_jacobian(self, link_index: int, reference_frame: Literal["body", "spatial"] = "body") -> wp.array:
        if not self.is_motion_subspace_computed:
            raise RuntimeError("Motion subspace has not been computed. Call robot.compute_motion_subspace() first.")

        wp.init()
        device = self.S_world.device
        requires_grad = self.S_world.requires_grad
        num_actuated = self.spec.num_actuated_joints

        total_dofs = (6 if self.has_floating_base else 0) + num_actuated
        J_out = wp.zeros((*self.shape, 6, total_dofs), dtype=wp.float32, device=device, requires_grad=requires_grad)
        spec_t = WarpRobot._get_spec_tensors(self.spec, str(device))  # type: ignore[arg-type]
        reference_frame_int = 1 if reference_frame == "body" else 0

        wp.launch(
            kernel=compute_link_jacobian_kernel,
            dim=(self.num_elements, total_dofs),
            inputs=[
                self.S_world.reshape((self.num_elements, 6, self.spec.num_joints)),
                self.T_world_link.xyz_wxyz.reshape((self.num_elements, self.spec.num_links)),
                self.T_world_base.xyz_wxyz.flatten(),
                spec_t.link_ancestor_joints_mask[link_index],
                spec_t.joints_to_actuated_mapping,
                reference_frame_int,
                self.has_floating_base,
                link_index,
                J_out.reshape((self.num_elements, 6, total_dofs)),
            ],
            device=device,
        )

        return J_out

    def integrate(self, velocity: ArrayLike, out: Optional["WarpRobotState"] = None) -> "WarpRobotState":
        if velocity.shape != (self.batch_size, self.tangent_dim):
            raise ValueError(f"Expected velocity shape {(self.batch_size, self.tangent_dim)}, but got {velocity.shape}")

        requires_grad = velocity.requires_grad or self.q.requires_grad
        if out is None:
            out = WarpRobotState(
                spec=self.spec,
                q=wp.empty_like(self.q, requires_grad=requires_grad),
                T_world_base=WarpSE3(wp.empty_like(self.T_world_base.xyz_wxyz, requires_grad=requires_grad))
                if self.has_floating_base
                else None,
            )

        num_frames = self.shape[1] if self.is_trajectory else 1
        single_tangent_dim = (6 if self.has_floating_base else 0) + self.spec.num_actuated_joints
        velocity_offset = 0
        if self.has_floating_base:
            wp.launch(
                kernel=integrate_floating_base_kernel,
                dim=self.num_elements,
                inputs=[
                    self.T_world_base.xyz_wxyz.flatten(),
                    velocity,
                    num_frames,
                    single_tangent_dim,
                    wp.float32(self.base_step_scale),
                ],
                outputs=[out.T_world_base.xyz_wxyz.flatten()],
                device=self.q.device,
            )
            velocity_offset = 6

        wp.launch(
            kernel=integrate_joint_positions_kernel,
            dim=(self.num_elements, self.spec.num_actuated_joints),
            inputs=[
                self.q.reshape((self.num_elements, self.spec.num_actuated_joints)),
                velocity,
                velocity_offset,
                num_frames,
                single_tangent_dim,
            ],
            outputs=[out.q.reshape((self.num_elements, self.spec.num_actuated_joints))],
            device=self.q.device,
        )

        out.base_step_scale = self.base_step_scale
        out.invalidate()
        return out

    def clone(self) -> "WarpRobotState":
        state = object.__new__(WarpRobotState)
        state.spec = self.spec
        state.q = wp.clone(self.q)
        state.T_world_base = self.T_world_base.clone()
        state.T_world_joint = self.T_world_joint.clone()
        state.T_world_link = self.T_world_link.clone()
        state.S_world = wp.clone(self.S_world)
        state.world_collision_sphere_centers = wp.clone(self.world_collision_sphere_centers)
        state.world_contact_point_centers = wp.clone(self.world_contact_point_centers)
        state.has_floating_base = self.has_floating_base
        state.is_fk_computed = self.is_fk_computed
        state.is_motion_subspace_computed = self.is_motion_subspace_computed
        state.is_collision_spheres_computed = self.is_collision_spheres_computed
        state.is_contact_points_computed = self.is_contact_points_computed
        state.base_step_scale = self.base_step_scale
        return state

    def gather(self, indices: ArrayLike, dest: Optional["WarpRobotState"] = None) -> "WarpRobotState":
        num_gather = indices.shape[0]
        device = self.q.device
        requires_grad = self.q.requires_grad

        if dest is None:
            new_shape = (num_gather,) + self.q.shape[1:]
            q_placeholder = wp.empty(new_shape, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self.has_floating_base:
                T_world_base_placeholder = WarpSE3(
                    wp.empty(new_shape[:-1], dtype=wp_vec7, device=device, requires_grad=requires_grad)
                )
            else:
                T_world_base_placeholder = None
            dest = WarpRobotState(spec=self.spec, q=q_placeholder, T_world_base=T_world_base_placeholder)

        warp_gather(self.q, indices, dest.q)
        self.T_world_base.gather(indices, dest.T_world_base)
        self.T_world_joint.gather(indices, dest.T_world_joint)
        self.T_world_link.gather(indices, dest.T_world_link)
        warp_gather(self.S_world, indices, dest.S_world)
        warp_gather(self.world_collision_sphere_centers, indices, dest.world_collision_sphere_centers)
        warp_gather(self.world_contact_point_centers, indices, dest.world_contact_point_centers)

        dest.has_floating_base = self.has_floating_base
        dest.is_fk_computed = self.is_fk_computed
        dest.is_motion_subspace_computed = self.is_motion_subspace_computed
        dest.is_collision_spheres_computed = self.is_collision_spheres_computed
        dest.is_contact_points_computed = self.is_contact_points_computed
        dest.base_step_scale = self.base_step_scale
        return dest

    def __repr__(self) -> str:
        return f"WarpRobotState(q={self.q.numpy()}, T_world_base={self.T_world_base}, is_fk_computed={self.is_fk_computed}, is_collision_spheres_computed={self.is_collision_spheres_computed})"


class WarpRobot(Robot):
    def __new__(cls, spec: RobotSpec, backend: Literal["warp"] = "warp") -> Self:
        return object.__new__(cls)

    def __init__(self, spec: RobotSpec, backend: Literal["warp"] = "warp"):
        self.spec = spec

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_spec_tensors(spec: RobotSpec, device: Optional[str] = None) -> WarpSpecTensors:
        return WarpSpecTensors(spec=spec, device=device)

    def get_spec_tensors(self, device: Optional[str] = None) -> WarpSpecTensors:
        return WarpRobot._get_spec_tensors(self.spec, device)

    def get_spec_tensors_torch(self, device: torch.types.Device = None) -> "TorchSpecTensors":
        from robokit.robo.torch_robot import TorchRobot

        return TorchRobot._get_spec_tensors(self.spec, device)

    def state(
        self, q: Optional[Union[wp.array, np.ndarray]] = None, T_world_base: Optional[WarpSE3] = None
    ) -> WarpRobotState:
        if q is not None and isinstance(q, np.ndarray):
            if T_world_base is not None:
                device = T_world_base.xyz_wxyz.device
            else:
                device = wp.get_device("cpu")
            q = wp.from_numpy(q.astype(np.float32), dtype=wp.float32, device=device)
        elif q is None:
            q = self.zero_q
        elif not isinstance(q, wp.array):
            raise TypeError(f"Expected q to be a wp.array or np.ndarray, but got {type(q)}")
        return WarpRobotState(spec=self.spec, q=q, T_world_base=T_world_base)

    def forward_kinematics(self, state: WarpRobotState) -> WarpRobotState:
        """
        Compute the forward kinematics of the robot for given joint positions statelessly.
        Returns a robot state containing the computed transforms of all links.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = WarpRobot.load(load_robot_description("panda_description"), backend="warp")
            >>> q = wp.from_numpy(np.array([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0]), dtype=wp.float32)
            >>> state = robot.state(q=q)
            >>> state = robot.forward_kinematics(state)
            >>> hand_idx = robot.spec.link_names.index("panda_hand")
            >>> link_pose = state.T_world_link[:, hand_idx]
            >>> expected = np.array([6.0860e-02, -4.7547e-12, 7.6373e-01, 0.0382, 0.9193, 0.3808, 0.0922], dtype=np.float32)
            >>> expected = WarpSE3(wp.from_numpy(expected, dtype=wp_vec7))
            >>> np.allclose(link_pose.as_matrix().numpy(), expected.as_matrix().numpy(), atol=1e-4)
            True
            >>> T_world_base = WarpSE3(wp.from_numpy(np.array([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0]), dtype=wp_vec7))
            >>> state = robot.state(q=q, T_world_base=T_world_base)
            >>> state = robot.forward_kinematics(state)
            >>> link_pose = state.T_world_link[:, hand_idx]
            >>> expected = np.array([1.76371783, 2.0, 2.93915576, -0.24222432, 0.71524162, 0.29626278, -0.58478125], dtype=np.float32)
            >>> expected = WarpSE3(wp.from_numpy(expected, dtype=wp_vec7))
            >>> np.allclose(link_pose.as_matrix().numpy(), expected.as_matrix().numpy(), atol=1e-4)
            True
        """
        spec_t = self.get_spec_tensors(str(state.q.device))

        spec_arrays = [
            spec_t.actuated_joint_indices,
            spec_t.mimic_actuated_joint_indices,
            spec_t.mimic_multipliers,
            spec_t.mimic_offsets,
            spec_t.joint_twists,
            spec_t.parent_joint_transforms,
            spec_t.topological_order_joint_indices,
            spec_t.parent_joint_indices,
            spec_t.link_parent_joint_indices,
        ]

        wp.launch(
            kernel=forward_kinematics_kernel,
            dim=state.num_elements,
            inputs=[
                state.q.reshape((state.num_elements, state.spec.num_actuated_joints)),
                state.T_world_base.xyz_wxyz.flatten(),
                *spec_arrays,
            ],
            outputs=[
                state.T_world_joint.xyz_wxyz.reshape((state.num_elements, self.spec.num_joints)),
                state.T_world_link.xyz_wxyz.reshape((state.num_elements, self.spec.num_links)),
            ],
            device=state.q.device,
        )

        state.is_fk_computed = True
        state.is_collision_spheres_computed = False
        return state

    def forward_kinematics_via_matrix_torch(
        self, q: Float[torch.Tensor, "... num_dofs"], T_world_base: Optional[Float[torch.Tensor, "... 4 4"]] = None
    ) -> Float[torch.Tensor, "... num_links 4 4"]:
        from robokit.robo.warp_robot_kernels import WarpForwardKinematicsMatrix

        if T_world_base is None:
            base = torch.eye(4, device=q.device, dtype=torch.float32)
            base = torch.broadcast_to(base, q.shape[:-1] + (4, 4))
        else:
            base = T_world_base.to(device=q.device, dtype=torch.float32)
            base = torch.broadcast_to(base, q.shape[:-1] + (4, 4))

        link_matrices = WarpForwardKinematicsMatrix.apply(self.spec, q, base)  # type: ignore
        return link_matrices  # type: ignore

    def transform_collision_spheres(self, state: WarpRobotState) -> WarpRobotState:
        """
        Compute the collision sphere centers in world frame.

        Args:
            state: The robot state.

        Returns:
            The robot state with updated collision sphere centers.
        """
        if not self.spec.has_collision_spheres:
            return state

        if not state.is_fk_computed:
            state = self.forward_kinematics(state)

        spec_t = self.get_spec_tensors(str(state.q.device))
        num_spheres = len(self.spec.local_collision_sphere_centers)

        wp.launch(
            kernel=transform_link_points_kernel,
            dim=(state.num_elements, num_spheres),
            inputs=[
                state.T_world_link.xyz_wxyz.reshape((state.num_elements, self.spec.num_links)),
                spec_t.local_collision_sphere_centers,
                spec_t.collision_spheres_link_indices,
                state.world_collision_sphere_centers.reshape((state.num_elements, num_spheres)),
            ],
            device=state.q.device,
        )

        state.is_collision_spheres_computed = True
        return state

    def transform_contact_points(self, state: WarpRobotState) -> WarpRobotState:
        """
        Compute the contact point centers in world frame.

        Args:
            state: The robot state.

        Returns:
            The robot state with updated contact point centers.
        """
        if not self.spec.has_contact_points:
            return state

        if not state.is_fk_computed:
            state = self.forward_kinematics(state)

        spec_t = self.get_spec_tensors(str(state.q.device))
        num_contact_points = len(self.spec.local_contact_point_centers)

        wp.launch(
            kernel=transform_link_points_kernel,
            dim=(state.num_elements, num_contact_points),
            inputs=[
                state.T_world_link.xyz_wxyz.reshape((state.num_elements, self.spec.num_links)),
                spec_t.local_contact_point_centers,
                spec_t.contact_points_link_indices,
                state.world_contact_point_centers.reshape((state.num_elements, num_contact_points)),
            ],
            device=state.q.device,
        )

        state.is_contact_points_computed = True
        return state

    def transform_link_points_torch(
        self,
        T_world_link: Float[torch.Tensor, "... num_links 4 4"],
        local_points: Float[torch.Tensor, "num_points 3"],
        point_link_indices: Float[torch.Tensor, "num_points"],
    ) -> Float[torch.Tensor, "... num_points 3"]:
        """Transform local points attached to links to world frame.

        Args:
            T_world_link: Link transforms in world frame [..., num_links, 4, 4]
            local_points: Local points in link frames [num_points, 3]
            point_link_indices: Link index for each point [num_points]

        Returns:
            World-frame points [..., num_points, 3]
        """
        from robokit.robo.warp_robot_kernels import WarpTransformLinkPoints

        return WarpTransformLinkPoints.apply(T_world_link, local_points, point_link_indices)  # type: ignore

    def compute_motion_subspace(self, state: WarpRobotState) -> WarpRobotState:
        """
        Compute the joint Jacobians for the robot and store them in the state.

        After calling this method, use state.get_link_jacobian() to retrieve Jacobians
        for specific links. When a floating base is provided, the returned Jacobian will
        have J_base prepended to J_joints (shape: [batch, 6, 6 + num_dofs]).

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = WarpRobot.load(load_robot_description("panda_description"), backend="warp")
            >>> T_world_base = WarpSE3(wp.from_numpy(np.array([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0]), dtype=wp_vec7))
            >>> q = wp.from_numpy(np.array([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0]), dtype=wp.float32)
            >>> state = robot.state(q=q, T_world_base=T_world_base)
            >>> state = robot.compute_motion_subspace(state)
            >>> J = state.get_link_jacobian(robot.link_names.index("panda_hand"), "body")
            >>> np.allclose(J.numpy()[0, 0, :6], np.array([0.693, 0.7071, 0.1405, -0.54, 0.5207, 0.043], dtype=np.float32), atol=1e-4)
            True
            >>> J = state.get_link_jacobian(robot.link_names.index("panda_hand"), "spatial")
            >>> np.allclose(J.numpy()[0, 0, 6:], np.array([0.0001, -3., 1.8641, 3.2647, -1.4346, 3.0467, -0.3974, 0.], dtype=np.float32), atol=1e-3)
            True
            >>> np.allclose(J.numpy()[0, 0, :6], np.array([0.0001, 0., 1., -2., -3., 0.0001], dtype=np.float32), atol=1e-3)
            True
        """
        if not state.is_fk_computed:
            state = self.forward_kinematics(state)
        assert state.q is not None and state.T_world_joint is not None

        wp.init()
        spec_t = self.get_spec_tensors(str(state.q.device))

        wp.launch(
            kernel=compute_motion_subspace_kernel,
            dim=(state.num_elements, self.spec.num_joints),
            inputs=[
                state.T_world_joint.xyz_wxyz.reshape((state.num_elements, self.spec.num_joints)),
                spec_t.joint_twists,
                state.S_world.reshape((state.num_elements, 6, self.spec.num_joints)),
            ],
            device=state.q.device,
        )

        state.is_motion_subspace_computed = True
        return state

    def build_inverse_kinematics_optimizer(
        self,
        frame_names: Union[str, Sequence[str]],
        T_world_target: Union[WarpSE3, Sequence[WarpSE3]],
        position_weight: float = 1.0,
        orientation_weight: float = 0.2,
        optimizer_config: Optional["WarpLMOptimizerConfig"] = None,
    ) -> "WarpLMOptimizer":
        """
        Build an inverse kinematics optimizer for the robot.

        Example:
            >>> from robokit.lie.warp_se3 import WarpSE3
            >>> from robokit.utils.warp_utils import wp_vec7
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("ur10_description"), backend="warp")
            >>> target_pose = WarpSE3(wp.from_numpy(np.array([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]), dtype=wp_vec7))
            >>> ik_optimizer = robot.build_inverse_kinematics_optimizer("ee_link", target_pose)
            >>> q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
            >>> state = robot.state(q=q_init)
            >>> state = ik_optimizer.solve(state)
            >>> state = robot.forward_kinematics(state)
            >>> achieved_pose = state.get_T_world_link(robot.link_names.index("ee_link"))
            >>> assert np.allclose(achieved_pose.xyz.numpy(), target_pose.xyz.numpy(), atol=1e-3)
            >>> assert np.allclose(achieved_pose.quat_wxyz.numpy(), target_pose.quat_wxyz.numpy(), atol=1e-3)
        """
        from robokit.opt.warp_optimizer import WarpLMOptimizer, WarpLMOptimizerConfig
        from robokit.terms.warp.frame_task import WarpFrameTask
        from robokit.terms.warp.position_limit import WarpPositionLimit

        frame_names = [frame_names] if isinstance(frame_names, str) else list(frame_names)
        T_world_target = [T_world_target] if isinstance(T_world_target, WarpSE3) else list(T_world_target)

        if len(frame_names) != len(T_world_target):
            raise ValueError("frame_names and T_world_target must have the same length")

        # infer device and batch size
        device = T_world_target[0].xyz_wxyz.device
        batch_size = T_world_target[0].xyz_wxyz.shape[0]

        terms: List["Term"] = []  # type: ignore[name-defined]
        for idx, frame_name in enumerate(frame_names):
            frame_index = self.link_names.index(frame_name)
            terms.append(
                WarpFrameTask(
                    robot=self,
                    frame_index=frame_index,
                    T_world_target=T_world_target[idx],
                    position_weight=position_weight,
                    orientation_weight=orientation_weight,
                )
            )

        # position limits
        terms.append(WarpPositionLimit(robot=self, weight=2.0, batch_size=batch_size))

        if optimizer_config is None:
            optimizer_config = WarpLMOptimizerConfig(lm_lambda=1.0)

        q_placeholder = wp.empty((batch_size, self.spec.num_actuated_joints), dtype=wp.float32, device=device)
        placeholder_state = self.state(q=q_placeholder)

        return WarpLMOptimizer(
            terms=terms,  # type: ignore[arg-type]
            device=device,
            config=optimizer_config,
            placeholder_var=placeholder_state,
        )

    @property
    def zero_q(self) -> wp.array:
        return self.get_spec_tensors().zero_q

    @property
    def midrange_q(self) -> wp.array:
        return self.get_spec_tensors().midrange_q

    def get_midrange_q(self, num_samples: int = 1, device: Optional[wp_device_type] = None) -> wp.array:
        return wp.from_numpy(np.tile(self.spec.midrange_q, (num_samples, 1)), dtype=wp.float32, device=device)

    def sample_q(
        self,
        num_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
        device: wp_device_type = None,
    ) -> wp.array:
        joint_limits = self.spec.actuated_joint_limits
        if rng is None:
            rng = np.random.default_rng()
        q_np = rng.uniform(joint_limits[:, 0], joint_limits[:, 1], size=(num_samples, joint_limits.shape[0]))
        return wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32, device=device)

    def map_to_full_joint_values_torch(
        self, actuated_values: Float[torch.Tensor, "... num_actuated_joints"]
    ) -> Float[torch.Tensor, "... num_joints"]:
        spec_t = self.get_spec_tensors_torch(device=actuated_values.device)
        is_mimic = spec_t.mimic_actuated_joint_indices != -1
        controlling_indices = torch.where(is_mimic, spec_t.mimic_actuated_joint_indices, spec_t.actuated_joint_indices)
        padded_actuated_values = torch.cat([actuated_values, torch.zeros_like(actuated_values[..., :1])], dim=-1)
        safe_controlling_indices = torch.where(controlling_indices == -1, self.num_actuated_joints, controlling_indices)
        controlling_values = padded_actuated_values[..., safe_controlling_indices]
        return controlling_values * spec_t.mimic_multipliers + spec_t.mimic_offsets

    def get_link_trimesh_meshes(self, mode: Literal["visual", "collision"] = "collision") -> Dict[str, trimesh.Trimesh]:
        geometries = self.spec.link_visual_geometries if mode == "visual" else self.spec.link_collision_geometries
        return {link_name: scene.to_mesh() for link_name, scene in geometries.items() if len(scene.geometry) > 0}

    def get_zero_joint_values(self, return_tensors: Literal["pt", "np"] = "pt") -> Union[torch.Tensor, np.ndarray]:
        zeros = np.zeros(self.num_actuated_joints, dtype=np.float32)
        if return_tensors == "pt":
            return torch.from_numpy(zeros)
        return zeros
