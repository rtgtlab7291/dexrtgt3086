from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast, no_type_check

import torch
from jaxtyping import Bool, Float, Int

from robokit.lie.torch_se3 import TorchSE3, _se3_multiply
from robokit.robo.robot import Robot, RobotState
from robokit.robo.robot_spec import JointType, RobotSpec
from robokit.xform.torch.transforms import rot_tl_to_tf_mat, transform_points


if TYPE_CHECKING:
    from robokit.opt.torch_optimizer import TorchOptimizer, TorchOptimizerConfig


@dataclass
class TorchSpecTensors:
    spec: RobotSpec
    device: torch.types.Device

    @cached_property
    def joint_twists(self) -> Float[torch.Tensor, "num_joints 6"]:
        return torch.from_numpy(self.spec.joint_twists).float().to(device=self.device)

    @cached_property
    def joint_types(self) -> Int[torch.Tensor, "num_joints"]:
        return torch.from_numpy(self.spec.joint_types).long().to(device=self.device)

    @cached_property
    def joint_axes(self) -> Float[torch.Tensor, "num_joints 3"]:
        return torch.from_numpy(self.spec.joint_axes).float().to(device=self.device)

    @cached_property
    def actuated_joint_indices(self) -> Int[torch.Tensor, "num_joints"]:
        return torch.from_numpy(self.spec.actuated_joint_indices).long().to(device=self.device)

    @cached_property
    def parent_joint_indices(self) -> Int[torch.Tensor, "num_joints"]:
        return torch.from_numpy(self.spec.parent_joint_indices).long().to(device=self.device)

    @cached_property
    def parent_joint_transforms(self) -> Float[torch.Tensor, "num_joints 7"]:
        return torch.from_numpy(self.spec.parent_joint_transforms).float().to(device=self.device)

    @cached_property
    def parent_joint_transforms_matrix(self) -> Float[torch.Tensor, "num_joints 4 4"]:
        return TorchSE3(self.parent_joint_transforms).as_matrix()

    @cached_property
    def mimic_actuated_joint_indices(self) -> Int[torch.Tensor, "num_joints"]:
        return torch.from_numpy(self.spec.mimic_actuated_joint_indices).long().to(device=self.device)

    @cached_property
    def mimic_multipliers(self) -> Float[torch.Tensor, "num_joints"]:
        return torch.from_numpy(self.spec.mimic_multipliers).float().to(device=self.device)

    @cached_property
    def mimic_offsets(self) -> Float[torch.Tensor, "num_joints"]:
        return torch.from_numpy(self.spec.mimic_offsets).float().to(device=self.device)

    @cached_property
    def joint_limits(self) -> Float[torch.Tensor, "num_joints 2"]:
        return torch.from_numpy(self.spec.joint_limits).float().to(device=self.device)

    @cached_property
    def joint_velocity_limits(self) -> Float[torch.Tensor, "num_joints"]:
        return torch.from_numpy(self.spec.joint_velocity_limits).float().to(device=self.device)

    @cached_property
    def actuated_joint_limits(self) -> Float[torch.Tensor, "num_actuated_joints 2"]:
        return torch.from_numpy(self.spec.actuated_joint_limits).float().to(device=self.device)

    @cached_property
    def actuated_joint_velocity_limits(self) -> Float[torch.Tensor, "num_actuated_joints"]:
        return torch.from_numpy(self.spec.actuated_joint_velocity_limits).float().to(device=self.device)

    @cached_property
    def link_parent_joint_indices(self) -> Int[torch.Tensor, "num_links"]:
        return torch.from_numpy(self.spec.link_parent_joint_indices).long().to(device=self.device)

    @cached_property
    def link_ancestor_joints_mask(self) -> Bool[torch.Tensor, "num_links num_joints"]:
        return torch.from_numpy(self.spec.link_ancestor_joints_mask).bool().to(device=self.device)

    @cached_property
    def joints_to_actuated_mapping(self) -> Float[torch.Tensor, "num_joints num_actuated_joints"]:
        return torch.from_numpy(self.spec.joints_to_actuated_mapping).float().to(device=self.device)

    @cached_property
    def topological_order_joint_indices(self) -> Int[torch.Tensor, "num_joints"]:
        return torch.from_numpy(self.spec.topological_order_joint_indices).long().to(device=self.device)

    @cached_property
    def zero_q(self) -> Float[torch.Tensor, "num_actuated_joints"]:
        return torch.from_numpy(self.spec.zero_q).float().to(device=self.device)

    @cached_property
    def midrange_q(self) -> Float[torch.Tensor, "num_actuated_joints"]:
        return torch.from_numpy(self.spec.midrange_q).float().to(device=self.device)


@dataclass
class TorchRobotState(RobotState):
    _q: Optional[Float[torch.Tensor, "... num_dofs"]] = None
    _T_world_base: Optional[TorchSE3] = None

    has_floating_base: Optional[bool] = None
    """Whether a floating base pose was provided."""
    T_world_joint: Optional[TorchSE3] = field(default=None, init=False)
    """SE3 transforms from each joint to the world frame."""
    T_world_link: Optional[TorchSE3] = field(default=None, init=False)
    """SE3 transforms from each link to the world frame."""
    S_world: Optional[Float[torch.Tensor, "... 6 num_joints"]] = field(default=None, init=False)
    """Motion subspace expressed in the world frame."""

    _link_jacobians_cache: Dict[Tuple[int, Literal["body", "spatial"]], Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self._q is not None:
            if self.has_floating_base is None:
                self.has_floating_base = self._T_world_base is not None
            if self._T_world_base is None:
                # For fixed base, create identity with proper batch dimension
                # If q is unbatched (1D), we want identity of dim=1, not dim=q.shape[0]
                batch_dim = 1 if self._q.ndim == 1 else self._q.shape[0]
                self._T_world_base = TorchSE3.identity(dim=batch_dim, device=self._q.device)
                self.has_floating_base = False
        self._link_jacobians_cache.clear()
        self.T_world_joint = None
        self.T_world_link = None
        self.S_world = None

    @property
    def is_initialized(self) -> bool:
        return self._q is not None

    @property
    def q(self) -> Float[torch.Tensor, "... num_dofs"]:
        if self._q is None:
            raise RuntimeError("TorchRobotState has not been initialized. Call robot.set_configuration() first.")
        return cast(torch.Tensor, self._q)

    @property
    def T_world_base(self) -> TorchSE3:
        if self._T_world_base is None:
            raise RuntimeError("TorchRobotState has not been initialized. Call robot.set_configuration() first.")
        return cast(TorchSE3, self._T_world_base)

    def set_configuration(self, q: Float[torch.Tensor, "... num_dofs"], T_world_base: Optional[TorchSE3] = None):
        self._q = q
        self._T_world_base = T_world_base
        self.__post_init__()

    @property
    def is_fk_computed(self) -> bool:
        return self.T_world_link is not None

    @property
    def is_motion_subspace_computed(self) -> bool:
        return self.S_world is not None

    @property
    def spec_tensors(self) -> TorchSpecTensors:
        device = self.q.device if self.q is not None else None
        return TorchRobot._get_spec_tensors(self.spec, device)

    @property
    def tangent_dim(self) -> int:
        return (6 if self.has_floating_base else 0) + self.spec.num_actuated_joints

    def get_T_world_link(self, link_index: int) -> TorchSE3:
        if not self.T_world_link:
            raise RuntimeError("Forward kinematics has not been computed. Call robot.forward_kinematics() first.")
        return self.T_world_link[..., link_index, :]

    def get_motion_subspace(self) -> Float[torch.Tensor, "6 num_joints"]:
        if self.S_world is None:
            raise RuntimeError("Motion subspace has not been computed. Call robot.compute_motion_subspace() first.")
        return self.S_world

    def get_link_jacobian(
        self, link_index: int, reference_frame: Literal["body", "spatial"] = "body"
    ) -> Float[torch.Tensor, "... 6 num_total_dofs"]:
        if not self.is_motion_subspace_computed:
            raise RuntimeError("Motion subspace has not been computed. Call robot.compute_motion_subspace() first.")
        if (link_index, reference_frame) in self._link_jacobians_cache:
            return self._link_jacobians_cache[(link_index, reference_frame)]
        spec_t = TorchRobot._get_spec_tensors(self.spec, self.q.device)  # type: ignore
        ancestor_mask = spec_t.link_ancestor_joints_mask[link_index]  # [num_joints]
        masked_S_world = self.S_world * ancestor_mask.to(self.S_world.dtype).unsqueeze(-2)  # pyright: ignore[reportOptionalOperand, reportOptionalMemberAccess]
        J_spatial = masked_S_world @ spec_t.joints_to_actuated_mapping  # [..., 6, num_actuated_joints]

        if reference_frame == "body":
            # Express Jacobian in the body frame of the target link
            T_world_frame = self.get_T_world_link(link_index)
            J_joints = torch.matmul(T_world_frame.inverse().adjoint(), J_spatial)
        else:
            J_joints = J_spatial
        if not self.has_floating_base:
            self._link_jacobians_cache[(link_index, reference_frame)] = J_joints
            return J_joints
        # Base Jacobian when a floating base is present
        if reference_frame == "spatial":
            J_base = self.T_world_base.adjoint()
        else:
            # J_base is adjoint of T_frame_base
            T_world_frame = self.get_T_world_link(link_index)
            T_frame_base = T_world_frame.inverse() @ self.T_world_base
            J_base = T_frame_base.adjoint()
        J_full = torch.cat([J_base, J_joints], dim=-1)
        self._link_jacobians_cache[(link_index, reference_frame)] = J_full
        return J_full

    def integrate(self, velocity: Float[torch.Tensor, "... tangent_dim"]) -> "TorchRobotState":
        if self.has_floating_base:
            xi_base = velocity[..., :6]
            dq = velocity[..., 6:]
            new_q = self.q + dq
            new_T_world_base = self.T_world_base @ TorchSE3.exp(xi_base)
            new_state = TorchRobotState(spec=self.spec)
            new_state.set_configuration(q=new_q, T_world_base=new_T_world_base)
        else:
            new_q = self.q + velocity
            new_state = TorchRobotState(spec=self.spec)
            new_state.set_configuration(q=new_q, T_world_base=None)
        return new_state

    def clone(self) -> "TorchRobotState":
        new_state = TorchRobotState(spec=self.spec)
        if self._q is not None:
            new_state.set_configuration(
                q=self._q.clone(),
                T_world_base=self._T_world_base.clone() if self._T_world_base is not None else None,
            )
        return new_state

    def __repr__(self) -> str:
        return f"TorchRobotState(q={self.q}, T_world_base={self.T_world_base}, is_fk_computed={self.is_fk_computed})"


class TorchRobot(Robot):
    def __init__(self, spec: RobotSpec, backend: Literal["torch"] = "torch"):
        self.spec = spec

    @staticmethod
    @no_type_check
    @lru_cache(maxsize=128)
    def _get_spec_tensors(spec: RobotSpec, device: torch.types.Device = None) -> TorchSpecTensors:
        return TorchSpecTensors(spec=spec, device=device)

    def get_spec_tensors(self, device: torch.types.Device = None) -> TorchSpecTensors:
        return TorchRobot._get_spec_tensors(self.spec, device)

    def state(
        self, q: Optional[Float[torch.Tensor, "... num_dofs"]] = None, T_world_base: Optional[TorchSE3] = None
    ) -> TorchRobotState:
        state = TorchRobotState(spec=self.spec, _q=q, _T_world_base=T_world_base)
        return state

    def map_to_full_joint_values(
        self, actuated_values: Float[torch.Tensor, "... num_actuated_joints"]
    ) -> Float[torch.Tensor, "... num_joints"]:
        spec_t = self.get_spec_tensors(actuated_values.device)
        is_mimic = spec_t.mimic_actuated_joint_indices != -1
        controlling_indices = torch.where(is_mimic, spec_t.mimic_actuated_joint_indices, spec_t.actuated_joint_indices)
        padded_actuated_values = torch.cat([actuated_values, torch.zeros_like(actuated_values[..., :1])], dim=-1)
        safe_controlling_indices = torch.where(controlling_indices == -1, self.num_actuated_joints, controlling_indices)
        controlling_values = padded_actuated_values[..., safe_controlling_indices]
        return controlling_values * spec_t.mimic_multipliers + spec_t.mimic_offsets

    def forward_kinematics(self, state: TorchRobotState) -> TorchRobotState:
        """
        Compute the forward kinematics of the robot for given joint positions statelessly.
        Returns a robot state containing the computed transforms of all links.

        Example:
            >>> from robokit.lie.torch_se3 import TorchSE3
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = TorchRobot.load(load_robot_description("panda_description"), backend="torch")
            >>> q = torch.tensor([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
            >>> state = robot.state(q=q)
            >>> state = robot.forward_kinematics(state)
            >>> link_pose = state.get_T_world_link(robot.link_names.index("panda_hand"))
            >>> expected_pose = TorchSE3(torch.tensor([0.0608599, 0.0, 0.7637312, 0.0382045, 0.9192637, 0.3807715, 0.0922340], dtype=torch.float32))
            >>> torch.allclose(link_pose.as_matrix(), expected_pose.as_matrix(), atol=1e-4)
            True
            >>> T_world_base = TorchSE3(torch.tensor([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0], dtype=torch.float32))
            >>> state = robot.state(q=q, T_world_base=T_world_base)
            >>> state = robot.forward_kinematics(state)
            >>> link_pose = state.get_T_world_link(robot.link_names.index("panda_hand"))
            >>> expected_pose = TorchSE3(torch.tensor([1.76371783, 2.0, 2.93915576, -0.24222432, 0.71524162, 0.29626278, -0.58478125], dtype=torch.float32))
            >>> torch.allclose(link_pose.as_matrix(), expected_pose.as_matrix(), atol=1e-4)
            True
        """
        T_world_base_arg = state.T_world_base if state.has_floating_base else None
        T_world_joint = self._forward_kinematics_joints_via_matrix(state.q, T_world_base_arg)
        T_world_link = self._link_poses_from_joint_poses(T_world_joint, T_world_base_arg)
        state.T_world_joint = TorchSE3(T_world_joint)
        state.T_world_link = TorchSE3(T_world_link)
        return state

    def _link_poses_from_joint_poses(
        self, T_world_joint: Float[torch.Tensor, "... num_joints 7"], T_world_base: Optional[TorchSE3] = None
    ) -> Float[torch.Tensor, "... num_links 7"]:
        spec_t = self.get_spec_tensors(T_world_joint.device)
        if T_world_base is None:
            T_world_base_xyz_wxyz = torch.zeros_like(T_world_joint[..., 0:1, :])
            T_world_base_xyz_wxyz[..., 3] = 1.0  # w = 1
        else:
            T_world_base_xyz_wxyz = T_world_base.xyz_wxyz.unsqueeze(-2).expand_as(T_world_joint[..., 0:1, :])
        return torch.cat([T_world_joint, T_world_base_xyz_wxyz], dim=-2)[..., spec_t.link_parent_joint_indices, :]

    @staticmethod
    def _axis_angle_to_matrix(
        axis: Float[torch.Tensor, "... 3"], angle: Float[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... 3 3"]:
        x, y, z = torch.unbind(axis, -1)
        s, c = torch.sin(angle), torch.cos(angle)
        C = 1 - c

        xs, ys, zs = x * s, y * s, z * s
        xC, yC, zC = x * C, y * C, z * C
        xyC, yzC, zxC = x * yC, y * zC, z * xC

        # fmt: off
        rot_mat = torch.stack([x * xC + c, xyC - zs, zxC + ys,
                            xyC + zs, y * yC + c, yzC - xs,
                            zxC - ys, yzC + xs, z * zC + c], dim=-1).reshape(angle.shape + (3, 3))
        # fmt: on
        return rot_mat

    def _compute_joint_transforms_matrix(
        self,
        q: Float[torch.Tensor, "... num_actuated_joints"],
        T_world_base_matrix: Float[torch.Tensor, "... 4 4"],
    ) -> Float[torch.Tensor, "... num_joints 4 4"]:
        batch_shape, device = q.shape[:-1], q.device
        spec_t = self.get_spec_tensors(device)
        full_q = self.map_to_full_joint_values(q)

        pris_jnt_transform = rot_tl_to_tf_mat(tl=spec_t.joint_axes * full_q.unsqueeze(-1))
        rev_jnt_transform = rot_tl_to_tf_mat(rot_mat=TorchRobot._axis_angle_to_matrix(spec_t.joint_axes, full_q))

        padded_T_world_joint = [None] * self.spec.num_joints + [T_world_base_matrix]
        safe_parent_indices = torch.where(
            spec_t.parent_joint_indices == -1, self.spec.num_joints, spec_t.parent_joint_indices
        )
        for joint_idx in spec_t.topological_order_joint_indices:
            parent_idx = safe_parent_indices[joint_idx]
            T_parent_joint = spec_t.parent_joint_transforms_matrix[joint_idx, :, :]

            jnt_type = spec_t.joint_types[joint_idx]
            if jnt_type == JointType.FIXED:
                jnt_transform = T_parent_joint
            elif jnt_type == JointType.REVOLUTE:
                jnt_transform = T_parent_joint @ rev_jnt_transform[..., joint_idx, :, :]
            elif jnt_type == JointType.PRISMATIC:
                jnt_transform = T_parent_joint @ pris_jnt_transform[..., joint_idx, :, :]
            else:
                raise ValueError(f"Unknown joint type: {jnt_type}")
            padded_T_world_joint[joint_idx] = padded_T_world_joint[parent_idx] @ jnt_transform  # type: ignore[operator]

        return torch.stack(padded_T_world_joint[:-1], dim=len(batch_shape))  # type: ignore

    def _forward_kinematics_joints_via_matrix(
        self, q: Float[torch.Tensor, "... num_actuated_joints"], T_world_base: Optional[TorchSE3] = None
    ) -> Float[torch.Tensor, "... num_joints 7"]:
        batch_shape, device, dtype = q.shape[:-1], q.device, q.dtype
        if T_world_base is not None:
            T_world_base_matrix = T_world_base.as_matrix().expand(batch_shape + (4, 4))
        else:
            T_world_base_matrix = torch.eye(4, device=device, dtype=dtype).expand(batch_shape + (4, 4))

        T_world_joint_matrix = self._compute_joint_transforms_matrix(q, T_world_base_matrix)
        return TorchSE3.from_matrix(T_world_joint_matrix).xyz_wxyz

    def _forward_kinematics_joints_via_se3(
        self, q: Float[torch.Tensor, "... num_actuated_joints"], T_world_base: Optional[TorchSE3] = None
    ) -> Float[torch.Tensor, "... num_joints 7"]:
        batch_shape, device, dtype = q.shape[:-1], q.device, q.dtype
        spec_t = self.get_spec_tensors(device)
        full_q = self.map_to_full_joint_values(q)
        tangents = spec_t.joint_twists * full_q.unsqueeze(-1)
        delta_tf = TorchSE3.exp(tangents).xyz_wxyz
        T_parent_child = _se3_multiply(spec_t.parent_joint_transforms, delta_tf)
        if T_world_base is not None:
            T_world_base_xyz_wxyz = T_world_base.xyz_wxyz
        else:
            T_world_base_xyz_wxyz = torch.zeros(batch_shape + (7,), device=device, dtype=dtype)
            T_world_base_xyz_wxyz[..., 3] = 1.0  # w = 1
        padded_T_world_joint = [None] * self.spec.num_joints + [T_world_base_xyz_wxyz]
        safe_parent_indices = torch.where(
            spec_t.parent_joint_indices == -1, self.spec.num_joints, spec_t.parent_joint_indices
        )
        for joint_idx in spec_t.topological_order_joint_indices:
            parent_idx = safe_parent_indices[joint_idx]
            padded_T_world_joint[joint_idx] = _se3_multiply(
                padded_T_world_joint[parent_idx],  # type: ignore
                T_parent_child[..., joint_idx, :],
            )
        return torch.stack(padded_T_world_joint[:-1], dim=-2)  # type: ignore

    def forward_kinematics_via_matrix(
        self,
        q: Float[torch.Tensor, "... num_dofs"],
        T_world_base: Optional[Float[torch.Tensor, "... 4 4"]] = None,
    ) -> Float[torch.Tensor, "... num_links 4 4"]:
        batch_shape, device, dtype = q.shape[:-1], q.device, q.dtype
        spec_t = self.get_spec_tensors(device)

        if T_world_base is not None:
            T_world_base_matrix = T_world_base.expand(batch_shape + (4, 4)).contiguous()
        else:
            T_world_base_matrix = torch.eye(4, device=device, dtype=dtype).expand(batch_shape + (4, 4))

        T_world_joint_matrix = self._compute_joint_transforms_matrix(q, T_world_base_matrix)
        padded_T_world_joint_matrix = torch.cat([T_world_joint_matrix, T_world_base_matrix.unsqueeze(-3)], dim=-3)
        return padded_T_world_joint_matrix[..., spec_t.link_parent_joint_indices, :, :]

    def transform_link_points(
        self,
        T_world_link: Float[torch.Tensor, "... num_links 4 4"],
        local_points: Float[torch.Tensor, "num_points 3"],
        point_link_indices: Int[torch.Tensor, "num_points"],
    ) -> Float[torch.Tensor, "... num_points 3"]:
        """Transform local points attached to links to world frame.

        Args:
            T_world_link: Link transforms in world frame [..., num_links, 4, 4]
            local_points: Local points in link frames [num_points, 3]
            point_link_indices: Link index for each point [num_points]

        Returns:
            World-frame points [..., num_points, 3]
        """

        selected_transforms = torch.index_select(T_world_link, dim=-3, index=point_link_indices)
        world_points = transform_points(local_points.unsqueeze(-2), selected_transforms).squeeze(-2)
        return world_points

    def compute_motion_subspace(self, state: TorchRobotState) -> TorchRobotState:
        """
        Compute the joint Jacobians for the robot and store them in the state.

        After calling this method, use state.get_link_jacobian() to retrieve Jacobians
        for specific links. When a floating base is provided, the returned Jacobian will
        have J_base prepended to J_joints (shape: [6, 6 + num_dofs]).

        Example:
            >>> from robokit.lie.torch_se3 import TorchSE3
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = TorchRobot.load(load_robot_description("panda_description"), backend="torch")
            >>> T_world_base = TorchSE3(torch.tensor([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0], dtype=torch.float32))
            >>> q = torch.tensor([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
            >>> state = robot.state(q=q, T_world_base=T_world_base)
            >>> state = robot.compute_motion_subspace(state)
            >>> J = state.get_link_jacobian(robot.link_names.index("panda_hand"), "body")
            >>> torch.allclose(J[0, :6], torch.tensor([0.693, 0.7071, 0.1405, -0.54, 0.5207, 0.043], dtype=torch.float32), atol=1e-4)
            True
            >>> J = state.get_link_jacobian(robot.link_names.index("panda_hand"), "spatial")
            >>> torch.allclose(J[0, 6:], torch.tensor([0.0001, -3., 1.8641, 3.2647, -1.4346, 3.0467, -0.3974, 0.], dtype=torch.float32), atol=1e-3)
            True
            >>> torch.allclose(J[0, :6], torch.tensor([0.0001, 0., 1., -2., -3., 0.0001], dtype=torch.float32), atol=1e-3)
            True
        """
        if not state.is_fk_computed:
            state = self.forward_kinematics(state)

        spec_t = self.get_spec_tensors(state.q.device)
        # Ad_T is [..., num_joints, 6, 6], twists is [num_joints, 6]
        # Compute per-joint world-frame twists then transpose to [..., 6, num_joints]
        assert state.T_world_joint is not None
        Ad_world_joint = state.T_world_joint.adjoint()
        S_per_joint = torch.matmul(Ad_world_joint, spec_t.joint_twists.unsqueeze(-1)).squeeze(-1)
        state.S_world = S_per_joint.transpose(-1, -2)  # type: ignore

        return state

    def sample_q(
        self, num_samples: int = 1, rng: Optional[torch.Generator] = None, device: torch.types.Device = None
    ) -> Float[torch.Tensor, "num_samples num_actuated_joints"]:
        q_limits = self.get_spec_tensors(device).actuated_joint_limits
        rand = torch.rand((num_samples, q_limits.shape[0]), generator=rng, device=device)
        return rand * (q_limits[:, 1] - q_limits[:, 0]) + q_limits[:, 0]

    def build_inverse_kinematics_optimizer(
        self,
        frame_names: Union[str, Sequence[str]],
        T_world_target: Union[TorchSE3, Sequence[TorchSE3]],
        position_weight: float = 1.0,
        orientation_weight: float = 0.2,
        limit_weight: float = 2.0,
        optimizer_config: Optional["TorchOptimizerConfig"] = None,
    ) -> "TorchOptimizer":
        """
        Build an inverse kinematics optimizer for the robot.

        Example:
            >>> from robokit.lie.torch_se3 import TorchSE3
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("ur10_description"), backend="torch")
            >>> target_pose = TorchSE3(torch.tensor([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]))
            >>> ik_optimizer = robot.build_inverse_kinematics_optimizer("ee_link", target_pose)
            >>> state = robot.state(q=robot.zero_q)
            >>> state = ik_optimizer.solve(state)
            >>> state = robot.forward_kinematics(state)
            >>> achieved_pose = state.get_T_world_link(robot.link_names.index("ee_link"))
            >>> torch.allclose(achieved_pose.xyz, target_pose.xyz, atol=1e-3)
            True
            >>> torch.allclose(achieved_pose.quat_wxyz, target_pose.quat_wxyz, atol=1e-3)
            True
        """
        from robokit.opt.torch_optimizer import TorchOptimizer, TorchOptimizerConfig
        from robokit.terms import Term
        from robokit.terms.torch.frame_task import TorchFrameTask
        from robokit.terms.torch.position_limit import TorchPositionLimit

        frame_names = [frame_names] if isinstance(frame_names, str) else frame_names
        T_world_target = [T_world_target] if isinstance(T_world_target, TorchSE3) else T_world_target

        terms: List[Term] = []
        for idx, frame_name in enumerate(frame_names):
            frame_index = self.link_names.index(frame_name)
            frame_task = TorchFrameTask(
                robot=self,
                frame_index=frame_index,
                T_world_target=T_world_target[idx],
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
            terms.append(frame_task)
        position_limit = TorchPositionLimit(robot=self, weight=limit_weight)
        terms.append(position_limit)
        if optimizer_config is None:
            optimizer_config = TorchOptimizerConfig(lm_lambda=1.0)
        return TorchOptimizer(terms=terms, config=optimizer_config)

    @property
    def zero_q(self) -> Float[torch.Tensor, "num_actuated_joints"]:
        return self.get_spec_tensors().zero_q

    @property
    def midrange_q(self) -> Float[torch.Tensor, "num_actuated_joints"]:
        return self.get_spec_tensors().midrange_q


__all__ = ["TorchRobot", "TorchRobotState", "TorchSpecTensors"]
