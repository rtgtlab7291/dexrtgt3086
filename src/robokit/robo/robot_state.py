# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
import math
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Optional, Tuple, cast

import warp as wp

from robokit.lie.se3 import se3_identity
from robokit.opt.var_values import Var
from robokit.robo.robot_kernels import (
    accept_state_forward_kinematics_floating_kernel,
    accept_state_forward_kinematics_kernel,
    compute_link_jacobian_kernel,
    integrate_floating_base_kernel,
    integrate_joint_positions_kernel,
)
from robokit.robo.robot_spec import RobotSpec
from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather as warp_gather
from robokit.utils.warp_utils import wp_device_type, wp_vec7


# lazily allocated buffers, copied in clone()/gather() only when materialized
_GATHER_BUFFERS = (
    "S_world",
    "collision_sphere_centers_world",
    "link_bounding_sphere_centers_world",
    "link_capsule_endpoint_a_world",
    "link_capsule_endpoint_b_world",
)
_CLONE_BUFFERS = _GATHER_BUFFERS + ("com_world",)


if TYPE_CHECKING:
    from robokit.robo.robot import Robot


class RobotState(Var):
    spec: RobotSpec
    q: wp.array  # [batch_size, num_dofs] or [batch_size, num_frames, num_dofs]
    T_world_base: wp.array  # wp_vec7
    T_world_joint: wp.array  # wp_vec7
    T_world_link: wp.array  # wp_vec7

    has_floating_base: bool
    is_fk_computed: bool
    is_motion_subspace_computed: bool
    is_collision_spheres_computed: bool
    is_capsules_computed: bool
    is_com_computed: bool

    def __init__(self, robot: "Robot", q: wp.array, T_world_base: Optional[wp.array] = None):
        self.robot = robot
        self.spec = robot.spec
        self.q = self._validate_q_shape(q).contiguous()
        requires_grad = self.q.requires_grad
        self.base_step_scale = 1e-4
        self.spec_tensors = robot.spec.get_tensors(str(self.q.device))

        if T_world_base is None:
            self.T_world_base = se3_identity(shape=self.shape, device=self.device, requires_grad=requires_grad)
            self.has_floating_base = False
        else:
            self.T_world_base = T_world_base.contiguous()
            self.has_floating_base = True

        self.T_world_joint = se3_identity(
            shape=self.shape + (self.spec.num_joints,), device=self.device, requires_grad=requires_grad
        )
        self.T_world_link = se3_identity(
            shape=self.shape + (self.spec.num_links,), device=self.device, requires_grad=requires_grad
        )
        self.invalidate()

    def __repr__(self) -> str:
        return f"RobotState(q={self.q.numpy()}, T_world_base={self.T_world_base}, is_fk_computed={self.is_fk_computed})"

    # --- lazily allocated buffers (safe with CUDA graphs: capture sites warm up first) ---
    @cached_property
    def X_local(self) -> wp.array:
        return wp.zeros(shape=self.shape + (self.spec.num_joints,), dtype=wp_vec7, device=self.device)

    @cached_property
    def S_world(self) -> wp.array:
        return wp.zeros(
            shape=self.shape + (6, self.spec.num_joints),
            dtype=wp.float32,
            device=self.device,
            requires_grad=self.q.requires_grad,
        )

    @cached_property
    def com_world(self) -> wp.array:
        return wp.zeros(shape=(self.num_elements,), dtype=wp.vec3, device=self.device)

    @cached_property
    def collision_sphere_centers_world(self) -> wp.array:
        num_spheres = len(self.spec.local_collision_sphere_centers) if self.spec.has_collision_spheres else 0
        return wp.zeros(
            shape=self.shape + (num_spheres,), dtype=wp.vec3, device=self.device, requires_grad=self.q.requires_grad
        )

    @cached_property
    def link_bounding_sphere_centers_world(self) -> wp.array:
        num_bounding = self.spec.num_links if self.spec.has_collision_spheres else 0
        return wp.zeros(
            shape=self.shape + (num_bounding,), dtype=wp.vec3, device=self.device, requires_grad=self.q.requires_grad
        )

    @cached_property
    def link_capsule_endpoint_a_world(self) -> wp.array:
        return wp.zeros(
            shape=self.shape + (self.spec.num_links,),
            dtype=wp.vec3,
            device=self.device,
            requires_grad=self.q.requires_grad,
        )

    @cached_property
    def link_capsule_endpoint_b_world(self) -> wp.array:
        return wp.zeros(
            shape=self.shape + (self.spec.num_links,),
            dtype=wp.vec3,
            device=self.device,
            requires_grad=self.q.requires_grad,
        )

    # --- shape and device properties ---
    def _validate_q_shape(self, q: wp.array) -> wp.array:
        if q.ndim == 1:
            q = q.reshape((1, q.shape[0]))
        elif q.ndim in {2, 3}:
            if q.shape[-1] != self.spec.num_actuated_joints:
                raise ValueError(f"Expected q to have {self.spec.num_actuated_joints} DOFs, but got {q.shape[1]}")
        else:
            raise ValueError(f"Expected q to have 1, 2, or 3 dimensions, but got {q.ndim}")
        return q

    @property
    def is_trajectory(self) -> bool:
        return self.q.ndim == 3  # [batch_size, num_frames, num_dofs]

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

    # --- computed results access ---
    def get_T_world_link(self, link_index: int) -> wp.array:
        if not self.is_fk_computed:
            raise RuntimeError("Forward kinematics has not been computed. Call robot.forward_kinematics() first.")
        xyz_wxyz = self.T_world_link
        index = tuple([slice(None)] * (xyz_wxyz.ndim - 1) + [link_index])
        # slicing out one link yields a strided view; downstream se3_* kernels flatten
        return cast(wp.array, xyz_wxyz[index]).contiguous()

    def get_link_jacobian(self, link_index: int, reference_frame: Literal["body", "spatial"] = "body") -> wp.array:
        if not self.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(self)

        wp.init()
        device = self.S_world.device
        requires_grad = self.S_world.requires_grad
        num_actuated = self.spec.num_actuated_joints
        ne = self.num_elements

        total_dofs = (6 if self.has_floating_base else 0) + num_actuated
        J_out = wp.zeros((*self.shape, 6, total_dofs), dtype=wp.float32, device=device, requires_grad=requires_grad)
        spec_t = self.spec.get_tensors(str(device))
        reference_frame_int = 1 if reference_frame == "body" else 0

        wp.launch(
            kernel=compute_link_jacobian_kernel,
            dim=(ne, total_dofs),
            inputs=[
                self.S_world.reshape((ne, 6, self.spec.num_joints)),
                self.T_world_link.reshape((ne, self.spec.num_links)),
                self.T_world_base.flatten(),
                spec_t.link_ancestor_joints_mask[link_index],
                spec_t.joints_to_actuated_mapping,
                reference_frame_int,
                self.has_floating_base,
                link_index,
                J_out.reshape((ne, 6, total_dofs)),
            ],
            device=device,
        )

        return J_out

    # --- solver interface (Var) ---
    def invalidate(self):
        self.is_fk_computed = False
        self.is_motion_subspace_computed = False
        self.is_collision_spheres_computed = False
        self.is_capsules_computed = False
        self.is_com_computed = False

    def precompute(self):
        if not self.is_fk_computed:
            self.robot.forward_kinematics(self)

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["RobotState"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
    ) -> "RobotState":
        """Integrate the state by a tangent-space velocity."""
        if velocity.shape != (self.batch_size, self.tangent_dim):
            raise ValueError(f"Expected velocity shape {(self.batch_size, self.tangent_dim)}, but got {velocity.shape}")

        requires_grad = velocity.requires_grad or self.q.requires_grad
        if out is None:
            out = type(self)(
                robot=self.robot,
                q=wp.empty_like(self.q, requires_grad=requires_grad),
                T_world_base=wp.empty_like(self.T_world_base, requires_grad=requires_grad)
                if self.has_floating_base
                else None,
            )

        ne = self.num_elements
        na = self.spec.num_actuated_joints
        num_frames = self.shape[1] if self.is_trajectory else 1
        single_tangent_dim = (6 if self.has_floating_base else 0) + na
        velocity_offset = 0
        if self.has_floating_base:
            wp.launch(
                kernel=integrate_floating_base_kernel,
                dim=ne,
                inputs=[
                    self.T_world_base.flatten(),
                    velocity,
                    tangent_mask,
                    tangent_mask is not None,
                    num_frames,
                    single_tangent_dim,
                    wp.float32(self.base_step_scale),
                ],
                outputs=[out.T_world_base.flatten()],
                device=self.device,
            )
            velocity_offset = 6
        elif out.T_world_base.ptr != self.T_world_base.ptr:
            wp.copy(out.T_world_base, self.T_world_base)

        wp.launch(
            kernel=integrate_joint_positions_kernel,
            dim=(ne, na),
            inputs=[
                self.q.reshape((ne, na)),
                velocity,
                velocity_offset,
                num_frames,
                single_tangent_dim,
                tangent_mask,
                tangent_mask is not None,
                weight_decay,
            ],
            outputs=[out.q.reshape((ne, na))],
            device=self.device,
        )

        out.base_step_scale = self.base_step_scale
        out.invalidate()
        return out

    def clone(self) -> "RobotState":
        state = object.__new__(type(self))
        state.spec = self.spec
        state.q = wp.clone(self.q)
        state.T_world_base = wp.clone(self.T_world_base)
        state.T_world_joint = wp.clone(self.T_world_joint)
        state.T_world_link = wp.clone(self.T_world_link)
        for name in _CLONE_BUFFERS:
            if name in self.__dict__:
                setattr(state, name, wp.clone(self.__dict__[name]))
        state.has_floating_base = self.has_floating_base
        state.is_fk_computed = self.is_fk_computed
        state.is_motion_subspace_computed = self.is_motion_subspace_computed
        state.is_collision_spheres_computed = self.is_collision_spheres_computed
        state.is_capsules_computed = self.is_capsules_computed
        state.is_com_computed = self.is_com_computed
        state.base_step_scale = self.base_step_scale
        state.robot = self.robot
        state.spec_tensors = self.spec_tensors
        return state

    def flatten(self) -> "RobotState":
        """Zero-copy view with the frame axis folded into the batch axis (self if not a trajectory).

        Lets per-pose kernels run unchanged over a trajectory: `q` becomes 2D, so
        `is_trajectory` is False and `tangent_dim` is the single-frame one.
        """
        if not self.is_trajectory:
            return self
        ne = self.num_elements
        state = object.__new__(type(self))
        state.spec = self.spec
        state.q = self.q.reshape((ne, self.q.shape[-1]))
        state.T_world_base = self.T_world_base.flatten()
        state.T_world_joint = self.T_world_joint.reshape((ne, self.spec.num_joints))
        state.T_world_link = self.T_world_link.reshape((ne, self.spec.num_links))
        for name in _CLONE_BUFFERS + ("X_local",):
            buf = self.__dict__.get(name)
            if buf is not None:
                # com_world is already flat; everything else leads with (batch, frames)
                shape = (ne,) + tuple(buf.shape[2:])
                setattr(state, name, buf.reshape(shape) if tuple(buf.shape[:2]) == self.shape else buf)
        state.has_floating_base = self.has_floating_base
        state.is_fk_computed = self.is_fk_computed
        state.is_motion_subspace_computed = self.is_motion_subspace_computed
        state.is_collision_spheres_computed = self.is_collision_spheres_computed
        state.is_capsules_computed = self.is_capsules_computed
        state.is_com_computed = self.is_com_computed
        state.base_step_scale = self.base_step_scale
        state.robot = self.robot
        state.spec_tensors = self.spec_tensors
        return state

    def gather(self, indices: ArrayLike, out: Optional["RobotState"] = None, q_only: bool = True) -> "RobotState":
        num_gather = indices.shape[0]
        device = self.q.device
        requires_grad = self.q.requires_grad

        if out is None:
            new_shape = (num_gather,) + self.q.shape[1:]
            q_placeholder = wp.empty(new_shape, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self.has_floating_base:
                T_world_base_placeholder = wp.empty(
                    new_shape[:-1], dtype=wp_vec7, device=device, requires_grad=requires_grad
                )
            else:
                T_world_base_placeholder = None
            out = type(self)(robot=self.robot, q=q_placeholder, T_world_base=T_world_base_placeholder)

        warp_gather(self.q, indices, out.q)
        if self.has_floating_base:
            warp_gather(self.T_world_base, indices, out.T_world_base)
        if not q_only:
            if not self.has_floating_base:
                warp_gather(self.T_world_base, indices, out.T_world_base)
            warp_gather(self.T_world_joint, indices, out.T_world_joint)
            warp_gather(self.T_world_link, indices, out.T_world_link)
            for name in _GATHER_BUFFERS:
                if name in self.__dict__:
                    warp_gather(self.__dict__[name], indices, getattr(out, name))

        out.has_floating_base = self.has_floating_base
        out.is_fk_computed = False if q_only else self.is_fk_computed
        out.is_motion_subspace_computed = False if q_only else self.is_motion_subspace_computed
        out.is_collision_spheres_computed = False if q_only else self.is_collision_spheres_computed
        out.is_capsules_computed = False if q_only else self.is_capsules_computed
        out.is_com_computed = False  # com_world is never gathered
        out.base_step_scale = self.base_step_scale
        out.robot = self.robot
        out.spec_tensors = self.spec_tensors
        return out

    def accept(self, accept_mask: ArrayLike, proposed: "RobotState") -> "RobotState":
        """Optimized accept: copies q + FK results only (1 kernel)."""
        ne = self.num_elements
        na, nj, nl = self.spec.num_actuated_joints, self.spec.num_joints, self.spec.num_links
        epb = ne // self.batch_size
        if self.has_floating_base:
            wp.launch(
                accept_state_forward_kinematics_floating_kernel,
                dim=ne,
                inputs=[
                    accept_mask,
                    proposed.q.reshape((ne, na)),
                    self.q.reshape((ne, na)),
                    proposed.T_world_base.flatten(),
                    self.T_world_base.flatten(),
                    proposed.T_world_joint.reshape((ne, nj)),
                    self.T_world_joint.reshape((ne, nj)),
                    proposed.T_world_link.reshape((ne, nl)),
                    self.T_world_link.reshape((ne, nl)),
                    epb,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                accept_state_forward_kinematics_kernel,
                dim=ne,
                inputs=[
                    accept_mask,
                    proposed.q.reshape((ne, na)),
                    self.q.reshape((ne, na)),
                    proposed.T_world_joint.reshape((ne, nj)),
                    self.T_world_joint.reshape((ne, nj)),
                    proposed.T_world_link.reshape((ne, nl)),
                    self.T_world_link.reshape((ne, nl)),
                    epb,
                ],
                device=self.device,
            )
        self.is_fk_computed = True
        self.is_motion_subspace_computed = False
        self.is_collision_spheres_computed = False
        self.is_capsules_computed = False
        self.is_com_computed = False
        return self


RobotCollisionState = RobotState
