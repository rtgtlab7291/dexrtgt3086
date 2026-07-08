# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIncompatibleVariableOverride=false
# pyright: reportOptionalMemberAccess=false
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.robo.warp_robot_kernels import se3_adjoint_multiply_vec6_func
from robokit.terms.terms import WarpTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_apply_func


if TYPE_CHECKING:
    from robokit.robo.warp_robot import WarpRobot, WarpRobotState


@wp.kernel
def _transform_contact_points_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    local_contact_points: wp.array2d(dtype=wp.vec3),
    contact_link_indices: wp.array2d(dtype=wp.int32),
    world_contact_points: wp.array2d(dtype=wp.vec3),
):
    instance, contact_idx = wp.tid()

    link_idx = contact_link_indices[instance, contact_idx]
    T_world = T_world_link[instance, link_idx]

    translation = wp.vec3(T_world[0], T_world[1], T_world[2])
    quat_wxyz = wp.vec4(T_world[3], T_world[4], T_world[5], T_world[6])

    local_pt = local_contact_points[instance, contact_idx]
    world_pt = quaternion_apply_func(quat_wxyz, local_pt) + translation

    world_contact_points[instance, contact_idx] = world_pt


@wp.kernel
def _sample_candidate_indices_kernel(
    random_states: wp.array2d(dtype=wp.uint32),
    num_candidates: int,
    sampled_indices: wp.array2d(dtype=wp.int32),
):
    """Randomly sample candidate indices (with replacement)."""
    instance, contact_idx = wp.tid()

    state = random_states[instance, contact_idx]
    sampled_indices[instance, contact_idx] = wp.randi(state, 0, num_candidates)
    random_states[instance, contact_idx] = state


@wp.kernel
def _transform_contact_points_from_indices_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    contact_candidates_link_frame: wp.array1d(dtype=wp.vec3),
    contact_candidates_link_indices: wp.array1d(dtype=wp.int32),
    contact_point_indices: wp.array2d(dtype=wp.int32),
    world_contact_points: wp.array2d(dtype=wp.vec3),
):
    instance, contact_idx = wp.tid()

    candidate_idx = contact_point_indices[instance, contact_idx]
    local_pt = contact_candidates_link_frame[candidate_idx]
    link_idx = contact_candidates_link_indices[candidate_idx]

    T_world = T_world_link[instance, link_idx]
    translation = wp.vec3(T_world[0], T_world[1], T_world[2])
    quat_wxyz = wp.vec4(T_world[3], T_world[4], T_world[5], T_world[6])
    world_pt = quaternion_apply_func(quat_wxyz, local_pt) + translation

    world_contact_points[instance, contact_idx] = world_pt


@wp.kernel
def _extract_link_indices_from_candidates_kernel(
    contact_candidates_link_indices: wp.array1d(dtype=wp.int32),
    contact_point_indices: wp.array2d(dtype=wp.int32),
    contact_link_indices: wp.array2d(dtype=wp.int32),
):
    instance, contact_idx = wp.tid()
    candidate_idx = contact_point_indices[instance, contact_idx]
    contact_link_indices[instance, contact_idx] = contact_candidates_link_indices[candidate_idx]


@wp.kernel
def query_sdf_for_contact_points_single_scene_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    inv_mesh_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_mesh_poses: wp.bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_mesh_scales: wp.bool,
    max_dist: float,
    signed_dists: wp.array2d(dtype=wp.float32),
):
    """Query SDF for contact points against multiple scene meshes.

    All instances query against the same set of meshes (single scene, multiple samples).
    """
    instance, point_idx = wp.tid()

    query_pt = contact_points[instance, point_idx]

    min_abs_dist = wp.float32(max_dist)
    min_sign = wp.float32(1.0)

    for mesh_idx in range(mesh_ids.shape[0]):
        query_pt_in_mesh_coord = query_pt
        if enable_inv_mesh_poses:
            query_pt_in_mesh_coord = wp.transform_point(inv_mesh_poses[mesh_idx], query_pt)
        if enable_mesh_scales:
            query_pt_in_mesh_coord = query_pt_in_mesh_coord / mesh_scales[mesh_idx]

        query = wp.mesh_query_point(mesh_ids[mesh_idx], query_pt_in_mesh_coord, max_dist)
        if query.result:
            closest_pt_in_mesh_coord = wp.mesh_eval_position(mesh_ids[mesh_idx], query.face, query.u, query.v)
            abs_dist = wp.length(query_pt_in_mesh_coord - closest_pt_in_mesh_coord)
            if enable_mesh_scales:
                abs_dist = abs_dist * mesh_scales[mesh_idx]

            if abs_dist < min_abs_dist:
                min_abs_dist = abs_dist
                min_sign = wp.float32(query.sign)

    signed_dists[instance, point_idx] = min_abs_dist * min_sign


@wp.kernel
def query_sdf_for_contact_points_multi_scene_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    scene_indices: wp.array1d(dtype=wp.int32),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    mesh_first_idx: wp.array1d(dtype=wp.int32),
    inv_mesh_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_mesh_poses: wp.bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_mesh_scales: wp.bool,
    max_dist: float,
    signed_dists: wp.array2d(dtype=wp.float32),
):
    """Query SDF for contact points - each instance queries only its scene's meshes."""
    instance, point_idx = wp.tid()

    query_pt = contact_points[instance, point_idx]
    scene_idx = scene_indices[instance]

    mesh_begin = mesh_first_idx[scene_idx]
    mesh_end = mesh_first_idx[scene_idx + 1]

    min_abs_dist = wp.float32(max_dist)
    min_sign = wp.float32(1.0)

    for mesh_idx in range(mesh_begin, mesh_end):
        query_pt_in_mesh_coord = query_pt
        if enable_inv_mesh_poses:
            query_pt_in_mesh_coord = wp.transform_point(inv_mesh_poses[mesh_idx], query_pt)
        if enable_mesh_scales:
            query_pt_in_mesh_coord = query_pt_in_mesh_coord / mesh_scales[mesh_idx]

        query = wp.mesh_query_point(mesh_ids[mesh_idx], query_pt_in_mesh_coord, max_dist)
        if query.result:
            closest_pt_in_mesh_coord = wp.mesh_eval_position(mesh_ids[mesh_idx], query.face, query.u, query.v)
            abs_dist = wp.length(query_pt_in_mesh_coord - closest_pt_in_mesh_coord)
            if enable_mesh_scales:
                abs_dist = abs_dist * mesh_scales[mesh_idx]

            if abs_dist < min_abs_dist:
                min_abs_dist = abs_dist
                min_sign = wp.float32(query.sign)

    signed_dists[instance, point_idx] = min_abs_dist * min_sign


@wp.kernel
def compute_distance_energy_residual_kernel(
    signed_dists: wp.array2d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
):
    """Compute per-contact-point abs(sdf) residual."""
    instance, point_idx = wp.tid()
    sdf = signed_dists[instance, point_idx]
    abs_sdf = wp.abs(sdf)
    residual = abs_sdf
    if use_sqrt:
        residual = wp.sqrt(abs_sdf + eps)
    residual_buffer[instance, row_offset + point_idx] = residual_weight[point_idx] * residual


@wp.kernel
def query_sdf_and_normal_for_contact_points_single_scene_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    inv_mesh_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_mesh_poses: wp.bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_mesh_scales: wp.bool,
    max_dist: float,
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
):
    """Query SDF and surface normal for contact points against multiple scene meshes."""
    instance, point_idx = wp.tid()

    query_pt = contact_points[instance, point_idx]

    min_abs_dist = wp.float32(max_dist)
    min_sign = wp.float32(1.0)
    min_closest_world = wp.vec3(0.0, 0.0, 0.0)
    any_found = wp.bool(False)

    for mesh_idx in range(mesh_ids.shape[0]):
        query_pt_in_mesh_coord = query_pt
        if enable_inv_mesh_poses:
            query_pt_in_mesh_coord = wp.transform_point(inv_mesh_poses[mesh_idx], query_pt)
        if enable_mesh_scales:
            query_pt_in_mesh_coord = query_pt_in_mesh_coord / mesh_scales[mesh_idx]

        query = wp.mesh_query_point(mesh_ids[mesh_idx], query_pt_in_mesh_coord, max_dist)
        if query.result:
            closest_pt_in_mesh_coord = wp.mesh_eval_position(mesh_ids[mesh_idx], query.face, query.u, query.v)
            abs_dist = wp.length(query_pt_in_mesh_coord - closest_pt_in_mesh_coord)
            if enable_mesh_scales:
                abs_dist = abs_dist * mesh_scales[mesh_idx]

            if abs_dist < min_abs_dist:
                min_abs_dist = abs_dist
                min_sign = wp.float32(query.sign)
                any_found = wp.bool(True)

                closest_pt_unscaled = closest_pt_in_mesh_coord
                if enable_mesh_scales:
                    closest_pt_unscaled = closest_pt_in_mesh_coord * mesh_scales[mesh_idx]

                closest_world = closest_pt_unscaled
                if enable_inv_mesh_poses:
                    mesh_pose = wp.inverse(inv_mesh_poses[mesh_idx])
                    closest_world = wp.transform_point(mesh_pose, closest_pt_unscaled)

                min_closest_world = closest_world

    signed_dists[instance, point_idx] = min_abs_dist * min_sign

    if any_found:
        diff = query_pt - min_closest_world
        dist = wp.length(diff)
        denom = wp.max(wp.float32(1e-8), dist)
        normal = diff / denom
        normal = normal * min_sign
        normals[instance, point_idx] = normal
    else:
        normals[instance, point_idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def query_sdf_and_normal_for_contact_points_multi_scene_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    scene_indices: wp.array1d(dtype=wp.int32),
    mesh_ids: wp.array1d(dtype=wp.uint64),
    mesh_first_idx: wp.array1d(dtype=wp.int32),
    inv_mesh_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_mesh_poses: wp.bool,
    mesh_scales: wp.array1d(dtype=wp.float32),
    enable_mesh_scales: wp.bool,
    max_dist: float,
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
):
    """Query SDF and surface normal - each instance queries only its scene's meshes."""
    instance, point_idx = wp.tid()

    query_pt = contact_points[instance, point_idx]
    scene_idx = scene_indices[instance]

    mesh_begin = mesh_first_idx[scene_idx]
    mesh_end = mesh_first_idx[scene_idx + 1]

    min_abs_dist = wp.float32(max_dist)
    min_sign = wp.float32(1.0)
    min_closest_world = wp.vec3(0.0, 0.0, 0.0)
    any_found = wp.bool(False)

    for mesh_idx in range(mesh_begin, mesh_end):
        query_pt_in_mesh_coord = query_pt
        if enable_inv_mesh_poses:
            query_pt_in_mesh_coord = wp.transform_point(inv_mesh_poses[mesh_idx], query_pt)
        if enable_mesh_scales:
            query_pt_in_mesh_coord = query_pt_in_mesh_coord / mesh_scales[mesh_idx]

        query = wp.mesh_query_point(mesh_ids[mesh_idx], query_pt_in_mesh_coord, max_dist)
        if query.result:
            closest_pt_in_mesh_coord = wp.mesh_eval_position(mesh_ids[mesh_idx], query.face, query.u, query.v)
            abs_dist = wp.length(query_pt_in_mesh_coord - closest_pt_in_mesh_coord)
            if enable_mesh_scales:
                abs_dist = abs_dist * mesh_scales[mesh_idx]

            if abs_dist < min_abs_dist:
                min_abs_dist = abs_dist
                min_sign = wp.float32(query.sign)
                any_found = wp.bool(True)

                closest_pt_unscaled = closest_pt_in_mesh_coord
                if enable_mesh_scales:
                    closest_pt_unscaled = closest_pt_in_mesh_coord * mesh_scales[mesh_idx]

                closest_world = closest_pt_unscaled
                if enable_inv_mesh_poses:
                    mesh_pose = wp.inverse(inv_mesh_poses[mesh_idx])
                    closest_world = wp.transform_point(mesh_pose, closest_pt_unscaled)

                min_closest_world = closest_world

    signed_dists[instance, point_idx] = min_abs_dist * min_sign

    if any_found:
        diff = query_pt - min_closest_world
        dist = wp.length(diff)
        denom = wp.max(wp.float32(1e-8), dist)
        normal = diff / denom
        normal = normal * min_sign
        normals[instance, point_idx] = normal
    else:
        normals[instance, point_idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_distance_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    contact_points_world: wp.array2d(dtype=wp.vec3),
    contact_points_link_indices: wp.array2d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),
):
    """Compute weighted Jacobian for distance task.

    Jacobian formula: dr_i/dq = sign(sdf_i) * n_i . (dp_i/dq)
    where n_i is the surface normal and dp_i/dq is the FK Jacobian for position.

    Note: contact_points_link_indices is 2D [batch_size, num_contact_points] to support
    per-sample contact point selection (e.g., when contact points are sampled from candidates).
    """
    instance, contact_idx, col_idx = wp.tid()
    base_dofs = 6 if has_floating_base else 0
    actuated_idx = col_idx - base_dofs

    contact_point = contact_points_world[instance, contact_idx]
    signed_dist = signed_dists[instance, contact_idx]
    normal = normals[instance, contact_idx]

    sign_sdf = wp.sign(signed_dist)
    abs_sdf = wp.abs(signed_dist)
    scale = wp.float32(1.0)
    if use_sqrt:
        scale = wp.float32(0.5) / wp.sqrt(abs_sdf + eps)

    spatial_twist = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if has_floating_base and col_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = wp.float32(1.0)
        spatial_twist = se3_adjoint_multiply_vec6_func(T_world_base[instance], unit_vec)
    else:
        link_idx = contact_points_link_indices[instance, contact_idx]
        num_joints = S_world.shape[2]
        for joint_idx in range(num_joints):
            if link_ancestor_joints_mask[link_idx, joint_idx]:
                weight = joints_to_actuated[joint_idx, actuated_idx]
                if weight != 0.0:
                    for row in range(6):
                        spatial_twist[row] = spatial_twist[row] + S_world[instance, row, joint_idx] * weight

    linear_vel = wp.vec3(spatial_twist[0], spatial_twist[1], spatial_twist[2])
    angular_vel = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])
    point_vel = linear_vel + wp.cross(angular_vel, contact_point)

    jacobian_value = scale * sign_sdf * wp.dot(normal, point_vel)

    jacobian_buffer[instance, row_offset + contact_idx, col_idx] = residual_weight[contact_idx] * jacobian_value


@dataclass
class WarpDistanceTask(WarpTask):
    """Distance to contact energy term for grasp optimization.

    E_dis = sum(abs(signed_distance)) for contact points on the hand.

    This task computes the sum of absolute signed distances from contact points
    to the target object surface. The residual is per-contact-point abs(sdf).

    All instances query against the same set of scene meshes. This is suitable for
    grasp optimization where multiple grasp samples are evaluated against a single
    target object.

    Contact data can be provided in two ways:
    1. Pass contact_points_link_frame and contact_link_indices to compute methods
    2. Pass contact_candidates_* to constructor for per-iteration random sampling
    """

    robot: "WarpRobot"
    scene_meshes: Sequence[wp.Mesh]
    num_contact_points: int
    weight: Optional[Union[float, Sequence[float]]] = None
    max_dist: float = 1e6
    batch_size: int = 1
    inv_mesh_poses: Optional[wp.array] = None
    mesh_scales: Optional[wp.array] = None
    contact_candidates_link_frame: Optional[wp.array] = None
    contact_candidates_link_indices: Optional[wp.array] = None
    candidates_seed: int = 42
    residual_mode: Literal["abs", "sqrt_abs"] = "abs"
    residual_eps: float = 1e-6
    fixed_contact_points_link_frame: Optional[wp.array] = None
    fixed_contact_link_indices: Optional[wp.array] = None
    shared_contact_point_indices: Optional[wp.array] = None
    shared_contact_link_indices: Optional[wp.array] = None
    sample_shared_contact_indices: bool = False

    # Multi-scene batching support
    scene_indices: Optional[wp.array] = None  # [batch_size] maps instance -> scene index
    mesh_first_idx: Optional[wp.array] = None  # [n_scene + 1] mesh boundaries per scene

    _device: Optional[wp_device_type] = None
    _multi_scene_mode: bool = False
    _mesh_ids: Optional[wp.array] = None
    _inv_mesh_poses: Optional[wp.array] = None
    _mesh_scales: Optional[wp.array] = None
    _enable_inv_mesh_poses: bool = False
    _enable_mesh_scales: bool = False
    _residual_weight_np: Optional[np.ndarray] = None
    residual_weight: Optional[wp.array] = None

    _random_states: Optional[wp.array] = None
    _sampled_contact_point_indices: Optional[wp.array] = None
    _sampled_contact_link_indices: Optional[wp.array] = None
    _use_candidates: bool = False

    def __post_init__(self):
        if len(self.scene_meshes) == 0:
            raise ValueError("scene_meshes must be non-empty.")
        if self.residual_mode not in {"abs", "sqrt_abs"}:
            raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")

        residual_weight = np.ones((self.num_contact_points,), dtype=np.float32)
        if self.weight is not None:
            if isinstance(self.weight, (float, int)):
                residual_weight[:] = float(self.weight)
            else:
                weight_arr = np.asarray(self.weight, dtype=np.float32)
                if weight_arr.shape != (self.num_contact_points,):
                    raise ValueError(f"Expected weight shape ({self.num_contact_points},), got {weight_arr.shape}.")
                residual_weight[:] = weight_arr
        self._residual_weight_np = residual_weight

        # Detect multi-scene mode
        self._multi_scene_mode = self.scene_indices is not None and self.mesh_first_idx is not None

    def _init_buffers(self, device: wp_device_type):
        self._device = device

        self._mesh_ids = wp.array([m.id for m in self.scene_meshes], dtype=wp.uint64, device=device)
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

        if self.inv_mesh_poses is not None:
            self._inv_mesh_poses = self.inv_mesh_poses
            self._enable_inv_mesh_poses = True
        else:
            self._inv_mesh_poses = wp.zeros((len(self.scene_meshes),), dtype=wp.mat44, device=device)
            self._enable_inv_mesh_poses = False

        if self.mesh_scales is not None:
            self._mesh_scales = self.mesh_scales
            self._enable_mesh_scales = True
        else:
            ones_np = np.ones((len(self.scene_meshes),), dtype=np.float32)
            self._mesh_scales = wp.from_numpy(ones_np, dtype=wp.float32, device=device)
            self._enable_mesh_scales = False

        if self.contact_candidates_link_frame is not None:
            self._use_candidates = True
            np.random.seed(self.candidates_seed)
            states_np = np.random.randint(0, 2**32, size=(self.batch_size, self.num_contact_points), dtype=np.uint32)
            self._random_states = wp.from_numpy(states_np, dtype=wp.uint32, device=device)
            self._sampled_contact_point_indices = wp.empty(
                (self.batch_size, self.num_contact_points), dtype=wp.int32, device=device
            )
            self._sampled_contact_link_indices = wp.empty(
                (self.batch_size, self.num_contact_points), dtype=wp.int32, device=device
            )

    @property
    def residual_dim(self) -> int:
        return self.num_contact_points

    def _transform_contact_points(
        self,
        var: "WarpRobotState",
        contact_points_link_frame: wp.array,
        contact_link_indices: wp.array,
    ) -> wp.array:
        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        device = var.q.device

        contact_points_world = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)

        wp.launch(
            kernel=_transform_contact_points_kernel,
            dim=(batch_size, self.num_contact_points),
            inputs=[
                var.T_world_link.xyz_wxyz.reshape((batch_size, num_links)),
                contact_points_link_frame,
                contact_link_indices,
            ],
            outputs=[contact_points_world],
            device=device,
        )

        return contact_points_world

    def compute_weighted_residual(
        self,
        var: "WarpRobotState",
        contact_points_link_frame: Optional[wp.array] = None,
        contact_link_indices: Optional[wp.array] = None,
        residual_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        """Compute the weighted residual for distance energy.

        Args:
            var: Robot state containing FK data.
            contact_points_link_frame: Contact points in link frame [batch_size, num_contact_points] of wp.vec3.
                If None, samples from candidates (if provided at construction).
            contact_link_indices: Link indices for each contact point [batch_size, num_contact_points] of int32.
                If None, samples from candidates (if provided at construction).
            residual_buffer: Optional buffer to write residuals into.
            row_offset: Row offset in the residual buffer.

        Returns:
            Weighted residual [batch_size, num_contact_points].
        """
        if self._device is None:
            self._init_buffers(var.q.device)

        if not var.is_fk_computed:
            var = self.robot.forward_kinematics(var)

        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        device = var.q.device

        if contact_points_link_frame is None and contact_link_indices is None:
            if self.fixed_contact_points_link_frame is not None and self.fixed_contact_link_indices is not None:
                contact_points_world = self._transform_contact_points(
                    var,
                    self.fixed_contact_points_link_frame,
                    self.fixed_contact_link_indices,
                )
            elif self.shared_contact_point_indices is not None:
                if self.shared_contact_link_indices is None:
                    raise ValueError("shared_contact_link_indices must be provided with shared_contact_point_indices.")
                if self.sample_shared_contact_indices:
                    num_candidates = self.contact_candidates_link_frame.shape[0]
                    wp.launch(
                        kernel=_sample_candidate_indices_kernel,
                        dim=(batch_size, self.num_contact_points),
                        inputs=[
                            self._random_states,
                            num_candidates,
                            self.shared_contact_point_indices,
                        ],
                        device=device,
                    )
                    wp.launch(
                        kernel=_extract_link_indices_from_candidates_kernel,
                        dim=(batch_size, self.num_contact_points),
                        inputs=[
                            self.contact_candidates_link_indices,
                            self.shared_contact_point_indices,
                        ],
                        outputs=[self.shared_contact_link_indices],
                        device=device,
                    )
                contact_points_world = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
                wp.launch(
                    kernel=_transform_contact_points_from_indices_kernel,
                    dim=(batch_size, self.num_contact_points),
                    inputs=[
                        var.T_world_link.xyz_wxyz.reshape((batch_size, num_links)),
                        self.contact_candidates_link_frame,
                        self.contact_candidates_link_indices,
                        self.shared_contact_point_indices,
                    ],
                    outputs=[contact_points_world],
                    device=device,
                )
            elif self._use_candidates:
                num_candidates = self.contact_candidates_link_frame.shape[0]
                wp.launch(
                    kernel=_sample_candidate_indices_kernel,
                    dim=(batch_size, self.num_contact_points),
                    inputs=[
                        self._random_states,
                        num_candidates,
                        self._sampled_contact_point_indices,
                    ],
                    device=device,
                )
                contact_points_world = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
                wp.launch(
                    kernel=_transform_contact_points_from_indices_kernel,
                    dim=(batch_size, self.num_contact_points),
                    inputs=[
                        var.T_world_link.xyz_wxyz.reshape((batch_size, num_links)),
                        self.contact_candidates_link_frame,
                        self.contact_candidates_link_indices,
                        self._sampled_contact_point_indices,
                    ],
                    outputs=[contact_points_world],
                    device=device,
                )
                wp.launch(
                    kernel=_extract_link_indices_from_candidates_kernel,
                    dim=(batch_size, self.num_contact_points),
                    inputs=[
                        self.contact_candidates_link_indices,
                        self._sampled_contact_point_indices,
                    ],
                    outputs=[self._sampled_contact_link_indices],
                    device=device,
                )
            else:
                raise ValueError(
                    "Contact data not provided. Either pass contact_points_link_frame and contact_link_indices, "
                    "or pass contact_candidates_* to constructor for sampling."
                )
        elif contact_points_link_frame is not None and contact_link_indices is not None:
            contact_points_world = self._transform_contact_points(var, contact_points_link_frame, contact_link_indices)
        else:
            raise ValueError(
                "Contact data not provided. Either pass contact_points_link_frame and contact_link_indices, "
                "or pass contact_candidates_* to constructor for sampling."
            )

        batch_size = var.batch_size
        kernel_device = residual_buffer.device if residual_buffer is not None else self._device

        if residual_buffer is None:
            residual_buffer = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        signed_dists = wp.empty((batch_size, self.num_contact_points), dtype=wp.float32, device=kernel_device)

        if self._multi_scene_mode:
            wp.launch(
                kernel=query_sdf_for_contact_points_multi_scene_kernel,
                dim=(batch_size, self.num_contact_points),
                inputs=[
                    contact_points_world,
                    self.scene_indices,
                    self._mesh_ids,
                    self.mesh_first_idx,
                    self._inv_mesh_poses,
                    self._enable_inv_mesh_poses,
                    self._mesh_scales,
                    self._enable_mesh_scales,
                    self.max_dist,
                    signed_dists,
                ],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=query_sdf_for_contact_points_single_scene_kernel,
                dim=(batch_size, self.num_contact_points),
                inputs=[
                    contact_points_world,
                    self._mesh_ids,
                    self._inv_mesh_poses,
                    self._enable_inv_mesh_poses,
                    self._mesh_scales,
                    self._enable_mesh_scales,
                    self.max_dist,
                    signed_dists,
                ],
                device=kernel_device,
            )

        use_sqrt = self.residual_mode == "sqrt_abs"
        wp.launch(
            kernel=compute_distance_energy_residual_kernel,
            dim=(batch_size, self.num_contact_points),
            inputs=[
                signed_dists,
                self.residual_weight,
                use_sqrt,
                self.residual_eps,
                row_offset,
                residual_buffer,
            ],
            device=kernel_device,
        )

        return residual_buffer

    def compute_weighted_jacobian_analytic(
        self,
        var: "WarpRobotState",
        contact_points_link_frame: Optional[wp.array] = None,
        contact_link_indices: Optional[wp.array] = None,
        jacobian_buffer: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        """Compute analytical Jacobian for distance task.

        Jacobian formula: dr_i/dq = sign(sdf_i) * n_i . (dp_i/dq)
        where:
            - sdf_i is the signed distance from contact point i to the mesh
            - n_i is the surface normal at the closest point (pointing outward from mesh)
            - dp_i/dq is the FK Jacobian for contact point position

        Args:
            var: Robot state containing FK data.
            contact_points_link_frame: Contact points in link frame [batch_size, num_contact_points] of wp.vec3.
                If None, uses sampled points from candidates (must have been sampled by compute_weighted_residual).
            contact_link_indices: Link indices for each contact point [batch_size, num_contact_points] of int32.
                If None, uses sampled points from candidates (must have been sampled by compute_weighted_residual).
            jacobian_buffer: Optional buffer to write Jacobian into [batch_size, residual_dim, total_dofs].
            row_offset: Row offset in the Jacobian buffer.

        Returns:
            Weighted Jacobian [batch_size, num_contact_points, total_dofs].
        """
        if self._device is None:
            self._init_buffers(var.q.device)

        if not var.is_fk_computed:
            var = self.robot.forward_kinematics(var)
        if not var.is_motion_subspace_computed:
            var = self.robot.compute_motion_subspace(var)

        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        device = var.q.device

        if contact_points_link_frame is None and contact_link_indices is None:
            if self.fixed_contact_points_link_frame is not None and self.fixed_contact_link_indices is not None:
                contact_points_world = self._transform_contact_points(
                    var,
                    self.fixed_contact_points_link_frame,
                    self.fixed_contact_link_indices,
                )
                contact_link_indices = self.fixed_contact_link_indices
            elif self.shared_contact_point_indices is not None:
                if self.shared_contact_link_indices is None:
                    raise ValueError("shared_contact_link_indices must be provided with shared_contact_point_indices.")
                contact_points_world = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
                wp.launch(
                    kernel=_transform_contact_points_from_indices_kernel,
                    dim=(batch_size, self.num_contact_points),
                    inputs=[
                        var.T_world_link.xyz_wxyz.reshape((batch_size, num_links)),
                        self.contact_candidates_link_frame,
                        self.contact_candidates_link_indices,
                        self.shared_contact_point_indices,
                    ],
                    outputs=[contact_points_world],
                    device=device,
                )
                contact_link_indices = self.shared_contact_link_indices
            elif self._use_candidates:
                contact_points_world = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
                wp.launch(
                    kernel=_transform_contact_points_from_indices_kernel,
                    dim=(batch_size, self.num_contact_points),
                    inputs=[
                        var.T_world_link.xyz_wxyz.reshape((batch_size, num_links)),
                        self.contact_candidates_link_frame,
                        self.contact_candidates_link_indices,
                        self._sampled_contact_point_indices,
                    ],
                    outputs=[contact_points_world],
                    device=device,
                )
                contact_link_indices = self._sampled_contact_link_indices
            else:
                raise ValueError(
                    "Contact data not provided. Either pass contact_points_link_frame and contact_link_indices, "
                    "or pass contact_candidates_* to constructor for sampling."
                )
        elif contact_points_link_frame is not None and contact_link_indices is not None:
            contact_points_world = self._transform_contact_points(var, contact_points_link_frame, contact_link_indices)
        else:
            raise ValueError(
                "Contact data not provided. Either pass contact_points_link_frame and contact_link_indices, "
                "or pass contact_candidates_* to constructor for sampling."
            )

        total_dofs = var.tangent_dim
        kernel_device = jacobian_buffer.device if jacobian_buffer is not None else self._device

        if jacobian_buffer is None:
            jacobian_buffer = wp.zeros(
                (batch_size, self.residual_dim, total_dofs),
                dtype=wp.float32,
                device=kernel_device,
            )
            row_offset = 0

        signed_dists = wp.empty((batch_size, self.num_contact_points), dtype=wp.float32, device=kernel_device)
        normals = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=kernel_device)

        if self._multi_scene_mode:
            wp.launch(
                kernel=query_sdf_and_normal_for_contact_points_multi_scene_kernel,
                dim=(batch_size, self.num_contact_points),
                inputs=[
                    contact_points_world,
                    self.scene_indices,
                    self._mesh_ids,
                    self.mesh_first_idx,
                    self._inv_mesh_poses,
                    self._enable_inv_mesh_poses,
                    self._mesh_scales,
                    self._enable_mesh_scales,
                    self.max_dist,
                    signed_dists,
                    normals,
                ],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=query_sdf_and_normal_for_contact_points_single_scene_kernel,
                dim=(batch_size, self.num_contact_points),
                inputs=[
                    contact_points_world,
                    self._mesh_ids,
                    self._inv_mesh_poses,
                    self._enable_inv_mesh_poses,
                    self._mesh_scales,
                    self._enable_mesh_scales,
                    self.max_dist,
                    signed_dists,
                    normals,
                ],
                device=kernel_device,
            )

        spec_tensors = var.spec_tensors

        use_sqrt = self.residual_mode == "sqrt_abs"
        wp.launch(
            kernel=compute_distance_task_weighted_jacobian_kernel,
            dim=(batch_size, self.num_contact_points, total_dofs),
            inputs=[
                var.S_world,
                var.T_world_base.xyz_wxyz.flatten(),
                contact_points_world,
                contact_link_indices,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                signed_dists,
                normals,
                var.has_floating_base,
                self.residual_weight,
                use_sqrt,
                self.residual_eps,
                row_offset,
                jacobian_buffer,
            ],
            device=kernel_device,
        )

        return jacobian_buffer


__all__ = ["WarpDistanceTask"]
