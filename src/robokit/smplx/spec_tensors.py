"""Lazy device-side tensors derived from a `BodyModelSpec`."""

# pyright: reportArgumentType=false
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import warp as wp

from robokit.smplx.spec import BodyModelSpec


@dataclass
class BodyModelSpecTensors:
    """Lazy device-side view of a `BodyModelSpec`."""

    spec: BodyModelSpec
    device: str

    @cached_property
    def parent_joint_indices(self) -> wp.array:
        """`[J]` int32, root = -1."""
        return wp.from_numpy(self.spec.parents.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def J_template(self) -> wp.array:
        """`J_regressor @ v_template` - `[J]` vec3 rest joints at zero shape."""
        J_np = self.spec.J_regressor.astype(np.float32) @ self.spec.v_template.astype(np.float32)
        return wp.from_numpy(J_np, device=self.device, dtype=wp.vec3)

    @cached_property
    def J_shapedirs(self) -> wp.array:
        """`J_regressor @ shapedirs` - `[J, num_betas]` vec3 rest-joint deltas per beta."""
        J_reg = self.spec.J_regressor.astype(np.float32)
        sd = self.spec.shapedirs.astype(np.float32)
        js = np.einsum("jv,vkl->jlk", J_reg, sd)
        return wp.from_numpy(np.ascontiguousarray(js), device=self.device, dtype=wp.vec3)

    @cached_property
    def v_template(self) -> wp.array:
        """`[V]` vec3 rest-pose vertices."""
        return wp.from_numpy(self.spec.v_template.astype(np.float32), device=self.device, dtype=wp.vec3)

    @cached_property
    def shapedirs_v3(self) -> wp.array:
        """`[V, num_betas]` vec3 shape-PCA basis (transposed from `(V, 3, num_betas)`)."""
        sd = self.spec.shapedirs.astype(np.float32)
        sd_v3 = np.ascontiguousarray(sd.transpose(0, 2, 1))
        return wp.from_numpy(sd_v3, device=self.device, dtype=wp.vec3)

    @cached_property
    def posedirs_v3(self) -> wp.array:
        """`[P, V]` vec3 pose-PCA basis (zero-copy view of `(P, V*3)`)."""
        P = self.spec.posedirs.shape[0]
        V = self.spec.v_template.shape[0]
        pd = np.ascontiguousarray(self.spec.posedirs.astype(np.float32).reshape(P, V, 3))
        return wp.from_numpy(pd, device=self.device, dtype=wp.vec3)

    @cached_property
    def lbs_weights(self) -> wp.array:
        """`[V, J]` float32 blend weights."""
        return wp.from_numpy(self.spec.lbs_weights.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def static_landmark_vertex_ids(self) -> wp.array:
        """`[L]` int32 landmark vertex indices."""
        assert self.spec.static_landmark_vertex_ids is not None, (
            f"Spec {self.spec.name!r} has no static_landmark_vertex_ids."
        )
        return wp.from_numpy(
            self.spec.static_landmark_vertex_ids.astype(np.int32),
            device=self.device,
            dtype=wp.int32,
        )

    # --- convenience scalars ---
    @property
    def num_joints(self) -> int:
        return int(self.spec.J_regressor.shape[0])

    @property
    def num_betas(self) -> int:
        return int(self.spec.num_betas)

    @property
    def num_vertices(self) -> int:
        return int(self.spec.v_template.shape[0])

    @property
    def num_pose_dirs(self) -> int:
        return int(self.spec.posedirs.shape[0])

    @property
    def num_static_landmarks(self) -> int:
        if self.spec.static_landmark_vertex_ids is None:
            return 0
        return int(self.spec.static_landmark_vertex_ids.shape[0])
