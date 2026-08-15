from typing import Optional, Sequence

import numpy as np
import warp as wp

from robokit.lie.se3 import se3_identity
from robokit.utils.warp_utils import tile, wp_vec7


# multi-seed layout: instance-major order (output_idx = instance_idx * num_seeds + seed_idx)
@wp.kernel
def sample_q_around_init_kernel(
    init_q: wp.array2d(dtype=wp.float32),
    offsets: wp.array2d(dtype=wp.float32),
    joint_limits: wp.array2d(dtype=wp.float32),
    sample_range: wp.float32,
    num_seeds: wp.int32,
    joint_mask: wp.array1d(dtype=wp.float32),
    out_q: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    instance_idx = tid // int(num_seeds)
    seed_idx = tid % int(num_seeds)
    num_joints = init_q.shape[1]

    for j in range(num_joints):
        joint_lower = joint_limits[j, 0]
        joint_upper = joint_limits[j, 1]
        joint_range = joint_upper - joint_lower
        offset = offsets[seed_idx, j] * sample_range * joint_range * joint_mask[j]
        q_val = init_q[instance_idx, j] + offset
        out_q[tid, j] = wp.clamp(q_val, joint_lower, joint_upper)


@wp.kernel
def pin_inactive_q_kernel(
    inactive_q: wp.array2d(dtype=wp.float32),
    joint_mask: wp.array1d(dtype=wp.float32),
    num_seeds: wp.int32,
    out_q: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    row = tid // out_q.shape[1]
    joint = tid % out_q.shape[1]
    if joint_mask[joint] == 0.0:
        out_q[row, joint] = inactive_q[row // num_seeds, joint]


# multi-seed layout: instance-major order (output_idx = instance_idx * num_seeds + seed_idx)
@wp.kernel
def sample_base_translation_kernel(
    init_base: wp.array(dtype=wp_vec7),
    offsets: wp.array2d(dtype=wp.float32),
    sample_range: wp.float32,
    num_seeds: wp.int32,
    config_mask: wp.array1d(dtype=wp.float32),
    active_mask: wp.array1d(dtype=wp.float32),
    out_base: wp.array(dtype=wp_vec7),
):
    """Sample base translation while preserving orientation and respecting both masks."""
    tid = wp.tid()
    instance_idx = tid // int(num_seeds)
    seed_idx = tid % int(num_seeds)

    base = init_base[instance_idx]
    out_base[tid] = wp_vec7(
        base[0] + offsets[seed_idx, 0] * sample_range * config_mask[0] * active_mask[0],
        base[1] + offsets[seed_idx, 1] * sample_range * config_mask[1] * active_mask[1],
        base[2] + offsets[seed_idx, 2] * sample_range * config_mask[2] * active_mask[2],
        base[3],
        base[4],
        base[5],
        base[6],
    )


class Sampler:
    """Generate quasi-random joint and base seeds for multi-seed optimization."""

    @staticmethod
    def _roberts_samples(num_points: int, dim: int, offset: int) -> np.ndarray:
        """Generate raw Roberts samples in [0, 1)."""
        root = 1.0
        for _ in range(10000):
            f = root ** (dim + 1) - root - 1
            root = root - f / ((dim + 1) * root**dim - 1)
            if abs(f) < 1e-10:
                break
        basis = 1 - (1 / root ** (1 + np.arange(dim)))
        n = np.arange(num_points) + offset
        samples, _ = np.modf(n[:, None] * basis[None, :])
        return samples

    @staticmethod
    def _build_offsets(samples: np.ndarray, keep_first: bool, device) -> wp.array:
        """Center Roberts samples at zero and optionally preserve seed zero."""
        offsets = (samples - 0.5).astype(np.float32)
        if keep_first:
            offsets = np.vstack([np.zeros_like(offsets[:1]), offsets[:-1]])
        return wp.from_numpy(offsets, dtype=wp.float32, device=device)

    def __init__(
        self,
        num_seeds: int,
        num_joints: int,
        joint_limits: np.ndarray,
        base_translation_mask: Sequence[float],
        seed: Optional[int],
        keep_init_seed: bool,
        device,
    ):
        self.num_seeds = num_seeds
        self.device = device
        roberts_offset = (seed % 10000) if seed is not None else 0

        q_samples = self._roberts_samples(num_seeds, num_joints, roberts_offset)
        base_samples = self._roberts_samples(num_seeds, 3, roberts_offset)
        self.offsets_q = self._build_offsets(q_samples, keep_init_seed, device)
        self.offsets_base = self._build_offsets(base_samples, keep_init_seed, device)

        # global joint-space seeds spanning the full joint limits (no init point needed)
        global_q = (joint_limits[:, 0] + q_samples * (joint_limits[:, 1] - joint_limits[:, 0])).astype(np.float32)
        self.global_q_seeds = wp.from_numpy(global_q, dtype=wp.float32, device=device)

        self.joint_limits = wp.from_numpy(joint_limits.astype(np.float32), dtype=wp.float32, device=device)
        self.config_base_mask = wp.from_numpy(
            np.array(base_translation_mask, dtype=np.float32), dtype=wp.float32, device=device
        )
        self.ones_joint_mask = wp.ones(num_joints, dtype=wp.float32, device=device)
        self.ones_base_mask = wp.ones(3, dtype=wp.float32, device=device)

    def warmup(self, batch_size: int):
        """Build the batch-expanded buffers used by the no-init seeding path."""
        self.batch_size = batch_size
        self.global_q_seeds_expanded = tile(self.global_q_seeds, batch_size)
        self.base_default_expanded = se3_identity(shape=(batch_size * self.num_seeds,), device=self.device)

    def sample_q(
        self,
        init_q: Optional[wp.array],
        sample_range: float,
        out_q: wp.array,
        joint_mask: Optional[wp.array] = None,
        inactive_q: Optional[wp.array] = None,
    ):
        """Fill seeds around `init_q` or globally, then pin inactive joints to `inactive_q`."""
        if init_q is None:
            wp.copy(out_q, self.global_q_seeds_expanded)
        else:
            wp.launch(
                sample_q_around_init_kernel,
                dim=self.batch_size * self.num_seeds,
                inputs=[
                    init_q,
                    self.offsets_q,
                    self.joint_limits,
                    sample_range,
                    self.num_seeds,
                    joint_mask if joint_mask is not None else self.ones_joint_mask,
                    out_q,
                ],
                device=self.device,
            )
        if inactive_q is not None and joint_mask is not None:
            wp.launch(
                pin_inactive_q_kernel,
                dim=out_q.shape[0] * out_q.shape[1],
                inputs=[inactive_q, joint_mask, self.num_seeds, out_q],
                device=self.device,
            )

    def sample_base(
        self,
        init_base: Optional[wp.array],
        sample_range: float,
        out_base: wp.array,
        base_mask: Optional[wp.array] = None,
    ):
        """Fill base seeds around `init_base`, or use identity without an initial base."""
        out_array = out_base
        if init_base is None:
            wp.copy(out_array, self.base_default_expanded)
            return
        init_array = init_base
        wp.launch(
            sample_base_translation_kernel,
            dim=self.batch_size * self.num_seeds,
            inputs=[
                init_array,
                self.offsets_base,
                sample_range,
                self.num_seeds,
                self.config_base_mask,
                base_mask if base_mask is not None else self.ones_base_mask,
                out_array,
            ],
            device=self.device,
        )
