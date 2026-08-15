import numpy as np
import pytest
import torch
import warp as wp
from torch.testing import assert_close

from robokit.utils.sampling_utils import Sampler
from robokit.utils.warp_utils import wp_vec7


pytestmark = pytest.mark.torch

wp.init()
DEVICE = "cpu"

NUM_SEEDS = 8
NUM_JOINTS = 4
SEED = 42
BATCH = 2
LIMITS = np.array([[-1.0, 1.0], [-2.0, 0.0], [0.0, 3.14], [-0.5, 0.5]], dtype=np.float64)
BASE_MASK = [1.0, 1.0, 1.0]
INIT_Q = np.array([[0.0, 0.0, 1.0, 0.0], [0.1, -1.0, 2.0, 0.2]], dtype=np.float32)
INIT_BASE = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], [0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)


def _make_sampler(seed=SEED, keep_init_seed=True, base_mask=BASE_MASK, warmup=True) -> Sampler:
    s = Sampler(NUM_SEEDS, NUM_JOINTS, LIMITS, base_mask, seed, keep_init_seed, DEVICE)
    if warmup:
        s.warmup(BATCH)
    return s


def _ref_roberts_samples(num_points: int, dim: int, offset: int) -> np.ndarray:
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


def _ref_offsets(num_seeds: int, dim: int, roberts_offset: int, keep_first: bool) -> np.ndarray:
    samples = _ref_roberts_samples(num_seeds, dim, roberts_offset)
    if keep_first:
        return np.vstack([np.zeros((1, dim), dtype=np.float32), (samples[:-1] - 0.5).astype(np.float32)])
    return (samples - 0.5).astype(np.float32)


class TestSampler:
    def test_offsets_match_reference(self):
        s = _make_sampler()
        roberts_offset = SEED % 10000
        exp_q = _ref_offsets(NUM_SEEDS, NUM_JOINTS, roberts_offset, True)
        exp_base = _ref_offsets(NUM_SEEDS, 3, roberts_offset, True)
        assert_close(torch.from_numpy(s.offsets_q.numpy()), torch.from_numpy(exp_q))
        assert_close(torch.from_numpy(s.offsets_base.numpy()), torch.from_numpy(exp_base))
        # keep_init_seed -> seed 0 row is all zeros (preserves init exactly)
        assert np.all(s.offsets_q.numpy()[0] == 0.0)
        assert np.all(s.offsets_base.numpy()[0] == 0.0)

    def test_offsets_no_keep_first(self):
        s = _make_sampler(keep_init_seed=False, warmup=False)
        exp = _ref_offsets(NUM_SEEDS, NUM_JOINTS, SEED % 10000, False)
        assert_close(torch.from_numpy(s.offsets_q.numpy()), torch.from_numpy(exp))
        assert not np.all(s.offsets_q.numpy()[0] == 0.0)

    def test_global_q_seeds(self):
        s = _make_sampler()
        samples = _ref_roberts_samples(NUM_SEEDS, NUM_JOINTS, SEED % 10000)
        exp = (LIMITS[:, 0] + samples * (LIMITS[:, 1] - LIMITS[:, 0])).astype(np.float32)
        assert_close(torch.from_numpy(s.global_q_seeds.numpy()), torch.from_numpy(exp))
        g = s.global_q_seeds.numpy()
        assert np.all(g >= LIMITS[:, 0].astype(np.float32) - 1e-5)
        assert np.all(g <= LIMITS[:, 1].astype(np.float32) + 1e-5)

    def test_sample_q_jitter(self):
        s = _make_sampler()
        init_wp = wp.from_numpy(INIT_Q, dtype=wp.float32, device=DEVICE)
        out = wp.zeros((BATCH * NUM_SEEDS, NUM_JOINTS), dtype=wp.float32, device=DEVICE)
        s.sample_q(init_wp, 0.1, out)

        offsets = s.offsets_q.numpy()
        ranges = (LIMITS[:, 1] - LIMITS[:, 0]).astype(np.float32)
        ref = np.zeros((BATCH * NUM_SEEDS, NUM_JOINTS), dtype=np.float32)
        for inst in range(BATCH):
            for seed in range(NUM_SEEDS):
                q = INIT_Q[inst] + offsets[seed] * np.float32(0.1) * ranges
                ref[inst * NUM_SEEDS + seed] = np.clip(q, LIMITS[:, 0], LIMITS[:, 1])
        assert_close(torch.from_numpy(out.numpy()), torch.from_numpy(ref), atol=1e-6, rtol=1e-6)
        # seed 0 preserves init exactly (within limits)
        for inst in range(BATCH):
            assert_close(torch.from_numpy(out.numpy()[inst * NUM_SEEDS]), torch.from_numpy(INIT_Q[inst]))

    def test_sample_q_no_init(self):
        s = _make_sampler()
        out = wp.zeros((BATCH * NUM_SEEDS, NUM_JOINTS), dtype=wp.float32, device=DEVICE)
        s.sample_q(None, 0.1, out)
        exp = np.tile(s.global_q_seeds.numpy(), (BATCH, 1))
        assert_close(torch.from_numpy(out.numpy()), torch.from_numpy(exp))

    def test_sample_q_joint_mask(self):
        s = _make_sampler()
        init_wp = wp.from_numpy(INIT_Q, dtype=wp.float32, device=DEVICE)
        out = wp.zeros((BATCH * NUM_SEEDS, NUM_JOINTS), dtype=wp.float32, device=DEVICE)
        mask = wp.from_numpy(np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32), dtype=wp.float32, device=DEVICE)
        s.sample_q(init_wp, 0.1, out, joint_mask=mask)
        out_np = out.numpy()
        # joint 1 is masked off -> unchanged from init for every seed
        for inst in range(BATCH):
            for seed in range(NUM_SEEDS):
                assert out_np[inst * NUM_SEEDS + seed, 1] == np.float32(INIT_Q[inst, 1])

        inactive_q = INIT_Q.copy()
        inactive_q[:, 1] -= 0.25
        s.sample_q(
            init_wp,
            0.1,
            out,
            joint_mask=mask,
            inactive_q=wp.from_numpy(inactive_q, dtype=wp.float32, device=DEVICE),
        )
        expected = np.broadcast_to(inactive_q[:, None, 1], (BATCH, NUM_SEEDS))
        np.testing.assert_array_equal(out.numpy().reshape(BATCH, NUM_SEEDS, NUM_JOINTS)[:, :, 1], expected)

    def test_sample_base_jitter(self):
        s = _make_sampler()
        init_wp = wp.from_numpy(INIT_BASE, dtype=wp_vec7, device=DEVICE)
        out = wp.zeros(BATCH * NUM_SEEDS, dtype=wp_vec7, device=DEVICE)
        base_mask = wp.from_numpy(np.array([1.0, 1.0, 0.0], dtype=np.float32), dtype=wp.float32, device=DEVICE)
        s.sample_base(init_wp, 0.2, out, base_mask=base_mask)

        out_np = out.numpy()
        offsets = s.offsets_base.numpy()
        config = np.array(BASE_MASK, dtype=np.float32)
        active = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        for inst in range(BATCH):
            for seed in range(NUM_SEEDS):
                row = out_np[inst * NUM_SEEDS + seed]
                for ax in range(3):
                    exp = INIT_BASE[inst, ax] + offsets[seed, ax] * np.float32(0.2) * config[ax] * active[ax]
                    assert abs(row[ax] - exp) < 1e-6
                # z axis is locked (active=0) -> unchanged
                assert row[2] == np.float32(INIT_BASE[inst, 2])
                # orientation preserved
                assert_close(torch.from_numpy(row[3:]), torch.from_numpy(INIT_BASE[inst, 3:]))

    def test_sample_base_no_init(self):
        s = _make_sampler()
        out = wp.zeros(BATCH * NUM_SEEDS, dtype=wp_vec7, device=DEVICE)
        s.sample_base(None, 0.2, out)
        identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        exp = np.tile(identity, (BATCH * NUM_SEEDS, 1))
        assert_close(torch.from_numpy(out.numpy()), torch.from_numpy(exp))

    def test_determinism(self):
        same = _make_sampler(seed=42, warmup=False)
        same2 = _make_sampler(seed=42, warmup=False)
        diff = _make_sampler(seed=7, warmup=False)
        assert np.array_equal(same.offsets_q.numpy(), same2.offsets_q.numpy())
        assert np.array_equal(same.global_q_seeds.numpy(), same2.global_q_seeds.numpy())
        assert not np.array_equal(same.offsets_q.numpy(), diff.offsets_q.numpy())
