import numpy as np
import pytest
import warp as wp

from robokit.utils.warp_utils import gather, repeat, stack, tile


wp.init()
DEVICE = "cpu"


def test_repeat_1d():
    data = [1.0, 2.0, 3.0]
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    repeated = repeat(source, 2)
    assert repeated.shape == (6,)
    assert repeated.dtype == wp.float32
    assert np.allclose(repeated.numpy(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])


def test_repeat_2d():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    repeated = repeat(source, 2)
    assert repeated.shape == (4, 2)
    assert np.allclose(repeated.numpy(), np.repeat(data, 2, axis=0))


def test_repeat_3d():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    repeated = repeat(source, 3)
    assert repeated.shape == (6, 3, 4)
    assert np.allclose(repeated.numpy(), np.repeat(data, 3, axis=0))


def test_repeat_4d():
    data = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    repeated = repeat(source, 2)
    assert repeated.shape == (4, 3, 4, 5)
    assert np.allclose(repeated.numpy(), np.repeat(data, 2, axis=0))


def test_tile_1d():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    tiled = tile(source, 3)
    assert tiled.shape == (9,)
    assert np.allclose(tiled.numpy(), np.tile(data, 3))


def test_tile_2d():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    tiled = tile(source, 4)
    assert tiled.shape == (12, 2)
    assert np.allclose(tiled.numpy(), np.tile(data, (4, 1)))


def test_tile_3d():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    tiled = tile(source, 3)
    assert tiled.shape == (6, 3, 4)
    assert np.allclose(tiled.numpy(), np.tile(data, (3, 1, 1)))


def test_tile_4d():
    data = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    tiled = tile(source, 2)
    assert tiled.shape == (4, 3, 4, 5)
    assert np.allclose(tiled.numpy(), np.tile(data, (2, 1, 1, 1)))


def test_gather_1d():
    source = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=wp.float32, device=DEVICE)
    indices = wp.array([3, 1, 4], dtype=wp.int32, device=DEVICE)
    gathered = gather(source, indices)
    assert gathered.shape == (3,)
    assert np.allclose(gathered.numpy(), [4.0, 2.0, 5.0])


def test_gather_2d():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    indices = wp.array([2, 0], dtype=wp.int32, device=DEVICE)
    gathered = gather(source, indices)
    assert gathered.shape == (2, 2)
    assert np.allclose(gathered.numpy(), data[[2, 0]])


def test_gather_3d():
    data = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    indices = wp.array([1, 2], dtype=wp.int32, device=DEVICE)
    gathered = gather(source, indices)
    assert gathered.shape == (2, 2, 4)
    assert np.allclose(gathered.numpy(), data[[1, 2]])


def test_gather_4d():
    data = np.arange(120, dtype=np.float32).reshape(3, 2, 4, 5)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    indices = wp.array([2, 0, 1], dtype=wp.int32, device=DEVICE)
    gathered = gather(source, indices)
    assert gathered.shape == (3, 2, 4, 5)
    assert np.allclose(gathered.numpy(), data[[2, 0, 1]])


def test_gather_dest_preallocated():
    data = np.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]], dtype=np.float32)
    source = wp.array(data, dtype=wp.float32, device=DEVICE)
    indices = wp.array([0, 2], dtype=wp.int32, device=DEVICE)
    dest = wp.empty((2, 2), dtype=wp.float32, device=DEVICE)  # type: ignore
    gathered = gather(source, indices, dest=dest)
    assert gathered is dest
    assert np.allclose(gathered.numpy(), data[[0, 2]])


def test_stack_1d_axis0():
    arrays_np = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    arrays_wp = [wp.array(a, dtype=wp.float32, device=DEVICE) for a in arrays_np]
    result = stack(arrays_wp, axis=0)
    assert result.shape == (2, 3)
    assert np.allclose(result.numpy(), np.stack(arrays_np, axis=0))


def test_stack_1d_axis1():
    arrays_np = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    arrays_wp = [wp.array(a, dtype=wp.float32, device=DEVICE) for a in arrays_np]
    result = stack(arrays_wp, axis=1)
    assert result.shape == (3, 2)
    assert np.allclose(result.numpy(), np.stack(arrays_np, axis=1))


def test_stack_2d_axis0():
    arrays_np = [np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(6, 12, dtype=np.float32).reshape(2, 3)]
    arrays_wp = [wp.array(a, dtype=wp.float32, device=DEVICE) for a in arrays_np]
    result = stack(arrays_wp, axis=0)
    assert result.shape == (2, 2, 3)
    assert np.allclose(result.numpy(), np.stack(arrays_np, axis=0))


def test_stack_2d_axis1():
    arrays_np = [np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(6, 12, dtype=np.float32).reshape(2, 3)]
    arrays_wp = [wp.array(a, dtype=wp.float32, device=DEVICE) for a in arrays_np]
    result = stack(arrays_wp, axis=1)
    assert result.shape == (2, 2, 3)
    assert np.allclose(result.numpy(), np.stack(arrays_np, axis=1))


def test_stack_2d_axis2():
    arrays_np = [np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(6, 12, dtype=np.float32).reshape(2, 3)]
    arrays_wp = [wp.array(a, dtype=wp.float32, device=DEVICE) for a in arrays_np]
    result = stack(arrays_wp, axis=2)
    assert result.shape == (2, 3, 2)
    assert np.allclose(result.numpy(), np.stack(arrays_np, axis=2))


def test_stack_3d_axis0():
    arrays_np = [np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.arange(24, 48, dtype=np.float32).reshape(2, 3, 4)]
    arrays_wp = [wp.array(a, dtype=wp.float32, device=DEVICE) for a in arrays_np]
    result = stack(arrays_wp, axis=0)
    assert result.shape == (2, 2, 3, 4)
    assert np.allclose(result.numpy(), np.stack(arrays_np, axis=0))


def test_stack_3d_axis_last():
    arrays_np = [np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.arange(24, 48, dtype=np.float32).reshape(2, 3, 4)]
    arrays_wp = [wp.array(a, dtype=wp.float32, device=DEVICE) for a in arrays_np]
    result = stack(arrays_wp, axis=3)
    assert result.shape == (2, 3, 4, 2)
    assert np.allclose(result.numpy(), np.stack(arrays_np, axis=3))


def test_stack_dest_preallocated():
    arrays_np = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    arrays_wp = [wp.array(a, dtype=wp.float32, device=DEVICE) for a in arrays_np]
    dest = wp.empty((2, 2), dtype=wp.float32, device=DEVICE)
    result = stack(arrays_wp, axis=1, dest=dest)
    assert result is dest
    assert np.allclose(result.numpy(), np.stack(arrays_np, axis=1))


if __name__ == "__main__":
    pytest.main([__file__])
