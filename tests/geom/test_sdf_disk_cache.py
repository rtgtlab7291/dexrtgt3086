# pyright: reportArgumentType=false
"""cached_sdf_volume disk cache + SdfVolume.save/load round-trip (CUDA-only)."""

import time

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from robokit.geom import VolumeGeom, WarpScene
from robokit.geom import sdf_volume as sdf_volume_module
from robokit.geom.sdf_volume import SdfVolume, cached_sdf_volume, mesh_to_sdf_volume


pytestmark = pytest.mark.torch

_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SDF volumes")


def _volume_sdf(vol: SdfVolume, points: np.ndarray) -> np.ndarray:
    scene = WarpScene(1, "cuda:0").add(VolumeGeom([vol], np.array([0, 1], np.int32)))
    points_wp = wp.from_numpy(points.astype(np.float32), dtype=wp.vec3, device="cuda:0")
    scene_offsets = wp.from_numpy(np.array([0, len(points)], np.int32), dtype=wp.int32, device="cuda:0")
    return scene.query_sdf(points_wp, scene_offsets, distance_only=True).numpy()


class TestSdfDiskCache:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        wp.init()
        self.cache_dir = tmp_path / "sdf_cache"
        monkeypatch.setattr(sdf_volume_module, "SDF_CACHE_DIR", self.cache_dir)

    @_REQUIRES_CUDA
    def test_bake_then_hit(self):
        """First call bakes .nvdb + .json into the cache dir; second call reads them back untouched."""
        box = trimesh.creation.box((1.0, 1.0, 1.0))
        vol1 = cached_sdf_volume(box, 0.02, padding=0.1, device="cuda:0")
        files = sorted(p.name for p in self.cache_dir.iterdir())
        assert len(files) == 2 and files[0].endswith(".json") and files[1].endswith(".nvdb")
        mtime = (self.cache_dir / files[1]).stat().st_mtime

        time.sleep(0.05)  # ensure mtime resolution is sufficient
        vol2 = cached_sdf_volume(box, 0.02, padding=0.1, device="cuda:0")
        assert (self.cache_dir / files[1]).stat().st_mtime == mtime, "cache file rebaked despite existing"
        np.testing.assert_allclose(vol2.aabb_min, vol1.aabb_min)
        np.testing.assert_allclose(vol2.aabb_max, vol1.aabb_max)
        assert vol2.padding == vol1.padding
        pts = np.array([[0.0, 0.0, 0.0], [0.55, 0.0, 0.0], [0.53, 0.1, -0.2]], np.float32)
        np.testing.assert_allclose(_volume_sdf(vol2, pts), _volume_sdf(vol1, pts), atol=1e-6)

    @_REQUIRES_CUDA
    def test_key_encodes_content_and_params(self, tmp_path):
        """Key = sha1(bytes) + bake params: same file content under a new name hits; new params miss."""
        mesh_path_a = str(tmp_path / "box_a.obj")
        trimesh.creation.box((1.0, 1.0, 1.0)).export(mesh_path_a)
        cached_sdf_volume(mesh_path_a, 0.02, padding=0.1, device="cuda:0")
        assert len(list(self.cache_dir.iterdir())) == 2

        mesh_path_b = str(tmp_path / "box_b.obj")
        trimesh.creation.box((1.0, 1.0, 1.0)).export(mesh_path_b)
        cached_sdf_volume(mesh_path_b, 0.02, padding=0.1, device="cuda:0")
        assert len(list(self.cache_dir.iterdir())) == 2, "identical bytes must reuse the cache entry"

        cached_sdf_volume(mesh_path_a, 0.04, padding=0.1, device="cuda:0")
        cached_sdf_volume(mesh_path_a, 0.02, padding=0.2, device="cuda:0")
        assert len(list(self.cache_dir.iterdir())) == 6, "voxel_size/padding must be part of the key"

    @_REQUIRES_CUDA
    def test_save_load_roundtrip(self, tmp_path):
        box = trimesh.creation.box((0.4, 0.4, 0.4))
        vol = mesh_to_sdf_volume(box, 0.01, padding=0.05, device="cuda:0")
        vol.save(str(tmp_path / "vol"))
        loaded = SdfVolume.load(str(tmp_path / "vol"), device="cuda:0")
        np.testing.assert_allclose(loaded.aabb_min, vol.aabb_min)
        np.testing.assert_allclose(loaded.aabb_max, vol.aabb_max)
        assert loaded.padding == vol.padding
        pts = np.array([[0.0, 0.0, 0.0], [0.24, 0.0, 0.0], [0.1, 0.1, 0.1]], np.float32)
        np.testing.assert_allclose(_volume_sdf(loaded, pts), _volume_sdf(vol, pts), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
