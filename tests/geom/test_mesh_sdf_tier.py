# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false
"""Tests for the mesh geom's baked SDF mid-tier (`MeshGeom(enable_sdf=True)`).

The cascade per element: outside the padded volume AABB an analytical lower bound answers;
inside, the O(1) volume sample answers; within `REFINE_BAND_VOXELS * voxel_size` of the
surface the exact mesh BVH answers. The mixed-scene fixture (2 tiered spheres + 3 boxes)
exercises the union so kernels can't pollute each other across attached geometries. SdfVolume
baking requires CUDA; the whole module is CUDA-only.
"""

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from robokit.geom import REFINE_BAND_VOXELS, BoxGeom, MeshGeom, VolumeGeom, WarpScene
from robokit.geom import sdf_volume as sdf_volume_module
from robokit.geom.sdf_volume import mesh_to_sdf_volume


# single-value pytestmark so robokit's conftest skips collection on the test-warp-only CI
# variant where torch isn't installed (it string-matches "pytestmark = pytest.mark.torch")
pytestmark = pytest.mark.torch
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SDF volumes")

DEVICE = "cuda:0"
VOXEL = 0.003
BAND = REFINE_BAND_VOXELS * VOXEL
PADDING = 0.1


def _sphere(radius: float, center: tuple, subdivisions: int = 4) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    m.apply_translation(center)
    return m


@pytest.fixture(autouse=True)
def _warp_init(tmp_path, monkeypatch):
    wp.init()
    monkeypatch.setattr(sdf_volume_module, "SDF_CACHE_DIR", tmp_path / "sdf_cache")


def _fixture_meshes() -> tuple:
    tier_a = _sphere(0.10, (0.0, 0.0, 0.0))
    tier_b = _sphere(0.15, (1.0, 0.0, 0.0))
    mesh_c = _sphere(0.08, (0.0, 1.0, 0.0))
    mesh_d = _sphere(0.12, (1.0, 1.0, 0.0))
    box_he = np.array([[0.10, 0.10, 0.10]] * 3, dtype=np.float32)
    box_centers = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    return tier_a, tier_b, mesh_c, mesh_d, box_he, box_centers


def _box_prims(scene: WarpScene, box_he: np.ndarray, box_centers: np.ndarray) -> WarpScene:
    box_poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    box_poses[:, :3, 3] = box_centers
    return scene.add(
        BoxGeom(
            box_he,
            np.array([0, 3], dtype=np.int32),
            poses=wp.from_numpy(box_poses, dtype=wp.mat44, device=DEVICE),
        )
    )


def _mixed_scene() -> tuple:
    """The canonical 5-object union: 2 SDF-tiered spheres in the mesh geom + 3 posed boxes in
    a box geom. Returns (scene, tier_meshes, box_he, box_centers)."""
    tier_a, tier_b, _, _, box_he, box_centers = _fixture_meshes()
    scene = WarpScene(num_scenes=1, device=DEVICE).add(
        MeshGeom(
            [tier_a, tier_b],
            np.array([0, 2], dtype=np.int32),
            enable_sdf=True,
            sdf_voxel_size=VOXEL,
            sdf_padding=PADDING,
        )
    )
    scene = _box_prims(scene, box_he, box_centers)
    return scene, [tier_a, tier_b], box_he, box_centers


def _gt_signed_dist(p: np.ndarray, sphere_meshes, box_he, box_centers) -> float:
    """Analytic min signed distance over all 7 shapes for a single point."""
    dists = []
    for sphere_mesh in sphere_meshes:
        center = sphere_mesh.vertices.mean(axis=0)  # icosphere center == centroid
        radius = np.linalg.norm(sphere_mesh.vertices - center, axis=1).mean()
        dists.append(float(np.linalg.norm(p - center) - radius))
    for he, c in zip(box_he, box_centers):
        # axis-aligned box, exterior euclidean / interior min-axis penetration
        d = np.abs(p - c) - he
        outside = np.linalg.norm(np.maximum(d, 0.0))
        inside = min(0.0, float(np.max(d)))
        dists.append(outside + inside)
    return min(dists)


def _query(scene: WarpScene, points: np.ndarray) -> np.ndarray:
    qp_wp = wp.from_numpy(points.astype(np.float32), dtype=wp.vec3, device=DEVICE)
    scene_offsets = wp.from_numpy(np.array([0, points.shape[0]], dtype=np.int32), dtype=wp.int32, device=DEVICE)
    sdf, _, _ = scene.query_sdf(qp_wp, scene_offsets)
    return sdf.numpy()


@_REQUIRES_CUDA
def test_near_surface_matches_exact_mesh():
    """Within the refine band the cascade answers with the mesh BVH - bit-comparable to a
    plain MeshGeom scene (both inside/penetrating and outside points)."""
    tier_a, tier_b, mesh_c, mesh_d, box_he, box_centers = _fixture_meshes()
    tiered = WarpScene(1, DEVICE).add(
        MeshGeom(
            [tier_a, tier_b], np.array([0, 2], np.int32), enable_sdf=True, sdf_voxel_size=VOXEL, sdf_padding=PADDING
        )
    )
    exact = WarpScene(1, DEVICE).add(MeshGeom([tier_a, tier_b], np.array([0, 2], np.int32)))
    eps = 0.4 * BAND
    pts = np.array(
        [
            [0.10 - eps, 0.0, 0.0],  # just inside sphere A
            [0.10 + eps, 0.0, 0.0],  # just outside sphere A
            [1.0 + 0.15 - eps, 0.0, 0.0],  # just inside sphere B
            [1.0, 0.15 + eps, 0.0],  # just outside sphere B
            [0.03, 0.0, 0.0],  # deep inside sphere A: penetration always refines to the mesh
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(_query(tiered, pts), _query(exact, pts), atol=1e-6)


@_REQUIRES_CUDA
def test_far_field_matches_sdf_volumes_reference():
    """Beyond the refine band (but inside the volume AABB) the cascade answers with the same
    NanoVDB sample a VolumeGeom scene of identically-baked volumes returns."""
    tier_a, tier_b, _, _, _, _ = _fixture_meshes()
    tiered = WarpScene(1, DEVICE).add(
        MeshGeom(
            [tier_a, tier_b], np.array([0, 2], np.int32), enable_sdf=True, sdf_voxel_size=VOXEL, sdf_padding=PADDING
        )
    )
    vols = [mesh_to_sdf_volume(m, voxel_size=VOXEL, padding=PADDING, device=DEVICE) for m in (tier_a, tier_b)]
    ref = WarpScene(1, DEVICE).add(VolumeGeom(vols, np.array([0, 2], np.int32)))
    # NOTE: only exterior points beyond the band read the volume - any penetrating point has
    # vol_dist <= refine_band, so interior points always refine to the exact mesh instead.
    pts = np.array(
        [
            [0.15, 0.0, 0.0],  # 0.05 clear of sphere A, inside its AABB
            [0.0, 0.17, 0.0],
            [1.0, 0.0, 0.19],  # 0.04 clear of sphere B
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(_query(tiered, pts), _query(ref, pts), atol=1e-6)


@_REQUIRES_CUDA
def test_mixed_scene_conservative_never_overestimates():
    """The out-of-AABB fallback is a CONSERVATIVE under-estimate; mesh BVH and primitives are
    exact - so the min-reduce over the whole union never over-estimates the analytic truth."""
    rng = np.random.default_rng(0)
    pts = [(0.10 + 0.02, 0.0, 0.0), (0.10 - 0.03, 0.0, 0.0), (1.0, 0.15 + 0.01, 0.0)]
    pts += [(0.5, 0.5, 0.5), (3.0, 0.0, 0.0)]
    pts += [(0.0, 0.0, 0.95), (0.05, 0.05, 1.05), (1.0, 0.0, 0.85)]
    pts += [tuple(rng.uniform(-0.3, 1.3, 3)) for _ in range(40)]
    pts = np.array(pts, dtype=np.float32)

    scene, tier, box_he, box_centers = _mixed_scene()
    got = _query(scene, pts)
    expect = np.array([_gt_signed_dist(p, tier, box_he, box_centers) for p in pts])
    over = np.where(got > expect + 3 * VOXEL)[0]
    assert len(over) == 0, f"over-estimates at indices {over}: got={got[over]} expect={expect[over]}"


@_REQUIRES_CUDA
def test_isolation_no_cross_object_pollution():
    """Points whose nearest surface is a box (exact primitive) or a tiered sphere inside the
    refine band (exact mesh): the tiered elements' far-field readings must not corrupt them."""
    pts = np.array(
        [
            [0.0, 0.0, 1.0 - 0.05],  # near box E top
            [1.0, 0.0, 1.0 + 0.20],  # well above box F
            [0.10 + 0.005, 0.0, 0.0],  # in sphere A's refine band -> exact mesh
            [1.0, 0.0, 0.15 - 0.005],  # in sphere B's refine band (inside)
        ],
        dtype=np.float32,
    )
    scene, tier, box_he, box_centers = _mixed_scene()
    got = _query(scene, pts)
    expect = np.array([_gt_signed_dist(p, tier, box_he, box_centers) for p in pts])
    # 1e-4 absorbs icosphere-triangulation roundoff in the mesh BVH.
    np.testing.assert_allclose(got, expect, atol=1e-4)


@_REQUIRES_CUDA
def test_geometry_kind_and_band():
    """A `MeshGeom` without `enable_sdf` has `refine_band == 0.0`; with `enable_sdf` it
    reports `enable_sdf == True` and the auto refine band."""
    tier_a, _, mesh_c, _, _, _ = _fixture_meshes()
    plain = WarpScene(1, DEVICE).add(MeshGeom([mesh_c], np.array([0, 1], np.int32)))
    tiered = WarpScene(1, DEVICE).add(
        MeshGeom([tier_a], np.array([0, 1], np.int32), enable_sdf=True, sdf_voxel_size=VOXEL)
    )
    plain_geom, tiered_geom = plain.geoms[0], tiered.geoms[0]
    assert isinstance(plain_geom, MeshGeom) and not plain_geom.enable_sdf and plain_geom.refine_band == 0.0
    assert isinstance(tiered_geom, MeshGeom) and tiered_geom.enable_sdf
    assert tiered_geom.refine_band == pytest.approx(BAND)
