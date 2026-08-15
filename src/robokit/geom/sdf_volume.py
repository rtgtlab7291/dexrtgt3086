# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
"""Mesh -> SDF volume (NanoVDB) via Warp; `cached_sdf_volume` adds a disk cache."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import trimesh
import warp as wp


SDF_CACHE_DIR = Path.home() / ".robokit" / "sdf_cache"


@dataclass
class SdfVolume:
    """Baked SDF grid + its padded AABB; outside the AABB, `aabb_dist + padding` never overestimates."""

    volume: wp.Volume
    aabb_min: np.ndarray  # float32 (3,) local-frame min
    aabb_max: np.ndarray  # float32 (3,) local-frame max
    padding: float = 0.0  # bake-time padding between mesh bbox and AABB walls

    def save(self, path_stem: str) -> None:
        """Save as `<path_stem>.nvdb` + `<path_stem>.json` (written last, marks a complete pair)."""
        self.volume.save_to_nvdb(f"{path_stem}.nvdb", codec="zip")
        meta = {
            "aabb_min": np.asarray(self.aabb_min, dtype=np.float32).tolist(),
            "aabb_max": np.asarray(self.aabb_max, dtype=np.float32).tolist(),
            "padding": float(self.padding),
        }
        Path(f"{path_stem}.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path_stem: str, device: str = "cuda:0") -> "SdfVolume":
        meta = json.loads(Path(f"{path_stem}.json").read_text())
        with open(f"{path_stem}.nvdb", "rb") as f:
            volume = wp.Volume.load_from_nvdb(f, device=device)
        return cls(
            volume=volume,
            aabb_min=np.asarray(meta["aabb_min"], dtype=np.float32),
            aabb_max=np.asarray(meta["aabb_max"], dtype=np.float32),
            padding=float(meta["padding"]),
        )


@wp.kernel
def _sdf_grid_kernel(
    mesh_id: wp.uint64,
    min_world: wp.vec3,
    voxel_size: float,
    max_dist: float,
    sdf: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    pos = min_world + wp.vec3(float(i), float(j), float(k)) * voxel_size
    query = wp.mesh_query_point(mesh_id, pos, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        sdf[i, j, k] = wp.length(closest - pos) * query.sign
    else:
        sdf[i, j, k] = max_dist


def mesh_to_sdf_volume(
    mesh: Union[str, trimesh.Trimesh, wp.Mesh],
    voxel_size: float,
    padding: float = 0.1,
    max_dist: float = 1e6,
    device: str = "cuda:0",
) -> SdfVolume:
    """Bake a mesh (path / trimesh / `wp.Mesh`) into an `SdfVolume`; `device` is ignored for a `wp.Mesh`."""
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh, process=False, force="mesh")  # pyright: ignore[reportAssignmentType]
    if isinstance(mesh, wp.Mesh):
        device_wp = mesh.points.device
        points_np = mesh.points.numpy().astype(np.float32)
        bbox_min = points_np.min(axis=0)
        bbox_max = points_np.max(axis=0)
    else:
        device_wp = wp.get_device(device)
        bbox_min = mesh.bounds[0].astype(np.float32)
        bbox_max = mesh.bounds[1].astype(np.float32)
        mesh = wp.Mesh(
            points=wp.array(mesh.vertices.view(np.ndarray), dtype=wp.vec3, device=device_wp),
            indices=wp.array(np.ravel(mesh.faces.view(np.ndarray)), dtype=int, device=device_wp),
        )
    padded_min = bbox_min - padding
    padded_max = bbox_max + padding
    extent = padded_max - padded_min

    ni = int(np.ceil(extent[0] / voxel_size)) + 1
    nj = int(np.ceil(extent[1] / voxel_size)) + 1
    nk = int(np.ceil(extent[2] / voxel_size)) + 1

    sdf_wp = wp.full((ni, nj, nk), max_dist, dtype=wp.float32, device=device_wp)
    wp.launch(
        kernel=_sdf_grid_kernel,
        dim=(ni, nj, nk),
        inputs=[
            mesh.id,
            wp.vec3(float(padded_min[0]), float(padded_min[1]), float(padded_min[2])),
            voxel_size,
            max_dist,
            sdf_wp,
        ],
        device=device_wp,
    )
    sdf_np = sdf_wp.numpy()
    volume = wp.Volume.load_from_numpy(
        sdf_np,
        min_world=tuple(float(x) for x in padded_min.tolist()),
        voxel_size=float(voxel_size),
        bg_value=float(max_dist),
        device=device_wp,
    )
    return SdfVolume(volume=volume, aabb_min=padded_min.copy(), aabb_max=padded_max.copy(), padding=float(padding))


def _float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def cached_sdf_volume(
    mesh: Union[str, trimesh.Trimesh, wp.Mesh],
    voxel_size: float,
    padding: float = 0.1,
    device: str = "cuda:0",
) -> SdfVolume:
    """Disk-cached `mesh_to_sdf_volume`: keyed by sha1(mesh bytes) + bake params under `~/.robokit/sdf_cache`."""
    if isinstance(mesh, str):
        raw = Path(mesh).read_bytes()
    elif isinstance(mesh, wp.Mesh):
        raw = mesh.points.numpy().tobytes() + mesh.indices.numpy().tobytes()
    else:
        raw = mesh.vertices.view(np.ndarray).tobytes() + mesh.faces.view(np.ndarray).tobytes()
    # bump _v1 whenever the bake code changes
    key = f"{hashlib.sha1(raw).hexdigest()}_vox{_float_token(voxel_size)}_pad{_float_token(padding)}_v1"
    stem = SDF_CACHE_DIR / key
    if Path(f"{stem}.json").exists():
        return SdfVolume.load(str(stem), device)
    volume = mesh_to_sdf_volume(mesh, voxel_size, padding=padding, device=device)
    SDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    volume.save(str(stem))
    return volume
