import warp as wp

from robokit.geom.geoms import (
    REFINE_BAND_VOXELS,
    BaseGeom,
    BoxGeom,
    CapsuleGeom,
    MeshGeom,
    PlaneGeom,
    SphereGeom,
    VolumeGeom,
)
from robokit.geom.sample_farthest_points import sample_farthest_points
from robokit.geom.scene import WarpScene
from robokit.geom.sdf_kernels import (
    SegmentClosest,
    closest_point_on_segment_func,
    closest_segment_to_segment_func,
)
from robokit.geom.sdf_volume import SdfVolume, cached_sdf_volume, mesh_to_sdf_volume


wp.config.quiet = True
# wp.init()  # disabled due to conflict with ZED camera SDK

__all__ = [
    "REFINE_BAND_VOXELS",
    "BaseGeom",
    "BoxGeom",
    "CapsuleGeom",
    "MeshGeom",
    "PlaneGeom",
    "SdfVolume",
    "SegmentClosest",
    "SphereGeom",
    "VolumeGeom",
    "WarpScene",
    "cached_sdf_volume",
    "closest_point_on_segment_func",
    "closest_segment_to_segment_func",
    "mesh_to_sdf_volume",
    "sample_farthest_points",
]
