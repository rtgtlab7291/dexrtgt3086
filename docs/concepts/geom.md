# Geometry

`robokit.geom` runs batched SDF queries against scenes built from mixed geometric
representations. Each query point gets its signed distance to the scene (negative inside,
positive outside), the surface normal, and the closest surface point. Collision terms,
motion planning, and grasp optimization build on it.

## Overview

```
BoxGeom / SphereGeom / PlaneGeom / CapsuleGeom / MeshGeom / VolumeGeom
    |  scene.add(geom)
    v
WarpScene.query_sdf(points, scene_offsets)
    1. route each point to its scene
    2. launch one Warp kernel per geom, aggregated through the min
    3. return signed distances (+ normals, closest points)
```

Example usage:

```python
geom = BoxGeom(half_extents, scene_offsets, poses=poses)  # store raw inputs (CPU)
scene = WarpScene(num_scenes=1, device="cuda").add(geom)  # allocate GPU arrays of the geom
geom.update(poses=new_poses)  # in-place copy
```

`update` never reallocates: buffers keep their identity, so a query captured in a CUDA graph
keeps working after the environment moves.

## Geom Types

| Geom          | Elements                      | SDF                               |
| ------------- | ----------------------------- | --------------------------------- |
| `BoxGeom`     | half extents                  | analytical                        |
| `SphereGeom`  | radii                         | analytical                        |
| `PlaneGeom`   | local `z=0` halfspaces, posed | analytical                        |
| `CapsuleGeom` | radii + half heights          | analytical                        |
| `MeshGeom`    | triangle meshes               | mesh BVH, optional baked-SDF tier |
| `VolumeGeom`  | pre-baked `SdfVolume` grids   | NanoVDB trilinear sample          |

Every geom takes `scene_offsets` (which scene owns which elements), optional `poses` / `scales`,
and an optional `capacity` to grow later without reallocating CUDA memory.

## Query

```python
sdf, normals, closest = scene.query_sdf(points, scene_offsets)  # min aggregate over all geoms
sdf = scene.query_sdf(points, scene_offsets, distance_only=True)  # query the whole scene, but skip normals/closest
sdf, _, _ = geom.query_sdf(points, scene_offsets)  # this geom only
```

- Batching: one `WarpScene` holds `num_scenes` independent scenes; `scene_offsets` (CSR) or a
  per-point `scene_indices` maps each query point to its scene.
- Masking: `query_mask` skips masked points.
- Gradients: `query_sdf_torch(points, offsets, poses={geom: T})` is differentiable with respect
  to the points and any pose/scale overrides.

### Mesh Level-of-Detail

Mesh queries speed up significantly with a baked signed distance grid:
`MeshGeom(enable_sdf=True, sdf_voxel_size=0.01)` bakes one per mesh, disk-cached under
`~/.robokit/sdf_cache`. Queries then cascade:

```
outside the padded AABB -> AABB distance + padding
inside the grid         -> O(1) grid sample
near the surface        -> exact mesh BVH
```
