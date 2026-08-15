"""Generate a collision-sphere YAML for any robot from its collision meshes.

For each link that has collision geometry: sample interior points, set each candidate sphere's radius to
its distance-to-surface, then greedily keep the largest spheres whose centers aren't already covered.
Useful for robots whose URDF ships meshes but no collision spheres (e.g. Fourier gr3). Links without
collision geometry are skipped. Output loads via `Robot.load(..., collision_spheres_path=<out>)`.

Sphere YAMLs live at `robots/collision_spheres/<robot>/<variant>.yaml` in the HF dataset, so to add
a robot: generate into the git clone (`~/.robokit/hf_repo`), then commit and push it.

    uv run python scripts/geom/gen_collision_spheres.py --urdf <path> --per-link 8 \\
        --out ~/.robokit/hf_repo/robots/collision_spheres/<robot>/full_body.yaml
    cd ~/.robokit/hf_repo && git add -A && git commit -m "add <robot> spheres" && git push
"""

import argparse
from typing import Dict

import numpy as np
import trimesh
import yaml

from robokit.robo import Robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-link", type=int, default=8, help="max spheres per link")
    parser.add_argument("--candidates", type=int, default=2000, help="interior points sampled per link")
    parser.add_argument("--min-radius", type=float, default=0.01, help="drop spheres smaller than this (m)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    robot = Robot.load(args.urdf, load_meshes=True)
    collision_spheres: Dict[str, list] = {}
    for link, mesh in robot.spec.get_link_meshes("collision").items():
        if not mesh.is_watertight:
            mesh = mesh.copy()
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        pts = trimesh.sample.volume_mesh(mesh, args.candidates)
        if len(pts) == 0:
            print(f"{link}: 0 spheres (no interior samples)")
            continue
        radii = np.abs(trimesh.proximity.signed_distance(mesh, pts))  # interior pt -> distance to surface
        spheres: list = []
        for i in np.argsort(-radii):
            if radii[i] < args.min_radius or len(spheres) >= args.per_link:
                break
            center = pts[i]
            covered = any(np.linalg.norm(center - np.array(s["center"])) < s["radius"] for s in spheres)
            if not covered:
                spheres.append({"center": [float(x) for x in center], "radius": float(radii[i])})
        if spheres:
            collision_spheres[link] = spheres
        print(f"{link}: {len(spheres)} spheres")

    with open(args.out, "w") as f:
        yaml.dump({"collision_spheres": collision_spheres}, f, default_flow_style=None, sort_keys=False)
    total = sum(len(v) for v in collision_spheres.values())
    print(f"wrote {total} spheres over {len(collision_spheres)} links -> {args.out}")


if __name__ == "__main__":
    main()
