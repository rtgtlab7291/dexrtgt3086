# robokit assets

Robot descriptions, body models, motion datasets, and benchmarks for robokit. They live in the
HuggingFace dataset [`rtgtlab7291/robokit-assets`](https://huggingface.co/datasets/rtgtlab7291/robokit-assets).

## Layout

```
robots/
  robot_description/   # URDFs + meshes: arms/ end_effectors/ humanoids/ assembly/
  collision_spheres/   # per-robot collision-sphere YAMLs
  contact_points/      # per-hand contact-point JSONs
  self_collision/  contact_avoid_points/
body_models/           # mano/ smplx/  (.npz)
motions/               # AMASS-style datasets (ACCAD, SFU, DanceDB, ...) + dex-retargeting/
benchmarks/            # dexycb/
contact_retarget/      # self-contained contact-aware bundle (g1 model + spheres + demo meshes)
objects/               # grasp/object datasets
```

Robot files are type-major: one robot's URDF, collision spheres, and contact points live under
different top-level `robots/` subdirs keyed by the same robot name.

## Consuming

robokit reads assets through `robokit.assets`. Each named asset is a small module that exposes
static path symbols. Importing the module downloads that asset into `~/.robokit/cache/`; the
symbols are then ready-to-use `Path`s:

```python
from robokit.assets.robots import unitree_g1  # downloads g1 on import

unitree_g1.URDF_PATH  # a Path
unitree_g1.COLLISION_SPHERE_PATH
```

For a path that is not known statically, `fetch(patterns)` downloads a subtree and returns the
snapshot dir: `fetch(["motions/SFU/**"]) / "motions/SFU/0005/x.npz"`.

The default revision is `main`. Pin a commit with the `ROBOKIT_ASSETS_REVISION` environment
variable or `fetch(revision=...)`.

## Authoring

The dataset is a plain git repo whose binary assets are stored with **git-LFS**, so install
`git-lfs` (`git lfs install`) before cloning or pushing. `robokit.assets.link()` clones it into
`~/.robokit/hf_repo` and symlinks `<cwd>/robokit_assets` to it, so `robokit_assets/` is a normal git checkout. From any repo root:

```shell
uv run python -c "import robokit.assets; robokit.assets.link()"
cd robokit_assets && git add -A && git commit -m "add <asset>" && git push
```

Or clone it directly with `git clone https://huggingface.co/datasets/rtgtlab7291/robokit-assets`.

Every mesh/texture (`.obj .glb .stl .dae .png .npz ...`) is LFS-tracked via `.gitattributes`. When
adding a new binary extension, run `git lfs track "*.<ext>"` first so it is not committed as a raw blob.

Keep the type-major layout above. To expose the new asset as a static symbol, add a module under
`src/robokit/assets/<kind>/<name>.py` following an existing one (e.g. `robots/franka_panda.py`).
