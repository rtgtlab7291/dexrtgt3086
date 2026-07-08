# AGENTS.md

## Commands

```shell
make install        # uv sync --all-extras
make ci             # format + lint + check + test
make format         # uvx ruff format
make lint           # uvx ruff check --fix
make check          # uv run pyright
make test           # uv run pytest . && uv run xdoctest robokit
```

Use `uv` to run scripts (not `python` or `pip` directly).

GPU scripts: find a free GPU first and set `CUDA_VISIBLE_DEVICES`. Visualization scripts with infinite loops need a timeout.

## Code Style

- Simple, correct code. Straightforward control flow.
- Descriptive names — no single-letter names (except loop counters), no heavy abbreviations (`scene_id` not `sid`).
- Prefer fewer, larger functions. Split only to remove real duplication.
- No comments unless logic is truly non-obvious. Comments in English.
- Type hints on function args and return types (except `None`, `Any`, `object`).
- No `try-except`, `hasattr`, or fallback mechanisms. Let code crash on errors.
- Ruff: line-length 120. Pyright: standard mode.
- Never use em dashes in comments. Use commas, semicolons, colons, or periods instead.
- Never use `from __future__ import annotations` for Python typing.

## Testing

- Public functions should have doctests (see `docs/dev.md` for format).
- Pytest: group test cases into a single class per feature. Use `_test_*` helpers to compare results across backends (NumPy/Torch/Warp).
- Use `torch.testing.assert_close` for numerical comparisons.

## Conventions

### SE(3)

- Quaternion: scalar-first `(q_w, q_x, q_y, q_z)`.
- 7D Pose: translation-first `(x, y, z, q_w, q_x, q_y, q_z)`.
- 6D Twist: translation-first, rotation-second. Velocity in body frame.
- Integration: right multiply — `T_new = T_old * exp(v_body)`.

### Naming

- Transforms: `T_dst_src` (maps `v_src` → `v_dst`). Components: `t_dst_src`, `q_wxyz_dst_src`. Keep `wxyz` in quaternion variable names. Avoid `pose`, `tf`.
- Joint positions: `q`.
- Batched transforms: still `T_dst_src` (no `Ts_` prefix).

### Multi-Seed Layout

Instance-major order: `output_idx = instance_idx * num_seeds + seed_idx`.

`warp_utils.repeat` expands instances for seeds; `warp_utils.tile` expands seeds for instances.

## Architecture

Source lives in `src/robokit/`. Key packages:

- **robo** — Robot model loading (URDF), kinematics
- **opt** — Optimization variables and solvers (Warp-based optimizer)
- **terms** — Cost/constraint terms for optimization (frame tasks, limits, smoothness)
- **lie** — SE(3) / SO(3) Lie group operations (NumPy, Torch, Warp backends)
- **xform** — Coordinate transform utilities
- **helpers** — Visualization and debugging helpers
- **geom** — Geometry utilities
- **retarget** — Motion retargeting
