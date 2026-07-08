# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies
uv sync
uv sync --extra cu128 --index pytorch-cu128  # for development with all extras

# Linting and formatting
uvx ruff check
uvx ruff format

# Type checking
uv run pyright

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/robo/test_forward_kinematics.py

# Run a specific test
uv run pytest tests/robo/test_forward_kinematics.py::test_forward_kinematics_warp

# Run docstring tests
uv run xdoctest robokit

# Run an example script
uv run python examples/ik/basic.py
```

## Architecture Overview

RoboKit is a robotics library for inverse kinematics and motion retargeting with multiple computational backends.

### Backend System

The library supports three backends that share the same API:
- **numpy** (CPU, via Pinocchio): Uses Pinocchio for kinematics
- **torch** (GPU): PyTorch-based with autodiff support
- **warp** (GPU): NVIDIA Warp for high-performance GPU computation with CUDA graphs

Robots are loaded via `Robot.load(urdf, backend="warp")` and automatically dispatch to the correct implementation.

### Core Modules

**`robo/`** - Robot representation and kinematics
- `Robot` / `RobotState`: Base classes with backend dispatch
- `RobotSpec`: Parsed URDF specification with joint limits, collision spheres, kinematic tree

**`opt/`** - Optimization solvers
- `WarpBeamSolver`: Multi-stage beam search optimizer for IK
- `NumpyOptimizer` / `TorchOptimizer`: Levenberg-Marquardt solvers

**`terms/`** - Optimization objectives
- `Task`: Base class for residual-based objectives (position error, orientation error, etc.)
- `WarpTask` / `NumpyTask` / `TorchTask`: Backend-specific implementations
- Key tasks: `FrameTask` (end-effector pose), `PositionLimit` (joint limits), `CollisionTask`, `SmoothnessTask`

**`helpers/`** - High-level APIs
- `IKHelper`: GPU-accelerated batched IK solver with multi-stage beam search
- `CPUIKHelper`: CPU-only IK solver using Pinocchio

**`lie/`** - Lie group operations (SO3, SE3) for each backend

**`xform/`** - Transform utilities and rotation conversions

**`retarget/`** - Motion retargeting from human motion to robot

### Typical IK Usage Pattern

```python
from robokit.robo import Robot
from robokit.helpers.ik import IKHelper
from robokit.lie.warp_se3 import WarpSE3

robot = Robot.load(urdf, backend="warp")
ik_helper = IKHelper(robot, target_link_name, placeholder_target)
state = ik_helper.solve_numpy(target_pos, target_quat_wxyz)
```

## Coding Guidelines

- Use `uv` to run scripts (not `python` or `pip` directly)
- Write simple code with clear names; avoid abbreviations (use `scene_id` not `sid`)
- Always include type hints for function arguments and return types
- Do not use `try-except` or `hasattr`; let errors crash to find issues faster
- Avoid comments; code should be self-documenting
- For GPU scripts, find a free GPU and set `CUDA_VISIBLE_DEVICES`
- For visualization scripts with infinite loops, set up a timeout when running
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
- Never add "Co-Authored-By: Claude" or similar lines in commit or PR messages.

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
