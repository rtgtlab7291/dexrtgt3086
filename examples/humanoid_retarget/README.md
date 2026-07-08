# Humanoid Retargeting

Retarget human motions (SMPL-X or BVH) to humanoid robots using GPU-accelerated IK.

## Supported Robots

| Robot | Shorthand | DOF | Config |
|-------|-----------|-----|--------|
| Unitree G1 | `g1` | 29 | `ik_config/smplx_g1.yaml` |
| Unitree G1 Calibrated| `g1_cali` | 29 | `ik_config/smplx_g1_cali.yaml` |
| Unitree H1-2 | `h1_2` | 27 | `ik_config/smplx_h1_2.yaml` |
| Fourier GR3 | `gr3` | 31 | `ik_config/smplx_gr3.yaml` |
| Booster K1 | `k1` | - | `ik_config/smplx_k1.yaml` |
| Berkeley Humanoid Lite | `bhl` | - | `ik_config/smplx_bhl.yaml` |

Use `--robot <shorthand>` in any script to select the target robot (defaults to `g1`).

## Quick Start

### 1. Download Assets

```bash
uv run python examples/humanoid_retarget/download_assets.py
```

This downloads robot URDFs, SMPL-X body models, and sample motions into `assets/`.

### 2. Retarget a Single Motion

```bash
uv run python examples/humanoid_retarget/smplx_to_robot.py \
    --smplx_file motion_data/raw/ACCAD/walk.npz

# Target a different robot
uv run python examples/humanoid_retarget/smplx_to_robot.py \
    --smplx_file motion_data/raw/ACCAD/walk.npz \
    --robot h1_2 --save_path output/walk_h1_2.pkl
```

### 3. View the Result

```bash
uv run python examples/humanoid_retarget/view_motion.py \
    --motion output/walk_h1_2.pkl --robot h1_2
```

Open the printed URL in a browser to see the playback.

## Scripts

### Retargeting

| Script | Purpose |
|--------|---------|
| `smplx_to_robot.py` | Retarget a single SMPL-X file to a robot |
| `smplx_to_robot_dataset.py` | Batch-retarget an entire folder of SMPL-X files |
| `bvh_retarget_stream.py` | Retarget a BVH file frame-by-frame (real-time streaming) |

### Visualization

| Script | Purpose |
|--------|---------|
| `view_motion.py` | Play back retargeted `.pkl` motions, optionally with SMPL-X mesh overlay |
| `humanoid_retarget_online.py` | Retarget + live visualization with human vs. robot keypoint comparison |
| `interactive_retarget.py` | Browse a dataset folder, pick a motion, retarget and visualize on-the-fly |
| `batch_motion_viewer.py` | Render 50+ robots playing the same motion (for paper videos) |

## Common Options

All scripts share a consistent CLI pattern:

```
--robot <shorthand>    Robot to use (g1, h1_2, gr3, k1, bhl)
--urdf <path>          Override URDF path (otherwise resolved from config)
--config <path>        Override config YAML (otherwise resolved from --robot)
--device <device>      Compute device (default: cuda:0)
--port <port>          Viser viewer port (default: 8080)
```

## Workflow Examples

### Batch-retarget a dataset

```bash
uv run python examples/humanoid_retarget/smplx_to_robot_dataset.py \
    --robot g1 \
    --src_folder motion_data/raw/ACCAD \
    --tgt_folder motion_data/retargeted/ACCAD \
    --num_workers 4
```

### Browse retargeted motions

```bash
uv run python examples/humanoid_retarget/view_motion.py \
    --robot g1 --dataset motion_data/retargeted/ACCAD
```

### Stream a BVH file with visualization

```bash
uv run python examples/humanoid_retarget/bvh_retarget_stream.py \
    --robot g1 --bvh motion_data/bvh/test_andy.bvh --visualize
```

### Compare human and robot keypoints

```bash
uv run python examples/humanoid_retarget/view_motion.py \
    --robot g1 \
    --motion motion_data/retargeted/ACCAD/walk.pkl \
    --smplx_file motion_data/raw/ACCAD/walk.npz \
    --smplx_body_model assets/body_models \
    --show_keypoints
```

## Config Structure

Each YAML config in `ik_config/` defines:

- **urdf_path** -- robot URDF (resolved relative to the YAML file)
- **link_mapping** -- which robot links track which human (SMPL-X) joints, with position/orientation weights
- **scale_table** -- per-joint scaling factors to map human proportions to the robot
- **stages** -- multi-stage beam search solver parameters (seeds, iterations, damping)
- **smoothness_weight / rest_weight** -- per-joint temporal smoothness and rest-pose regularization

## Output Format

Retargeted motions are saved as `.pkl` files containing:

```python
{
    "fps": float,
    "root_pos": ndarray,      # (T, 3) root translation
    "root_rot": ndarray,      # (T, 4) root rotation (xyzw quaternion)
    "dof_pos": ndarray,       # (T, num_dof) joint positions
    "human_height": float,    # original human height in meters
}
```
