# Calibration Guide

Calibration optimizes the retargeting config so that SMPL-X human keypoints align precisely with the robot's FK link positions. It learns **scale_table** (per-joint size ratios) and **position_offset** (per-link local corrections) to minimize the gap between "where the human skeleton says the target should be" and "where the robot link actually is."

## Table of Contents

- [Why Calibrate?](#why-calibrate)
- [How Calibration Works](#how-calibration-works)
  - [The Transformation Pipeline](#the-transformation-pipeline)
  - [What Gets Optimized](#what-gets-optimized)
  - [Multi-Pose Strategy](#multi-pose-strategy)
  - [Loss Function](#loss-function)
- [Quick Start (G1)](#quick-start-g1)
- [Workflow](#workflow)
  - [Step 1: Inspect Current Errors](#step-1-inspect-current-errors)
  - [Step 2: Run Optimization](#step-2-run-optimization)
  - [Step 3: Visualize Results](#step-3-visualize-results)
  - [Step 4: Verify Correctness](#step-4-verify-correctness)
- [CLI Reference](#cli-reference)
- [Config Parameters Explained](#config-parameters-explained)
  - [scale_table](#scale_table)
  - [position_offset](#position_offset)
  - [rotation_offset](#rotation_offset)
  - [human_height_assumption](#human_height_assumption)
  - [link_mapping](#link_mapping)
- [Adapting to a New Robot](#adapting-to-a-new-robot)
  - [Overview](#overview)
  - [Step 1: Create the YAML Config](#step-1-create-the-yaml-config)
  - [Step 2: Determine rotation_offset for Each Link](#step-2-determine-rotation_offset-for-each-link)
  - [Step 3: Create a Calibration Script](#step-3-create-a-calibration-script)
  - [Step 4: Run Calibration](#step-4-run-calibration)
  - [Step 5: Set human_height_assumption](#step-5-set-human_height_assumption)
- [Troubleshooting](#troubleshooting)
- [Scripts in This Directory](#scripts-in-this-directory)

---

## Why Calibrate?

Without calibration, there are systematic mismatches between where SMPL-X says a joint should be and where the robot's physical link sits:

| Source of Error | Example |
|----------------|---------|
| Bone length mismatch | SMPL-X average femur ≠ G1 thigh link length |
| Joint center mismatch | SMPL-X "left_knee" center ≠ G1 knee link origin |
| Proportional difference | Human arm-to-leg ratio ≠ robot arm-to-leg ratio |

A typical uncalibrated config has **~0.08m mean error** across keypoints. After calibration, this drops to **~0.02m** — a 4x improvement that directly translates to more accurate retargeted motion.

---

## How Calibration Works

### The Transformation Pipeline

At runtime, the retargeting system transforms SMPL-X keypoints through this chain before passing them to the IK solver:

```
SMPL-X joint positions (Y-up)
    │
    ▼
scale_table ─── scale each joint relative to root (pelvis)
    │
    ▼
rotation_offset ─── align SMPL-X frame → robot frame (per link)
    │
    ▼
position_offset ─── apply local correction in rotated frame
    │
    ▼
Target positions for IK solver
```

Calibration compares these target positions against the robot's actual FK positions (from MuJoCo) and optimizes `scale_table` and `position_offset` to minimize the difference.

### What Gets Optimized

| Parameter | Learned? | What it does |
|-----------|:--------:|-------------|
| `scale_table` | Yes | Per-joint size ratio relative to pelvis |
| `position_offset` | Yes | Per-link local position correction (meters) |
| `rotation_offset` | No | Frame alignment quaternion (set manually) |
| `position_weight` | No | How strictly IK tracks this keypoint |
| `orientation_weight` | No | How strictly IK tracks orientation |

### Multi-Pose Strategy

The optimizer runs across **9 calibration poses simultaneously** to ensure learned values generalize:

| Pose | Purpose |
|------|---------|
| T-pose | Baseline reference, arms horizontal |
| A-pose | Relaxed standing, arms ~34° down |
| Arms-down | Natural hang, tests arm length |
| Arms-forward | Forward reach, tests shoulder offset |
| Squat | Knees bent, tests leg proportions |
| Arms-up | Overhead reach, tests full arm range |
| Half-T | Arms at 45°, intermediate check |
| Walking | Asymmetric, tests real-world pose |
| Lunge | Deep split stance, tests extreme range |

Each SMPL-X pose has a corresponding set of robot joint angles so that the robot is placed in a matching configuration. Both the SMPL-X body and robot FK are evaluated in the same frame (SMPL-X native Y-up), with the robot's base placed at the SMPL-X pelvis using the pelvis `rotation_offset`.

### Loss Function

```
L = L_reconstruction + L_regularization + L_symmetry

L_reconstruction = Σ  w[link] * ‖target[link] - robot_fk[link]‖²
                   poses,links

L_regularization = λ_scale * Σ (scale - 1.0)²    # pull scales toward identity
                 + λ_offset * Σ ‖offset‖²         # keep offsets small

L_symmetry = λ_sym * Σ (scale_left - scale_right)²           # equal scaling
           + λ_sym * Σ ‖offset_left + offset_right‖²         # mirror offsets
```

End-effectors (wrists, ankles) get extra weight via `--ee-weight` since hand/foot accuracy matters most for interaction and balance.

---

## Quick Start (G1)

```bash
cd robokit-internal

# Full calibration (recommended settings)
uv run python examples/humanoid_retarget/calibration/calibrate_g1.py \
    --config examples/humanoid_retarget/ik_config/smplx_g1.yaml \
    --optimize-all --multi-pose --iters 3000 \
    --save examples/humanoid_retarget/ik_config/smplx_g1_calibrated.yaml

# Visualize before vs after
uv run python examples/humanoid_retarget/calibration/visualize_calibration.py \
    --config examples/humanoid_retarget/ik_config/smplx_g1.yaml \
    --calibrated examples/humanoid_retarget/ik_config/smplx_g1_calibrated.yaml \
    --save calibration_result.png
```

---

## Workflow

### Step 1: Inspect Current Errors

Run without any `--optimize-*` flag to see how well the current config aligns:

```bash
uv run python examples/humanoid_retarget/calibration/calibrate_g1.py \
    --config examples/humanoid_retarget/ik_config/smplx_g1.yaml \
    --multi-pose
```

Output shows per-link, per-pose errors:
```
================================================================================
  T-pose (before)
================================================================================
  Robot Link                      Error (m)    Target pos                    G1 pos
  pelvis                             0.0000    [ 0.003, -0.316,  0.011]     [ 0.003, -0.316,  0.011]
  left_toe_link                      0.0838    [ 0.108, -1.156,  0.057]     [ 0.121, -1.093,  0.111]
  ...
  Mean error: 0.0847 m
```

### Step 2: Run Optimization

Three optimization modes, from simplest to most expressive:

```bash
# Scale only (fast, coarse correction)
--optimize-scales --iters 1000

# Offset only (fine position correction)
--optimize-offsets --iters 1000

# Both jointly (recommended, best results)
--optimize-all --multi-pose --iters 3000
```

Recommended command:

```bash
uv run python examples/humanoid_retarget/calibration/calibrate_g1.py \
    --config examples/humanoid_retarget/ik_config/smplx_g1.yaml \
    --optimize-all --multi-pose --iters 3000 --ee-weight 3.0 \
    --save examples/humanoid_retarget/ik_config/smplx_g1_calibrated.yaml
```

The script prints before/after errors and outputs YAML-ready values.

### Step 3: Visualize Results

```bash
uv run python examples/humanoid_retarget/calibration/visualize_calibration.py \
    --config examples/humanoid_retarget/ik_config/smplx_g1.yaml \
    --calibrated examples/humanoid_retarget/ik_config/smplx_g1_calibrated.yaml \
    --save calibration_comparison.png
```

Generates a grid of 3D plots: rows = original vs calibrated, columns = 9 poses. Red dots = SMPL-X targets, blue dots = robot FK, colored lines = error magnitude.

---

## CLI Reference

```
calibrate_g1.py [OPTIONS]

Options:
  --config PATH           Input YAML config (default: smplx_g1.yaml)
  --optimize-scales       Optimize scale_table only
  --optimize-offsets      Optimize position_offset only
  --optimize-all          Optimize both (recommended)
  --multi-pose            Use all 9 poses (recommended; default: T-pose only)
  --iters N               Optimization iterations (default: 1000)
  --lr FLOAT              Adam learning rate (default: 0.005)
  --ee-weight FLOAT       End-effector weight multiplier (default: 3.0)
  --save PATH             Save calibrated config to file
  --body-model-path PATH  SMPL-X body model directory
```

---

## Config Parameters Explained

### scale_table

Scales each SMPL-X joint position **relative to the root (pelvis)** to compensate for limb length differences.

```yaml
scale_table:
  pelvis: 1.0714      # root scaled absolutely
  left_knee: 0.8395   # 16% shorter than SMPL-X average
```

**Math:**
```
For root:     scaled = pos[root] * scale[root]
For others:   scaled = pos[root] * scale[root] + (pos[joint] - pos[root]) * scale[joint]
```

Each joint is scaled independently relative to the root. This avoids cascading errors where scaling a hip would also shift the knee and foot.

### position_offset

A per-link 3D vector (in meters) applied in the joint's **local rotated frame**. Corrects residual misalignment that scaling alone cannot fix.

```yaml
left_toe_link:
  position_offset: [0.055, 0.022, 0.052]   # 55mm forward, 22mm up, 52mm lateral
```

**Math:**
```
global_offset = rotate(position_offset, joint_quaternion)
target = scaled_position + global_offset
```

Because the offset is rotated by the joint orientation, it remains consistent regardless of the overall pose.

### rotation_offset

A quaternion (wxyz) that aligns the SMPL-X joint frame to the robot link frame. This is **not optimized** — it is set once when creating the config and encodes the fixed coordinate frame relationship.

```yaml
pelvis:
  rotation_offset: [0.5, -0.5, -0.5, -0.5]  # SMPL-X Y-up → G1 frame
```

For the root link, this quaternion also determines how the robot base is oriented in the SMPL-X frame during calibration: `base_quat = smplx_pelvis_quat * rotation_offset`.

### human_height_assumption

The reference human height (in meters) that `scale_table` was calibrated against. At runtime, if the actual human height differs, scales are adjusted inversely:

```yaml
human_height_assumption: 1.66  # calibrated for neutral betas (1.66m)
```

**Runtime adjustment:**
```
ratio = human_height_assumption / actual_height
adjusted_scale[joint] = scale[joint] * ratio
```

A taller person produces larger SMPL-X positions, so scales are reduced to keep targets at the robot's fixed link positions.

For calibration with SMPL-X neutral betas (all zeros), the T-pose mesh height is **~1.66m**. Set `human_height_assumption: 1.66` for configs calibrated this way.

### link_mapping

Connects each robot link to its corresponding SMPL-X joint and stores all per-link parameters:

```yaml
link_mapping:
  left_knee_link:              # Robot link name (from URDF/MuJoCo)
    human_joint: left_knee     # SMPL-X joint name
    position_weight: 10        # IK position tracking weight
    orientation_weight: 15     # IK orientation tracking weight
    position_offset: [...]     # Learned correction (meters)
    rotation_offset: [...]     # Fixed frame alignment (wxyz quaternion)
```

The `position_weight` and `orientation_weight` control how strictly the IK solver tracks this keypoint at runtime. They also influence calibration loss weighting.

---

## Adapting to a New Robot

### Overview

Adding a new robot requires:

1. **Create the YAML config** — define link_mapping, URDF path, initial scale_table
2. **Determine rotation_offset** — align SMPL-X frames to robot frames
3. **Create a calibration script** — define robot-specific pose overrides
4. **Run calibration** — optimize scale_table and position_offset
5. **Set human_height_assumption** — match the calibration reference height

### Step 1: Create the YAML Config

Start from an existing config (e.g., `smplx_g1.yaml`) and modify:

```yaml
# Point to the new robot's URDF
urdf_path: ../../../assets/robot_description/my_robot/my_robot.urdf

# Root link names (check URDF for the base/pelvis link name)
human_root_name: pelvis
robot_root_name: base_link  # varies by robot

# Initial values (will be optimized)
human_height_assumption: 1.66
scale_table:
  pelvis: 1.0
  spine3: 1.0
  left_hip: 1.0
  # ... one entry per SMPL-X joint used in link_mapping

# Map SMPL-X joints to robot links
link_mapping:
  my_robot_pelvis_link:
    human_joint: pelvis
    position_weight: 2000
    orientation_weight: 600
    position_offset: [0.0, 0.0, 0.0]       # start at zero
    rotation_offset: [1.0, 0.0, 0.0, 0.0]  # TBD in step 2
  my_robot_left_hip_link:
    human_joint: left_hip
    position_weight: 10
    orientation_weight: 15
    position_offset: [0.0, 0.0, 0.0]
    rotation_offset: [1.0, 0.0, 0.0, 0.0]  # TBD
  # ... repeat for each link you want to track
```

**Which links to map:**
- Pelvis (root) — always required
- Hips, knees, feet — for locomotion
- Torso/spine — for upper body orientation
- Shoulders, elbows, wrists — for arm motions
- Head — optional, for gaze tracking

Typically 12-16 link mappings cover the full body.

### Step 2: Determine rotation_offset for Each Link

The `rotation_offset` aligns SMPL-X's joint orientation to the robot link's orientation. This must be set correctly **before** calibration — the optimizer cannot learn it.

**Method: Manual alignment at T-pose**

1. Load the robot in MuJoCo at its zero configuration
2. Load SMPL-X at T-pose
3. For each mapped link, compute the quaternion that rotates the SMPL-X frame to match the robot frame:
   ```
   rotation_offset = inverse(smplx_joint_quat) * robot_link_quat
   ```

**Common patterns:**
- SMPL-X is Y-up; many robots are Z-up. The pelvis rotation_offset often encodes a 90° rotation.
- Left/right limbs often share the same rotation_offset (mirrored by the skeleton, not the offset).
- For G1: `pelvis: [0.5, -0.5, -0.5, -0.5]` corresponds to a specific Y-up to G1-frame rotation.

**Tip:** Start by copying rotation_offsets from a similar robot config and adjust if the resulting visualization looks rotated or flipped.

### Step 3: Create a Calibration Script

Copy `calibrate_g1.py` as `calibrate_<robot>.py` and modify the robot-specific parts:

**a) Define pose overrides.** Each calibration pose needs robot joint angles that match the SMPL-X pose. This is the hardest part because it depends on the robot's **zero configuration**:

```python
# Example: robot with arms-down zero pose (like GR3)
MY_ROBOT_POSE_OVERRIDES = {
    "T-pose": {
        "left_shoulder_roll_joint": 1.5708,   # arms out to sides
        "right_shoulder_roll_joint": -1.5708,
    },
    "A-pose": {
        "left_shoulder_roll_joint": 0.785,
        "right_shoulder_roll_joint": -0.785,
    },
    # ...
}
```

The key principle: **for each named pose, the robot joint angles must produce the same physical pose as the SMPL-X body_pose.** For example, if T-pose means "arms straight out to the sides", the robot joint values must achieve that regardless of whether the robot's zero config has arms forward, down, or elsewhere.

**b) Update end-effector links:**

```python
END_EFFECTOR_LINKS = {
    "my_robot_left_foot_link",
    "my_robot_right_foot_link",
    "my_robot_left_hand_link",
    "my_robot_right_hand_link",
}
```

**c) Update `find_mujoco_xml()`** to locate the robot's MuJoCo XML, or hardcode the path.

### Step 4: Run Calibration

```bash
uv run python examples/humanoid_retarget/calibration/calibrate_my_robot.py \
    --config examples/humanoid_retarget/ik_config/smplx_my_robot.yaml \
    --optimize-all --multi-pose --iters 3000 \
    --save examples/humanoid_retarget/ik_config/smplx_my_robot_calibrated.yaml
```

**What to look for:**
- Before errors: typically 0.05-0.15m per keypoint
- After errors: should drop to 0.01-0.04m
- If errors don't improve: check rotation_offsets and pose overrides
- If a single link has large error: its rotation_offset is likely wrong

### Step 5: Set human_height_assumption

The calibration uses SMPL-X with neutral betas (all zeros), which produces a body of ~1.66m height. Set:

```yaml
human_height_assumption: 1.66
```

This tells the runtime system: "the scale_table was calibrated for a 1.66m person." When retargeting a different-height person, scales are adjusted automatically via `config.with_actual_height(actual_height)`.

---

## Troubleshooting

**"No MuJoCo XML found"**
- The script looks for the XML near the URDF path in the config. For a new robot, ensure the MuJoCo XML exists in the robot's asset directory, or modify `find_mujoco_xml()` in your calibration script.

**Large errors persist after optimization**
- Check `rotation_offset` — wrong frame alignment causes systematic errors that scaling cannot fix.
- Check pose overrides — if the robot pose doesn't match the SMPL-X pose, the optimizer fits to wrong targets.
- Try `--optimize-scales` alone first to isolate whether the issue is scale or offset.

**Asymmetric results (left != right)**
- The optimizer uses symmetry constraints but they're soft. Increase `symmetry_weight` in the `JointOptimizer` constructor if needed.
- Check that the robot's URDF/XML is actually symmetric.

**Torso link has high error**
- The torso/spine link often has `position_weight: 0` (orientation-only tracking). Its position error in calibration output is expected to be larger since it's not penalized.

**Scale values far from 1.0 (e.g., 0.5 or 2.0)**
- May indicate wrong `human_height_assumption` or a mismatch between the SMPL-X reference and robot size.
- Try setting all scales to 1.0 and re-running calibration from scratch.

---

## Scripts in This Directory

| Script | Purpose |
|--------|---------|
| `calibrate_g1.py` | G1-specific calibration optimizer. Defines 9 SMPL-X poses with matching G1 joint overrides, optimizes scale_table and position_offset via PyTorch Adam. |
| `visualize_calibration.py` | Generates 3D comparison plots (original vs calibrated) across all 9 poses. Red = SMPL-X targets, blue = robot FK, colored lines = error. |
| `compare_height_methods.py` | Compares 4 approaches for estimating human height from SMPL-X data. Validates that T-pose mesh vertex measurement is the gold standard. |
| `verify_calibration.py` | Automated tests: (1) verifies calibration math matches runtime math, (2) checks height invariance — that `with_actual_height()` produces consistent targets across different heights. |
