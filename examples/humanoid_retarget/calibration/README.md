# Calibration Guide

Calibration optimizes a humanoid retargeting config (any robot with a preset: `--robot g1|gr3|h1_2|bhl`) so that SMPL-X human keypoints align precisely with the robot's FK link positions. It learns **scale_table** (per-joint size ratios) and **position_offset** (per-link local corrections) to minimize the gap between "where the human skeleton says the target should be" and "where the robot link actually is."

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
- [Authoring Calibration Pose Pairs](#authoring-calibration-pose-pairs)
- [CLI Reference](#cli-reference)
- [Config Parameters Explained](#config-parameters-explained)
  - [scale_table](#scale_table)
  - [position_offset](#position_offset)
  - [rotation_offset](#rotation_offset)
  - [human_height_assumption](#human_height_assumption)
  - [link_mapping](#link_mapping)
- [Calibrating Another Robot](#calibrating-another-robot)
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

On real motion clips (3 SFU clips, `calibration_eval.py`), the uncalibrated G1 config tracks its keypoints at **~93 mm mean error with the robot floating ~9–25 cm above the ground**; the calibrated preset tracks at **~53 mm with feet on the floor**. Wrists benefit the most from correctly matched pose pairs (144/125 → 99/73 mm).

---

## How Calibration Works

### The Transformation Pipeline

At runtime, the retargeting system transforms SMPL-X keypoints through this chain before passing them to the IK solver:

```
SMPL-X joint positions (Z-up world, ground at z=0)
    │
    ▼
human_heights input ─── multiply all scales by assumption/actual_height
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

Calibration compares these target positions against the robot's actual FK positions (robokit's own Warp FK on the preset URDF) and optimizes `scale_table` and `position_offset` to minimize the difference.

**Frame standard (important):** calibration runs in the exact frame the runtime sees - **Z-up, with both the human (mesh min-z) and the robot standing on the ground plane z=0**. The robot is grounded by the minimum z over ALL its links' geometry (collision where a link has any, else visual) - the lowest point of a standing humanoid is its sole, so no per-robot foot configuration is needed. The root scale formula `pos[root] * scale[root]` is not translation-invariant, so learning it in any other frame (e.g. SMPL-X native Y-up with the pelvis at the origin) produces a root scale that misplaces the pelvis at runtime - the robot floats above the ground with its feet in the air. Grounded calibration makes the learned root scale encode the true robot-vs-human stature ratio (~0.78 for G1), which puts the feet on the floor and scales stride lengths correctly.

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
| A-pose | Relaxed standing, arms 34° below horizontal |
| Arms-down | Natural hang, tests arm length |
| Arms-forward | Forward reach, tests shoulder offset |
| Squat | Knees bent, tests leg proportions |
| Arms-up | Lateral raise 34° above horizontal (the G1 shoulder_roll limit caps higher raises) |
| Half-T | Lateral raise 20° above horizontal, intermediate check |
| Walking | Asymmetric, tests real-world pose |
| Lunge | Deep split stance, tests extreme range |

The SMPL-X side of the pairs lives in `SMPLX_POSES` in `calibrate.py` (robot-independent); the per-robot joint overrides come from `poses/<robot>.yaml` (G1's curated set lives in code). Replaceable via `--poses` - see [Authoring calibration pose pairs](#authoring-calibration-pose-pairs).

Each SMPL-X pose has a corresponding set of robot joint angles so that the robot is placed in a matching configuration. Both the SMPL-X body and robot FK are evaluated in the runtime frame (Z-up, grounded on z=0): the human is grounded by its mesh, the robot base sits at the human pelvis xy with quat = `smplx_pelvis_quat * rotation_offset_pelvis`, and its z is whatever puts its lowest geometry point (the foot soles) on the floor.

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

# Full calibration (recommended settings; default input config is presets.g1)
uv run python examples/humanoid_retarget/calibration/calibrate.py \
    --optimize-all --multi-pose --iters 3000 \
    --save /tmp/smplx_g1_calibrated.yaml

# Quantitative check on real clips (tracking error + foot-float metric)
uv run python examples/humanoid_retarget/calibration/calibration_eval.py \
    --yaml /tmp/smplx_g1_calibrated.yaml

# Visual before/after check (SMPL-X targets snapping onto the robot, per pose)
uv run python examples/humanoid_retarget/calibration/viz/viz_skeleton_alignment.py \
    --robot g1 --calibrated /tmp/smplx_g1_calibrated.yaml --out /tmp/g1_alignment.png
```

---

## Workflow

### Step 1: Inspect Current Errors

Run without any `--optimize-*` flag to see how well the current config aligns:

```bash
uv run python examples/humanoid_retarget/calibration/calibrate.py --multi-pose
```

Output shows the pose-pair audit table (limb-direction mismatch with ⚠ flags - see
[Authoring calibration pose pairs](#authoring-calibration-pose-pairs)) followed by
per-link, per-pose errors in the grounded Z-up frame:
```
--- Pose-pair audit: human-vs-robot limb direction (deg, ⚠ = exceeds structural floor + 15°) ---
  pose              L uarm    L farm    R uarm    R farm   L thigh    L shin   R thigh    R shin
  T-pose            25.7       5.3      18.3       1.7       7.9       8.0       7.1       7.3
  ...
================================================================================
  T-pose (before)
================================================================================
  Robot Link                      Error (m)    Target pos                    G1 pos
  pelvis                             0.0301    [ 0.002,  -0.010,  0.770]    [ 0.003,  -0.012,  0.797]
  left_toe_link                      0.0742    [ 0.119,   0.099,  0.064]    [ 0.184,   0.097,  0.029]
  ...
  Mean error: 0.1136 m
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

Recommended command (add `--edit` to review/fix the pose pairs in the browser first -
calibration continues in the same run after you click "Apply & continue calibration"):

```bash
uv run python examples/humanoid_retarget/calibration/calibrate.py \
    --optimize-all --multi-pose --iters 3000 --ee-weight 3.0 \
    --save /tmp/smplx_g1_calibrated.yaml
```

The script prints before/after errors and outputs YAML-ready values (including the
measured `human_height_assumption`).

### Step 3: Visualize Results

```bash
# dump the uncalibrated baseline preset to YAML for the comparison
uv run python -c "import yaml; from robokit.helpers.humanoid_retarget.presets.g1 import g1; \
yaml.dump(g1.to_dict(), open('/tmp/g1_orig.yaml','w'), sort_keys=False)"

uv run python examples/humanoid_retarget/calibration/viz/visualize_calibration.py \
    --config /tmp/g1_orig.yaml \
    --calibrated /tmp/smplx_g1_calibrated.yaml \
    --save calibration_comparison.png
```

Generates a grid of 3D plots: rows = original vs calibrated, columns = 9 poses. Red dots = SMPL-X targets, blue dots = robot FK, colored lines = error magnitude.

---

## Authoring Calibration Pose Pairs

A pose pair = (SMPL-X `body_pose` axis-angle, robot joint overrides) that put both bodies in the **same physical configuration**. Mismatched pairs (e.g. the old Half-T: human arms 40° above horizontal, robot arms 21° below) bias the calibration - fixing the G1 pairs cut runtime wrist tracking error from ~135 mm to ~85 mm.

Use the interactive builder to create or verify pairs:

```bash
uv run python examples/humanoid_retarget/calibration/build_poses.py \
    --save examples/humanoid_retarget/calibration/outputs/pose_pairs.yaml

# optionally grab poses from a real motion clip ("motion prior"):
uv run python examples/humanoid_retarget/calibration/build_poses.py \
    --motion ~/.robokit/assets/motions/SFU/0008/0008_Yoga001_poses.npz
```

In the browser UI:
1. Pick a pose from the dropdown (built-ins, a previous YAML via `--poses`, or grab any motion frame).
2. Press **IK init** to seed the robot joints from the retargeting IK, then correct with the per-joint sliders (URDF limits enforced). Watch:
   - red spheres + lines = the scaled human targets vs robot links (what the optimizer sees; depends on the current calibration),
   - the **limb direction mismatch** readout - calibration-independent angles between matching human/robot limb segments. This is the true pair-match metric; segments flagged ⚠ (>15° above the structural floor) need fixing. Expect a constant floor from joint-center placement differences (G1: upper arms ~18–26°, forearms ~2–5°, legs ~7–8°).
3. Step through poses with **◀ Prev / Next ▶** and tick **Mark done ✓**; edits **auto-save** to `poses/<robot>.yaml` on every pose switch (status line shows ● unsaved / ✓ saved). Then pre-check and calibrate:

```bash
# Pre-check: prints the limb-direction audit table and renders all pairs to a PNG,
# then exits WITHOUT optimizing - fix any ⚠ pose in build_poses.py first.
uv run python examples/humanoid_retarget/calibration/calibrate.py \
    --robot g1 --multi-pose --precheck /tmp/precheck.png

# Calibrate and emit the preset in one go (--emit-preset writes presets/g1_cali.py):
uv run python examples/humanoid_retarget/calibration/calibrate.py \
    --robot g1 --optimize-all --multi-pose --iters 3000 \
    --save /tmp/calibrated.yaml --emit-preset
```

(The audit table also prints on every calibration run, so a mismatched pair is flagged even without `--precheck`.)

Or do it all in ONE run with `--edit`: the calibration opens the pose editor in the browser first, you review/fix the pairs (same UI as `build_poses.py`), and clicking **Apply & continue calibration** auto-saves your edits and proceeds straight into the optimization:

```bash
uv run python examples/humanoid_retarget/calibration/calibrate.py \
    --edit --optimize-all --multi-pose --iters 3000 --save /tmp/calibrated.yaml
```

YAML format:

```yaml
poses:
  T-pose:
    body_pose_aa: [0.0, 0.0, ...]   # 63 floats (21 body joints x 3 axis-angle)
    g1_joint_overrides: {left_shoulder_roll_joint: 1.5708, ...}
```

---

## CLI Reference

```
calibrate.py [OPTIONS]

Options:
  --robot NAME            Robot preset: g1, gr3, h1_2, bhl (default: g1)
  --config PATH           Input YAML config (overrides --robot)
  --optimize-scales       Optimize scale_table only
  --optimize-offsets      Optimize position_offset only
  --optimize-all          Optimize both (recommended)
  --multi-pose            Use all 9 poses (recommended; default: T-pose only)
  --iters N               Optimization iterations (default: 1000)
  --lr FLOAT              Adam learning rate (default: 0.005)
  --ee-weight FLOAT       End-effector weight multiplier (default: 3.0)
  --save PATH             Save calibrated config to file
  --body-model-path PATH  SMPL-X body model directory
  --poses PATH            Pose-pair YAML from build_poses.py (default: poses/<robot>.yaml)
  --precheck PATH.png     Render all pose pairs + audit table, then exit (no optimization)
  --edit                  Open the viser pose editor first; continue after 'Apply & continue'
```

---

## Config Parameters Explained

### scale_table

Scales each SMPL-X joint position **relative to the root (pelvis)** to compensate for limb length differences.

```yaml
scale_table:
  pelvis: 0.7788      # root scaled absolutely (≈ robot/human stature ratio)
  left_knee: 0.8482   # 15% shorter than SMPL-X average
```

**Math:**
```
For root:     scaled = pos[root] * scale[root]
For others:   scaled = pos[root] * scale[root] + (pos[joint] - pos[root]) * scale[joint]
```

Each joint is scaled independently relative to the root. This avoids cascading errors where scaling a hip would also shift the knee and foot.

The root scale multiplies the pelvis **world position**, so it controls both the vertical placement (pelvis height above the ground) and the horizontal stride scaling. This is exactly why calibration must run grounded in the runtime frame - see [The Transformation Pipeline](#the-transformation-pipeline).

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

A quaternion (wxyz) that aligns the SMPL-X joint frame to the robot link frame. This is **not optimized** - it is set once when creating the config and encodes the fixed coordinate frame relationship.

```yaml
pelvis:
  rotation_offset: [0.5, -0.5, -0.5, -0.5]  # SMPL-X Y-up → G1 frame
```

For the root link, this quaternion also determines how the robot base is oriented in the SMPL-X frame during calibration: `base_quat = smplx_pelvis_quat * rotation_offset`.

### human_height_assumption

The reference human height (in meters) that `scale_table` was calibrated against. The calibration script **measures and exports this automatically** - it is the T-pose mesh extent of the calibration body (betas=0), the same convention `smplx_loader._compute_tpose_height` uses to measure a motion's actual height.

```yaml
human_height_assumption: 1.7189  # measured T-pose mesh height, neutral betas
```

**Runtime adjustment:**
```
ratio = human_height_assumption / actual_height
adjusted_scale[joint] = scale[joint] * ratio
```

A taller person produces larger SMPL-X positions, so scales are reduced to keep targets at the robot's fixed link positions. Pass `motion.human_height` through the solver's `human_heights` argument; it is computed from the motion file's betas.

Do **not** hand-set this value: a mismatch between the assumption and the true calibration-body height is a systematic scale error on every motion (the old hand-set 1.66 vs the measured 1.7189 was a ~3.5% error).

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

## Calibrating Another Robot

The calibration tooling is universal - `--robot g1|gr3|h1_2|bhl` pulls everything robot-specific from the retarget preset (`src/robokit/helpers/humanoid_retarget/presets/`): URDF path, link_mapping, rotation offsets, root names. End-effector links and foot links are derived from the link_mapping, and grounding uses the robot's own geometry. The ONLY thing a new robot needs is its **pose-pair joint overrides**, which you build interactively.

### Robot already has a preset (gr3, h1_2, bhl)

```bash
# 1. Build the pose pairs in the browser, calibrate, and emit the preset in one run:
#    for each pose: press "IK init", correct with the sliders until the robot
#    visually matches the semi-transparent human, then "Apply & continue".
#    Pairs auto-save to poses/<robot>.yaml; --emit-preset writes presets/gr3_cali.py.
uv run python examples/humanoid_retarget/calibration/calibrate.py \
    --robot gr3 --edit --optimize-all --multi-pose --iters 3000 \
    --save /tmp/gr3_cali.yaml --emit-preset

# 2. Register the new preset in examples/humanoid_retarget/10_gallery.py
#    (--emit-preset prints the two lines to paste), then:

# 3. Quantitative check on real clips (vs the uncalibrated preset):
uv run python examples/humanoid_retarget/calibration/calibration_eval.py \
    --robot gr3 --yaml /tmp/gr3_cali.yaml

# 4. Math/height sanity:
uv run python examples/humanoid_retarget/calibration/verify_calibration.py \
    --robot gr3 --config /tmp/gr3_cali.yaml
```

No hand-built pairs? Drop `--edit` and add `--ik-bootstrap` to seed all 9 pose pairs
automatically from the IK-init solve, then calibrate - useful as a first pass on a new robot.

You can also build/refine pairs separately with `build_poses.py --robot gr3` (IK init uses the robot's best available preset as the prior), and pre-check them with `calibrate.py --robot gr3 --multi-pose --precheck /tmp/precheck.png`. The limb-direction audit floors were measured on the G1 - treat the ⚠ flags as heuristics for other robots and trust the visual overlay.

**What to look for:** before-errors typically 0.05–0.15 m per keypoint; after-errors 0.03–0.06 m. If a single link stays bad, its `rotation_offset` is likely wrong; if everything is bad, the pose pairs don't match (rebuild with `--edit`).

### Robot has no preset yet

Author one first (then the section above applies):

1. **Create the preset** (copy `presets/g1.py`): `urdf_path`, `human_root_name`, a `link_mapping` from SMPL-X joints to robot links (pelvis always; hips/knees/feet for locomotion; torso; shoulders/elbows/wrists for arms - 11–16 links typical), `scale_table` initialized to 1.0, `human_height_assumption` placeholder (the calibration measures and exports the real value).
2. **Set each `rotation_offset`** (the optimizer cannot learn these): the quaternion aligning the SMPL-X joint frame to the robot link frame, `rotation_offset = inverse(smplx_joint_quat) * robot_link_quat` at matching poses. Start by copying from a similar robot (G1/H1-2 pelvis: `[0.5, -0.5, -0.5, -0.5]`; GR3: `[-0.5, 0.5, 0.5, 0.5]`) and fix anything that renders rotated/flipped in `build_poses.py`.
3. Register the preset in `PRESETS` in `calibrate.py` and `examples/humanoid_retarget/10_gallery.py`.

---

## Troubleshooting

**"Meshes are not loaded"**
- Grounding reads the robot's geometry: load the robot with `Robot.load(urdf, load_meshes=True)`. Any link geometry works (collision preferred, visual as the per-link fallback) - no per-robot foot configuration is needed.

**Retargeted robot floats above the ground (or sinks into it)**
- The config was calibrated in the wrong frame, or its `human_height_assumption` doesn't match the calibration body. Re-run the calibration (grounded Z-up is the default behavior of `calibrate.py`) and check the float metric with `calibration_eval.py` - stance-frame foot height should be ≈ 0.

**Large errors persist after optimization**
- Check `rotation_offset` - wrong frame alignment causes systematic errors that scaling cannot fix.
- Check pose overrides - if the robot pose doesn't match the SMPL-X pose, the optimizer fits to wrong targets.
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
| `calibrate.py` | Universal calibration optimizer (`--robot g1\|gr3\|h1_2\|bhl`). 9 SMPL-X poses paired with per-robot joint overrides (`poses/<robot>.yaml`; grounded Z-up), optimizes scale_table and position_offset via PyTorch Adam, exports the measured `human_height_assumption`. `--emit-preset` writes the `<robot>_cali.py` preset; `--ik-bootstrap` seeds pairs from IK. |
| `build_poses.py` | Interactive Viser pose-pair builder (any `--robot`): joint sliders + IK init + limb-direction match metric, auto-saves `poses/<robot>.yaml`; can grab poses from real motion clips. |
| `calibration_eval.py` | Quantitative eval on real motion clips (any `--robot`): per-keypoint tracking error (mm) and stance-frame foot-float metric (m). Run directly to compare uncalibrated / calibrated / a fresh YAML. |
| `cali_configs.py` | Config helper: `uncalibrated(base)` (scale 1, zero offsets). |
| `verify_calibration.py` | Automated tests: (1) verifies calibration math matches runtime math, (2) checks height invariance for the `human_heights` input. |
| `viz/` | Optional post-calibration figures (deletable; the core above has no viz dependency): `viz/visualize_calibration.py` and `viz/viz_skeleton_alignment.py` render per-pose before/after target-vs-FK alignment for any `--robot`. |
