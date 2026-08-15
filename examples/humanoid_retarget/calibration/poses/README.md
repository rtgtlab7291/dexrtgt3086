# Calibration pose pairs (per robot)

`<robot>.yaml` files written by `build_poses.py` / `calibrate.py --edit`:
the 9 SMPL-X calibration poses paired with that robot's joint overrides.
`calibrate.py --robot <name>` picks up `<name>.yaml` here automatically (every
robot, including G1, lives here - there is no in-code fallback).

Format: `poses: {name: {body_pose_aa: [63 floats], joint_overrides: {joint: val}}}`
