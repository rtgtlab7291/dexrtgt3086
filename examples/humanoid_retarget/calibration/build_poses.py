"""Interactive calibration pose-pair builder (Viser) - works for any robot preset.

A calibration pose pair must put the SMPL-X body and the robot in the SAME physical
configuration - the calibration loss matches human keypoints onto the robot's
fixed FK, so a mismatched pair (e.g. human arms raised, robot arms lowered)
biases the learned scales/offsets. This tool makes pairs visually verifiable:

- pick a pose (built-ins, a YAML from a previous session, or any frame of a real
  motion clip via --motion as the "motion prior"),
- pose the robot with per-joint sliders (URDF limits enforced), or press
  "IK init" to seed the sliders from the retargeting IK and then correct,
- watch the per-pair match live: red spheres = the scaled+offset human targets
  the optimizer would see, lines = residual to the robot links, plus a
  calibration-independent limb-direction angle readout (the true match metric),
- Save writes a pose-pair YAML that `calibrate.py --poses` consumes.

Usage (from repo root, then open the printed URL / SSH-forward the port):
    uv run python examples/humanoid_retarget/calibration/build_poses.py --robot gr3
"""

import argparse
import dataclasses
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import viser
import warp as wp
import yaml
import yourdfpy
from calibrate import (
    PRESETS,
    _body_model_dir,
    _load_smplx_pose,
    apply_scale_and_offset_np,
    bone_angles,
    get_link_mapping,
    load_pose_pairs,
    mapped_bones,
    prepare_pose,
)
from viser.extras import ViserUrdf

from robokit.helpers.humanoid_retarget import HumanoidRetargetingOnline
from robokit.helpers.humanoid_retarget.presets.g1_cali import g1_cali
from robokit.opt.multi_seed_solver import StageConfig
from robokit.robo import Robot


# best available config per robot for the IK-init prior (calibrated where one exists)
IK_PRESETS = {**PRESETS, "g1": g1_cali}


# floor on every mapped link's position_weight for the IK-init seed (below)
_IK_INIT_MIN_POS_W = 20.0


def ik_init_config(cfg):
    """Static single-frame IK config for the editor's "IK init" button.

    Two robot-agnostic adjustments vs the runtime preset:

    1. Zero every temporal/regularization term (smoothness, rest, velocity,
       clamp) and oversample seeds + iterations. The presets are tuned for 30 fps
       online retargeting, where those terms keep frame-to-frame motion small  -
       for a from-scratch single-pose solve they just pin the result near the
       reset pose and make the init look bad.

    2. Floor EVERY mapped link's position_weight to _IK_INIT_MIN_POS_W. Presets
       often position-weight only a few links (e.g. pelvis + feet) and leave
       arms/wrists orientation-only for runtime - which leaves the hands
       positionally unconstrained, so the IK seed reaches the right *direction*
       but the wrong *place* and looks awkward. For building pose pairs we want
       the seed to track ALL mapped links. This applies to any current or future
       robot, so a new robot needs no per-preset weight surgery to get a clean
       IK init (calibration/runtime weights still come from its preset).
    """
    link_mapping = {
        rl: dataclasses.replace(m, position_weight=max(m.position_weight, _IK_INIT_MIN_POS_W))
        for rl, m in cfg.link_mapping.items()
    }
    return dataclasses.replace(
        cfg,
        smoothness_weight=0.0,
        base_smoothness_weight=0.0,
        rest_weight=0.0,
        velocity_limit_weight=0.0,
        velocity_clamp_scale=0.0,
        limit_warmup_frames=0,
        link_mapping=link_mapping,
        stages=[
            StageConfig(num_seeds=64, iters=10, lm_lambda=10.0),
            StageConfig(num_seeds=8, iters=20, lm_lambda=1.0),
            StageConfig(num_seeds=1, iters=40, lm_lambda=0.3),
        ],
    )


def ik_bootstrap_pairs(ik_config, robot, body_model_path, pairs, device):
    """Seed each pose pair's robot overrides from the IK-init solve - no editing.

    "IK init as default": for a robot with no hand-built pose pairs, this gives
    usable pairs automatically - the robot configuration that best matches each
    SMPL-X calibration pose under the (track-everything) IK-init config.
    """
    helper = HumanoidRetargetingOnline(ik_init_config(ik_config), device=device, robot=robot)
    helper.warmup(1)
    joint_names = list(robot.spec.actuated_joint_names)
    out = {}
    for name, (bp, _ov) in pairs.items():
        pos, quat, _ = _load_smplx_pose(body_model_path, bp)
        helper.reset()
        frame = np.stack([np.concatenate([pos[name], quat[name]]) for name in helper.human_joint_names])
        qpos = helper.solve_numpy(frame[None])[0]
        out[name] = (bp, {j: round(float(qpos[7 + i]), 4) for i, j in enumerate(joint_names)})
    return out


def save_pose_pairs(pairs, path):
    """Write pose pairs in the poses/<robot>.yaml format that load_pose_pairs reads."""
    out = {
        "poses": {
            name: {
                "body_pose_aa": [round(float(x), 4) for x in bp],
                "joint_overrides": {j: round(float(v), 4) for j, v in ov.items()},
            }
            for name, (bp, ov) in pairs.items()
        }
    }
    with open(path, "w") as f:
        yaml.safe_dump(out, f)


# joint-name-agnostic grouping (robots differ: G1 left_hip_*, GR3 left_thigh_*, BHL leg_left_*)
_SLIDER_GROUPS = [("Left", "left"), ("Right", "right")]


def edit_pose_pairs(
    pairs: Dict[str, Tuple[np.ndarray, Dict[str, float]]],
    config: dict,
    robot: Robot,
    body_model_path: str,
    save_path: str,
    port: int = 8080,
    motion=None,
    finish_label: Optional[str] = None,
    ik_config=None,
) -> Dict[str, Tuple[np.ndarray, Dict[str, float]]]:
    """Run the interactive pose-pair editor; return the edited pairs.

    With `finish_label` set, an extra button with that label ends the session:
    the current edits are committed, auto-saved to `save_path`, the viser
    server shuts down and the (edited) pairs are returned - this is how
    `calibrate.py --edit` reviews pairs and continues in the same run.
    Without it the editor runs forever (standalone `build_poses.py`).
    """
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    link_mapping = get_link_mapping(config)
    root_name = config["human_root_name"]
    pw_links = [rl for rl, _, _, _, pw, _ in link_mapping if pw > 0]
    hj2rl = {hj: rl for rl, hj, _, _, _, _ in link_mapping}
    joint_names = list(robot.spec.actuated_joint_names)
    limits = robot.spec.actuated_joint_limits
    ik_cfg = ik_init_config(ik_config if ik_config is not None else g1_cali)
    pose_names = list(pairs)

    smplx_cache: Dict[str, tuple] = {}  # name -> (positions, quaternions, verts, faces)

    def smplx_data(name: str):
        if name not in smplx_cache:
            pos, quat, _, verts, faces = _load_smplx_pose(body_model_path, pairs[name][0], return_mesh=True)
            smplx_cache[name] = (pos, quat, verts, faces)
        return smplx_cache[name]

    # --- GUI: status and sectioned controls ---
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.add_grid("/ground", width=4, height=4)

    status = server.gui.add_markdown("")
    with server.gui.add_folder("Pose"):
        pose_dropdown = server.gui.add_dropdown("Select", options=tuple(pose_names), initial_value=pose_names[0])
        prev_button = server.gui.add_button("◀ Prev")
        next_button = server.gui.add_button("Next ▶")
        done_toggle = server.gui.add_checkbox("Mark done ✓", initial_value=False)
        progress = server.gui.add_markdown("")
    with server.gui.add_folder("Actions"):
        ik_button = server.gui.add_button("IK init")
        reset_button = server.gui.add_button("Reset to saved")
        if finish_label is not None:
            finish_button = server.gui.add_button(finish_label, color="green")
    match_text = server.gui.add_markdown("")

    sliders: Dict[str, viser.GuiInputHandle] = {}
    for group, pat in _SLIDER_GROUPS + [("Other", "")]:
        members = [j for j in joint_names if (pat in j.lower() if pat else j not in sliders)]
        if not members:
            continue
        with server.gui.add_folder(group, expand_by_default=True):
            for j in members:
                i = joint_names.index(j)
                sliders[j] = server.gui.add_slider(
                    j.replace("_joint", ""),
                    min=float(limits[i, 0]),
                    max=float(limits[i, 1]),
                    step=1e-3,
                    initial_value=0.0,
                )

    if motion is not None:
        n_frames = motion.state.full_pose_aa.shape[0]
        with server.gui.add_folder("Motion tools", expand_by_default=False):
            frame_slider = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
            new_name = server.gui.add_text("New pose name", initial_value="Grabbed-1")
            grab_button = server.gui.add_button("Grab frame as new pose")

    # --- scene ---
    human_mesh = server.scene.add_mesh_simple(
        "/human", np.zeros((3, 3), np.float32), np.array([[0, 1, 2]]), color=(90, 200, 255), opacity=0.45
    )
    base_frame = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, yourdfpy.URDF.load(config["urdf_path"]), root_node_name="/robot")
    viser_joint_order = list(urdf_vis.get_actuated_joint_limits())
    k = len(pw_links)
    target_spheres = [server.scene.add_icosphere(f"/tgt/{rl}", radius=0.02, color=(228, 74, 51)) for rl in pw_links]
    err_lines = server.scene.add_line_segments(
        "/err", points=np.zeros((k, 2, 3), np.float32), colors=(60, 60, 60), line_width=3.0
    )

    done: set = set()
    finish_event = threading.Event()
    ui = {"loading": False, "cur": None, "pos": None, "quat": None, "targets": None}

    def set_sliders(overrides: Dict[str, float]):
        ui["loading"] = True
        for j, s in sliders.items():
            s.value = float(np.clip(overrides.get(j, 0.0), s.min, s.max))
        ui["loading"] = False

    def slider_overrides() -> Dict[str, float]:
        return {j: round(float(s.value), 4) for j, s in sliders.items() if abs(s.value) > 1e-4}

    def set_status(saved: bool):
        status.content = f"### {'✓ saved' if saved else '● unsaved'} - `{save_path}`"
        progress.content = "  ".join(("✓ " if n in done else "○ ") + n for n in pose_names)

    def save_now():
        out = {
            "poses": {
                n: {"body_pose_aa": [round(float(x), 4) for x in bp], "joint_overrides": dict(ov)}
                for n, (bp, ov) in pairs.items()
            }
        }
        with open(save_path, "w") as f:
            yaml.dump(out, f, default_flow_style=None, sort_keys=False)
        set_status(True)

    def commit_current():
        if ui["cur"] is not None:
            pairs[ui["cur"]] = (pairs[ui["cur"]][0], slider_overrides())

    def refresh():
        if ui["loading"]:
            return
        overrides = {j: float(s.value) for j, s in sliders.items() if abs(s.value) > 1e-6}
        pose_data = prepare_pose(ui["cur"], ui["pos"], ui["quat"], robot, config, link_mapping, overrides)
        robot_pos = pose_data["robot_pos"]
        with server.atomic():
            base_frame.position = pose_data["robot_base_pos"]
            base_frame.wxyz = pose_data["robot_base_quat"]
            urdf_vis.update_cfg(np.array([sliders[j].value for j in viser_joint_order], dtype=np.float32))
            line_pts = np.zeros((k, 2, 3), np.float32)
            for i, rl in enumerate(pw_links):
                line_pts[i, 0] = ui["targets"][rl]
                line_pts[i, 1] = robot_pos[rl]
            err_lines.points = line_pts
        resid = float(np.mean([np.linalg.norm(ui["targets"][rl] - robot_pos[rl]) for rl in pw_links])) * 1000.0
        angs = bone_angles(pose_data, hj2rl)
        bones = mapped_bones(hj2rl)
        bad = [lab for lab, _a, _b, floor in bones if angs[lab] > floor + 15.0]
        head = f"**{ui['cur']}** - residual {resid:.0f} mm · {len(bad)} bone(s) need correction\n\n"
        head += "limb direction mismatch (⚠ = above structural floor + 15°):\n"
        match_text.content = head + "".join(
            f"- {lab}: {angs[lab]:.0f}° (floor {floor:.0f}°){' ⚠' if lab in bad else ''}\n"
            for lab, _a, _b, floor in bones
        )

    def load_pose(name: str):
        commit_current()  # keep the leaving pose's edits
        ui["cur"] = name
        pos, quat, verts, faces = smplx_data(name)
        ui["pos"], ui["quat"] = pos, quat
        ui["targets"] = apply_scale_and_offset_np(pos, quat, root_name, config["scale_table"], link_mapping)
        set_sliders(pairs[name][1])
        ui["loading"] = True
        done_toggle.value = name in done
        ui["loading"] = False
        with server.atomic():
            human_mesh.vertices = verts
            human_mesh.faces = faces
            for i, rl in enumerate(pw_links):
                target_spheres[i].position = tuple(ui["targets"][rl])
        refresh()

    def select(name: str):
        if name == ui["cur"]:
            return
        load_pose(name)
        pose_dropdown.value = name  # keep display in sync; re-entry no-ops (name == ui["cur"])
        save_now()  # persist on pose switch

    @pose_dropdown.on_update
    def _(_event):
        select(pose_dropdown.value)

    @prev_button.on_click
    def _(_event):
        select(pose_names[(pose_names.index(ui["cur"]) - 1) % len(pose_names)])

    @next_button.on_click
    def _(_event):
        select(pose_names[(pose_names.index(ui["cur"]) + 1) % len(pose_names)])

    @done_toggle.on_update
    def _(_event):
        if ui["loading"]:
            return
        (done.add if done_toggle.value else done.discard)(ui["cur"])
        save_now()

    for _s in sliders.values():

        @_s.on_update
        def _(_event):
            if ui["loading"]:
                return
            refresh()
            set_status(False)

    @ik_button.on_click
    def _(_event):
        helper = HumanoidRetargetingOnline(ik_cfg, device=device, robot=robot)
        helper.warmup(1)
        helper.reset()
        frame = np.stack([np.concatenate([ui["pos"][name], ui["quat"][name]]) for name in helper.human_joint_names])
        qpos = helper.solve_numpy(frame[None])[0]
        set_sliders({j: float(qpos[7 + i]) for i, j in enumerate(joint_names)})
        refresh()
        set_status(False)

    @reset_button.on_click
    def _(_event):
        set_sliders(pairs[ui["cur"]][1])
        refresh()
        set_status(False)

    if finish_label is not None:

        @finish_button.on_click
        def _(_event):
            finish_event.set()

    if motion is not None:

        @grab_button.on_click
        def _(_event):
            bp = motion.state.full_pose_aa[int(frame_slider.value), 1:22].reshape(63).astype(np.float32)
            pairs[new_name.value] = (bp, {})
            pose_names.append(new_name.value)
            pose_dropdown.options = tuple(pose_names)
            select(new_name.value)

    load_pose(pose_names[0])
    set_status(True)
    print(f"\nViser running - open http://127.0.0.1:{port}  (SSH-forward the port if remote)\n")

    finish_event.wait()  # finish button ends --edit; standalone runs until Ctrl+C (edits persist on switch)
    commit_current()
    save_now()
    server.stop()
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="g1", choices=sorted(PRESETS), help="Robot preset to pose")
    parser.add_argument("--poses", default=None, help="Start from a pose-pair YAML (default: poses/<robot>.yaml)")
    parser.add_argument("--save", default=None, help="Output YAML (default: poses/<robot>.yaml)")
    parser.add_argument("--motion", default=None, help="Optional SMPL-X clip to grab new poses from")
    parser.add_argument("--body-model-path", default=str(_body_model_dir()))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    wp.init()
    ik_cfg = IK_PRESETS[args.robot]
    config = ik_cfg.to_dict()
    robot = Robot.load(config["urdf_path"], load_meshes=True)
    pairs = {name: (bp.copy(), dict(ov)) for name, (bp, ov) in load_pose_pairs(args.poses, args.robot).items()}
    save_path = args.save or f"examples/humanoid_retarget/calibration/poses/{args.robot}.yaml"

    motion = None
    if args.motion:
        from robokit.helpers.humanoid_retarget.loaders import load_smplx_file

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        motion = load_smplx_file(args.motion, args.body_model_path, device=device, compute_vertices=False)

    edit_pose_pairs(
        pairs, config, robot, args.body_model_path, save_path, port=args.port, motion=motion, ik_config=ik_cfg
    )


if __name__ == "__main__":
    main()
