"""Calibration example: a checkbox flips between the raw g1 mapping and the calibrated g1_cali preset."""

import time

import numpy as np
import viser
import warp as wp
import yourdfpy
from viser.extras import ViserUrdf

from robokit.helpers.humanoid_retarget import HumanoidRetargetingOnline
from robokit.helpers.humanoid_retarget.loaders import fetch_smplx_clip, get_smplx_motion
from robokit.helpers.humanoid_retarget.presets.g1 import g1
from robokit.helpers.humanoid_retarget.presets.g1_cali import g1_cali


MOTION = "motions/SFU/0007/0007_Walking001_poses.npz"


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- retarget: one helper per preset (concurrent helpers must not share a Robot instance) ---
    helpers = {}
    for name, preset in (("raw", g1), ("calibrated", g1_cali)):
        helpers[name] = HumanoidRetargetingOnline(preset, device=device, robot=preset.load_robot())

    smplx, raw_motion, fps = fetch_smplx_clip(MOTION, helpers["raw"].human_joint_names)
    human_arrays = {"raw": raw_motion}
    human_arrays["calibrated"], _ = get_smplx_motion(smplx, helpers["calibrated"].human_joint_names)
    human_heights = np.array([smplx.human_height], dtype=np.float32)
    for name, helper in helpers.items():
        helper.warmup(1)
        helper.solve_numpy(human_arrays[name][None, 0], human_heights)  # warm up the CUDA graph
        helper.reset()

    # --- viewer: watch the feet — the raw mapping floats, the calibrated one stands on the ground ---
    assert g1.urdf_path is not None
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4, height=4)
    base = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, yourdfpy.URDF.load(str(g1.urdf_path)), root_node_name="/robot")
    human_joints = helpers["raw"].human_joint_names
    keypoints = [
        server.scene.add_icosphere(f"/human/{name}", radius=0.02, color=(255, 60, 60)) for name in human_joints
    ]
    calibrated = server.gui.add_checkbox("calibrated (g1_cali)", True)

    i = 0
    while True:
        # --- retarget: both presets every frame; the checkbox picks which result to show ---
        t0 = time.time()
        results = {
            name: helper.solve_numpy(human_arrays[name][None, i], human_heights)[0] for name, helper in helpers.items()
        }
        active = "calibrated" if calibrated.value else "raw"
        qpos = results[active]

        # --- viewer: visualize the selected result ---
        with server.atomic():
            base.position, base.wxyz = qpos[:3], qpos[3:7]
            urdf_vis.update_cfg(qpos[7:])
            for j, handle in enumerate(keypoints):
                handle.position = human_arrays[active][i, j, :3]
        i = (i + 1) % len(human_arrays[active])
        if i == 0:
            for helper in helpers.values():
                helper.reset()
        time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))


if __name__ == "__main__":
    main()
