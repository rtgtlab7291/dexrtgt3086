"""IK gallery: pick a fixed-base robot, set instance count, tune every weight live."""

import importlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IK, IKConfig
from robokit.robo import Robot, RobotState
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rest_task import RestTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.utils.visualize_utils import ViserBatchUrdf


# kind: "urdf" (robot_descriptions URDF), "mjcf" (robot_descriptions MJCF module)
# links: end-effector frames to task; one draggable target gizmo is created per frame.
ROBOT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # --- arms: single frame ---
    "panda (urdf)": {"kind": "urdf", "ref": "panda_description", "links": ["panda_hand"]},
    "panda (mjcf)": {"kind": "mjcf", "ref": "panda_mj_description", "links": ["hand"]},
    "ur5e (mjcf)": {"kind": "mjcf", "ref": "ur5e_mj_description", "links": ["wrist_3_link"]},
    "iiwa14 (urdf)": {"kind": "urdf", "ref": "iiwa14_description", "links": ["iiwa_link_ee"]},
    "iiwa14 (mjcf)": {"kind": "mjcf", "ref": "iiwa14_mj_description", "links": ["link7"]},
    "xarm7 (mjcf)": {"kind": "mjcf", "ref": "xarm7_mj_description", "links": ["link7"]},
    "gen3 (mjcf)": {"kind": "mjcf", "ref": "gen3_mj_description", "links": ["bracelet_link"]},
    "fr3 (mjcf)": {"kind": "mjcf", "ref": "fr3_mj_description", "links": ["fr3_link7"]},
    # --- dual arms: two frames ---
    "yumi (urdf, dual)": {"kind": "urdf", "ref": "yumi_description", "links": ["gripper_l_base", "gripper_r_base"]},
    "aloha (mjcf, dual)": {
        "kind": "mjcf",
        "ref": "aloha_mj_description",
        "links": ["left/gripper_base", "right/gripper_base"],
    },
    # --- dexterous hand: four fingertips ---
    # position_only: fingertip orientation is over-constrained on a few-DOF finger, so task position only.
    "allegro (urdf, hand)": {
        "kind": "urdf",
        "ref": "allegro_hand_description",
        "links": ["link_15_tip", "link_3_tip", "link_7_tip", "link_11_tip"],
        "position_only": True,
    },
    "allegro (mjcf, hand)": {
        "kind": "mjcf",
        "ref": "allegro_hand_mj_description",
        "links": ["th_tip", "ff_tip", "mf_tip", "rf_tip"],
        "position_only": True,
    },
}

GRID_COLS = 6
GRID_SPACING = 1.5
VEL_LIMIT_DT = 0.05

# (term name, default, min, max, step). Cost terms = what the LM optimizer minimizes per seed.
COST_SLIDERS: List[Tuple[str, float, float, float, float]] = [
    ("Position", 20.0, 0.0, 100.0, 0.5),
    ("Rotation", 10.0, 0.0, 100.0, 0.5),
    ("PositionLimit", 50.0, 0.0, 200.0, 0.5),
    ("Rest", 0.0, 0.0, 5.0, 0.05),
    ("SmoothCost", 0.1, 0.0, 5.0, 0.05),
    ("VelocityLimit", 0.0, 0.0, 10.0, 0.05),
]


def load_robot(spec: Dict[str, Any]) -> Robot:
    ref = spec["ref"]
    src: Any = (
        load_robot_description(ref)
        if spec["kind"] == "urdf"
        else importlib.import_module(f"robot_descriptions.{ref}").MJCF_PATH
    )
    return Robot.load(src, load_meshes=True)


@dataclass
class Scene:
    robot_name: str
    batch_size: int
    robot: Robot
    ik: IK
    batch_urdf: ViserBatchUrdf
    transform_controls: List[Any]
    T_world_base_grid: np.ndarray
    per_robot_offset: np.ndarray  # (batch, 3) deterministic per-instance target offset, base frame
    ground_z: float
    rest_state: RobotState
    out_state: RobotState
    prev_state: Optional[RobotState] = None


def build_scene(
    server: viser.ViserServer,
    robot_name: str,
    batch_size: int,
    weights: Dict[str, float],
    device: str,
    prior: Optional[Scene],
) -> Scene:
    if prior is not None:
        prior.batch_urdf.remove()
        for ctrl in prior.transform_controls:
            ctrl.remove()

    spec = ROBOT_CONFIGS[robot_name]
    robot = load_robot(spec)
    links = spec["links"]

    # --- IK ---
    # Cost-only: the multi-seed solver ranks candidates by total cost (no separate score terms).
    # Hands task position only: a few-DOF fingertip can't hold a dragged position and an orientation
    # at once, so rotation would just pull it short of target.
    rot_w = 0.0 if spec.get("position_only", False) else weights["Rotation"]
    config = (
        IKConfig()
        .add(PositionTask(weight=weights["Position"]), name="Position")
        .add(RotationTask(weight=rot_w), name="Rotation")
        .add(PositionLimit(weight=weights["PositionLimit"]), name="PositionLimit")
        .add(RestTask(weight=weights["Rest"]), name="Rest")
        .add(SmoothnessTask(weight=weights["SmoothCost"]), name="SmoothCost")
        .add(VelocityLimitTask(dt=VEL_LIMIT_DT, weight=weights["VelocityLimit"]), name="VelocityLimit")
    )
    ik = IK(config, robot=robot, link=links, device=device)
    ik.warmup(batch_size=batch_size)

    cols = min(GRID_COLS, batch_size)
    num_rows = (batch_size + cols - 1) // cols
    T_world_base = np.zeros((batch_size, 7), dtype=np.float32)
    T_world_base[:, 3] = 1.0
    for i in range(batch_size):
        row, col = i // cols, i % cols
        T_world_base[i, 0] = -col * GRID_SPACING
        T_world_base[i, 1] = (row - (num_rows - 1) / 2) * GRID_SPACING

    # deterministic per-instance target offset (like 03_batch) so the instances reach slightly
    # different targets instead of all identical. Position only, centered, bounded near the gizmo.
    per_robot_offset = np.zeros((batch_size, 3), dtype=np.float32)
    if batch_size > 1:
        idx = np.arange(batch_size)
        per_robot_offset[:, 0] = 0.05 * (idx % cols - (cols - 1) / 2.0)
        per_robot_offset[:, 1] = 0.05 * (idx // cols - (num_rows - 1) / 2.0)

    # Rest pose: the robot's natural zero pose (clipped to limits), so arms/hands look natural.
    # Its FK places one gizmo per tasked frame and lifts the base so the lowest link rests at z=0.
    jl = robot.spec.actuated_joint_limits
    q_rest = np.clip(0.0, jl[:, 0], jl[:, 1]).astype(np.float32)
    rest = robot.forward_kinematics(robot.state(q=wp.from_numpy(q_rest[None], dtype=wp.float32, device=device)))
    link_poses = rest.T_world_link.numpy()[0]  # (num_links, 7), base at origin
    ground_z = -float(link_poses[:, 2].min())
    T_world_base[:, 2] = ground_z

    # --- viewer ---
    # One draggable gizmo per tasked frame is the target (like 01_basic); the robot moving to it
    # is the feedback, so no extra target axes are drawn.
    transform_controls: List[Any] = []
    for link, link_idx in zip(links, ik.target_link_indices):
        node = link.replace("/", "_")
        pos = link_poses[link_idx, :3] + np.array([0.0, 0.0, ground_z], dtype=np.float32)
        if batch_size > 1:
            # Push the shared gizmo out in front of the grid (offset scales with reach) so it's grabbable.
            reach = float(np.linalg.norm(link_poses[link_idx, :3]))
            pos = pos + np.array([0.45 * reach, 0.0, -0.45 * reach], dtype=np.float32)
        transform_controls.append(
            server.scene.add_transform_controls(
                f"/target/{node}", scale=0.15, position=pos, wxyz=link_poses[link_idx, 3:7]
            )
        )

    batch_urdf = ViserBatchUrdf(target=server, robot=robot, batch_size=batch_size, root_node_name="/robots")
    out_state = robot.state(
        q=wp.empty((batch_size, ik.num_joints), dtype=wp.float32, device=device)  # type: ignore[arg-type]
    )
    # Rest state seeds the solve and anchors untasked (null-space) joints to the rest pose.
    rest_q = np.broadcast_to(q_rest, (batch_size, ik.num_joints)).copy()
    rest_state = robot.state(q=wp.from_numpy(rest_q, dtype=wp.float32, device=device))

    return Scene(
        robot_name=robot_name,
        batch_size=batch_size,
        robot=robot,
        ik=ik,
        batch_urdf=batch_urdf,
        transform_controls=transform_controls,
        T_world_base_grid=T_world_base,
        per_robot_offset=per_robot_offset,
        ground_z=ground_z,
        rest_state=rest_state,
        out_state=out_state,
        prev_state=None,
    )


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=20, height=20, cell_size=1.0)

    robot_dropdown = server.gui.add_dropdown("Robot", options=list(ROBOT_CONFIGS), initial_value="panda (urdf)")
    instances_slider = server.gui.add_slider("Instances", min=1, max=64, step=1, initial_value=1)
    weight_handles: Dict[str, Any] = {}
    with server.gui.add_folder("Cost weights"):
        for name, default, lo, hi, step in COST_SLIDERS:
            weight_handles[name] = server.gui.add_slider(f"w: {name}", min=lo, max=hi, step=step, initial_value=default)
    # seed spread for the multi-seed solver, as a fraction of each joint's travel (live)
    sample_range_slider = server.gui.add_slider("Sample range", min=0.0, max=1.0, step=0.01, initial_value=1.0)
    timing = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    scene: Optional[Scene] = None

    def push_weight(name: str):
        # Push one slider into the live IK only when it moves. Hands skip rotation (always 0).
        if scene is None:
            return
        pos_only = ROBOT_CONFIGS[scene.robot_name].get("position_only", False)
        if name == "Rotation" and pos_only:
            return
        scene.ik.set_weight(name, float(weight_handles[name].value))

    for name, handle in weight_handles.items():
        handle.on_update(lambda _, n=name: push_weight(n))

    while True:
        # Rebuild whenever the live selection differs from what's built (loop reads widgets directly).
        robot_name = robot_dropdown.value
        batch_size = int(instances_slider.value)
        if scene is None or robot_name != scene.robot_name or batch_size != scene.batch_size:
            weights = {name: float(handle.value) for name, handle in weight_handles.items()}
            scene = build_scene(server, robot_name, batch_size, weights, device, scene)

        assert scene is not None
        start = time.time()

        # --- IK: solve ---
        # Weights are pushed by slider on_update callbacks, not every frame.
        # Undo the base lift to get the base-frame target; each instance gets a deterministic offset.
        ground_offset = np.array([0.0, 0.0, scene.ground_z], dtype=np.float32)
        positions: List[np.ndarray] = []
        wxyzs: List[np.ndarray] = []
        for ctrl in scene.transform_controls:
            wxyz = np.asarray(ctrl.wxyz, dtype=np.float32)
            wxyz /= np.linalg.norm(wxyz) + 1e-12
            base_pos = np.asarray(ctrl.position, dtype=np.float32) - ground_offset
            positions.append(base_pos[None] + scene.per_robot_offset)  # (batch, 3)
            wxyzs.append(np.tile(wxyz, (scene.batch_size, 1)))  # (batch, 4)

        # Warm-start from the previous solution, else the rest pose.
        init_state = scene.prev_state if scene.prev_state is not None else scene.rest_state
        scene.ik.solve_numpy(
            np.stack([np.concatenate([p, q], axis=-1) for p, q in zip(positions, wxyzs)], axis=-2),
            init_state=init_state,
            rest_state=scene.rest_state,
            prev_state=scene.prev_state,
            out_state=scene.out_state,
            init_sample_range=float(sample_range_slider.value),
        )
        wp.synchronize()

        elapsed_ms = (time.time() - start) * 1000.0
        timing.value = 0.99 * timing.value + 0.01 * elapsed_ms

        # --- viewer: render ---
        q_np = scene.out_state.q.numpy()
        scene.batch_urdf.update_cfg(q_np, T_world_base=scene.T_world_base_grid)

        scene.prev_state = scene.robot.state(q=wp.clone(scene.out_state.q))

        time.sleep(0.001)


if __name__ == "__main__":
    main()
