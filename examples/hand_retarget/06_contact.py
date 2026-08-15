"""Offline xArm7-XHand contact retargeting around an object mesh."""

import time
from dataclasses import replace

import numpy as np
import trimesh
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets.robots.arms import xarm7_xhand
from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.hand_retargeting import HandRetargetingOffline
from robokit.helpers.hand_retargeting.dexycb_utils import load_dexycb_grasp
from robokit.helpers.hand_retargeting.presets.xarm7_xhand import offline, spec
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_apply, quaternion_invert, quaternion_multiply


BASE_POSE = np.array([0.9, 0.9, 0.0, 0.70710678, 0.0, 0.0, -0.70710678], np.float32)
CONTACT_START_DISTANCE = 0.020
CONTACT_FULL_DISTANCE = 0.006
CONTACT_EMBED_DEPTH = 0.005
CONTACT_MARKER_RADIUS = 0.004


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    robot = Robot.load(
        str(xarm7_xhand.URDF_PATH),
        load_meshes=True,
        load_collision_spheres=True,
        collision_spheres_path=xarm7_xhand.COLLISION_SPHERE_PATH,
    )
    data = load_dexycb_grasp(0, device=device)
    keypoints = data["keypoints"]
    object_pos = data["object_pos"]
    object_quat_wxyz = data["object_quat_wxyz"]
    num_frames = len(keypoints)
    contact_target_indices = np.asarray(
        [spec.target_names.index(name) for name in spec.contact_target_names], dtype=np.int32
    )
    contact_link_indices = np.asarray(
        [robot.spec.link_names.index(spec.target_link_names[name]) for name in spec.contact_target_names],
        dtype=np.int32,
    )
    q_wxyz_object_world = quaternion_invert(object_quat_wxyz)
    keypoints_object = quaternion_apply(q_wxyz_object_world[:, None], keypoints - object_pos[:, None]).astype(
        np.float32
    )
    fingertips_object = keypoints_object[:, contact_target_indices]
    closest, distance, _ = trimesh.proximity.closest_point(data["object_mesh"], fingertips_object.reshape(-1, 3))
    contact_points_object = closest.reshape(fingertips_object.shape).astype(np.float32)
    contact_mask = distance.reshape(fingertips_object.shape[:2]) < CONTACT_START_DISTANCE

    # --- direct contact-aware retargeting in the fixed robot-base frame ---
    q_wxyz_base_world = quaternion_invert(BASE_POSE[3:])
    keypoints_base = quaternion_apply(q_wxyz_base_world, keypoints - BASE_POSE[:3]).astype(np.float32)
    contact_points_world = object_pos[:, None] + quaternion_apply(object_quat_wxyz[:, None], contact_points_object)
    contact_points_base = quaternion_apply(q_wxyz_base_world, contact_points_world - BASE_POSE[:3]).astype(np.float32)
    direct = HandRetargetingOffline(robot, spec, replace(offline, collision_weight=0.0), device=device)
    packed = direct.solve_with_contact_numpy(
        keypoints_base,
        contact_points_base,
        contact_mask,
    )

    # --- anchored local-contact refinement against a static object-frame scene ---
    T_object_base = np.empty((num_frames, 7), dtype=np.float32)
    T_object_base[:, :3] = quaternion_apply(q_wxyz_object_world, BASE_POSE[:3] - object_pos)
    T_object_base[:, 3:] = quaternion_multiply(q_wxyz_object_world, BASE_POSE[3:])
    init_state = robot.state(
        q=wp.from_numpy(packed[None, :, 7:], dtype=wp.float32, device=device),
        T_world_base=wp.from_numpy(T_object_base[None], dtype=wp_vec7, device=device),
    )
    robot.forward_kinematics(init_state)
    T_object_link = init_state.T_world_link.numpy()[0][:, contact_link_indices]
    contact_outward = fingertips_object - contact_points_object
    contact_outward /= np.maximum(np.linalg.norm(contact_outward, axis=-1, keepdims=True), 1e-8)
    contact_inward_local = quaternion_apply(quaternion_invert(T_object_link[..., 3:]), -contact_outward)
    local_contact_points = np.zeros_like(contact_points_object)
    for contact_index, link_index in enumerate(contact_link_indices):
        link_mesh = robot.spec.link_visual_geometries[robot.spec.link_names[link_index]].to_mesh()
        vertices = np.asarray(link_mesh.vertices, np.float32)
        local_contact_points[:, contact_index] = vertices[
            np.argmax(contact_inward_local[:, contact_index] @ vertices.T, axis=1)
        ]

    root_link_index = robot.spec.link_names.index(spec.target_link_names[spec.root_target_name])
    arm_joint_mask = robot.spec.link_ancestor_joints_mask[root_link_index]
    arm_dof_mask = np.any(robot.spec.joints_to_actuated_mapping[arm_joint_mask] != 0.0, axis=0)
    hand_joint_names = tuple(np.asarray(robot.spec.actuated_joint_names)[~arm_dof_mask])
    identity = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
    scene = WarpScene(1, device).add(MeshGeom([data["object_mesh"]], np.array([0, 1], dtype=np.int32), poses=identity))
    refine_config = replace(
        offline,
        contact_weight=300.0,
        contact_margin=0.0,
        anchor_q_weight=10.0,
        acceleration_weight=20.0,
        collision_weight=100.0,
        collision_margin=0.0,
        optimize_base=False,
        active_joint_names=hand_joint_names,
        locked_prefix_frames=2,
    )
    contact_points_robot = T_object_link[..., :3] + quaternion_apply(T_object_link[..., 3:], local_contact_points)
    contact_strength = np.clip(
        (CONTACT_START_DISTANCE - distance.reshape(contact_mask.shape))
        / (CONTACT_START_DISTANCE - CONTACT_FULL_DISTANCE),
        0.0,
        1.0,
    ).astype(np.float32)
    contact_strength = contact_strength * contact_strength * (3.0 - 2.0 * contact_strength)
    contact_points_target = contact_points_object - CONTACT_EMBED_DEPTH * contact_outward
    contact_points_target = contact_points_robot + contact_strength[..., None] * (
        contact_points_target - contact_points_robot
    )
    refine = HandRetargetingOffline(robot, spec, refine_config, scene=scene, device=device)
    packed = refine.solve_with_contact_numpy(
        keypoints_object,
        contact_points_target,
        contact_mask,
        init_state=init_state,
        local_contact_points=local_contact_points,
    )
    joints, T_object_base = packed[:, 7:], packed[:, :7]
    T_world_base = np.empty_like(T_object_base)
    T_world_base[:, :3] = object_pos + quaternion_apply(object_quat_wxyz, T_object_base[:, :3])
    T_world_base[:, 3:] = quaternion_multiply(object_quat_wxyz, T_object_base[:, 3:])
    print("contact retargeting of %d frames (%d touching)" % (num_frames, int(contact_mask.any(1).sum())))

    # --- viewer: show the grasp and active contacts ---
    server = viser.ViserServer()
    server.scene.add_grid("/grid", 1.0, 1.0, position=(0.5, 0.5, 0.0))
    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, xarm7_xhand.URDF_PATH, root_node_name="/robot")
    object_frame = server.scene.add_frame("/object", show_axes=False)
    server.scene.add_mesh_simple(
        "/object/mesh",
        data["object_mesh"].vertices,
        data["object_mesh"].faces,
        color=(180, 180, 190),
        opacity=0.6,
    )
    mano = server.scene.add_mesh_simple(
        "/mano", data["mano_vertices"][0], data["mano_faces"], color=(120, 200, 130), opacity=0.35
    )
    spheres = [
        server.scene.add_icosphere("/contact_%d" % i, radius=CONTACT_MARKER_RADIUS, color=(255, 40, 40))
        for i in range(len(spec.contact_target_names))
    ]
    playing = server.gui.add_checkbox("Play", True)
    frame_slider = server.gui.add_slider("Frame", 0, num_frames - 1, 1, 0)

    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % num_frames
        f = frame_slider.value
        with server.atomic():
            robot_frame.position, robot_frame.wxyz = T_world_base[f, :3], T_world_base[f, 3:]
            urdf_vis.update_cfg(joints[f])
            object_frame.position, object_frame.wxyz = object_pos[f], object_quat_wxyz[f]
            mano.vertices = data["mano_vertices"][f]
            for i, sphere in enumerate(spheres):
                sphere.visible = bool(contact_mask[f, i])
                sphere.position = contact_points_world[f, i]
        time.sleep(0.03)


if __name__ == "__main__":
    main()
