import hashlib
import io
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import trimesh
import yaml
import yourdfpy
from jaxtyping import Bool, Float, Int
from yourdfpy.urdf import filename_handler_null

from robokit.robo.yourdfpy import URDFWrapper
from robokit.xform.numpy import matrix_to_quaternion, quaternion_to_matrix, rot_tl_to_tf_mat


logger = logging.getLogger("robokit")


class JointType(IntEnum):
    FIXED = 0
    """Fixed joint, no motion."""
    PRISMATIC = 1
    """Prismatic joint, translation along a single axis."""
    REVOLUTE = 2
    """Revolute joint, rotation around a single axis."""


@dataclass
class RobotSpec:
    """Robot specification.

    Examples:
        >>> spec = RobotSpec.parse("assets/robot_description/wx250s.urdf")
        >>> spec.num_actuated_joints
        8
        >>> spec.num_links
        14
    """

    # ------------------------------ general specification ------------------------------
    name: str
    """Name of the robot."""
    robot_description: str
    """Robot description"""
    format: Literal["urdf", "mjcf"]
    """Format of the robot description."""

    # ------------------------------ joint specification ------------------------------
    joint_names: List[str]
    """Names of all joints, including actuated, mimic and fixed joints."""
    nonfixed_joint_names: List[str]
    """Names of non-fixed joints, including actuated and mimic joints."""
    actuated_joint_names: List[str]
    """Names of actuated joints."""
    joint_types: Int[np.ndarray, "num_joints"]
    """Types of all joints, see `JointType`."""
    joint_axes: Float[np.ndarray, "num_joints 3"]
    """Axes of all joints, [x, y, z]."""
    joint_twists: Float[np.ndarray, "num_joints 6"]
    """Twists of all joints, [linear (x, y, z), angular (x, y, z)]."""
    actuated_joint_indices: Int[np.ndarray, "num_joints"]
    """Indices of actuated joints, -1 for mimic and non-actuated joints."""
    parent_joint_indices: Int[np.ndarray, "num_joints"]
    """Indices of parent joints, -1 for joints attached to the base link."""
    parent_joint_transforms: Float[np.ndarray, "num_joints 7"]
    """Transforms of parent joints, [pos (x, y, z), quat (w, x, y, z)]."""

    mimic_actuated_joint_indices: Int[np.ndarray, "num_joints"]
    """Indices of mimicked actuated joints, -1 for non-mimic joints."""
    mimic_multipliers: Float[np.ndarray, "num_joints"]
    """Multipliers of mimicked actuated joints, 1.0 for non-mimic joints."""
    mimic_offsets: Float[np.ndarray, "num_joints"]
    """Offsets of mimicked actuated joints, 0.0 for non-mimic joints."""

    joint_limits: Float[np.ndarray, "num_joints 2"]
    """Limits of all joints, [lower, upper]."""
    joint_velocity_limits: Float[np.ndarray, "num_joints"]
    """Velocity limits of all joints."""
    actuated_joint_limits: Float[np.ndarray, "num_actuated_joints 2"]
    """Limits of actuated joints, [lower, upper]."""
    actuated_joint_velocity_limits: Float[np.ndarray, "num_actuated_joints"]
    """Velocity limits of actuated joints."""

    # ------------------------------ link specification ------------------------------
    link_names: List[str]
    """Names of all links."""
    link_parent_joint_indices: Int[np.ndarray, "num_links"]
    """Indices of parent joints of all links, -1 for the base link."""

    # ------------------------------ geometry specification ------------------------------
    link_visual_geometries: Dict[str, trimesh.Scene]
    """Visual geometries of all links."""
    link_collision_geometries: Dict[str, trimesh.Scene]
    """Collision geometries of all links."""

    local_collision_sphere_centers: Float[np.ndarray, "num_collision_spheres 3"]
    """Collision sphere centers in link-local coordinates."""
    local_collision_sphere_radii: Float[np.ndarray, "num_collision_spheres"]
    """Collision sphere radii."""
    collision_spheres_link_indices: Int[np.ndarray, "num_collision_spheres"]
    """Link indices (in link_names) for each collision sphere."""

    local_contact_point_centers: Float[np.ndarray, "num_contact_points 3"]
    """Contact point centers in link-local coordinates."""
    contact_points_link_indices: Int[np.ndarray, "num_contact_points"]
    """Link indices (in link_names) for each contact point."""

    # ------------------------------ parse funcs ------------------------------
    @staticmethod
    def parse(
        robot_description_or_path: Union[str, Path, yourdfpy.URDF],
        load_meshes: bool = False,
        mesh_dir: Optional[Union[str, Path]] = None,
        load_collision_spheres: bool = False,
        collision_spheres_path: Optional[str] = None,
        load_contact_points: bool = False,
        contact_points_path: Optional[str] = None,
        base_link_name: Optional[str] = None,
        ee_link_names: Optional[List[str]] = None,
    ) -> "RobotSpec":
        format: Literal["urdf", "mjcf"]

        if isinstance(robot_description_or_path, yourdfpy.URDF):
            format = "urdf"
        elif isinstance(robot_description_or_path, Path):
            if not robot_description_or_path.exists():
                raise FileNotFoundError(f"Robot description file not found: {robot_description_or_path}")
            suffix = robot_description_or_path.suffix.lower()
            if suffix == ".urdf":
                format = "urdf"
            elif suffix == ".mjcf":
                format = "mjcf"
            else:
                raise ValueError(f"Unsupported file extension: {suffix}")
        elif isinstance(robot_description_or_path, str):
            if "<robot" in robot_description_or_path:
                format = "urdf"
            elif "<mujoco" in robot_description_or_path:
                format = "mjcf"
            else:
                path = Path(robot_description_or_path)
                if not path.exists():
                    raise FileNotFoundError(f"Robot description file not found: {robot_description_or_path}")
                suffix = path.suffix.lower()
                if suffix == ".urdf":
                    format = "urdf"
                elif suffix == ".mjcf":
                    format = "mjcf"
                else:
                    raise ValueError(f"Unsupported file extension: {suffix}")
        else:
            raise ValueError(f"Invalid robot description type: {type(robot_description_or_path)}")

        if format == "urdf":
            return RobotSpec.parse_urdf(
                robot_description_or_path,
                load_meshes=load_meshes,
                mesh_dir=mesh_dir,
                load_collision_spheres=load_collision_spheres,
                collision_spheres_path=collision_spheres_path,
                load_contact_points=load_contact_points,
                contact_points_path=contact_points_path,
                base_link_name=base_link_name,
                ee_link_names=ee_link_names,
            )
        elif format == "mjcf":
            return RobotSpec.parse_mjcf(robot_description_or_path, load_meshes=load_meshes)  # type: ignore
        else:
            raise ValueError(f"Unsupported robot description format: {format}")

    @staticmethod
    def parse_urdf(
        robot_description_or_path: Union[str, Path, yourdfpy.URDF],
        load_meshes: bool = False,
        mesh_dir: Optional[Union[str, Path]] = None,
        load_collision_spheres: bool = False,
        collision_spheres_path: Optional[str] = None,
        load_contact_points: bool = False,
        contact_points_path: Optional[str] = None,
        base_link_name: Optional[str] = None,
        ee_link_names: Optional[List[str]] = None,
    ) -> "RobotSpec":
        # helper functions
        def get_joint_type(joint: yourdfpy.Joint) -> JointType:
            if joint.type in ("revolute", "continuous"):
                return JointType.REVOLUTE
            elif joint.type == "prismatic":
                return JointType.PRISMATIC
            elif joint.type == "fixed":
                return JointType.FIXED
            else:
                logger.warning(
                    f"Unsupported joint type {joint.type} for joint '{joint.name}'. Falling back to fixed joint."
                )
                return JointType.FIXED

        def get_joint_axis(joint: yourdfpy.Joint) -> Float[np.ndarray, "3"]:
            if joint.type in ("revolute", "continuous", "prismatic"):
                return joint.axis
            elif joint.type == "fixed":
                return np.zeros(3)
            else:
                logger.warning(
                    f"Unsupported joint type {joint.type} for joint '{joint.name}'. Falling back to zero axis."
                )
                return np.zeros(3)

        def get_joint_twist(joint: yourdfpy.Joint) -> Float[np.ndarray, "6"]:
            if joint.type in ("revolute", "continuous"):
                return np.concatenate([np.zeros(3), joint.axis])
            elif joint.type == "prismatic":
                return np.concatenate([joint.axis, np.zeros(3)])
            elif joint.type == "fixed":
                return np.zeros(6)
            else:
                logger.warning(
                    f"Unsupported joint type {joint.type} for joint '{joint.name}'. Falling back to fixed joint."
                )
                return np.zeros(6)

        def get_actuated_joint_index(urdf: yourdfpy.URDF, joint: yourdfpy.Joint) -> int:
            if joint.mimic is not None:
                return -1
            elif joint.name in urdf.actuated_joint_names:
                return urdf.actuated_joint_names.index(joint.name)
            else:
                return -1

        def get_mimic_joint_info(urdf: yourdfpy.URDF, joint: yourdfpy.Joint) -> Tuple[int, float, float]:
            mimic_actuated_joint_index, mimic_multiplier, mimic_offset = -1, 1.0, 0.0
            if joint.mimic is not None:
                if joint.mimic.multiplier is not None:
                    mimic_multiplier = joint.mimic.multiplier
                if joint.mimic.offset is not None:
                    mimic_offset = joint.mimic.offset
                mimicked_joint = urdf.joint_map[joint.mimic.joint]
                mimic_actuated_joint_index = urdf.actuated_joint_names.index(mimicked_joint.name)
            return mimic_actuated_joint_index, mimic_multiplier, mimic_offset

        def get_joint_limits(joint: yourdfpy.Joint) -> Tuple[float, float]:
            if joint.limit is not None and joint.limit.lower is not None and joint.limit.upper is not None:
                return joint.limit.lower, joint.limit.upper
            elif joint.type in ("continuous", "revolute"):
                logger.warning(
                    f"Joint '{joint.name}' ({joint.type}) has no specified limits. Falling back to [-pi, pi]."
                )
                return -math.pi, math.pi
            elif joint.type == "prismatic":
                logger.warning(
                    f"Joint '{joint.name}' ({joint.type}) has no specified limits. Falling back to [-1.0, 1.0]."
                )
                return -1.0, 1.0
            elif joint.type == "fixed":
                return 0.0, 0.0
            else:
                raise ValueError(f"Joint '{joint.name}' ({joint.type}) has no specified limits.")

        def get_joint_velocity_limit(joint: yourdfpy.Joint) -> float:
            if joint.limit is not None and joint.limit.velocity is not None:
                return joint.limit.velocity
            elif joint.type == "fixed":
                return 0.0
            else:
                logger.warning(
                    f"Joint '{joint.name}' ({joint.type}) has no specified velocity limit. Falling back to 10.0."
                )
                return 10.0

        def get_parent_joint_info(
            urdf: yourdfpy.URDF,
            joint: yourdfpy.Joint,
        ) -> Tuple[int, Float[np.ndarray, "7"]]:
            parent_joint_transform = np.eye(4)
            if joint.origin is not None:
                parent_joint_transform = joint.origin

            child_link_to_joint = {j.child: j for j in urdf.joint_map.values()}
            joint_name_to_idx = {joint_name: i for i, joint_name in enumerate(urdf.joint_names)}

            if joint.parent not in child_link_to_joint:
                parent_index = -1
            else:
                parent_joint = child_link_to_joint[joint.parent]
                parent_index = joint_name_to_idx[parent_joint.name]

            parent_joint_quat = matrix_to_quaternion(parent_joint_transform[:3, :3])
            parent_joint_pos = parent_joint_transform[:3, 3]
            parent_joint_transform = np.concatenate([parent_joint_pos, parent_joint_quat], axis=-1)
            return parent_index, parent_joint_transform

        def get_link_geometry(
            urdf: yourdfpy.URDF, link_name: str, mode: Literal["visual", "collision"]
        ) -> trimesh.Scene:
            source_scene = urdf.scene if mode == "visual" else urdf.collision_scene
            link_scene = trimesh.Scene()
            if source_scene is None or (link_name not in source_scene.graph.nodes):
                return link_scene

            children = source_scene.graph.transforms.children.get(link_name, [])

            for child_node in children:
                if child_node in source_scene.geometry:
                    geometry = source_scene.geometry[child_node]
                    transform = source_scene.graph.get(child_node, frame_from=link_name)[0]
                    link_scene.add_geometry(geometry, transform=transform)

            return link_scene

        def load_collision_spheres_from_yaml(
            collision_spheres_path: str, robot_link_names: List[str]
        ) -> Tuple[
            Float[np.ndarray, "num_collision_spheres 3"],
            Float[np.ndarray, "num_collision_spheres"],
            Int[np.ndarray, "num_collision_spheres"],
        ]:
            with open(collision_spheres_path, "r") as f:
                sphere_data = yaml.safe_load(f)
            collision_spheres_dict = sphere_data.get("collision_spheres", {})

            all_centers, all_radii, all_link_indices = [], [], []
            for link_name, spheres in collision_spheres_dict.items():
                if link_name not in robot_link_names:
                    continue
                link_idx = robot_link_names.index(link_name)
                for sphere in spheres:
                    center, radius = sphere["center"], sphere["radius"]
                    all_centers.append([center[0], center[1], center[2]])
                    all_radii.append(radius)
                    all_link_indices.append(link_idx)

            local_collision_sphere_centers = (
                np.array(all_centers, dtype=np.float32) if all_centers else np.empty((0, 3), dtype=np.float32)
            )
            local_collision_sphere_radii = (
                np.array(all_radii, dtype=np.float32) if all_radii else np.empty(0, dtype=np.float32)
            )
            collision_spheres_link_indices = (
                np.array(all_link_indices, dtype=np.int32) if all_link_indices else np.empty(0, dtype=np.int32)
            )
            return local_collision_sphere_centers, local_collision_sphere_radii, collision_spheres_link_indices

        def load_contact_points_from_json(
            contact_points_path: str, robot_link_names: List[str]
        ) -> Tuple[
            Float[np.ndarray, "num_contact_points 3"],
            Int[np.ndarray, "num_contact_points"],
        ]:
            with open(contact_points_path, "r") as f:
                contact_points_dict = json.load(f)

            all_centers, all_link_indices = [], []
            for link_name, points in contact_points_dict.items():
                if link_name not in robot_link_names:
                    continue
                link_idx = robot_link_names.index(link_name)
                for point in points:
                    all_centers.append([point[0], point[1], point[2]])
                    all_link_indices.append(link_idx)

            local_contact_point_centers = (
                np.array(all_centers, dtype=np.float32) if all_centers else np.empty((0, 3), dtype=np.float32)
            )
            contact_points_link_indices = (
                np.array(all_link_indices, dtype=np.int32) if all_link_indices else np.empty(0, dtype=np.int32)
            )
            return local_contact_point_centers, contact_points_link_indices

        if isinstance(robot_description_or_path, yourdfpy.URDF):
            urdf = robot_description_or_path
        else:
            if isinstance(robot_description_or_path, Path) or os.path.isfile(robot_description_or_path):
                source = robot_description_or_path
            elif isinstance(robot_description_or_path, str):
                source = io.BytesIO(robot_description_or_path.encode())
            else:
                raise ValueError(f"Invalid robot description type: {type(robot_description_or_path)}")
            kwargs: Dict[str, Any] = {
                "build_scene_graph": load_meshes,
                "build_collision_scene_graph": load_meshes,
                "load_meshes": load_meshes,
                "load_collision_meshes": load_meshes,
            }
            if mesh_dir is not None:
                kwargs["mesh_dir"] = str(mesh_dir)
            logging.getLogger("yourdfpy").setLevel(logging.ERROR)
            urdf = yourdfpy.URDF.load(source, **kwargs)

        # filter links and joints based on base_link_name and/or ee_link_names
        if base_link_name is not None or ee_link_names is not None:
            child_to_parent = {j.child: j.parent for j in urdf.robot.joints}
            parent_to_children: Dict[str, List[str]] = defaultdict(list)
            for j in urdf.robot.joints:
                parent_to_children[j.parent].append(j.child)

            filtered_links = {l.name for l in urdf.robot.links}

            if base_link_name is not None:
                # keep only descendants of base_link_name
                descendants, stack = set(), [base_link_name]
                while stack:
                    link = stack.pop()
                    descendants.add(link)
                    stack.extend(parent_to_children.get(link, []))
                filtered_links &= descendants

            if ee_link_names is not None:
                # keep only ancestors of ee_link_names (up to base_link_name or root)
                ancestors = {base_link_name} if base_link_name else set()
                for ee in ee_link_names:
                    current: Optional[str] = ee
                    while current is not None and current != base_link_name:
                        ancestors.add(current)
                        current = child_to_parent.get(current)
                filtered_links &= ancestors

            urdf.robot.links = [l for l in urdf.robot.links if l.name in filtered_links]
            urdf.robot.joints = [
                j for j in urdf.robot.joints if j.parent in filtered_links and j.child in filtered_links
            ]
            urdf._create_maps()
            urdf._update_actuated_joints()

        # collect joint specification
        joint_names, actuated_joint_names = urdf.joint_names, urdf.actuated_joint_names
        nonfixed_joint_names = []
        joint_types, joint_axes = [], []
        joint_twists, actuated_joint_indices = [], []
        mimic_actuated_joint_indices, mimic_multipliers, mimic_offsets = [], [], []
        joint_limits, joint_velocity_limits = [], []
        parent_joint_indices, parent_joint_transforms = [], []
        for joint_name in joint_names:
            joint = urdf.joint_map[joint_name]
            joint_types.append(get_joint_type(joint))
            joint_axes.append(get_joint_axis(joint))
            joint_twists.append(get_joint_twist(joint))
            actuated_joint_indices.append(get_actuated_joint_index(urdf, joint))
            mimic_actuated_joint_index, mimic_multiplier, mimic_offset = get_mimic_joint_info(urdf, joint)
            mimic_actuated_joint_indices.append(mimic_actuated_joint_index)
            mimic_multipliers.append(mimic_multiplier)
            mimic_offsets.append(mimic_offset)
            joint_limits.append(get_joint_limits(joint))
            joint_velocity_limits.append(get_joint_velocity_limit(joint))
            parent_joint_index, parent_joint_transform = get_parent_joint_info(urdf, joint)
            parent_joint_indices.append(parent_joint_index)
            parent_joint_transforms.append(parent_joint_transform)
            if joint.type != "fixed":
                nonfixed_joint_names.append(joint_name)

        actuated_joint_limits = [get_joint_limits(urdf.joint_map[joint_name]) for joint_name in actuated_joint_names]
        actuated_joint_velocity_limits = [
            get_joint_velocity_limit(urdf.joint_map[joint_name]) for joint_name in actuated_joint_names
        ]

        # collect link specification
        link_parent_joint_indices = []
        child_link_to_joint = {j.child: j for j in urdf.joint_map.values()}
        link_names = list(urdf.link_map.keys())
        for link_name in link_names:
            if link_name in child_link_to_joint:
                joint_idx = joint_names.index(child_link_to_joint[link_name].name)
                link_parent_joint_indices.append(joint_idx)
            else:
                link_parent_joint_indices.append(-1)

        # collect geometry specification
        if load_meshes:
            link_visual_geometries = {
                link_name: get_link_geometry(urdf, link_name, "visual") for link_name in link_names
            }
            link_collision_geometries = {
                link_name: get_link_geometry(urdf, link_name, "collision") for link_name in link_names
            }
        else:
            link_visual_geometries, link_collision_geometries = {}, {}
        if load_collision_spheres:
            if collision_spheres_path is None:
                raise ValueError("Collision spheres path must be provided when loading collision spheres.")
            local_collision_sphere_centers, local_collision_sphere_radii, collision_spheres_link_indices = (
                load_collision_spheres_from_yaml(collision_spheres_path, link_names)
            )
        else:
            local_collision_sphere_centers, local_collision_sphere_radii, collision_spheres_link_indices = (
                np.empty((0, 3), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32),
            )
        if load_contact_points:
            if contact_points_path is None:
                raise ValueError("Contact points path must be provided when loading contact points.")
            local_contact_point_centers, contact_points_link_indices = load_contact_points_from_json(
                contact_points_path, link_names
            )
        else:
            local_contact_point_centers, contact_points_link_indices = (
                np.empty((0, 3), dtype=np.float32),
                np.empty(0, dtype=np.int32),
            )

        # NOTE different from the original yourdfpy, we add mimic joint writing in the URDFWrapper
        urdf.__class__ = URDFWrapper
        if not mesh_dir:
            urdf._filename_handler = filename_handler_null
        return RobotSpec(
            name=urdf.robot.name,
            robot_description=urdf.write_xml_string().decode(),
            format="urdf",
            joint_names=joint_names,
            nonfixed_joint_names=nonfixed_joint_names,
            actuated_joint_names=actuated_joint_names,
            joint_types=np.asarray(joint_types, dtype=np.int32),
            joint_axes=np.stack(joint_axes),
            joint_twists=np.stack(joint_twists),
            actuated_joint_indices=np.asarray(actuated_joint_indices),
            parent_joint_indices=np.asarray(parent_joint_indices),
            parent_joint_transforms=np.stack(parent_joint_transforms),
            mimic_actuated_joint_indices=np.asarray(mimic_actuated_joint_indices),
            mimic_multipliers=np.asarray(mimic_multipliers),
            mimic_offsets=np.asarray(mimic_offsets),
            joint_limits=np.asarray(joint_limits),
            joint_velocity_limits=np.asarray(joint_velocity_limits),
            actuated_joint_limits=np.asarray(actuated_joint_limits),
            actuated_joint_velocity_limits=np.asarray(actuated_joint_velocity_limits),
            link_names=link_names,
            link_parent_joint_indices=np.asarray(link_parent_joint_indices),
            link_visual_geometries=link_visual_geometries,
            link_collision_geometries=link_collision_geometries,
            local_collision_sphere_centers=local_collision_sphere_centers,
            local_collision_sphere_radii=local_collision_sphere_radii,
            collision_spheres_link_indices=collision_spheres_link_indices,
            local_contact_point_centers=local_contact_point_centers,
            contact_points_link_indices=contact_points_link_indices,
        )

    @staticmethod
    def parse_mjcf(robot_description_or_path: Union[str, Path], load_mesh: bool) -> "RobotSpec":  # type: ignore
        pass

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_actuated_joints(self) -> int:
        return len(self.actuated_joint_names)

    @property
    def num_dofs(self) -> int:
        return self.num_actuated_joints

    @property
    def num_nonfixed_joints(self) -> int:
        return len(self.nonfixed_joint_names)

    @property
    def num_links(self) -> int:
        return len(self.link_names)

    @property
    def zero_q(self) -> Float[np.ndarray, "num_actuated_joints"]:
        return np.zeros(self.num_actuated_joints, dtype=np.float32)

    @property
    def midrange_q(self) -> Float[np.ndarray, "num_actuated_joints"]:
        return 0.5 * (self.actuated_joint_limits[:, 0] + self.actuated_joint_limits[:, 1])

    @cached_property
    def parent_joint_transforms_matrix(self) -> Float[np.ndarray, "num_joints 4 4"]:
        """Transforms of parent joints as 4x4 matrices."""
        return rot_tl_to_tf_mat(
            quaternion_to_matrix(self.parent_joint_transforms[:, 3:]), self.parent_joint_transforms[:, :3]
        )

    @cached_property
    def has_mimic_joints(self) -> bool:
        return bool(np.any(self.mimic_actuated_joint_indices >= 0))

    @cached_property
    def mimic_joint_names(self) -> List[str]:
        """Names of mimic joints."""
        return [
            self.joint_names[joint_idx]
            for joint_idx in range(self.num_joints)
            if self.mimic_actuated_joint_indices[joint_idx] >= 0
        ]

    @cached_property
    def has_collision_spheres(self) -> bool:
        return len(self.local_collision_sphere_centers) > 0 and len(self.collision_spheres_link_indices) > 0

    @cached_property
    def has_contact_points(self) -> bool:
        return len(self.local_contact_point_centers) > 0 and len(self.contact_points_link_indices) > 0

    @cached_property
    def topological_order_joint_indices(self) -> Int[np.ndarray, "num_joints"]:
        dependencies = [[] for _ in range(self.num_joints)]
        in_degree = [0] * self.num_joints

        for joint_idx, parent_idx in enumerate(self.parent_joint_indices):
            if parent_idx >= 0:
                dependencies[parent_idx].append(joint_idx)
                in_degree[joint_idx] += 1
        for joint_idx, mimic_idx in enumerate(self.mimic_actuated_joint_indices):
            if mimic_idx >= 0:
                dependencies[mimic_idx].append(joint_idx)
                in_degree[joint_idx] += 1

        topological_order = []
        queue = [joint_idx for joint_idx in range(self.num_joints) if in_degree[joint_idx] == 0]

        while queue:
            joint_idx = queue.pop(0)
            topological_order.append(joint_idx)
            for dependent_idx in dependencies[joint_idx]:
                in_degree[dependent_idx] -= 1
                if in_degree[dependent_idx] == 0:
                    queue.append(dependent_idx)

        if len(topological_order) != self.num_joints:
            raise ValueError("Circular dependency detected in joint relationships")

        return np.asarray(topological_order, dtype=np.int32)

    @cached_property
    def kinematic_tree_str(self) -> str:
        joint_to_child_link_indices = {pj: li for li, pj in enumerate(self.link_parent_joint_indices) if pj >= 0}
        base_links = [li for li, pj in enumerate(self.link_parent_joint_indices) if pj < 0]
        link_to_child_joint_indices = defaultdict(list)
        for j in range(self.num_joints):
            pj = int(self.parent_joint_indices[j])
            if pj >= 0:
                parent_link = joint_to_child_link_indices.get(pj)
            else:
                parent_link = base_links[0] if base_links else None
            if parent_link is None:
                continue
            link_to_child_joint_indices[parent_link].append(j)

        def _joint_type_str(twist: Float[np.ndarray, "6"]) -> str:
            if np.allclose(twist, 0.0):
                return "F"
            elif np.any(twist[3:] != 0):
                return "R"
            elif np.any(twist[:3] != 0):
                return "P"
            return "?"

        def _chain_str(link_idx: int, prefix: str = "", is_last: bool = True) -> str:
            link_name = self.link_names[link_idx]
            parent_joint_idx = int(self.link_parent_joint_indices[link_idx])
            current_prefix = "└──" if is_last else "├──"
            next_prefix = prefix + ("  " if is_last else "│ ")
            if parent_joint_idx >= 0:
                joint_name = self.joint_names[parent_joint_idx]
                joint_type = _joint_type_str(self.joint_twists[parent_joint_idx])
                chain = f"{prefix}{current_prefix} {link_name} ({joint_name}, {joint_type})\n"
            else:
                chain = f"{prefix}{current_prefix} {link_name}\n"
            for i, cj in enumerate(link_to_child_joint_indices.get(link_idx, [])):
                is_last_child = i == len(link_to_child_joint_indices[link_idx]) - 1
                child_link_idx = joint_to_child_link_indices.get(cj)
                if child_link_idx is None:
                    continue
                chain += _chain_str(child_link_idx, next_prefix, is_last_child)
            return chain

        kin_tree_str = ""
        for i, base_link in enumerate(base_links):
            is_last = i == len(base_links) - 1
            kin_tree_str += _chain_str(base_link, "", is_last)
        return kin_tree_str

    @cached_property
    def joints_to_actuated_mapping(self) -> Float[np.ndarray, "num_joints num_actuated_joints"]:
        """Mapping from all joints to actuated joints."""
        mapping = np.zeros((self.num_joints, self.num_actuated_joints), dtype=np.float32)
        for joint_idx in range(self.num_joints):
            actuated_idx = self.actuated_joint_indices[joint_idx]
            mimic_actuated_idx = self.mimic_actuated_joint_indices[joint_idx]
            if actuated_idx >= 0:
                mapping[joint_idx, actuated_idx] = 1.0
            elif mimic_actuated_idx >= 0:
                multiplier = self.mimic_multipliers[joint_idx]
                mapping[joint_idx, mimic_actuated_idx] = multiplier
        return mapping

    @cached_property
    def nonfixed_joint_mask(self) -> Bool[np.ndarray, "num_joints"]:
        """Boolean mask for non-fixed joints in all joints."""
        mask = np.zeros(self.num_joints, dtype=bool)
        for nonfixed_name in self.nonfixed_joint_names:
            mask[self.joint_names.index(nonfixed_name)] = True
        return mask

    @cached_property
    def link_ancestor_joints_mask(self) -> Bool[np.ndarray, "num_links num_joints"]:
        """Boolean mask for ancestor joints of links."""
        mask = np.zeros((self.num_links, self.num_joints), dtype=bool)
        for link_idx in range(self.num_links):
            j = int(self.link_parent_joint_indices[link_idx])
            while j != -1:
                mask[link_idx, j] = True
                j = int(self.parent_joint_indices[j])
        return mask

    def get_link_mesh(
        self, link_name: str, mode: Literal["visual", "collision"] = "collision", return_mesh: bool = True
    ) -> Union[trimesh.Scene, trimesh.Trimesh]:
        if not self.link_visual_geometries or not self.link_collision_geometries:
            raise ValueError("Meshes are not loaded. Please set load_meshes=True when loading the robot.")
        scene = (
            self.link_visual_geometries[link_name] if mode == "visual" else self.link_collision_geometries[link_name]
        )
        if return_mesh:
            return scene.to_mesh()
        else:
            return scene

    def __repr__(self) -> str:
        return (
            f"RobotSpec(name={self.name}, num_actuated_joints={self.num_actuated_joints}, num_links={self.num_links})\n"
            + self.kinematic_tree_str
        )

    def __str__(self) -> str:
        return self.__repr__()

    @cached_property
    def _spec_hash(self) -> int:
        hasher = hashlib.sha256()
        hasher.update(self.robot_description.encode())
        hasher.update(self.local_collision_sphere_centers.tobytes())
        hasher.update(self.local_collision_sphere_radii.tobytes())
        hasher.update(self.collision_spheres_link_indices.tobytes())
        hasher.update(self.local_contact_point_centers.tobytes())
        hasher.update(self.contact_points_link_indices.tobytes())
        return int(hasher.hexdigest(), 16)

    def __hash__(self) -> int:
        return self._spec_hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RobotSpec):
            return False
        return self._spec_hash == other._spec_hash
