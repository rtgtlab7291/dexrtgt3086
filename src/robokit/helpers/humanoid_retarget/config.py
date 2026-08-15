"""Shared humanoid retargeting configuration dataclasses for online mappings and offline optimizer settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from robokit.opt.multi_seed_solver import StageConfig


if TYPE_CHECKING:
    from robokit.robo import Robot


@dataclass
class LinkMapping:
    human_joint: str
    position_weight: float
    orientation_weight: float
    position_offset: np.ndarray
    rotation_offset: np.ndarray


@dataclass
class CorrespondenceEdge:
    """Drive the robot link vector toward the corresponding human joint vector each frame."""

    origin_link: str  # robot link
    task_link: str  # robot link
    origin_joint: str  # human joint (drives the per-frame target)
    task_joint: str  # human joint
    weight: float = 1.0


@dataclass
class InteractionConfig:
    """Interaction-mesh settings for contact-aware online retargeting."""

    laplacian_weight: float = 100.0
    num_object_points: int = 100


@dataclass
class TargetArrays:
    """Per-target inputs to `compute_targets_kernel`, resolved from a config's `link_mapping`."""

    joint_indices: np.ndarray  # (N,) index into the caller's human joint order
    scales: np.ndarray  # (N,) per-target scale from scale_table
    rotation_offsets: np.ndarray  # (N, 4) normalized wxyz
    position_offsets: np.ndarray  # (N, 3)
    root_joint_index: int
    root_target_index: int  # -1 when the root is not itself a mapped target
    root_scale: float


def _default_retarget_stages() -> List[StageConfig]:
    return [
        StageConfig(num_seeds=16, iters=4, lm_lambda=10.0),
        StageConfig(num_seeds=4, iters=6, lm_lambda=1.0),
        StageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
    ]


@dataclass
class HumanoidRetargetingOnlineConfig:
    urdf_path: Optional[str] = None
    robot_height: Optional[float] = None

    human_root_name: str = "pelvis"
    human_height_assumption: float = 1.8
    ground_height: float = 0.0

    collision_spheres_path: Optional[Union[str, Path]] = None
    scene_collision_weight: float = 0.0
    scene_collision_margin: float = 0.02

    # self-collision penalty over the collision spheres; weight > 0 needs collision_spheres_path
    self_collision_weight: float = 0.0
    self_collision_margin: float = 0.01
    self_collision_max_active_pairs: int = 50

    scale_table: Dict[str, float] = field(default_factory=dict)

    link_mapping: Dict[str, LinkMapping] = field(default_factory=dict)

    # optional vector-pair correspondence objective (None = off)
    correspondence_edges: Optional[List[CorrespondenceEdge]] = None
    interaction: Optional[InteractionConfig] = None

    position_limit_weight: float = 30.0
    rest_weight: Union[float, Sequence[float]] = 0.0
    smoothness_weight: Union[float, Sequence[float]] = 1.0
    base_smoothness_weight: Union[float, Sequence[float]] = 1.5

    velocity_limit_weight: float = 3.0
    velocity_limit_dt: float = 1.0 / 30.0
    velocity_clamp_scale: float = 0.6  # clamp output at this fraction of URDF velocity limits; 0 = disabled
    velocity_limit_override: Optional[Sequence[Union[int, float]]] = (
        None  # per-joint velocity limits (rad/s); overrides URDF
    )

    cuda_graph_mode: Literal["none", "full", "iter"] = "full"

    # joint-limit residual mode; per-DOF mask weights in [0, 1]
    position_limit_mode: Literal["abs", "sqrt_abs", "exp_barrier"] = "exp_barrier"
    per_dof_limit_mask: Optional[Dict[str, float]] = None

    # ramp the limit penalty 0 -> full over this many frames (0 = disabled)
    limit_warmup_frames: int = 0

    stages: List[StageConfig] = field(default_factory=_default_retarget_stages)

    @property
    def human_joint_names(self) -> Tuple[str, ...]:
        """Return the ordered human joints expected by the solve APIs.

        Example:
            >>> import numpy as np
            >>> config = HumanoidRetargetingOnlineConfig(
            ...     link_mapping={"torso": LinkMapping("spine", 1.0, 1.0, np.zeros(3), np.zeros(4))},
            ... )
            >>> config.human_joint_names  # mapped joints first, then the root
            ('spine', 'pelvis')
        """
        mapping_joints = [entry.human_joint for entry in self.link_mapping.values()]
        edge_joints = [
            joint for edge in self.correspondence_edges or [] for joint in (edge.origin_joint, edge.task_joint)
        ]
        return tuple(dict.fromkeys([*mapping_joints, self.human_root_name, *edge_joints]))

    @property
    def uses_collision_spheres(self) -> bool:
        """Return whether an enabled online term needs robot collision spheres.

        Example:
            >>> HumanoidRetargetingOnlineConfig().uses_collision_spheres
            False
            >>> HumanoidRetargetingOnlineConfig(self_collision_weight=100.0).uses_collision_spheres
            True
        """
        return self.scene_collision_weight > 0 or self.self_collision_weight > 0

    def target_arrays(self, human_joint_names: Sequence[str]) -> "TargetArrays":
        """Resolve `link_mapping` into the per-target arrays both solvers feed to `compute_targets_kernel`.

        `human_joint_names` is the caller's input joint order (see `human_joint_names`).

        Example:
            >>> import numpy as np
            >>> config = HumanoidRetargetingOnlineConfig(
            ...     link_mapping={
            ...         "torso": LinkMapping("spine", 1.0, 1.0, np.zeros(3), np.zeros(4)),
            ...         "hip": LinkMapping("pelvis", 1.0, 0.0, np.zeros(3), np.zeros(4)),
            ...     },
            ...     scale_table={"spine": 0.9},
            ... )
            >>> arrays = config.target_arrays(config.human_joint_names)
            >>> arrays.joint_indices.tolist(), arrays.scales.tolist()
            ([0, 1], [0.8999999761581421, 1.0])
            >>> arrays.root_joint_index, arrays.root_target_index
            (1, 1)
            >>> arrays.rotation_offsets[0].tolist()  # zero offset normalizes to identity
            [1.0, 0.0, 0.0, 0.0]
        """
        mapping_joints = [entry.human_joint for entry in self.link_mapping.values()]
        joint_index = {name: i for i, name in enumerate(human_joint_names)}
        rotation_offsets = np.zeros((len(mapping_joints), 4), dtype=np.float32)
        rotation_offsets[:, 0] = 1.0
        position_offsets = np.zeros((len(mapping_joints), 3), dtype=np.float32)
        for i, entry in enumerate(self.link_mapping.values()):
            norm = float(np.linalg.norm(entry.rotation_offset))
            if norm > 1e-6:
                rotation_offsets[i] = entry.rotation_offset / norm
            if np.linalg.norm(entry.position_offset) > 1e-6:
                position_offsets[i] = entry.position_offset
        return TargetArrays(
            joint_indices=np.array([joint_index[name] for name in mapping_joints], dtype=np.int32),
            scales=np.array([self.scale_table.get(name, 1.0) for name in mapping_joints], dtype=np.float32),
            rotation_offsets=rotation_offsets,
            position_offsets=position_offsets,
            root_joint_index=joint_index[self.human_root_name],
            root_target_index=(
                mapping_joints.index(self.human_root_name) if self.human_root_name in mapping_joints else -1
            ),
            root_scale=float(self.scale_table.get(self.human_root_name, 1.0)),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "HumanoidRetargetingOnlineConfig":
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        urdf_path: Optional[str] = None
        if "urdf_path" in data:
            from robokit.assets import fetch

            urdf_path = str(fetch(["robots/**"]) / "robots" / data["urdf_path"])

        config = cls(
            urdf_path=urdf_path,
            robot_height=float(data["robot_height"]) if "robot_height" in data else None,
            human_root_name=data.get("human_root_name", "pelvis"),
            human_height_assumption=float(data.get("human_height_assumption", 1.8)),
            ground_height=float(data.get("ground_height", 0.0)),
        )

        config.scale_table = {k: float(v) for k, v in data.get("scale_table", {}).items()}

        for robot_link, entry in data.get("link_mapping", {}).items():
            config.link_mapping[robot_link] = LinkMapping(
                human_joint=entry["human_joint"],
                position_weight=float(entry["position_weight"]),
                orientation_weight=float(entry["orientation_weight"]),
                position_offset=np.array(entry["position_offset"], dtype=np.float32),
                rotation_offset=np.array(entry["rotation_offset"], dtype=np.float32),
            )

        for key in (
            "position_limit_weight",
            "velocity_limit_weight",
            "velocity_limit_dt",
            "velocity_clamp_scale",
            "scene_collision_weight",
            "scene_collision_margin",
            "self_collision_weight",
            "self_collision_margin",
        ):
            if key in data:
                setattr(config, key, float(data[key]))
        if "collision_spheres_path" in data:
            config.collision_spheres_path = data["collision_spheres_path"]
        if "self_collision_max_active_pairs" in data:
            config.self_collision_max_active_pairs = int(data["self_collision_max_active_pairs"])
        for key in ("smoothness_weight", "base_smoothness_weight", "rest_weight"):  # scalar or per-DOF list
            if key in data:
                val = data[key]
                setattr(config, key, np.array(val, dtype=np.float32) if isinstance(val, list) else float(val))
        if "velocity_limit_override" in data:
            config.velocity_limit_override = data["velocity_limit_override"]
        if "cuda_graph_mode" in data:
            config.cuda_graph_mode = str(data["cuda_graph_mode"])
        if "position_limit_mode" in data:
            config.position_limit_mode = str(data["position_limit_mode"])
        if "per_dof_limit_mask" in data:
            config.per_dof_limit_mask = {k: float(v) for k, v in data["per_dof_limit_mask"].items()}
        if "limit_warmup_frames" in data:
            config.limit_warmup_frames = int(data["limit_warmup_frames"])
        if "correspondence_edges" in data:
            config.correspondence_edges = [
                CorrespondenceEdge(
                    origin_link=e["origin_link"],
                    task_link=e["task_link"],
                    origin_joint=e["origin_joint"],
                    task_joint=e["task_joint"],
                    weight=float(e["weight"]),
                )
                for e in data["correspondence_edges"]
            ]
        if "interaction" in data:
            config.interaction = InteractionConfig(**data["interaction"])
        if "stages" in data:
            config.stages = [
                StageConfig(
                    num_seeds=int(s["num_seeds"]),
                    iters=int(s["iters"]),
                    lm_lambda=float(s.get("lm_lambda", 10.0)),
                )
                for s in data["stages"]
            ]

        return config

    def _weight_to_yaml(self, val: Union[float, int, Sequence[float], np.ndarray]) -> Union[float, List[float]]:
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (list, tuple)):
            return list(val)
        return float(val)  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        data: dict = {}
        if self.urdf_path is not None:
            data["urdf_path"] = self.urdf_path
        if self.robot_height is not None:
            data["robot_height"] = self.robot_height
        if self.collision_spheres_path is not None:
            data["collision_spheres_path"] = str(self.collision_spheres_path)
        data.update(
            {
                "human_root_name": self.human_root_name,
                "human_height_assumption": self.human_height_assumption,
                "ground_height": self.ground_height,
                "scale_table": self.scale_table,
                "link_mapping": {
                    robot_link: {
                        "human_joint": m.human_joint,
                        "position_weight": m.position_weight,
                        "orientation_weight": m.orientation_weight,
                        "position_offset": m.position_offset.tolist(),
                        "rotation_offset": m.rotation_offset.tolist(),
                    }
                    for robot_link, m in self.link_mapping.items()
                },
                "position_limit_weight": self.position_limit_weight,
                "smoothness_weight": self._weight_to_yaml(self.smoothness_weight),
                "base_smoothness_weight": self._weight_to_yaml(self.base_smoothness_weight),
                "rest_weight": self._weight_to_yaml(self.rest_weight),
                "velocity_limit_weight": self.velocity_limit_weight,
                "velocity_limit_dt": self.velocity_limit_dt,
                "velocity_clamp_scale": self.velocity_clamp_scale,
                "scene_collision_weight": self.scene_collision_weight,
                "scene_collision_margin": self.scene_collision_margin,
                "self_collision_weight": self.self_collision_weight,
                "self_collision_margin": self.self_collision_margin,
                "self_collision_max_active_pairs": self.self_collision_max_active_pairs,
                "cuda_graph_mode": self.cuda_graph_mode,
                "position_limit_mode": self.position_limit_mode,
                "stages": [{"num_seeds": s.num_seeds, "iters": s.iters, "lm_lambda": s.lm_lambda} for s in self.stages],
            }
        )
        if self.velocity_limit_override is not None:
            data["velocity_limit_override"] = self._weight_to_yaml(
                np.asarray(self.velocity_limit_override)
                if not isinstance(self.velocity_limit_override, (float, int))
                else self.velocity_limit_override
            )
        if self.per_dof_limit_mask:
            data["per_dof_limit_mask"] = self.per_dof_limit_mask
        if self.limit_warmup_frames > 0:
            data["limit_warmup_frames"] = self.limit_warmup_frames
        if self.correspondence_edges:
            data["correspondence_edges"] = [
                {
                    "origin_link": e.origin_link,
                    "task_link": e.task_link,
                    "origin_joint": e.origin_joint,
                    "task_joint": e.task_joint,
                    "weight": e.weight,
                }
                for e in self.correspondence_edges
            ]
        if self.interaction is not None:
            data["interaction"] = {
                "laplacian_weight": self.interaction.laplacian_weight,
                "num_object_points": self.interaction.num_object_points,
            }
        return data

    def to_yaml(self, yaml_path: Union[str, Path]):
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def load_robot(self, load_meshes: bool = False) -> "Robot":
        """Load the config's robot and collision spheres required by enabled terms."""
        from robokit.robo import Robot

        assert self.urdf_path is not None
        return Robot.load(
            str(self.urdf_path),
            load_meshes=load_meshes,
            load_collision_spheres=self.uses_collision_spheres,
            collision_spheres_path=self.collision_spheres_path,
        )


@dataclass
class HumanoidRetargetingOfflineConfig:
    # Frame-tracking weights come from the online config's `link_mapping`, not from here.
    # Self-collision adds every sphere pair at every frame to the sparse system; enable only when needed.
    self_collision_weight: float = 0.0
    self_collision_margin: float = 0.01
    joint_smoothness_weight: Union[float, List[float]] = 20.0
    root_position_smoothness_weight: float = 10.0
    root_orientation_smoothness_weight: float = 5.0
    limit_weight: float = 100.0
    rest_weight: Union[float, List[float]] = 1.0
    base_rest_weight: Optional[Union[float, List[float]]] = None
    velocity_limit_override: Optional[List[float]] = None
    velocity_limit_weight: float = 0.0
    velocity_limit_dt: float = 1.0 / 30.0
    velocity_clamp_scale: float = 0.0
    lm_lambda: float = 0.5
    max_iter: int = 25
    lambda_factor: float = 2.0
    lambda_min: float = 1e-6
    lambda_max: float = 1e6
    rho_min: float = 1e-4
    cuda_graph_mode: Literal["none", "full", "iter"] = "full"

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "HumanoidRetargetingOfflineConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        config = cls()
        list_or_scalar_keys = {"rest_weight", "joint_smoothness_weight"}
        for key in (
            "self_collision_weight",
            "self_collision_margin",
            "joint_smoothness_weight",
            "root_position_smoothness_weight",
            "root_orientation_smoothness_weight",
            "limit_weight",
            "rest_weight",
            "velocity_limit_weight",
            "velocity_limit_dt",
            "velocity_clamp_scale",
            "lm_lambda",
            "max_iter",
            "lambda_factor",
            "lambda_min",
            "lambda_max",
            "rho_min",
        ):
            if key not in data:
                continue
            val = data[key]
            if key in list_or_scalar_keys and isinstance(val, list):
                setattr(config, key, val)
            else:
                setattr(config, key, type(getattr(config, key))(val))
        if "velocity_limit_override" in data:
            config.velocity_limit_override = data["velocity_limit_override"]
        if "cuda_graph_mode" in data:
            config.cuda_graph_mode = str(data["cuda_graph_mode"])
        return config

    def to_yaml(self, yaml_path: Union[str, Path]):
        data = {
            "self_collision_weight": self.self_collision_weight,
            "self_collision_margin": self.self_collision_margin,
            "joint_smoothness_weight": self.joint_smoothness_weight,
            "root_position_smoothness_weight": self.root_position_smoothness_weight,
            "root_orientation_smoothness_weight": self.root_orientation_smoothness_weight,
            "limit_weight": self.limit_weight,
            "rest_weight": self.rest_weight,
            "velocity_limit_weight": self.velocity_limit_weight,
            "velocity_limit_dt": self.velocity_limit_dt,
            "velocity_clamp_scale": self.velocity_clamp_scale,
            "lm_lambda": self.lm_lambda,
            "max_iter": self.max_iter,
            "lambda_factor": self.lambda_factor,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "rho_min": self.rho_min,
            "cuda_graph_mode": self.cuda_graph_mode,
        }
        if self.velocity_limit_override is not None:
            data["velocity_limit_override"] = self.velocity_limit_override
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


__all__ = [
    "CorrespondenceEdge",
    "HumanoidRetargetingOfflineConfig",
    "HumanoidRetargetingOnlineConfig",
    "InteractionConfig",
    "LinkMapping",
    "TargetArrays",
]
