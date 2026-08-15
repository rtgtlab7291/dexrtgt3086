"""Shared construction for source-specific interaction retargeting presets."""

from typing import Dict

import numpy as np

from robokit.helpers.humanoid_retarget.config import (
    HumanoidRetargetingOnlineConfig,
    InteractionConfig,
    LinkMapping,
)
from robokit.opt.multi_seed_solver import StageConfig


def _interaction_preset(
    link_map: Dict[str, str], robot_height: float, collision_spheres_path: str, urdf_path: str
) -> HumanoidRetargetingOnlineConfig:
    """Convert an ordered source-joint map into the shared online configuration."""
    human_root_name = next(iter(link_map))
    return HumanoidRetargetingOnlineConfig(
        urdf_path=urdf_path,
        robot_height=robot_height,
        human_root_name=human_root_name,
        collision_spheres_path=collision_spheres_path,
        scene_collision_weight=200.0,
        scene_collision_margin=0.012,
        link_mapping={
            robot_link: LinkMapping(
                human_joint,
                0.0,
                0.0,
                np.zeros(3, dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            for human_joint, robot_link in link_map.items()
        },
        interaction=InteractionConfig(),
        position_limit_weight=30.0,
        rest_weight=0.0,
        smoothness_weight=20.0,
        base_smoothness_weight=1.0,
        velocity_limit_weight=0.0,
        velocity_clamp_scale=0.0,
        cuda_graph_mode="none",
        position_limit_mode="abs",
        stages=[StageConfig(num_seeds=1, iters=30, lm_lambda=1.0)],
    )
