"""Public hand-retargeting configuration and preset invariants."""

from dataclasses import is_dataclass, replace

import numpy as np
import warp as wp

from robokit.assets.robots.arms import xarm7_ability as xarm7_ability_asset
from robokit.assets.robots.arms import xarm7_mano as xarm7_mano_asset
from robokit.assets.robots.arms import xarm7_xhand as xarm7_xhand_asset
from robokit.assets.robots.hands import (
    ability_hand,
    allegro_hand,
    barrett_hand,
    dclaw_gripper,
    inspire_hand,
    leap_hand,
    paxini_hand,
    schunk_hand,
    shadow_hand,
    sharpa_hand,
)
from robokit.assets.robots.hands import xhand as xhand_asset
from robokit.helpers.hand_retargeting import HandRetargetingOffline, HandRetargetingOnline, config, presets
from robokit.robo import Robot


_PRESETS = (
    (
        "ability",
        presets.ability.spec,
        presets.ability.online,
        presets.ability.offline,
        ability_hand.URDF_PATH,
        ability_hand.COLLISION_SPHERE_PATH,
        None,
    ),
    (
        "ability_left",
        presets.ability.spec_left,
        presets.ability.online_left,
        None,
        ability_hand.URDF_PATH_LEFT,
        None,
        None,
    ),
    (
        "allegro",
        presets.allegro.spec,
        presets.allegro.online,
        presets.allegro.offline,
        allegro_hand.URDF_PATH,
        allegro_hand.COLLISION_SPHERE_PATH,
        None,
    ),
    (
        "allegro_left",
        presets.allegro.spec_left,
        presets.allegro.online_left,
        None,
        allegro_hand.URDF_PATH_LEFT,
        None,
        None,
    ),
    ("barrett", presets.barrett.spec, presets.barrett.online, None, barrett_hand.URDF_PATH, None, None),
    ("dclaw", presets.dclaw.spec, presets.dclaw.online, None, dclaw_gripper.URDF_PATH, None, None),
    (
        "inspire",
        presets.inspire.spec,
        presets.inspire.online,
        presets.inspire.offline,
        inspire_hand.URDF_PATH,
        inspire_hand.COLLISION_SPHERE_PATH,
        inspire_hand.SELF_COLLISION_IGNORE_PATH,
    ),
    (
        "inspire_left",
        presets.inspire.spec_left,
        presets.inspire.online_left,
        None,
        inspire_hand.URDF_PATH_LEFT,
        None,
        None,
    ),
    ("leap", presets.leap.spec, presets.leap.online, presets.leap.offline, leap_hand.URDF_PATH, None, None),
    ("leap_left", presets.leap.spec_left, presets.leap.online_left, None, leap_hand.URDF_PATH_LEFT, None, None),
    ("paxini", presets.paxini.spec, presets.paxini.online, presets.paxini.offline, paxini_hand.URDF_PATH, None, None),
    ("paxini_left", presets.paxini.spec_left, presets.paxini.online_left, None, paxini_hand.URDF_PATH_LEFT, None, None),
    (
        "shadow",
        presets.shadow.spec,
        presets.shadow.online,
        presets.shadow.offline,
        shadow_hand.URDF_PATH,
        shadow_hand.COLLISION_SPHERE_PATH,
        None,
    ),
    ("shadow_left", presets.shadow.spec_left, presets.shadow.online_left, None, shadow_hand.URDF_PATH_LEFT, None, None),
    (
        "sharpa",
        presets.sharpa.spec,
        presets.sharpa.online,
        presets.sharpa.offline,
        sharpa_hand.URDF_PATH,
        sharpa_hand.COLLISION_SPHERE_PATH,
        sharpa_hand.SELF_COLLISION_IGNORE_PATH,
    ),
    ("svh", presets.svh.spec, presets.svh.online, presets.svh.offline, schunk_hand.URDF_PATH, None, None),
    ("svh_left", presets.svh.spec_left, presets.svh.online_left, None, schunk_hand.URDF_PATH_LEFT, None, None),
    (
        "xhand",
        presets.xhand.spec,
        presets.xhand.online,
        presets.xhand.offline,
        xhand_asset.URDF_PATH,
        xhand_asset.COLLISION_SPHERE_PATH,
        xhand_asset.SELF_COLLISION_IGNORE_PATH,
    ),
    ("xhand_left", presets.xhand.spec_left, presets.xhand.online_left, None, xhand_asset.URDF_PATH_LEFT, None, None),
    (
        "xarm7_xhand",
        presets.xarm7_xhand.spec,
        presets.xarm7_xhand.online,
        presets.xarm7_xhand.offline,
        xarm7_xhand_asset.URDF_PATH,
        xarm7_xhand_asset.COLLISION_SPHERE_PATH,
        xarm7_xhand_asset.SELF_COLLISION_IGNORE_PATH,
    ),
    (
        "xarm7_ability",
        presets.xarm7_ability.spec,
        None,
        presets.xarm7_ability.offline,
        xarm7_ability_asset.URDF_PATH,
        xarm7_ability_asset.COLLISION_SPHERE_PATH,
        None,
    ),
    (
        "xarm7_ability_no_mimic",
        presets.xarm7_ability.spec_no_mimic,
        None,
        presets.xarm7_ability.offline_no_mimic,
        xarm7_ability_asset.URDF_NO_MIMIC_PATH,
        xarm7_ability_asset.COLLISION_SPHERE_PATH,
        None,
    ),
    (
        "xarm7_mano",
        presets.xarm7_mano.spec,
        None,
        presets.xarm7_mano.offline,
        xarm7_mano_asset.URDF_PATH,
        xarm7_mano_asset.COLLISION_SPHERE_PATH,
        None,
    ),
)


class TestHandRetargetingPresets:
    """Every built-in workflow shares one valid named robot specification."""

    def test_public_config_dataclasses(self) -> None:
        public_dataclasses = {
            name
            for name in config.__all__
            if is_dataclass(getattr(config, name)) and getattr(config, name).__module__ == config.__name__
        }
        assert public_dataclasses == {
            "HandRetargetingOfflineConfig",
            "HandRetargetingOnlineConfig",
            "HandSpec",
        }

    def test_all_named_topologies_resolve_and_solve(self) -> None:
        for name, spec, online, offline, urdf_path, collision_spheres_path, ignore_path in _PRESETS:
            robot = Robot.load(
                str(urdf_path),
                load_collision_spheres=collision_spheres_path is not None,
                collision_spheres_path=collision_spheres_path,
                self_collision_ignore_path=ignore_path,
            )
            links = set(robot.spec.link_names)
            joints = set(robot.spec.actuated_joint_names)
            assert set(spec.target_link_names.values()) <= links, name
            assert set(spec.contact_target_names) <= set(spec.target_link_names), name
            assert set(spec.init_q_by_name) <= joints, name
            assert set(spec.rest_q_by_name) <= joints, name

            if online is not None:
                correspondences = (*online.vector_weights, *online.direction_weights)
                assert {link for pair in correspondences for link in pair[:2]} <= links, name
                assert {target for pair in correspondences for target in pair[2:]} <= set(spec.target_names), name
                assert set(online.pinch_correspondences) <= set(online.vector_weights), name
                online_solver = replace(
                    online.solver,
                    stages=(replace(online.solver.stages[-1], num_seeds=1, iters=0),),
                    cuda_graph_mode="none",
                )
                online_helper = HandRetargetingOnline(
                    robot,
                    spec,
                    replace(online, solver=online_solver),
                    device="cpu",
                )
                online_helper.warmup(1)
                online_helper.solve(wp.zeros((1, len(spec.target_names), 3), dtype=wp.float32, device="cpu"))

            if offline is not None:
                offline_solver = replace(
                    offline.solver,
                    max_iter=0,
                    use_early_stopping=False,
                    use_cuda_graph=False,
                )
                offline_helper = HandRetargetingOffline(
                    robot, spec, replace(offline, solver=offline_solver), device="cpu"
                )
                offline_helper.warmup(1, 3)
                points = wp.zeros((1, 3, len(spec.target_names), 3), dtype=wp.float32, device="cpu")
                root_quat_wxyz = np.zeros((1, 3, 4), dtype=np.float32)
                root_quat_wxyz[:, :, 0] = 1.0
                offline_helper.solve(
                    points,
                    wp.from_numpy(root_quat_wxyz, dtype=wp.float32, device="cpu")
                    if offline.root_orientation_weight > 0
                    else None,
                )
