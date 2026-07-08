"""Humanoid retargeting helper."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp
import yaml
from scipy.spatial.transform import Rotation as R

from robokit.lie.warp_se3 import WarpSE3
from robokit.opt.warp_solver import WarpSolver, WarpSolverConfig, WarpStageConfig
from robokit.robo.warp_robot import WarpRobot, WarpRobotState
from robokit.terms import WarpFrameTask
from robokit.terms.terms import WarpTask
from robokit.terms.warp.position_limit import WarpPositionLimit
from robokit.terms.warp.rest_task import WarpRestTask
from robokit.terms.warp.smoothness_task import WarpSmoothnessTask
from robokit.terms.warp.velocity_limit_task import WarpVelocityLimitTask
from robokit.utils.warp_utils import repeat, wp_vec7


WarpFrameTaskType = WarpFrameTask


@dataclass
class LinkMapping:
    human_joint: str
    position_weight: float
    orientation_weight: float
    position_offset: np.ndarray
    rotation_offset: np.ndarray


def _default_retarget_stages() -> List[WarpStageConfig]:
    return [
        WarpStageConfig(num_seeds=16, iters=4, lm_lambda=10.0),
        WarpStageConfig(num_seeds=4, iters=6, lm_lambda=1.0),
        WarpStageConfig(num_seeds=1, iters=0, lm_lambda=1.0),
    ]


@dataclass
class RetargetConfig:
    urdf_path: Optional[str] = None

    human_root_name: str = "pelvis"
    robot_root_name: str = "pelvis"
    human_height_assumption: float = 1.8
    ground_height: float = 0.0

    scale_table: Dict[str, float] = field(default_factory=dict)

    link_mapping: Dict[str, LinkMapping] = field(default_factory=dict)

    position_limit_weight: float = 30.0
    rest_weight: Union[float, Sequence[float]] = 0.0
    smoothness_weight: Union[float, Sequence[float]] = 1.0
    base_smoothness_weight: Union[float, Sequence[float]] = 1.5

    # Adaptive smoothness: scale = max(min_scale, 1/(1 + prev_cost/threshold)).
    smoothness_cost_threshold: float = 50.0
    smoothness_min_scale: float = 0.3  # floor for adaptive smoothness; 0 = no floor, 1 = disable adaptive

    velocity_limit_weight: float = 3.0
    velocity_limit_dt: float = 1.0 / 30.0
    velocity_clamp_scale: float = 0.6  # clamp output at this fraction of URDF velocity limits; 0 = disabled
    velocity_limit_override: Optional[Sequence[float]] = None  # per-joint velocity limits (rad/s); overrides URDF

    use_cuda_graph: bool = True

    stages: List[WarpStageConfig] = field(default_factory=_default_retarget_stages)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RetargetConfig":
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        urdf_path: Optional[str] = None
        if "urdf_path" in data:
            resolved = (yaml_path.parent / data["urdf_path"]).resolve()
            urdf_path = str(resolved)

        config = cls(
            urdf_path=urdf_path,
            human_root_name=data.get("human_root_name", "pelvis"),
            robot_root_name=data.get("robot_root_name", "pelvis"),
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

        if "position_limit_weight" in data:
            config.position_limit_weight = float(data["position_limit_weight"])
        if "smoothness_weight" in data:
            val = data["smoothness_weight"]
            config.smoothness_weight = np.array(val, dtype=np.float32) if isinstance(val, list) else float(val)
        if "base_smoothness_weight" in data:
            val = data["base_smoothness_weight"]
            config.base_smoothness_weight = np.array(val, dtype=np.float32) if isinstance(val, list) else float(val)
        if "smoothness_cost_threshold" in data:
            config.smoothness_cost_threshold = float(data["smoothness_cost_threshold"])
        if "smoothness_min_scale" in data:
            config.smoothness_min_scale = float(data["smoothness_min_scale"])
        if "rest_weight" in data:
            val = data["rest_weight"]
            config.rest_weight = np.array(val, dtype=np.float32) if isinstance(val, list) else float(val)
        if "velocity_limit_weight" in data:
            config.velocity_limit_weight = float(data["velocity_limit_weight"])
        if "velocity_limit_dt" in data:
            config.velocity_limit_dt = float(data["velocity_limit_dt"])
        if "velocity_clamp_scale" in data:
            config.velocity_clamp_scale = float(data["velocity_clamp_scale"])
        if "velocity_limit_override" in data:
            config.velocity_limit_override = data["velocity_limit_override"]
        if "use_cuda_graph" in data:
            config.use_cuda_graph = bool(data["use_cuda_graph"])
        if "stages" in data:
            config.stages = [
                WarpStageConfig(
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

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        data = {}
        if self.urdf_path is not None:
            data["urdf_path"] = self.urdf_path
        data.update({
            "human_root_name": self.human_root_name,
            "robot_root_name": self.robot_root_name,
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
            "smoothness_cost_threshold": self.smoothness_cost_threshold,
            "smoothness_min_scale": self.smoothness_min_scale,
            "rest_weight": self._weight_to_yaml(self.rest_weight),
            "velocity_limit_weight": self.velocity_limit_weight,
            "velocity_limit_dt": self.velocity_limit_dt,
            "velocity_clamp_scale": self.velocity_clamp_scale,
            "stages": [{"num_seeds": s.num_seeds, "iters": s.iters, "lm_lambda": s.lm_lambda} for s in self.stages],
        })
        if self.velocity_limit_override is not None:
            data["velocity_limit_override"] = self._weight_to_yaml(
                np.asarray(self.velocity_limit_override)
                if not isinstance(self.velocity_limit_override, (float, int))
                else self.velocity_limit_override
            )
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def with_actual_height(self, actual_human_height: float) -> "RetargetConfig":
        # Scale_table was calibrated at human_height_assumption. For a different
        # height, SMPL-X positions are proportionally larger/smaller, so we
        # adjust scales inversely to keep targets at the robot's body positions.
        # Taller person → smaller scales (more shrinking), shorter → larger.
        # Matches neural_retargeting: height_scale = canonical / actual.
        ratio = self.human_height_assumption / actual_human_height
        new_config = RetargetConfig(
            urdf_path=self.urdf_path,
            human_root_name=self.human_root_name,
            robot_root_name=self.robot_root_name,
            human_height_assumption=self.human_height_assumption,
            ground_height=self.ground_height,
            scale_table={k: v * ratio for k, v in self.scale_table.items()},
            link_mapping=self.link_mapping.copy(),
            position_limit_weight=self.position_limit_weight,
            rest_weight=self.rest_weight,
            smoothness_weight=self.smoothness_weight,
            base_smoothness_weight=self.base_smoothness_weight,
            smoothness_cost_threshold=self.smoothness_cost_threshold,
            smoothness_min_scale=self.smoothness_min_scale,
            velocity_limit_weight=self.velocity_limit_weight,
            velocity_limit_dt=self.velocity_limit_dt,
            velocity_clamp_scale=self.velocity_clamp_scale,
            use_cuda_graph=self.use_cuda_graph,
            stages=self.stages,
            velocity_limit_override=self.velocity_limit_override,
        )
        return new_config


def _is_nonzero_weight(weight: Union[float, Sequence[float]]) -> bool:
    if isinstance(weight, (int, float)):
        return weight > 0
    return bool(np.any(weight))


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    result = r1 * r2
    xyzw = result.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.apply(v).astype(np.float32)


class HumanoidRetargetingHelper:
    def __init__(
        self,
        robot: WarpRobot,
        config: RetargetConfig,
        device: str = "cuda:0",
    ) -> None:
        self.robot = robot
        self.config = config
        self.device = wp.get_device(device)
        self.batch_size = 1

        self.link_name_to_index = {name: i for i, name in enumerate(self.robot.link_names)}
        self._validate_config()

        self.ground_offset = config.ground_height

        self._prev_q: Optional[np.ndarray] = None
        self._prev_cost: float = 0.0
        self._base_smoothness_weights: List[np.ndarray] = []

        (
            self._solver,
            self._frame_tasks,
            self._smoothness_tasks,
            self._velocity_tasks,
        ) = self._init_solver(self.config.link_mapping)

        self._prev_state: Optional[WarpRobotState] = None

        self._se3_tmp = np.zeros((1, 7), dtype=np.float32)

        self._target_buffers: List[wp.array] = []
        for _ in self.config.link_mapping:
            buf = wp.zeros((self.batch_size,), dtype=wp_vec7, device=self.device)
            self._target_buffers.append(buf)

    def _validate_config(self) -> None:
        for robot_link in self.config.link_mapping:
            if robot_link not in self.link_name_to_index:
                raise KeyError(f"Link '{robot_link}' not found in robot links.")
        stages = self.config.stages
        if len(stages) < 1:
            raise ValueError("stages must have at least 1 entry")
        if stages[-1].num_seeds != 1:
            raise ValueError("stages must end with num_seeds=1")

    def _init_solver(
        self, mapping: Dict[str, LinkMapping]
    ) -> Tuple[
        WarpSolver,
        List[WarpFrameTaskType],
        List[Optional[WarpSmoothnessTask]],
        List[Optional[WarpVelocityLimitTask]],
    ]:
        stages = self.config.stages
        wp_device = self.device

        frame_indices: List[int] = []
        position_weights: List[float] = []
        orientation_weights: List[float] = []
        for robot_link, entry in mapping.items():
            frame_indices.append(self.link_name_to_index[robot_link])
            position_weights.append(entry.position_weight)
            orientation_weights.append(entry.orientation_weight)

        placeholder_targets: List[WarpSE3] = []
        identity_pose = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        for _ in frame_indices:
            target = WarpSE3(wp.from_numpy(identity_pose, dtype=wp_vec7, device=wp_device))
            placeholder_targets.append(target)

        # Create single placeholder state; solver auto-creates per-stage vars via gather
        placeholder_q = wp.zeros((self.batch_size, self.robot.num_actuated_joints), dtype=wp.float32, device=wp_device)
        placeholder_base = WarpSE3.identity(shape=(self.batch_size,), device=wp_device)
        placeholder_state = self.robot.state(q=placeholder_q, T_world_base=placeholder_base)

        stage_terms: List[List[WarpTask]] = []
        stage_frame_tasks: List[WarpFrameTaskType] = []
        stage_smoothness_tasks: List[Optional[WarpSmoothnessTask]] = []
        stage_velocity_tasks: List[Optional[WarpVelocityLimitTask]] = []

        for stage_config in stages:
            total_batch = self.batch_size * stage_config.num_seeds

            terms_for_stage: List[WarpTask] = []
            frame_task = WarpFrameTask(
                robot=self.robot,
                frame_index=frame_indices,
                T_world_target=placeholder_targets,
                position_weight=position_weights,
                orientation_weight=orientation_weights,
                num_seeds=stage_config.num_seeds,
            )
            terms_for_stage.append(frame_task)
            stage_frame_tasks.append(frame_task)

            position_limit = WarpPositionLimit(
                robot=self.robot, weight=self.config.position_limit_weight, batch_size=total_batch
            )
            position_limit.init_buffers(wp_device)
            terms_for_stage.append(position_limit)

            if _is_nonzero_weight(self.config.rest_weight):
                rest_task = WarpRestTask(
                    robot=self.robot,
                    rest_q=self.robot.spec.midrange_q,
                    weight=self.config.rest_weight,
                    batch_size=total_batch,
                )
                rest_task.init_buffers(wp_device)
                terms_for_stage.append(rest_task)

            smoothness_task: Optional[WarpSmoothnessTask] = None
            if _is_nonzero_weight(self.config.smoothness_weight) or _is_nonzero_weight(
                self.config.base_smoothness_weight
            ):
                prev_base_placeholder = WarpSE3(repeat(placeholder_base.xyz_wxyz, self.batch_size))
                prev_q_placeholder = wp.zeros(
                    (self.batch_size, self.robot.num_actuated_joints), dtype=wp.float32, device=wp_device
                )
                prev_state_placeholder = self.robot.state(q=prev_q_placeholder, T_world_base=prev_base_placeholder)
                smoothness_task = WarpSmoothnessTask(
                    robot=self.robot,
                    prev_var=prev_state_placeholder,
                    weight=self.config.smoothness_weight,
                    base_weight=self.config.base_smoothness_weight,
                    batch_size=total_batch,
                    num_seeds=stage_config.num_seeds,
                )
                smoothness_task.init_buffers(wp_device)
                terms_for_stage.append(smoothness_task)
                self._base_smoothness_weights.append(smoothness_task._residual_weight_np.copy())
            else:
                self._base_smoothness_weights.append(np.array([], dtype=np.float32))
            stage_smoothness_tasks.append(smoothness_task)

            velocity_task: Optional[WarpVelocityLimitTask] = None
            if self.config.velocity_limit_weight > 0:
                prev_q_placeholder_v = wp.zeros(
                    (self.batch_size, self.robot.num_actuated_joints), dtype=wp.float32, device=wp_device
                )
                prev_base_placeholder_v = WarpSE3(repeat(placeholder_base.xyz_wxyz, self.batch_size))
                prev_state_placeholder_v = self.robot.state(
                    q=prev_q_placeholder_v, T_world_base=prev_base_placeholder_v
                )
                vel_limits_np = (
                    np.array(self.config.velocity_limit_override, dtype=np.float32)
                    if self.config.velocity_limit_override is not None
                    else None
                )
                velocity_task = WarpVelocityLimitTask(
                    robot=self.robot,
                    dt=self.config.velocity_limit_dt,
                    prev_state_var=prev_state_placeholder_v,
                    velocity_limits=vel_limits_np,
                    weight=self.config.velocity_limit_weight,
                    batch_size=total_batch,
                    num_seeds=stage_config.num_seeds,
                )
                velocity_task.init_buffers(wp_device)
                terms_for_stage.append(velocity_task)
            stage_velocity_tasks.append(velocity_task)

            stage_terms.append(terms_for_stage)

        solver_config = WarpSolverConfig(
            stages=list(stages),
            use_cuda_graph=self.config.use_cuda_graph,
        )
        solver = WarpSolver(
            config=solver_config,
            placeholder_var=placeholder_state,
            terms=stage_terms,
            score_terms=None,
        )

        return solver, stage_frame_tasks, stage_smoothness_tasks, stage_velocity_tasks

    def scale_human_frame(
        self,
        human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        root_name = self.config.human_root_name
        if root_name not in human_frame:
            raise ValueError(f"Root joint '{root_name}' not found in human frame")

        root_pos_orig = human_frame[root_name][0].copy()
        root_scale = self.config.scale_table.get(root_name, 1.0)

        scaled_frame: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for joint_name, (pos, quat) in human_frame.items():
            pos = pos.copy().astype(np.float32)
            quat = quat.copy().astype(np.float32)

            scale = self.config.scale_table.get(joint_name, 1.0)

            if joint_name == root_name:
                scaled_pos = pos * scale
            else:
                relative_pos = pos - root_pos_orig
                scaled_root = root_pos_orig * root_scale
                scaled_pos = scaled_root + relative_pos * scale

            scaled_frame[joint_name] = (scaled_pos, quat)

        return scaled_frame

    def apply_offsets(
        self,
        scaled_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
        mapping: Dict[str, LinkMapping],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        targets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for entry in mapping.values():
            human_joint = entry.human_joint
            if human_joint not in scaled_frame:
                continue

            pos, quat = scaled_frame[human_joint]
            pos = pos.copy()
            quat = quat.copy()

            rot_offset = entry.rotation_offset
            if np.linalg.norm(rot_offset) > 1e-6:
                rot_offset = rot_offset / np.linalg.norm(rot_offset)
                quat = _quat_multiply(quat, rot_offset)

            pos_offset = entry.position_offset
            if np.linalg.norm(pos_offset) > 1e-6:
                global_offset = _rotate_vector_by_quat(pos_offset, quat)
                pos = pos + global_offset

            pos[2] -= self.config.ground_height

            targets[human_joint] = (pos, quat)

        return targets

    def _set_targets_for_solver(
        self,
        frame_tasks: List[WarpFrameTaskType],
        mapping: Dict[str, LinkMapping],
        targets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        target_buffers: List[wp.array],
    ) -> None:
        target_se3_list: List[WarpSE3] = []
        for idx, entry in enumerate(mapping.values()):
            human_joint = entry.human_joint
            if human_joint in targets:
                pos, quat = targets[human_joint]
                self._se3_tmp[0, :3] = pos
                self._se3_tmp[0, 3:7] = quat
            else:
                self._se3_tmp[0, :] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

            first_batch = target_buffers[idx].shape[0]
            se3_tiled = np.tile(self._se3_tmp, (first_batch, 1))
            wp.copy(target_buffers[idx], wp.from_numpy(se3_tiled, dtype=wp_vec7, device=self.device))
            target_se3_list.append(WarpSE3(target_buffers[idx]))

        for frame_task in frame_tasks:
            frame_task.set_target(target_se3_list)

    def _set_prev_state_for_smoothness(
        self,
        smoothness_tasks: List[Optional[WarpSmoothnessTask]],
        prev_state: Optional[WarpRobotState],
    ) -> None:
        if prev_state is None:
            return
        for task in smoothness_tasks:
            if task is not None:
                task.set_prev_state(prev_state)
        for task in self._velocity_tasks:
            if task is not None:
                task.set_prev_state(prev_state)

    def _fill_initial_var(
        self,
        solver: WarpSolver,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        init_q: np.ndarray,
    ) -> None:
        initial_var = solver.initial_var
        num_seeds = initial_var.batch_size  # batch_size * first_stage_num_seeds

        self._se3_tmp[0, :3] = root_pos
        self._se3_tmp[0, 3:7] = root_quat
        base_tiled = np.tile(self._se3_tmp, (num_seeds, 1))
        wp.copy(initial_var.T_world_base.xyz_wxyz, wp.from_numpy(base_tiled, dtype=wp_vec7, device=self.device))

        q_tiled = np.tile(init_q.reshape(1, -1), (num_seeds, 1))
        if num_seeds > 1:
            joint_limits = self.robot.spec.actuated_joint_limits
            joint_range = joint_limits[:, 1] - joint_limits[:, 0]
            midrange_q = self.robot.spec.midrange_q

            # Seed 0: warm start (prev_q). Local seeds: refine around prev_q. Midrange seeds: escape local minima.
            n_local = (num_seeds - 1) // 2
            n_midrange = num_seeds - 1 - n_local

            local_pert = np.random.uniform(-0.1, 0.1, (n_local, len(init_q))) * joint_range
            q_tiled[1 : 1 + n_local] = np.clip(init_q + local_pert, joint_limits[:, 0], joint_limits[:, 1])

            mid_pert = np.random.uniform(-0.1, 0.1, (n_midrange, len(init_q))) * joint_range
            q_tiled[1 + n_local :] = np.clip(midrange_q + mid_pert, joint_limits[:, 0], joint_limits[:, 1])
        wp.copy(initial_var.q, wp.from_numpy(q_tiled.astype(np.float32), dtype=wp.float32, device=self.device))

    def retarget(
        self,
        human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
        offset_to_ground: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        scaled_frame = self.scale_human_frame(human_frame)

        if offset_to_ground:
            foot_joints = [j for j in scaled_frame.keys() if "foot" in j.lower()]
            if foot_joints:
                min_z = min(scaled_frame[j][0][2] for j in foot_joints)
                for joint_name in scaled_frame:
                    pos, quat = scaled_frame[joint_name]
                    pos = pos.copy()
                    pos[2] -= min_z
                    scaled_frame[joint_name] = (pos, quat)

        targets = self.apply_offsets(scaled_frame, self.config.link_mapping)

        root_name = self.config.human_root_name
        root_pos, root_quat = targets.get(root_name, scaled_frame[root_name])
        init_q = self._prev_q if self._prev_q is not None else self.robot.spec.zero_q

        self._set_targets_for_solver(self._frame_tasks, self.config.link_mapping, targets, self._target_buffers)
        self._set_prev_state_for_smoothness(self._smoothness_tasks, self._prev_state)

        # Reduce smoothness when stuck (high prev_cost) so solver can accept large corrective moves
        smoothness_scale = max(
            self.config.smoothness_min_scale,
            1.0 / (1.0 + self._prev_cost / self.config.smoothness_cost_threshold),
        )
        for i, task in enumerate(self._smoothness_tasks):
            if task is not None and task.residual_weight is not None:
                scaled = (self._base_smoothness_weights[i] * smoothness_scale).astype(np.float32)
                wp.copy(task.residual_weight, wp.from_numpy(scaled, dtype=wp.float32, device=self.device))  # type: ignore[arg-type]

        self._fill_initial_var(self._solver, root_pos, root_quat, init_q)
        best_state, best_costs = self._solver.solve()
        self._prev_cost = float(best_costs.numpy()[0])
        root_xyz = best_state.T_world_base.xyz.numpy()[0]
        root_quat_final = best_state.T_world_base.quat_wxyz.numpy()[0]
        q = best_state.q.numpy()[0]

        if self._prev_q is not None and self.config.velocity_clamp_scale > 0:
            dt = self.config.velocity_limit_dt
            v_limits = (
                np.array(self.config.velocity_limit_override, dtype=np.float32)
                if self.config.velocity_limit_override is not None
                else self.robot.spec.actuated_joint_velocity_limits
            )
            dq = q - self._prev_q
            max_dq = v_limits * dt * self.config.velocity_clamp_scale
            q = self._prev_q + np.clip(dq, -max_dq, max_dq)

        self._prev_q = q.copy()

        needs_prev_state = (
            _is_nonzero_weight(self.config.smoothness_weight)
            or _is_nonzero_weight(self.config.base_smoothness_weight)
            or self.config.velocity_limit_weight > 0
        )
        if needs_prev_state:
            self._se3_tmp[0, :3] = root_xyz
            self._se3_tmp[0, 3:7] = root_quat_final
            base_se3 = WarpSE3(wp.from_numpy(self._se3_tmp, dtype=wp_vec7, device=self.device))
            q_wp = wp.from_numpy(q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=self.device)
            if self._prev_state is None:
                self._prev_state = self.robot.state(q=q_wp, T_world_base=base_se3)
            else:
                wp.copy(self._prev_state.q, q_wp)
                wp.copy(self._prev_state.T_world_base.xyz_wxyz, base_se3.xyz_wxyz)

        robot_pose = np.concatenate([root_xyz, root_quat_final, q]).astype(np.float32)

        return robot_pose, targets

    def reset(self) -> None:
        self._prev_q = None
        self._prev_state = None
        self._prev_cost = 0.0
