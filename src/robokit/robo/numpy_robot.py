# pyright: reportIncompatibleVariableOverride=false
# pyright: reportOperatorIssue=false
"""
NOTE: As of development time, pinocchio 3.7 does not handle mimic joints correctly.
Therefore, mimic joints are processed manually in this file, with most logic implemented in numpy/python.
This reduces efficiency (potentially 5x slower than pure pinocchio functions).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pinocchio as pin
from jaxtyping import Float

from robokit.lie.pinocchio_se3 import PinocchioSE3
from robokit.robo.robot import Robot, RobotState
from robokit.robo.robot_spec import RobotSpec


if TYPE_CHECKING:
    from robokit.opt.numpy_optimizer import NumpyOptimizer, NumpyOptimizerConfig


@dataclass(frozen=True)
class PinocchioAuxiliaryMapping:
    pin_link_frame_ids: List[int]
    """Frame IDs in the Pinocchio model."""
    pin_q_start_indices: List[int]
    """Start indices in the Pinocchio configuration vector q for each non-universe joint."""
    pin_q_sizes: List[int]
    """Configuration sizes (nq per joint) for each non-universe joint."""
    pin_to_spec_actuated_joint_indices: Optional[np.ndarray] = None
    """Mapping from Pinocchio actuated joint indices to spec actuated joint indices."""
    pin_to_spec_nonfixed_joint_indices: Optional[np.ndarray] = None
    """Mapping from Pinocchio actuated joint indices to spec non-fixed joint indices."""
    pin_mimic_multipliers: Optional[np.ndarray] = None
    """Multipliers for mimic joints in Pinocchio ordering."""
    pin_mimic_offsets: Optional[np.ndarray] = None
    """Offsets for mimic joints in Pinocchio ordering."""
    pin_to_spec_jac_map: Optional[np.ndarray] = None
    """Mapping matrix for converting Jacobians from Pinocchio to spec ordering."""


@dataclass
class NumpyRobotState(RobotState):
    _pin_model: pin.Model
    _pin_data: pin.Data
    _pin_aux: PinocchioAuxiliaryMapping
    _pin_q: Optional[np.ndarray] = None

    _q: Optional[np.ndarray] = None
    _T_world_base: Optional[PinocchioSE3] = None
    S_world: Optional[Float[np.ndarray, "6 num_joints"]] = field(default=None, init=False)

    _T_world_base_adjoint: Optional[PinocchioSE3] = field(default=None, init=False)
    _T_world_link_cache: Dict[int, PinocchioSE3] = field(default_factory=dict, init=False)
    _link_jacobians_cache: Dict[Tuple[int, Literal["body", "spatial"]], Any] = field(default_factory=dict, init=False)

    is_fk_computed: bool = field(default=False, init=False)
    is_motion_subspace_computed: bool = field(default=False, init=False)

    def __post_init__(self):
        if self._q is not None:
            self._pin_q = self._to_pin_q(self._q)
            if self._T_world_base is not None:
                self._T_world_base_adjoint = self._T_world_base.adjoint()
        self._T_world_link_cache.clear()
        self._link_jacobians_cache.clear()
        self.is_fk_computed = False
        self.is_motion_subspace_computed = False

    @property
    def q(self) -> np.ndarray:
        if self._q is None:
            raise RuntimeError("NumpyRobotState has not been initialized. Call robot.set_configuration() first.")
        return self._q

    @property
    def is_initialized(self) -> bool:
        return self._q is not None

    @property
    def T_world_base(self) -> Optional[PinocchioSE3]:
        return self._T_world_base

    @property
    def has_floating_base(self) -> bool:
        return self._T_world_base is not None

    @property
    def tangent_dim(self) -> int:
        return (6 if self.has_floating_base else 0) + self.spec.num_actuated_joints

    def set_configuration(
        self, q: Float[np.ndarray, "num_actuated_joints"], T_world_base: Optional[PinocchioSE3] = None
    ):
        if q is None:
            raise ValueError("q cannot be None")
        self._q = q
        self._T_world_base = T_world_base
        self.__post_init__()

    def integrate(self, velocity: Float[np.ndarray, "tangent_dim"]) -> "NumpyRobotState":
        if self.has_floating_base:
            xi_base = velocity[:6]
            dq = velocity[6:]
            new_state = NumpyRobotState(
                spec=self.spec,
                _q=self._q + dq,
                _T_world_base=PinocchioSE3.exp(xi_base) @ self._T_world_base,
                _pin_model=self._pin_model,
                _pin_data=self._pin_data,
                _pin_aux=self._pin_aux,
            )
            return new_state
        else:
            return NumpyRobotState(
                spec=self.spec,
                _q=self._q + velocity,
                _T_world_base=self._T_world_base,
                _pin_model=self._pin_model,
                _pin_data=self._pin_data,
                _pin_aux=self._pin_aux,
            )

    def clone(self) -> "NumpyRobotState":
        return NumpyRobotState(
            spec=self.spec,
            _q=self._q.copy() if self._q is not None else None,
            _T_world_base=self._T_world_base,
            _pin_model=self._pin_model,
            _pin_data=self._pin_model.createData(),
            _pin_aux=self._pin_aux,
        )

    def get_T_world_link(self, link_index: int) -> PinocchioSE3:
        if not self.is_fk_computed:
            raise RuntimeError("Forward kinematics has not been computed. Call robot.forward_kinematics() first.")
        if link_index in self._T_world_link_cache:
            return self._T_world_link_cache[link_index]
        pin_frame_id = self._pin_aux.pin_link_frame_ids[link_index]
        T_base_frame = PinocchioSE3(self._pin_data.oMf[pin_frame_id])
        T_world_link = self._T_world_base @ T_base_frame if self._T_world_base is not None else T_base_frame
        self._T_world_link_cache[link_index] = T_world_link
        return T_world_link

    def get_motion_subspace(self) -> Float[np.ndarray, "6 num_joints"]:
        """Returns the stack of all motion subspace expressed in the world frame."""
        if not self.is_motion_subspace_computed:
            raise RuntimeError("Motion subspace has not been computed. Call robot.compute_motion_subspace() first.")
        if self.S_world is not None:
            return self.S_world
        self.S_world = np.zeros((6, self.spec.num_joints), dtype=np.float32)
        if self._pin_aux.pin_to_spec_nonfixed_joint_indices is None:  # no re-indexing needed
            S_nonfixed = self._pin_data.J
        else:  # re-index from pinocchio non-fixed joint order to spec non-fixed joint order
            S_nonfixed = self._pin_data.J[:, self._pin_aux.pin_to_spec_nonfixed_joint_indices]
        self.S_world[:, self.spec.nonfixed_joint_mask] = S_nonfixed
        return self.S_world

    def get_link_jacobian(
        self, link_index: int, reference_frame: Literal["body", "spatial"] = "body"
    ) -> Float[np.ndarray, "6 num_total_dofs"]:
        if not self.is_motion_subspace_computed:
            raise RuntimeError("Motion subspace has not been computed. Call robot.compute_motion_subspace() first.")
        if (link_index, reference_frame) in self._link_jacobians_cache:
            return self._link_jacobians_cache[(link_index, reference_frame)]
        pin_frame_id = self._pin_aux.pin_link_frame_ids[link_index]
        pin_ref_frame = pin.ReferenceFrame.LOCAL if reference_frame == "body" else pin.ReferenceFrame.WORLD
        J = pin.getFrameJacobian(self._pin_model, self._pin_data, pin_frame_id, pin_ref_frame)
        J_joints = self._to_spec_jacobian(J)

        if self._T_world_base is None:
            self._link_jacobians_cache[(link_index, reference_frame)] = J_joints
            return J_joints

        if reference_frame == "body":
            J_base = self._pin_data.oMf[pin_frame_id].inverse().action
        else:  # spatial
            J_joints = self._T_world_base_adjoint @ J_joints
            J_base = self._T_world_base_adjoint
        J_full = np.empty((6, 6 + J_joints.shape[1]), dtype=J_joints.dtype)
        J_full[:, :6] = J_base
        J_full[:, 6:] = J_joints
        self._link_jacobians_cache[(link_index, reference_frame)] = J_full
        return J_full

    def _to_pin_q(self, q: Float[np.ndarray, "num_actuated_joints"]) -> Float[np.ndarray, "nq"]:
        """Build a full-length Pinocchio configuration vector from spec actuated q.

        - Reorders to Pinocchio joint ordering
        - Applies mimic multipliers/offsets per Pinocchio joint
        - Fills the model's nq-sized configuration vector, keeping extra coordinates neutral
        """
        if self._pin_aux.pin_to_spec_actuated_joint_indices is not None:
            q_pin = q[self._pin_aux.pin_to_spec_actuated_joint_indices]
        else:
            q_pin = q
        if self._pin_aux.pin_mimic_multipliers is not None:
            q_pin = q_pin * self._pin_aux.pin_mimic_multipliers + self._pin_aux.pin_mimic_offsets

        q_full = pin.neutral(self._pin_model)
        # - nq==1 (revolute/prismatic): write scalar
        # - nq==2 (revolute unbounded/SO2): write [cos(theta), sin(theta)]
        for k, start in enumerate(self._pin_aux.pin_q_start_indices):
            if k >= q_pin.shape[0]:
                break
            nqk = self._pin_aux.pin_q_sizes[k]
            if nqk == 1:
                q_full[start] = q_pin[k]
            elif nqk == 2:
                theta = float(q_pin[k])
                q_full[start : start + 2] = np.array([np.cos(theta), np.sin(theta)], dtype=q_full.dtype)
            else:
                q_full[start] = q_pin[k]
        return q_full

    def _to_spec_jacobian(
        self, jacobian: Float[np.ndarray, "6 num_nonfixed_joints"]
    ) -> Float[np.ndarray, "6 num_actuated_joints"]:
        if self._pin_aux.pin_to_spec_actuated_joint_indices is None:  # identical joint sets, nothing to do
            return jacobian
        elif self._pin_aux.pin_to_spec_jac_map is None:  # no mimic joints, just re-index
            return jacobian[:, self._pin_aux.pin_to_spec_nonfixed_joint_indices]
        else:  # has mimic joints, need to apply mapping
            return jacobian @ self._pin_aux.pin_to_spec_jac_map

    def __repr__(self) -> str:
        return f"NumpyRobotState(q={self.q}, T_world_base={self.T_world_base}, is_initialized={self.is_initialized}, is_fk_computed={self.is_fk_computed}, is_motion_subspace_computed={self.is_motion_subspace_computed})"


class NumpyRobot(Robot):
    def __init__(self, spec: RobotSpec, backend: Literal["numpy"] = "numpy"):
        self.spec = spec
        self._pin_model = pin.buildModelFromXML(spec.robot_description)
        pin_act_joint_ids = [i for i, j in enumerate(self._pin_model.joints) if j.nq > 0 and i != 0]
        self._pin_actuated_joint_names = [self._pin_model.names[i] for i in pin_act_joint_ids]
        if set(self._pin_actuated_joint_names) != set(self.spec.nonfixed_joint_names):
            missing_joints = set(self._pin_actuated_joint_names) - set(self.spec.nonfixed_joint_names)
            raise ValueError(f"Actuated joint names from Pinocchio model not found in robot spec: {missing_joints}")
        self._pin_aux = self._get_auxiliary_mapping()

    def _get_auxiliary_mapping(self) -> PinocchioAuxiliaryMapping:
        # Pinocchio joint meta
        pin_act_joint_ids = [i for i, j in enumerate(self._pin_model.joints) if j.nq > 0 and i != 0]
        num_pin_act_joints = len(pin_act_joint_ids)
        pin_link_frame_ids = [self._pin_model.getFrameId(name, pin.FrameType.BODY) for name in self.spec.link_names]
        pin_q_start_indices = [self._pin_model.joints[jid].idx_q for jid in pin_act_joint_ids]
        pin_q_sizes = [self._pin_model.joints[jid].nq for jid in pin_act_joint_ids]

        to_pin_indices = np.array([self.spec.joint_names.index(name) for name in self._pin_actuated_joint_names])
        pin_to_spec_actuated_joint_indices = np.where(
            self.spec.mimic_actuated_joint_indices[to_pin_indices] >= 0,
            self.spec.mimic_actuated_joint_indices[to_pin_indices],
            self.spec.actuated_joint_indices[to_pin_indices],
        )
        pin_to_spec_nonfixed_joint_indices = np.array(
            [self._pin_actuated_joint_names.index(name) for name in self.spec.nonfixed_joint_names]
        )

        if np.array_equal(pin_to_spec_actuated_joint_indices, np.arange(len(self.spec.actuated_joint_names))):
            return PinocchioAuxiliaryMapping(
                pin_link_frame_ids=pin_link_frame_ids,
                pin_q_start_indices=pin_q_start_indices,
                pin_q_sizes=pin_q_sizes,
                pin_to_spec_actuated_joint_indices=None,
            )
        if np.array_equal(pin_to_spec_nonfixed_joint_indices, np.arange(len(self.spec.nonfixed_joint_names))):
            pin_to_spec_nonfixed_joint_indices = None

        pin_mimic_multipliers, pin_mimic_offsets, pin_to_spec_jac_map = None, None, None
        if self.spec.has_mimic_joints:
            pin_mimic_multipliers = self.spec.mimic_multipliers[to_pin_indices]
            pin_mimic_offsets = self.spec.mimic_offsets[to_pin_indices]
            pin_to_spec_jac_map = np.zeros((num_pin_act_joints, self.spec.num_actuated_joints), dtype=np.float32)
            pin_to_spec_jac_map[np.arange(num_pin_act_joints), pin_to_spec_actuated_joint_indices] = (
                pin_mimic_multipliers
            )

        return PinocchioAuxiliaryMapping(
            pin_link_frame_ids=pin_link_frame_ids,
            pin_q_start_indices=pin_q_start_indices,
            pin_q_sizes=pin_q_sizes,
            pin_to_spec_actuated_joint_indices=pin_to_spec_actuated_joint_indices,
            pin_to_spec_nonfixed_joint_indices=pin_to_spec_nonfixed_joint_indices,
            pin_mimic_multipliers=pin_mimic_multipliers,
            pin_mimic_offsets=pin_mimic_offsets,
            pin_to_spec_jac_map=pin_to_spec_jac_map,
        )

    def state(
        self, q: Optional[Float[np.ndarray, "num_dofs"]] = None, T_world_base: Optional[PinocchioSE3] = None
    ) -> NumpyRobotState:
        pin_data = self._pin_model.createData()
        state = NumpyRobotState(
            spec=self.spec,
            _q=q,
            _T_world_base=T_world_base,
            _pin_model=self._pin_model,
            _pin_data=pin_data,
            _pin_aux=self._pin_aux,
        )
        return state

    def forward_kinematics(self, state: NumpyRobotState) -> NumpyRobotState:
        """
        Compute the forward kinematics of the robot for given joint positions statelessly.
        Returns a robot state containing the computed transforms of all links.

        Example:
            >>> from robokit.lie.pinocchio_se3 import PinocchioSE3
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"), backend="numpy")
            >>> q = np.array([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            >>> state = robot.state(q=q)
            >>> state = robot.forward_kinematics(state)
            >>> link_pose = state.get_T_world_link(robot.link_names.index("panda_hand"))
            >>> expected_pose = PinocchioSE3(np.array([0.0608599, 0.0, 0.7637312, 0.0382045, 0.9192637, 0.3807715, 0.0922340]))
            >>> np.allclose(link_pose.as_matrix(), expected_pose.as_matrix(), atol=1e-4)
            True
            >>> T_world_base = PinocchioSE3(np.array([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0]))
            >>> state = robot.state(q=q, T_world_base=T_world_base)
            >>> state = robot.forward_kinematics(state)
            >>> link_pose = state.get_T_world_link(robot.link_names.index("panda_hand"))
            >>> expected_pose = PinocchioSE3(np.array([1.76371783, 2.0, 2.93915576, -0.24222432, 0.71524162, 0.29626278, -0.58478125]))
            >>> np.allclose(link_pose.as_matrix(), expected_pose.as_matrix(), atol=1e-4)
            True
        """
        pin.framesForwardKinematics(self._pin_model, state._pin_data, state._pin_q)
        state.is_fk_computed = True
        return state

    def compute_motion_subspace(self, state: NumpyRobotState) -> NumpyRobotState:
        """
        Compute the joint Jacobians for the robot and store them in the state.

        After calling this method, use state.get_link_jacobian() to retrieve Jacobians
        for specific links. When a floating base is provided, the returned Jacobian will
        have J_base prepended to J_joints (shape: [6, 6 + num_dofs]).

        Example:
            >>> from robokit.lie.pinocchio_se3 import PinocchioSE3
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = NumpyRobot.load(load_robot_description("panda_description"), backend="numpy")
            >>> T_world_base = PinocchioSE3(np.array([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0]))
            >>> q = np.array([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            >>> state = robot.state(q=q, T_world_base=T_world_base)
            >>> state = robot.compute_motion_subspace(state)
            >>> J = state.get_link_jacobian(robot.link_names.index("panda_hand"), "body")
            >>> np.allclose(J[0, :6], np.array([0.693, 0.7071, 0.1405, -0.54, 0.5207, 0.043]), atol=1e-4)
            True
            >>> J = state.get_link_jacobian(robot.link_names.index("panda_hand"), "spatial")
            >>> np.allclose(J[0, 6:], np.array([0.0001, -3., 1.8641, 3.2647, -1.4346, 3.0467, -0.3974, 0.]), atol=1e-3)
            True
            >>> np.allclose(J[0, :6], np.array([0.0001, 0., 1., -2., -3., 0.0001]), atol=1e-3)
            True
        """
        if not state.is_fk_computed:
            state = self.forward_kinematics(state)
        pin.computeJointJacobians(self._pin_model, state._pin_data, state._pin_q)
        state.is_motion_subspace_computed = True
        return state

    def build_inverse_kinematics_optimizer(
        self,
        frame_names: Union[str, Sequence[str]],
        T_world_target: Union[PinocchioSE3, Sequence[PinocchioSE3]],
        position_weight: float = 1.0,
        orientation_weight: float = 0.2,
        limit_weight: float = 2.0,
        optimizer_config: Optional["NumpyOptimizerConfig"] = None,
    ) -> "NumpyOptimizer":
        """
        Build an inverse kinematics optimizer for the robot.

        Example:
            >>> from robokit.lie.pinocchio_se3 import PinocchioSE3
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("ur10_description"), backend="numpy")
            >>> target_pose = PinocchioSE3(np.array([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]))
            >>> ik_optimizer = robot.build_inverse_kinematics_optimizer("ee_link", target_pose)
            >>> state = robot.state(q=robot.zero_q)
            >>> state = ik_optimizer.solve(state)
            >>> state = robot.forward_kinematics(state)
            >>> achieved_pose = state.get_T_world_link(robot.link_names.index("ee_link"))
            >>> np.allclose(achieved_pose.xyz, target_pose.xyz, atol=1e-3)
            True
            >>> np.allclose(achieved_pose.quat_wxyz, target_pose.quat_wxyz, atol=1e-3)
            True
        """
        from robokit.opt.numpy_optimizer import NumpyOptimizer, NumpyOptimizerConfig
        from robokit.terms import Term
        from robokit.terms.numpy.frame_task import PinocchioFrameTask
        from robokit.terms.numpy.position_limit import PinocchioPositionLimit

        frame_names = [frame_names] if isinstance(frame_names, str) else frame_names
        T_world_target = [T_world_target] if isinstance(T_world_target, PinocchioSE3) else T_world_target

        terms: List[Term] = []
        for idx, frame_name in enumerate(frame_names):
            frame_index = self.link_names.index(frame_name)
            frame_task = PinocchioFrameTask(
                robot=self,
                frame_index=frame_index,
                T_world_target=T_world_target[idx],
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
            terms.append(frame_task)
        position_limit = PinocchioPositionLimit(robot=self, weight=limit_weight)
        terms.append(position_limit)
        if optimizer_config is None:
            optimizer_config = NumpyOptimizerConfig(
                use_qpsolver=False,
                lm_lambda=1.0,
                use_early_stopping=True,
            )
        return NumpyOptimizer(terms=terms, config=optimizer_config)

    def sample_q(
        self, num_samples: int = 1, rng: Optional[np.random.Generator] = None
    ) -> Float[np.ndarray, "num_samples num_dofs"]:
        joint_limits = self.spec.actuated_joint_limits
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(joint_limits[:, 0], joint_limits[:, 1], size=(num_samples, joint_limits.shape[0]))
