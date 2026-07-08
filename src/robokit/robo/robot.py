import abc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import yourdfpy

from robokit.lie.se3 import SE3
from robokit.opt.variables import Var
from robokit.robo.robot_spec import RobotSpec
from robokit.types import ArrayLike


if TYPE_CHECKING:
    from robokit.robo.numpy_robot import NumpyRobot
    from robokit.robo.torch_robot import TorchRobot
    from robokit.robo.warp_robot import WarpRobot

logger = logging.getLogger("robokit")

Backend = Literal["numpy", "torch", "warp"]


@dataclass
class RobotState(Var):
    spec: RobotSpec

    @abc.abstractmethod
    def set_configuration(self, q: ArrayLike, T_world_base: Optional[SE3] = None): ...

    @abc.abstractmethod
    def get_T_world_link(self, link_index: int) -> SE3: ...

    @abc.abstractmethod
    def get_link_jacobian(
        self, link_index: int, reference_frame: Literal["body", "spatial"] = "body"
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        For fixed-base robots, returns only the joint Jacobian.
        For floating-base robots (when T_world_base is provided), returns both joint and base Jacobians.
        """
        ...

    def __repr__(self) -> str:
        return f"RobotState(q={self.q}, T_world_base={self.T_world_base})"

    def __str__(self) -> str:
        return self.__repr__()


class Robot(abc.ABC):
    """
    Robot model with multiple backends.

    Examples:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="numpy")
        >>> q = np.array([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        >>> state = robot.state(q=q)
        >>> state = robot.forward_kinematics(state)
        >>> ee_pose = state.get_T_world_link(robot.link_names.index("panda_hand"))
        >>> expected_pose = np.array([0.0608599, 0.0, 0.7637312, 0.0382045, 0.9192637, 0.3807715, 0.0922340])
        >>> np.allclose(ee_pose.xyz_wxyz, expected_pose, atol=1e-4)
        True

        >>> import torch
        >>> robot = Robot.load(load_robot_description("panda_description"), backend="torch")
        >>> q = torch.tensor([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        >>> state = robot.state(q=q)
        >>> state = robot.forward_kinematics(state)
        >>> ee_pose = state.get_T_world_link(robot.link_names.index("panda_hand"))
        >>> torch.allclose(ee_pose.xyz_wxyz, torch.tensor([0.0608599, 0.0, 0.7637312, 0.0382045, 0.9192637, 0.3807715, 0.0922340], dtype=torch.float32), atol=1e-4)
        True
    """

    spec: RobotSpec

    @staticmethod
    def _get_robot_class(backend: Backend) -> Union["type[NumpyRobot]", "type[TorchRobot]", "type[WarpRobot]"]:
        # fmt: off
        if backend == "numpy":
            from robokit.robo.numpy_robot import NumpyRobot
            return NumpyRobot
        elif backend == "torch":
            from robokit.robo.torch_robot import TorchRobot
            return TorchRobot
        elif backend == "warp":
            from robokit.robo.warp_robot import WarpRobot
            return WarpRobot
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        # fmt: on

    @overload
    def __new__(cls, spec: RobotSpec, backend: Literal["numpy"] = ...) -> "NumpyRobot": ...
    @overload
    def __new__(cls, spec: RobotSpec, backend: Literal["torch"] = ...) -> "TorchRobot": ...
    @overload
    def __new__(cls, spec: RobotSpec, backend: Literal["warp"] = ...) -> "WarpRobot": ...
    def __new__(cls, spec: RobotSpec, backend: Backend = "numpy") -> "Robot":
        if cls is Robot:
            robot_cls = cls._get_robot_class(backend)
            return robot_cls.__new__(robot_cls, spec, backend)  # type: ignore
        else:
            # Called on subclass, use normal instantiation
            return super().__new__(cls)

    # fmt: off
    @overload
    @staticmethod
    def load(robot_description_or_path: Union[str, Path, yourdfpy.URDF], backend: Literal["numpy"] = ..., load_meshes: bool = ..., mesh_dir: Optional[Union[str, Path]] = ..., load_collision_spheres: bool = ..., collision_spheres_path: Optional[str] = ..., base_link_name: Optional[str] = ..., ee_link_names: Optional[List[str]] = ...) -> "NumpyRobot": ...
    @overload
    @staticmethod
    def load(robot_description_or_path: Union[str, Path, yourdfpy.URDF], backend: Literal["torch"] = ..., load_meshes: bool = ..., mesh_dir: Optional[Union[str, Path]] = ..., load_collision_spheres: bool = ..., collision_spheres_path: Optional[str] = ..., base_link_name: Optional[str] = ..., ee_link_names: Optional[List[str]] = ...) -> "TorchRobot": ...
    @overload
    @staticmethod
    def load(robot_description_or_path: Union[str, Path, yourdfpy.URDF], backend: Literal["warp"] = ..., load_meshes: bool = ..., mesh_dir: Optional[Union[str, Path]] = ..., load_collision_spheres: bool = ..., collision_spheres_path: Optional[str] = ..., base_link_name: Optional[str] = ..., ee_link_names: Optional[List[str]] = ...) -> "WarpRobot": ...
    # fmt: on
    @staticmethod
    def load(
        robot_description_or_path: Union[str, Path, yourdfpy.URDF],
        backend: Backend = "numpy",
        load_meshes: bool = False,
        mesh_dir: Optional[Union[str, Path]] = None,
        load_collision_spheres: bool = False,
        collision_spheres_path: Optional[str] = None,
        base_link_name: Optional[str] = None,
        ee_link_names: Optional[List[str]] = None,
    ) -> "Robot":
        return Robot(
            spec=RobotSpec.parse(
                robot_description_or_path=robot_description_or_path,
                load_meshes=load_meshes,
                mesh_dir=mesh_dir,
                load_collision_spheres=load_collision_spheres,
                collision_spheres_path=collision_spheres_path,
                base_link_name=base_link_name,
                ee_link_names=ee_link_names,
            ),
            backend=backend,
        )

    @abc.abstractmethod
    def state(self, q: Optional[ArrayLike] = None, T_world_base: Optional[SE3] = None) -> RobotState: ...

    @abc.abstractmethod
    def forward_kinematics(self, state: RobotState) -> RobotState: ...

    @abc.abstractmethod
    def compute_motion_subspace(self, state: RobotState) -> RobotState: ...

    def __repr__(self) -> str:
        return (
            f"Robot(name={self.spec.name}, num_actuated_joints={self.spec.num_actuated_joints}, num_links={self.spec.num_links})\n"
            + self.spec.kinematic_tree_str
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def num_joints(self) -> int:
        return self.spec.num_joints

    @property
    def num_actuated_joints(self) -> int:
        return self.spec.num_actuated_joints

    @property
    def num_nonfixed_joints(self) -> int:
        return self.spec.num_nonfixed_joints

    @property
    def num_dofs(self) -> int:
        return self.spec.num_dofs

    @property
    def num_links(self) -> int:
        return self.spec.num_links

    @property
    def link_names(self) -> List[str]:
        return self.spec.link_names

    @property
    def joint_names(self) -> List[str]:
        return self.spec.joint_names

    @property
    def actuated_joint_names(self) -> List[str]:
        return self.spec.actuated_joint_names

    @property
    def actuated_joint_limits(self) -> np.ndarray:
        return self.spec.actuated_joint_limits

    @property
    def midrange_q(self) -> np.ndarray:
        return self.spec.midrange_q

    @property
    def zero_q(self) -> np.ndarray:
        return self.spec.zero_q


__all__ = ["Robot", "RobotState"]
