from typing import TYPE_CHECKING, Optional, cast

from robokit.opt.var_values import VarValues
from robokit.robo.robot_state import RobotState


if TYPE_CHECKING:
    from robokit.robo.robot import Robot


class RobotTask:
    """Serial preparation shared by terms that consume a robot state."""

    robot: Optional["Robot"]
    var_key: str
    precompute_collision_geometry: str = ""

    def precompute(self, var_values: VarValues, need_gradient: bool = False):
        """Prepare kinematics and geometry required by the task."""
        robot = cast("Robot", self.robot)
        state = cast(RobotState, var_values.get(self.var_key))
        if not state.is_fk_computed:
            robot.forward_kinematics(state)
        if self.precompute_collision_geometry == "sphere" and not state.is_collision_spheres_computed:
            robot.transform_collision_spheres(state)
        if self.precompute_collision_geometry == "capsule" and not state.is_capsules_computed:
            robot.transform_collision_capsules(state)
        if need_gradient and not state.is_motion_subspace_computed:
            robot.compute_motion_subspace(state)
