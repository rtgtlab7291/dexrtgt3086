from unittest.mock import MagicMock

import numpy as np
import pytest
import trimesh

from robokit.utils.visualize_utils import ViserBatchUrdf, ViserMjcf


MJCF = """
<mujoco model="visualizer_test">
  <worldbody>
    <body name="driver_body">
      <joint name="driver" type="hinge"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="dependent_body">
      <joint name="dependent" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="dependent" joint2="driver" polycoef="0.05 0.5 0 0 0"/>
  </equality>
</mujoco>
"""


class TestViserBatchUrdf:
    def test_api_and_scaling(self):
        robot = MagicMock()
        robot.spec.link_names = ["base"]
        robot.spec.link_visual_geometries = {"base": trimesh.Scene(trimesh.creation.box())}
        robot.spec.actuated_joint_names = ["joint"]
        robot.spec.actuated_joint_limits = np.array([[-1.0, 1.0]])

        state = MagicMock()
        state.T_world_link.numpy.return_value = np.array([[[11.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
        state.T_world_base.numpy.return_value = np.array([[10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        robot.state.return_value = state
        robot.forward_kinematics.return_value = state

        handle = MagicMock(visible=True)
        target = MagicMock()
        target.scene.add_batched_meshes_simple.return_value = handle
        visualizer = ViserBatchUrdf(target, robot, batch_size=1, scale=2.0)

        T_world_base = np.array([[10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        visualizer.update_cfg(configuration=np.zeros((1, 1)), T_world_base=T_world_base)

        assert visualizer.get_actuated_joint_names() == ("joint",)
        np.testing.assert_allclose(handle.batched_positions, [[12.0, 0.0, 0.0]])
        visualizer.show_visual = False
        assert not visualizer.show_visual


class TestViserMjcf:
    def test_api_limits_and_mimic(self):
        pytest.importorskip("mujoco")
        handles = [MagicMock(), MagicMock()]
        target = MagicMock()
        target.scene.add_mesh_trimesh.side_effect = handles
        visualizer = ViserMjcf(target, MJCF)

        assert visualizer.get_actuated_joint_names() == ("driver",)
        assert visualizer.get_actuated_joint_limits()["driver"] == pytest.approx((-np.pi, np.pi))

        visualizer.update_cfg(np.array([0.6]))
        assert handles[1].position[0] == pytest.approx(0.35)

        with pytest.raises(ValueError):
            visualizer.update_cfg(np.empty(0))
