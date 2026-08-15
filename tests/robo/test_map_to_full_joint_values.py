import pytest
import torch
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.robo import Robot


pytestmark = pytest.mark.torch


_robot_cache: dict = {}


def _get_robot(desc_name: str) -> Robot:
    if desc_name not in _robot_cache:
        _robot_cache[desc_name] = Robot.load(load_robot_description(desc_name))
    return _robot_cache[desc_name]


def test_map_to_full_joint_values_consistency():
    robot = _get_robot("panda_description")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_actuated = robot.spec.num_actuated_joints

    torch.manual_seed(42)

    for shape in [(num_actuated,), (4, num_actuated), (2, 10, num_actuated)]:
        actuated_values = torch.randn(shape, device=device, dtype=torch.float32)
        full_values = robot.map_to_full_joint_values_torch(actuated_values)

        assert full_values.shape[-1] == robot.spec.num_joints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
