import pytest
import torch
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.robo import Robot


def test_map_to_full_joint_values_consistency():
    urdf_path = load_robot_description("panda_description")
    robot_torch = Robot.load(urdf_path, backend="torch")
    robot_warp = Robot.load(urdf_path, backend="warp")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_actuated = robot_torch.num_actuated_joints

    torch.manual_seed(42)

    for shape in [(num_actuated,), (4, num_actuated), (2, 10, num_actuated)]:
        actuated_values = torch.randn(shape, device=device, dtype=torch.float32)
        full_values_torch = robot_torch.map_to_full_joint_values(actuated_values)
        full_values_warp = robot_warp.map_to_full_joint_values_torch(actuated_values)

        assert full_values_torch.shape == full_values_warp.shape
        assert full_values_torch.shape[-1] == robot_torch.spec.num_joints
        assert torch.allclose(full_values_torch, full_values_warp, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
