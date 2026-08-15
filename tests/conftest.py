"""Session-scoped robot fixtures shared across the entire test suite."""

from pathlib import Path

import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except (ImportError, ValueError, OSError):
        return False


_HAS_TORCH = _torch_available()


def pytest_ignore_collect(collection_path, config):
    if not _HAS_TORCH and collection_path.suffix == ".py" and collection_path.name.startswith("test_"):
        text = collection_path.read_text()
        if "pytestmark = pytest.mark.torch" in text:
            return True


_FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def panda_robot():
    return Robot.load(load_robot_description("panda_description"))


@pytest.fixture(scope="session")
def g1_robot():
    return Robot.load(load_robot_description("g1_description"))


@pytest.fixture(scope="session")
def ur10_robot():
    return Robot.load(load_robot_description("ur10_description"))


@pytest.fixture(scope="session")
def fetch_robot():
    return Robot.load(load_robot_description("fetch_description"))


@pytest.fixture(scope="session")
def yumi_robot():
    return Robot.load(load_robot_description("yumi_description"))


@pytest.fixture(scope="session")
def ability_hand_robot():
    return Robot.load(load_robot_description("ability_hand_description"))


@pytest.fixture(scope="session")
def shadow_hand_robot():
    urdf = _FIXTURES / "shadow_hand_right.urdf"
    return Robot(RobotSpec.parse(urdf, load_meshes=False, mesh_dir=None))


@pytest.fixture(scope="session")
def panda_robot_with_collision():
    return Robot.load(
        load_robot_description("panda_description"),
        load_collision_spheres=True,
        collision_spheres_path=str(_FIXTURES / "franka_collision_spheres.yaml"),
    )


@pytest.fixture(scope="session")
def panda_robot_with_capsules():
    # load_meshes=True fits one capsule per link from the collision meshes (lazily).
    return Robot.load(load_robot_description("panda_description"), load_meshes=True)
