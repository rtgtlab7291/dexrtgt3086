from pathlib import Path

from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.robo import Robot


def test_robot_load():
    urdf_str = """<robot name="simple_robot">
    <link name="link1"/>
    <joint name="joint1" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0.05 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14159" upper="3.14159" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint2" type="prismatic">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0.05 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.0" upper="1.0" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
</robot>"""
    robot = Robot.load(urdf_str)
    assert robot.spec.name == "simple_robot"
    assert robot.spec.num_links == 3
    assert robot.spec.num_actuated_joints == 2

    urdf_path = Path(__file__).parent.parent.parent / "assets" / "robot_description" / "panda.urdf"
    robot = Robot.load(urdf_path)
    assert robot.spec.name == "panda"
    assert robot.spec.num_links == 12
    assert robot.spec.num_actuated_joints == 8

    robot = Robot.load(Path(urdf_path))
    assert robot.spec.name == "panda"
    assert robot.spec.num_links == 12
    assert robot.spec.num_actuated_joints == 8

    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf)
    assert robot.spec.name == "panda"
    assert robot.spec.num_links == 13
    assert robot.spec.num_actuated_joints == 8


def test_robot_load_with_base_and_ee():
    urdf_str = """<robot name="chain_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint3" type="revolute">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
    <joint name="joint4" type="revolute">
        <parent link="link3"/>
        <child link="link4"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link4"/>
</robot>"""
    robot_full = Robot.load(urdf_str)
    assert robot_full.spec.num_links == 5
    assert robot_full.spec.num_actuated_joints == 4
    assert robot_full.spec.link_names == ["base", "link1", "link2", "link3", "link4"]
    assert robot_full.spec.actuated_joint_names == ["joint1", "joint2", "joint3", "joint4"]

    robot_sub = Robot.load(urdf_str, base_link_name="link1", ee_link_names=["link3"])
    assert robot_sub.spec.num_links == 3
    assert robot_sub.spec.num_actuated_joints == 2
    assert set(robot_sub.spec.link_names) == {"link1", "link2", "link3"}
    assert set(robot_sub.spec.actuated_joint_names) == {"joint2", "joint3"}

    robot_sub2 = Robot.load(urdf_str, base_link_name="base", ee_link_names=["link2"])
    assert robot_sub2.spec.num_links == 3
    assert robot_sub2.spec.num_actuated_joints == 2
    assert set(robot_sub2.spec.link_names) == {"base", "link1", "link2"}
    assert set(robot_sub2.spec.actuated_joint_names) == {"joint1", "joint2"}


def test_robot_load_with_base_only():
    urdf_str = """<robot name="tree_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint3" type="revolute">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
    <joint name="joint2b" type="revolute">
        <parent link="link1"/>
        <child link="link2b"/>
        <origin xyz="0.1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2b"/>
</robot>"""
    robot_full = Robot.load(urdf_str)
    assert robot_full.spec.num_links == 5
    assert robot_full.spec.num_actuated_joints == 4

    robot_sub = Robot.load(urdf_str, base_link_name="link1")
    assert robot_sub.spec.num_links == 4
    assert robot_sub.spec.num_actuated_joints == 3
    assert set(robot_sub.spec.link_names) == {"link1", "link2", "link3", "link2b"}
    assert set(robot_sub.spec.actuated_joint_names) == {"joint2", "joint3", "joint2b"}

    robot_sub2 = Robot.load(urdf_str, base_link_name="link2")
    assert robot_sub2.spec.num_links == 2
    assert robot_sub2.spec.num_actuated_joints == 1
    assert set(robot_sub2.spec.link_names) == {"link2", "link3"}
    assert set(robot_sub2.spec.actuated_joint_names) == {"joint3"}


def test_robot_load_with_ee_only():
    urdf_str = """<robot name="chain_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint3" type="revolute">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
</robot>"""
    robot_sub = Robot.load(urdf_str, ee_link_names=["link2"])
    assert robot_sub.spec.num_links == 3
    assert robot_sub.spec.num_actuated_joints == 2
    assert set(robot_sub.spec.link_names) == {"base", "link1", "link2"}
    assert set(robot_sub.spec.actuated_joint_names) == {"joint1", "joint2"}


def test_mimic_joint_names():
    urdf_str = """<robot name="mimic_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0.1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
        <mimic joint="joint1" multiplier="2.0" offset="0.1"/>
    </joint>
    <link name="link2"/>
</robot>"""
    robot = Robot.load(urdf_str)
    assert robot.spec.num_actuated_joints == 1
    assert robot.spec.actuated_joint_names == ["joint1"]
    assert robot.spec.has_mimic_joints is True
    assert robot.spec.mimic_joint_names == ["joint2"]


if __name__ == "__main__":
    test_robot_load()
    test_robot_load_with_base_and_ee()
    test_robot_load_with_base_only()
    test_robot_load_with_ee_only()
    test_mimic_joint_names()
