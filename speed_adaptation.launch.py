from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import xacro
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, TimerAction,
)
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project = Path(__file__).resolve().parent
    python = LaunchConfiguration("python")
    lateral_offset = LaunchConfiguration("human_lateral_offset")
    mesh_default = (Path.home() / ".gz/fuel/fuel.gazebosim.org/openrobotics/models"
                    / "standing person/3/meshes/standing.dae")

    log_output = {"both": "log"}

    def start(context):
        executable = python.perform(context)
        mesh = Path(LaunchConfiguration("human_mesh").perform(context)).expanduser().resolve()
        if which(executable) is None or not mesh.is_file():
            return []

        # gazebo загружает модели с задержкой поэтому файлы хранятся до конца запуска
        models = TemporaryDirectory(prefix="speed_adaptation_")
        model_dir = Path(models.name)
        robot = xacro.process_file(str(project / "robot.urdf.xacro")).toxml()
        robot_file = model_dir / "robot.urdf"
        robot_file.write_text(robot, encoding="utf-8")

        human = ElementTree.parse(project / "human.sdf")
        human.find(".//mesh/uri").text = f"file://{mesh}"
        human_file = model_dir / "human.sdf"
        human.write(human_file, encoding="utf-8", xml_declaration=True)

        def cleanup(_context):
            models.cleanup()

        gazebo = ExecuteProcess(cmd=[
            "gz", "sim", "-s", "-r", "--headless-rendering", "-v", "2",
            str(project / "world.sdf"),
        ], output=log_output)
        robot_state = Node(
            package="robot_state_publisher", executable="robot_state_publisher",
            ros_arguments=["--log-level", "warn"],
            parameters=[{"robot_description": robot, "use_sim_time": True}],
        )
        bridge = Node(
            package="ros_gz_bridge", executable="parameter_bridge",
            ros_arguments=["--log-level", "warn"],
            remappings=[("/world/actors/clock", "/clock")],
            arguments=[
                "/world/actors/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
                "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                "/lidar/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
                "/model/fun_robot/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                "/model/fun_robot/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            ],
            output=log_output,
        )
        spawn_robot = Node(
            package="ros_gz_sim", executable="create",
            ros_arguments=["--log-level", "warn"],
            arguments=["--world", "actors", "--file", str(robot_file),
                       "--name", "fun_robot", "-x", "2.0", "-y", "0.0", "-z", "0.16"],
            output=log_output,
        )
        spawn_human = ExecuteProcess(cmd=[
            "gz", "service", "-s", "/world/actors/create",
            "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000", "--req",
            [
                f'sdf_filename: "{human_file}", name: "human_static", '
                'pose: {position: {x: -0.1, y: ', lateral_offset,
                ', z: 0.0}, orientation: {z: 0.707107, w: 0.707107}}',
            ],
        ], output=log_output)

        def run_script(filename, *args):
            return ExecuteProcess(
                cmd=[python, str(project / filename), "--ros-args", *args],
                cwd=str(project), output=log_output,
            )

        return [
            RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])),
            gazebo, robot_state, bridge,
            TimerAction(period=5.0, actions=[spawn_robot]),
            TimerAction(period=7.0, actions=[spawn_human]),
            TimerAction(period=10.0, actions=[
                run_script("human_tracking.py", "-p", "use_sim_time:=true", "-p",
                           ["show_window:=", LaunchConfiguration("show_window")],
                           "-p", "inference_size:=320"),
                run_script("speed_controller.py", "-p", "use_sim_time:=true",
                           "-p", "kf_r:=0.0004", "-p", "position_topic:=/human/position"),
                run_script("human_motion.py", "-p", ["lateral_offset:=", lateral_offset]),
            ]),
        ]

    return LaunchDescription([
        DeclareLaunchArgument("python", default_value=str(project / "yolo_venv/bin/python")),
        DeclareLaunchArgument("show_window", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("human_lateral_offset", default_value="0.0",
                              description="Поперечное смещение человека в метрах"),
        DeclareLaunchArgument("human_mesh", default_value=str(mesh_default),
                              description="Путь к мешу Standing Person из Gazebo Fuel"),
        OpaqueFunction(function=start),
    ])