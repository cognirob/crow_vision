import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
import pyrealsense2 as rs
import rclpy
import re
import yaml


def generate_launch_description():
    launchConfigs = []
    # launchConfigs.append(LogInfo(msg=PathJoinSubstitution([get_package_share_directory("crow_vision_ros2"), "launch", "all_cameras.launch.py"])))

    all_cam_launcher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([get_package_share_directory("crow_vision_ros2"), "launch", "all_cameras.launch.py"])),
        launch_arguments={'camera_config': LaunchConfiguration('camera_config')}.items()
    )
    launchConfigs.append(LogInfo(msg="Launching all available cameras"))
    launchConfigs.append(all_cam_launcher)

    action_collector_node = Node(
        package='collectors',
        node_executable='image_collector',
        output='screen',
        emulate_tty=True,
        parameters=[{
                    }],
        arguments=[
        ],
    )

    launchConfigs.append(action_collector_node)

    return LaunchDescription([
        DeclareLaunchArgument("camera_config", default_value="rs_60.yaml")
    ] + launchConfigs
    )