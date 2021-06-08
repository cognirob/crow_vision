from launch import LaunchDescription
import launch_ros.actions
from launch.actions import LogInfo, IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir, LaunchConfiguration
import os
import rclpy
from ros2topic import api

def generate_launch_description():
    launchConfigs = []

    #2. detector (2D vision)
    detector_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='detector',
        output='screen',
        node_name="detector_crow",
        parameters=[{
                    "weights": "data/yolact/weights/weights_yolact_kuka_30/crow_base_25_133333.pth",
                    "config": "data/yolact/weights/weights_yolact_kuka_30/config_train.obj"
                    }]
    )
    launchConfigs.append(detector_node)

    #3. locator (3D segmented pcl)
    locator_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='locator',
        output='screen'
    )
    launchConfigs.append(locator_node)

    #TODO merger (merge pcl from cameras)

    # #4. filter
    # filter_node = launch_ros.actions.Node(
    #     package='crow_vision_ros2',
    #     node_executable='filter',
    #     output='screen'
    # )
    # launchConfigs.append(filter_node)

    launchDescription = LaunchDescription(launchConfigs)

    return LaunchDescription([
        DeclareLaunchArgument("camera_config", default_value="rs_native.yaml")
    ] + launchConfigs)
