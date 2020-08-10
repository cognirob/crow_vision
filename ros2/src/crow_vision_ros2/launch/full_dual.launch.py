from launch import LaunchDescription
import launch_ros.actions
from launch.actions import LogInfo, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir, ThisLaunchFile
import os
import rclpy
from ros2topic import api

def generate_launch_description():
    launchConfigs = []

    all_cam_launcher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ThisLaunchFileDir(), os.path.sep, "all_cameras.launch.py"]),
        # launch_arguments={'node_name': 'bar'}.items()
    )
    launchConfigs.append(LogInfo(msg="Launching all available cameras"))
    launchConfigs.append(all_cam_launcher)

    detector_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='detector',
        output='screen',
        node_name="detector_coco",
        parameters=[{
                    "weights": "data/yolact/weights/yolact_base_54_800000.pth",
                    "config": "yolact_base_config"
                    # "config": "data/yolact/weights/config_train.obj"
                    }]
    )
    launchConfigs.append(detector_node)
    detector_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='detector',
        output='screen',
        node_name="detector_crow",
        parameters=[{
                    "weights": "data/yolact/weights/crow_base_25_133333.pth",
                    "config": "data/yolact/weights/config_train.obj"
                    }]
    )
    launchConfigs.append(detector_node)
    locator_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='locator',
        output='screen'
    )
    launchConfigs.append(locator_node)

    launchDescription = LaunchDescription(launchConfigs)

    return launchDescription

# rclpy.init()
# executor = rclpy.get_global_executor()

# print("> " + str(executor.get_nodes()))
