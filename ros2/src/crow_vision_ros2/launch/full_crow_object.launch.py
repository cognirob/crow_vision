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

    #1. cams
    all_cam_launcher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ThisLaunchFileDir(), os.path.sep, "all_cameras.launch.py"]),
        launch_arguments={'camera_config': LaunchConfiguration('camera_config')}.items()
    )
    launchConfigs.append(LogInfo(msg="Launching all available cameras"))
    launchConfigs.append(all_cam_launcher)

    #2. detector (2D vision)
    detector_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='detector',
        output='screen',
        node_name="detector_crow",
        parameters=[{
                    "weights": "data/yolact/weights/weights_yolact_kuka_21/crow_base_25_133333.pth",
                    "config": "data/yolact/weights/weights_yolact_kuka_21/config_train.obj"
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

    #4. matcher (IPC object fitting)
    matcher_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='matcher',
        output='screen'
    )
    launchConfigs.append(matcher_node)

    #TODO tracker (trajectory tracking)


    launchDescription = LaunchDescription(launchConfigs)

    return LaunchDescription([
        DeclareLaunchArgument("camera_config", default_value="rs_native.yaml")
    ] + launchConfigs)
