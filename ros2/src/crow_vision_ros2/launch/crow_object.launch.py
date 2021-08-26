import launch_ros.actions
from launch.conditions import IfCondition, UnlessCondition
from launch import LaunchDescription, LaunchContext
from launch.actions import LogInfo, IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir, LaunchConfiguration
import os
import rclpy
from ros2topic import api

def generate_launch_description():
    launchConfigs = []

    # context = LaunchContext()
    use_edge = LaunchConfiguration("use_edge")
    edge_arg = DeclareLaunchArgument("use_edge", default_value="True")
    use_pose = LaunchConfiguration("use_pose")
    pose_arg = DeclareLaunchArgument("use_pose", default_value="True")

    #2. detector (2D vision)
    edge_detector_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='detector_edge',
        output='log',
        condition=IfCondition(use_edge),
        node_name="detector_edge",
        parameters=[{
                    "weights": "data/yolact/weights/weights_yolact_kuka_30/crow_base_25_133333.pth",
                    "config": "data/yolact/weights/weights_yolact_kuka_30/config_train.obj"
                    }]
    )
    launchConfigs.append(edge_detector_node)
    detector_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='detector',
        output='log',
        condition=UnlessCondition(use_edge),
        node_name="detector",
        parameters=[{
                    "weights": "data/yolact/weights/weights_yolact_kuka_30/crow_base_25_133333.pth",
                    "config": "data/yolact/weights/weights_yolact_kuka_30/config_train.obj"
                    }]
    )
    launchConfigs.append(detector_node)

    # afford_detector_node = launch_ros.actions.Node(
    #     package='crow_vision_ros2',
    #     node_executable=yolact_node,
    #     output='log',
    #     node_name="afford_detector_edge",
    #     parameters=[{
    #                 "weights": "data/yolact/weights/weights_yolact_kuka_26/crow_base_52_133333.pth",
    #                 "config": "data/yolact/weights/weights_yolact_kuka_26/config_train.obj"
    #                 }]
    # )
    # launchConfigs.append(afford_detector_node)

    pose_detector_node = launch_ros.actions.Node(
        package='crow_vision_ros2',
        node_executable='detector_pose',
        output='screen',
        node_name="detector_pose",
        condition=IfCondition(use_pose)
        # parameters=[{
        #             "weights": "data/yolact/weights/weights_yolact_kuka_30/crow_base_25_133333.pth",
        #             "config": "data/yolact/weights/weights_yolact_kuka_30/config_train.obj"
        #             }]
    )
    launchConfigs.append(pose_detector_node)

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
        edge_arg,
        pose_arg,
        LogInfo(msg=["Will use Edge? ", LaunchConfiguration('use_edge')]),
        LogInfo(msg=["Will use Pose? ", LaunchConfiguration('use_pose')])
    ] + launchConfigs)
