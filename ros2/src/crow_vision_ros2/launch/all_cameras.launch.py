import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import pyrealsense2 as rs
import rclpy
import re
import yaml
from crow_vision_ros2.utils.get_camera_transformation import CameraGlobalTFGetter


def launch_cameras(launchContext, globalTFGetter=None):

    frames = ["accel_frame_id",
              "accel_optical_frame_id",
              "aligned_depth_to_color_frame_id",
              "aligned_depth_to_fisheye1_frame_id",
              "aligned_depth_to_fisheye_frame_id",
              "aligned_depth_to_infra1_frame_id",
              "aligned_depth_to_infra_frame_id",
              "base_frame_id",
              "color_frame_id",
              "color_optical_frame_id",
              "depth_frame_id",
              "depth_optical_frame_id",
              "fisheye1_frame_id",
              "fisheye1_optical_frame_id",
              "fisheye2_frame_id",
              "fisheye2_optical_frame_id",
              "fisheye_frame_id",
              "fisheye_optical_frame_id",
              "gyro_frame_id",
              "gyro_optical_frame_id",
              "imu_optical_frame_id",
              "infra1_frame_id",
              "infra1_optical_frame_id",
              "infra2_frame_id",
              "infra2_optical_frame_id",
              "infra_frame_id",
              "infra_optical_frame_id",
              "odom_frame_id",
              "pose_frame_id",
              "pose_optical_frame_id"
              ]
    frame_regex = re.compile(r"(\w+_frame)_id")

    camera_configs = []
    devices = list(rs.context().query_devices())

    camera_namespaces = []
    camera_serials = []
    camera_transforms = []
    for cam_id, device in enumerate(devices):
        cam_name = device.get_info(rs.camera_info.name)
        if cam_name.lower() == 'platform camera':
            continue

        camera_namespace = f"camera{cam_id + 1}"
        camera_namespaces.append("/" + camera_namespace)
        device_serial = str(device.get_info(rs.camera_info.serial_number))
        camera_serials.append(device_serial)
        camera_configs.append(LogInfo(msg=f"Launching device {cam_name} with serial number {device_serial} in namespace /{camera_namespace}."))

        camera_frames_dict = {f: f'camera{cam_id + 1}_' + frame_regex.search(f).group(1) for f in frames}
        camera_frames_dict['base_frame_id'] = f'camera{cam_id + 1}_link'

        if globalTFGetter is not None:
            transform = globalTFGetter.get_camera_transformation(device_serial)
            camera_configs.append(LogInfo(msg=f"Adding workspace transform for {device_serial} ({camera_namespace}) with args {transform}."))
            camera_configs.append(LogInfo(msg=f"Transform for frame {camera_frames_dict['base_frame_id']}."))
            # camera_configs.append(
            #     Node(
            #         package='tf2_ros',
            #         node_executable='static_transform_publisher',
            #         arguments=transform.split() + [
            #                 "workspace_frame",
            #                 camera_frames_dict['base_frame_id'],
            #                 # camera_frames_dict['color_optical_frame_id'],
            #         ],
            #         output='screen',
            #         emulate_tty=True
            #     )
            # )
            camera_transforms.append(transform)
        else:
            camera_transforms.append(None)


        config_file = os.path.join(
            get_package_share_directory('crow_vision_ros2'),
            'config',
            launch.substitutions.LaunchConfiguration('camera_config').perform(launchContext)
        )
        with open(config_file, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)

        # print(device_serial)
        launchParams = {
            'align_depth': True,
                        'enable_infra1': False,
                        'enable_infra2': False,
                        'serial_no': "_" + device_serial,
                        }

        launchParams = {**launchParams, **camera_frames_dict, **config_dict}

        camera_node = Node(
            package='realsense2_camera',
            node_executable='realsense2_camera_node',
            node_namespace=camera_namespace,
            parameters=[launchParams],
            output='log',
            emulate_tty=True
        )
        camera_configs.append(camera_node)

    calibrator_node = Node(
        package='crow_vision_ros2',
        node_executable='calibrator',
        output='screen',
        emulate_tty=True,
        parameters=[{
                    "halt_calibration": True
                    }],
        arguments=[
            "--camera_namespaces", ' '.join(camera_namespaces),
            "--camera_serials", ' '.join(camera_serials),
            "--camera_transforms", ' | '.join(camera_transforms),
        ],
    )
    camera_configs.append(calibrator_node)
    return [
        LogInfo(msg=f"Found devices: {devices}"),
    ] + camera_configs


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('crow_vision_ros2'),
        'config',
        'camera_transformation_data.yaml'
        # launch.substitutions.LaunchConfiguration('camera_config').perform(launchContext)
    )
    cgtfg = CameraGlobalTFGetter(config_file)

    return LaunchDescription([
        DeclareLaunchArgument("camera_config", default_value="rs_native.yaml"),
        LogInfo(msg=["Configuration file used for cameras: ", LaunchConfiguration('camera_config')]),
        # ,
        OpaqueFunction(function=launch_cameras, kwargs={'globalTFGetter': cgtfg})
    ])
