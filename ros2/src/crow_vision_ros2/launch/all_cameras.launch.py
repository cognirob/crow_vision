# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /* Author: Gary Liu */

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


def launch_cameras(launchContext):

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
    for cam_id, device in enumerate(devices):
        cam_name = device.get_info(rs.camera_info.name)
        if cam_name.lower() == 'platform camera':
            continue

        camera_namespace = f"camera{cam_id + 1}"
        camera_namespaces.append("/" + camera_namespace)
        device_serial = str(device.get_info(rs.camera_info.serial_number))
        camera_configs.append(LogInfo(msg=f"Launching device {cam_name} with serial number {device_serial} in namespace /{camera_namespace}."))

        camera_frames_dict = {f: f'camera{cam_id + 1}_' + frame_regex.search(f).group(1) for f in frames}
        camera_frames_dict['base_frame_id'] = f'camera{cam_id + 1}_link'


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
        ],
    )
    camera_configs.append(calibrator_node)
    return [
        LogInfo(msg=f"Found devices: {devices}"),
    ] + camera_configs


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument("camera_config", default_value="rs_native.yaml"),
        LogInfo(msg=["Configuration file used for cameras: ", LaunchConfiguration('camera_config')]),
        # ,
        OpaqueFunction(function=launch_cameras)
    ])
