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
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import pyrealsense2 as rs
import rclpy
import re


def generate_launch_description():
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
    for cam_id, device in enumerate(rs.context().devices):
        if device.get_info(rs.camera_info.name).lower() != 'platform camera':
            camera_frames_dict = {f: f'camera{cam_id + 1}_' + frame_regex.search(f).group(1) for f in frames}
            camera_frames_dict['base_frame_id'] = f'camera{cam_id + 1}_link'

            launchParams = {'align_depth': True,
                            'enable_pointcloud': True,
                            'dense_pointcloud': True,
                            'serial_no': str(device.get_info(rs.camera_info.serial_number)),
                            }

            launchParams = {**launchParams, **camera_frames_dict}

            camera_node = Node(
                package='realsense2_node',
                node_executable='realsense2_node',
                node_namespace=f"camera{cam_id + 1}",
                parameters=[launchParams],
                output='screen',
                emulate_tty=True
            )
            camera_configs.append(camera_node)

    calibrator_node = Node(
        package='crow_vision_ros2',
        node_executable='calibrator',
        output='screen',
        parameters=[{
                    "halt_calibration": True
                    }]
    )
    camera_configs.append(calibrator_node)
    return launch.LaunchDescription(camera_configs)
