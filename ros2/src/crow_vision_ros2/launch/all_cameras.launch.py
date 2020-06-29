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


def generate_launch_description():
    camera_configs = []
    for i, device in enumerate(rs.context().devices):
        if device.get_info(rs.camera_info.name).lower() != 'platform camera':
            camera_serial_no = LaunchConfiguration('serial_no', default=device.get_info(rs.camera_info.serial_number))
            camera_base_frame_id = LaunchConfiguration('base_frame_id', default=f'camera{i + 1}_link')

            camera_node = Node(
                package='realsense_node',
                node_executable='realsense_node',
                node_namespace=f"/camera{i + 1}",
                output='screen',
                parameters=[{'serial_no': camera_serial_no,
                             'base_frame_id': camera_base_frame_id,
                             'align_depth': True,
                             'enable_pointcloud': True,
                             'dense_pointcloud': True,
                             }]
            )
            camera_configs.append(camera_node)
    return launch.LaunchDescription(camera_configs)
