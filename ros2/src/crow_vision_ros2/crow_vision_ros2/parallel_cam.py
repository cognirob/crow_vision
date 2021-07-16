import rclpy #add to package.xml deps
from rclpy.node import Node
from ros2param.api import call_get_parameters

import sensor_msgs
import std_msgs

from crow_msgs.msg import DetectionMask, DetectionBBox, BBox
from cv_bridge import CvBridge

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import cv2
import torch
import numpy as np

import commentjson as json
import pkg_resources
import argparse
import time
from time import sleep
import copy

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from collections import deque


qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ParallelCam(Node):
    TIMER_DELAY = 0.1
    TIMER_SLEEP = 1
    CAMERA_SLEEP = 1
    BUFF_LEN = 100

    def __init__(self, config='config.json'):
        super().__init__('parallel_cam')

        self.cameras, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]
        while len(self.cameras) == 0:
            self.get_logger().warn("Waiting for any cameras!")
            time.sleep(2)
            self.cameras, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]

        self.last_stamps = {}
        self.buffers = {}
        for cam in self.cameras:
            camera_topic=cam+"/color/image_raw"
            listener = self.create_subscription(msg_type=sensor_msgs.msg.Image,
                                                topic=camera_topic,
                                                callback=lambda msg, cam=cam: self.input_callback(msg, cam),
                                                callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
                                                qos_profile=10)
            self.get_logger().info(f"lambda mode")
            self.get_logger().info('Input listener created on topic: "%s"' % camera_topic)
            self.last_stamps[cam] = self.get_clock().now()
            self.buffers[cam] = deque(maxlen=self.BUFF_LEN)

        # timer
        self.create_timer(self.TIMER_DELAY, self.timer_cb, callback_group=rclpy.callback_groups.ReentrantCallbackGroup())
        # self.create_timer(self.TIMER_DELAY, self.timer_cb, callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup())
        self.last_stamps["timer"] = self.get_clock().now()

        # Init neural net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        self.cnn = NeuralNetwork().to(self.device)

    def timer_cb(self):
        # self.get_logger().info(f">>>>>>> timer")
        # new_time = self.get_clock().now()
        # self.get_logger().info(f"delay {(new_time - self.last_stamps['timer']).nanoseconds / 1e9:.3f}")
        # self.last_stamps["timer"] = new_time

        for cam in self.cameras:
            if len(self.buffers[cam]) == self.BUFF_LEN:
                self.get_logger().info(f">>>>>>> processing buffer {cam}")
                # tmp_buff = copy.deepcopy(self.buffers[cam])  # temp buffer copy
                data = torch.stack(list(self.buffers[cam])).squeeze()
                self.buffers[cam].clear()
                out = self.cnn(data)
                sleep(self.TIMER_SLEEP)
                self.get_logger().info(f"buffer processed {cam} <<<<<<<")


    def input_callback(self, msg, cam):
        # self.get_logger().info(f">>>>>>> cam {cam}, {msg.header.frame_id}")
        new_time = rclpy.time.Time.from_msg(msg.header.stamp)
        self.get_logger().info(f">>>>>>> cam {cam}, delay {(new_time - self.last_stamps[cam]).nanoseconds / 1e9:.3f}")
        # self.get_logger().info(f"delay {(new_time - self.last_stamps[cam]).nanoseconds / 1e9:.3f}")
        self.last_stamps[cam] = new_time
        # self.get_logger().info(f">>>>>>> cam {cam}: {self.cnn(torch.rand(1, 28, 28, device=self.device))}")
        self.buffers[cam].append(torch.rand(1, 28, 28, device=self.device))
        # sleep(self.CAMERA_SLEEP)
        self.get_logger().info(f"\tcam {cam} <<<<<")


def main(args=None):
    rclpy.init(args=args)
    try:
        pc = ParallelCam()
        n_threads = len(pc.cameras) + 2  # +1 for timer, +1 as backup
        mte = rclpy.executors.MultiThreadedExecutor(num_threads=n_threads, context=rclpy.get_default_context())
        rclpy.spin(pc, executor=mte)
        pc.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
