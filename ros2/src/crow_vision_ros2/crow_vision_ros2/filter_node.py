import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

#PointCloud2
from crow_msgs.msg import SegmentedPointcloud
import open3d as o3d
from crow_vision_ros2.utils import make_vector3, ftl_pcl2numpy, ftl_numpy2pcl
from crow_vision_ros2.filters import ParticleFilter, object_properties

#TF
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import numpy as np

import pkg_resources
import time


class ParticleFilterNode(Node):
    UPDATE_INTERVAL = 0.05

    def __init__(self, node_name="particle_filter"):
        super().__init__(node_name)
        self.cameras = []
        while(len(self.cameras) == 0):
            try:
                self.cameras , self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]
            except:
                self.get_logger().error("getting cameras failed. Retrying in 2s")
                time.sleep(2)
        assert len(self.cameras) > 0

        self.seg_pcl_topics = [cam + "/" + "detections/segmented_pointcloud" for cam in self.cameras] #input segmented pcl data

        self.particle_filter = ParticleFilter()

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        for i, (cam, pclTopic) in enumerate(zip(self.cameras, self.seg_pcl_topics)):
            self.get_logger().info("Created subscriber for segmented_pcl \"{}\"".format(pclTopic))
            # self.create_subscription(SegmentedPointcloud, pclTopic, self.detection_cb, qos_profile=qos)
            sub = message_filters.Subscriber(self, SegmentedPointcloud, pclTopic, qos_profile=qos)
            self.cache = message_filters.Cache(sub, 15, allow_headerless=True)
            # self.cache.registerCallback(self.cache_cb)

        self.lastFilterUpdate = self.get_clock().now()
        self.lastMeasurement = self.get_clock().now()
        self.updateWindowDuration = rclpy.time.Duration(seconds=0.05)
        self.timeSlipWindow = rclpy.time.Duration(seconds=1.5)
        self.measurementTolerance = rclpy.time.Duration(seconds=0.00001)
        self.lastUpdateMeasurementDDiff = rclpy.time.Duration(seconds=2)
        self.filtered_publisher = self.create_publisher(PoseArray, "filtered_poses", qos)
        self.timer = self.create_timer(self.UPDATE_INTERVAL, self.filter_update)

    def add_and_process(self, messages):
        if type(messages) is not list:
            messages = [messages]

        if len(messages) == 0:
            return

        self.get_logger().info(f"Adding {len(messages)} measurements to the filter")
        latest = self.lastMeasurement
        for pcl_msg in messages:
            self.frame_id = pcl_msg.header.frame_id
            if latest < rclpy.time.Time.from_msg(pcl_msg.header.stamp):
                latest = rclpy.time.Time.from_msg(pcl_msg.header.stamp)
            label = pcl_msg.label
            try:
                class_id = next((k for k, v in object_properties.items() if label in v["name"]))
            except StopIteration as e:
                class_id = -1

            pcl, _, _ = ftl_pcl2numpy(pcl_msg.pcl)
            self.particle_filter.add_measurement((pcl, class_id))

        now = self.get_clock().now()
        self.lastMeasurement = latest
        self.lastUpdateMeasurementDDiff = now - self.lastMeasurement
        self.update(now)

    def update(self, now=None):
        self.particle_filter.update()
        if now is not None:
            self.lastFilterUpdate = now
        else:
            self.lastFilterUpdate = self.get_clock().now()

        if self.particle_filter.n_models > 0:
            estimates = self.particle_filter.getEstimates()
            self.get_logger().info(str(estimates))
            poses = []
            for pose, label in estimates:
                pose_msg = Pose()
                pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = pose.tolist()
                poses.append(pose_msg)
            pose_array_msg = PoseArray(poses=poses)
            pose_array_msg.header.stamp = self.get_clock().now().to_msg()
            pose_array_msg.header.frame_id = self.frame_id
            self.filtered_publisher.publish(pose_array_msg)

    def filter_update(self):
        latest_time = self.cache.getLastestTime()
        if latest_time is not None:

            oldest_time = self.cache.getOldestTime()
            # orig_oldest = oldest_time.seconds_nanoseconds
            # if oldest_time <= (self.lastFilterUpdate - self.timeSlipWindow):
            #     oldest_time = self.lastFilterUpdate - self.timeSlipWindow
            # # if oldest_time <= (self.lastFilterUpdate - self.lastUpdateMeasurementDDiff):
            # #     oldest_time = self.lastFilterUpdate - self.lastUpdateMeasurementDDiff
            if oldest_time < self.lastMeasurement:
                oldest_time = self.lastMeasurement
                # oldest_time += self.measurementTolerance

            anyupdate = False
            while oldest_time <= latest_time:
                next_time = oldest_time + self.updateWindowDuration
                messages = self.cache.getInterval(oldest_time, next_time)
                oldest_time = next_time
                if len(messages) == 0:
                    continue
                self.add_and_process(messages)
                anyupdate = True

            # if anyupdate:
            #     self.lastMeasurement += self.measurementTolerance
        else:
            self.update()


    def cache_cb(self, *args):
        pass

    def detection_cb(self, pcl_msg):
        self.get_logger().info("got some pcl")
        print(pcl_msg.label)
        self.frame_id = pcl_msg.header.frame_id
        label = pcl_msg.label
        try:
            class_id = next((k for k, v in object_properties.items() if label in v["name"]))
        except StopIteration as e:
            class_id = -1

        pcl, _, _ = ftl_pcl2numpy(pcl_msg.pcl)
        self.particle_filter.add_measurement((pcl, class_id))


def main():
    rclpy.init()

    pfilter = ParticleFilterNode()

    rclpy.spin(pfilter)
    pfilter.destroy_node()


if __name__ == "__main__":
    main()