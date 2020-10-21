import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

#PointCloud2
from crow_msgs.msg import DetectionMask, SegmentedPointcloud
from sensor_msgs.msg import PointCloud2, PointField
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
            self.create_subscription(SegmentedPointcloud, pclTopic, self.detection_callback, qos_profile=qos)

        self.filtered_publisher = self.create_publisher(PoseArray, "filtered_poses", qos)
        self.timer = self.create_timer(self.UPDATE_INTERVAL, self.filter_update)

    def filter_update(self):
        self.particle_filter.update()
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


    def detection_callback(self, pcl_msg):
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
