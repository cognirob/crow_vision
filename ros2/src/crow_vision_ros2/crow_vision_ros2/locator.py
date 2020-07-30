import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters
from crow_msgs.msg import DetectionMask
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
import json
import numpy as np

import pkg_resources
from .utils.convertor_ros_open3d import convertCloudFromOpen3dToRos, convertCloudFromRosToOpen3d
import open3d as o3d


class Locator(Node):

    def __init__(self, node_name="locator"):
        super().__init__(node_name)
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_nodes", "camera_intrinsics", "camera_frames"]).values]
        self.camera_instrinsics = [json.loads(cintr) for cintr in self.camera_instrinsics]
        self.mask_topics = [cam + "/" + "detections/masks" for cam in self.cameras]
        self.pcl_topics = [cam + "/" + "pointcloud" for cam in self.cameras]

        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)
        for cam, pclTopic, maskTopic in zip(self.cameras, self.pcl_topics, self.mask_topics):
            self.subPCL = message_filters.Subscriber(self, PointCloud2, pclTopic, qos_profile=10)
            self.subMasks = message_filters.Subscriber(self, DetectionMask, maskTopic, qos_profile=10)
            self.get_logger().info("LOCATOR: Created Subscriber for masks at topic: {}".format(maskTopic))
            self.sync = message_filters.ApproximateTimeSynchronizer([self.subPCL, self.subMasks], 20, 0.005)
            self.sync.registerCallback(lambda pcl_msg, mask_msg, cam=cam: self.detection_callback(pcl_msg, mask_msg, cam))

    def detection_callback(self, pcl_msg, mask_msg, camera):
        #print(self.getCameraData(camera))
        # get pointcloud data from ROS2 msg to open3d format
        pcd = convertCloudFromRosToOpen3d(pcl_msg)
        # optimizations for performance:
        pcd = pcd.voxel_down_sample(voxel_size=0.02) #optional, downsampling for speed up
        #optional, remove plane for speedup
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        pcd = pcd.select_by_index(inliers, invert=True) #drop plane from the pcl
        #o3d.visualization.draw_geometries([pcd], zoom=0.8, #FIXME neumim to vykreslit
        #                          front=[-0.4999, -0.1659, -0.8499],
        #                          lookat=[2.1813, 2.0619, 2.0999],
        #                          up=[0.1204, -0.9852, 0.1215])
        print(pcd)

        #process pcd
        mean, cov = pcd.compute_mean_and_covariance()
        print(mean)
        bbox3d = pcd.get_axis_aligned_bounding_box()
        print(bbox3d.get_print_info())

    def getCameraData(self, camera):
        idx = self.cameras.index(camera)
        return {
            "camera": camera,
            "image_topic": self.image_topics[idx],
            "camera_matrix": np.array(self.camera_instrinsics[idx]["camera_matrix"]),
            "distortion_coefficients": np.array(self.camera_instrinsics[idx]["distortion_coefficients"]),
            "optical_frame": self.camera_frames[idx],
            "mask_topic": self.mask_topics[idx],
            "pcl_topic": self.pcl_topics[idx],
        }

def main():
    rclpy.init()

    locator = Locator()

    rclpy.spin(locator)
    locator.destroy_node()


if __name__ == "__main__":
    main()
