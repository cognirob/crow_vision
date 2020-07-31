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
import cv_bridge
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from crow_vision_ros2.utils import make_vector3

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

        self.cvb = cv_bridge.CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

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
        if not mask_msg.masks:
            print("no masks, no party. Quitting early.")
            return  # no mask detections (for some reason)

        cameraData = self.getCameraData(camera)
        masks = [self.cvb.imgmsg_to_cv2(mask, "mono8") for mask in mask_msg.masks]
        class_names, scores = mask_msg.class_names, mask_msg.scores

        # get pointcloud data from ROS2 msg to open3d format
        pcd = convertCloudFromRosToOpen3d(pcl_msg)
        # optimizations for performance:
        pcd = pcd.voxel_down_sample(voxel_size=0.02) #optional, downsampling for speed up
        #optional, remove plane for speedup #Note: may not be reasonable to use, as RS is noisy, problem with detection of objects ON table, ...
        #plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        #pcd = pcd.select_by_index(inliers, invert=True) #drop plane from the pcl

        #o3d.visualization.draw_geometries([pcd])
        print(pcd)

        ##process pcd
        # 1. convert 3d pcl to 2d image-space
        point_cloud = np.asarray(pcd.points).T
        camera_matrix = cameraData["camera_matrix"]
        # assert camera_matrix.shape[1] == camera_matrix.shape[0] == point_cloud.shape[0] == 3, 'matrix must be 3x3, pcl 3xN'
        imspace = np.dot(camera_matrix, point_cloud) # converts pcl (shape 3,N) of [x,y,z] (3D) into image space (with cam_projection matrix) -> [u,v,w] -> [u/w, v/w] which is in 2D
        imspace = imspace / imspace[2] # [u,v,w] -> [u/w, v/w, w/w] -> [u',v'] = 2D
        # assert imspace[2].all() == 1
        imspace = imspace[:2]
        assert imspace.ndim == 2,'should now be in 2D'

        for i, (mask, class_name, score) in enumerate(zip(masks, class_names, scores)):
            # segment PCL & compute median

            a = imspace.astype(int)
            b = np.array(np.where(mask))
            seg_pcd = point_cloud[:, np.where(imspace.T[:, None].astype(int) == np.where(mask))]
            # seg_pcd = point_cloud[:, np.where(imspace.T[:, None].astype(int) == np.where(mask))]

            mean = seg_pcd.mean(axis=1)
            assert len(mean) == 3, 'incorrect mean dim'
            self.sendPosition(cameraData["optical_frame"], class_name + f"_{i}", mask_msg.header.stamp, mean)

            #TODO 3d bbox?
            #bbox3d = pcd.get_axis_aligned_bounding_box()
            #print(bbox3d.get_print_info())

            #TODO if we wanted, create back a pcl from seg_pcd and publish it as ROS PointCloud2
            pcd.points = o3d.utility.Vector3dVector(seg_pcd)
            o3d.visualization.draw_geometries([pcd])
            print(seg_pcd.shape)

    def sendPosition(self, camera_frame, object_frame, time, xyz):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = time
        tf_msg.header.frame_id = camera_frame
        tf_msg.child_frame_id = object_frame
        tf_msg.transform.translation = make_vector3(xyz)

        self.tf_broadcaster.sendTransform(tf_msg)

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
