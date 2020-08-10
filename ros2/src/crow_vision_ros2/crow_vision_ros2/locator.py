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
import cv2

import pkg_resources
from .utils.convertor_ros_open3d import convertCloudFromOpen3dToRos, convertCloudFromRosToOpen3d
import open3d as o3d
from time import time
from crow_vision_ros2.utils import point_cloud2 as pc2
from ctypes import * # convert float to uint32


class Locator(Node):

    def __init__(self, node_name="locator"):
        super().__init__(node_name)
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_nodes", "camera_intrinsics", "camera_frames"]).values]
        self.camera_instrinsics = [json.loads(cintr) for cintr in self.camera_instrinsics]
        self.mask_topics = [cam + "/" + "detections/masks" for cam in self.cameras]
        self.pcl_topics = [cam + "/" + "pointcloud" for cam in self.cameras]

        self.cvb = cv_bridge.CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.mask_dtype = {'names':['f{}'.format(i) for i in range(2)], 'formats':2 * [np.int32]}

        self.pubPCL = self.create_publisher(PointCloud2, "test_pcl", qos_profile=10)
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

        start = time()
        field_names=[field.name for field in pcl_msg.fields]
        ## convert from ROS pcl_msg to np.array , drop nans
        cloud_data = list(pc2.read_points(pcl_msg, skip_nans=True))
        #cloud_data = np.array(pcl_msg.data).view(np.float32).reshape(4, -1).T #TODO try replacing the above with this and drop pc2 file
        #cloud_data = cloud_data[np.where(~np.isnan(cloud_data).any(axis=1))]
        # xyz = np.array([(x,y,z) for x,y,z,rgb in cloud_data])
        xyz = np.array(cloud_data)[:, :3]
        end = time()
        print("Convert: ", end - start)
        ## get pointcloud data from ROS2 msg to open3d format
        # pcd = convertCloudFromRosToOpen3d(pcl_msg)
        # optimizations for performance:
        # pcd = pcd.voxel_down_sample(voxel_size=0.02) #optional, downsampling for speed up
        #optional, remove plane for speedup #Note: may not be reasonable to use, as RS is noisy, problem with detection of objects ON table, ...
        #plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        #pcd = pcd.select_by_index(inliers, invert=True) #drop plane from the pcl

        #o3d.visualization.draw_geometries([pcd])

        ##process pcd
        # 1. convert 3d pcl to 2d image-space
        start = time()
        point_cloud = xyz.T
        camera_matrix = cameraData["camera_matrix"]
        imspace = np.dot(camera_matrix, point_cloud) # converts pcl (shape 3,N) of [x,y,z] (3D) into image space (with cam_projection matrix) -> [u,v,w] -> [u/w, v/w] which is in 2D
        imspace = imspace / imspace[2] # [u,v,w] -> [u/w, v/w, w/w] -> [u',v'] = 2D
        imspace[np.where(np.isnan(imspace))] = -1
        assert np.isnan(imspace).any() == False, 'must not have NaN element'
        #imspace[np.where(np.isnan(imspace))] = -1
        imspace = imspace[:2, :].astype(np.int32)
        end = time()
        print("Transform: ", end - start)

        # IDX_RGB_IN_FIELD = 3
        # convert_rgbUint32_to_tuple = lambda rgb_uint32: (
        #   (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
        # )
        # convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
        #   int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
        # )
        # if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
        #     rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        # else:
        #     rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        # colors = np.array(rgb)

        # img = np.zeros((480, 640, 3), dtype=np.uint8)
        # for i in range(imspace.shape[1]):
        #     # if (imspace[:, i] > np.r_[640, 480]).any() or (imspace[:, i] < np.r_[0, 0]).any():
        #     #     continue
        #     img[imspace[1, i], imspace[0, i], :] = colors[i, :]
        # cv2.imshow("img", img)
        # cv2.imshow("mask", masks[0] * 255)
        # cv2.waitKey(1)

        for i, (mask, class_name, score) in enumerate(zip(masks, class_names, scores)):
            # segment PCL & compute median
            start = time()
            where = self.compareMaskPCL(np.array(np.where(mask.T)), imspace)
            seg_pcd = point_cloud[:, where]

            mean = np.median(seg_pcd, axis=1)
            print("{} Centroid: {} accuracy: {}".format(class_name, mean, score))
            assert len(mean) == 3, 'incorrect mean dim'
            self.sendPosition(cameraData["optical_frame"], class_name + f"_{i}", mask_msg.header.stamp, mean)
            end = time()
            print("Filter: ", end - start)
            #TODO 3d bbox?
            #bbox3d = pcd.get_axis_aligned_bounding_box()
            #print(bbox3d.get_print_info())

            #TODO if we wanted, create back a pcl from seg_pcd and publish it as ROS PointCloud2
            # new_pcl = o3d.geometry.PointCloud()
            # new_pcl.points = o3d.utility.Vector3dVector(seg_pcd.T)
            # print(new_pcl)
            # o3d.visualization.draw_geometries([pcd])
            # print(seg_pcd.shape)
            # self.pubPCL.publish(convertCloudFromOpen3dToRos(new_pcl, pcl_msg.header.stamp, cameraData["optical_frame"]))

    def compareMaskPCL(self, mask_array, projected_points):
        a = mask_array.T.astype(np.int32).copy()
        b = projected_points.T.copy()
        self.mask_dtype = {'names':['f{}'.format(i) for i in range(2)], 'formats':2 * [np.int32]}
        result = np.intersect1d(a.view(self.mask_dtype), b.view(self.mask_dtype), return_indices=True)

        return result[2]

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
            # "camera_matrix": np.array([383.591, 0, 318.739, 0, 383.591, 237.591, 0, 0, 1]).reshape(3, 3),
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
