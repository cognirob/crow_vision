import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from ros2param.api import call_get_parameters
import message_filters

#Pointcloud
from crow_msgs.msg import DetectionMask, SegmentedPointcloud
from sensor_msgs.msg import PointCloud2
from crow_vision_ros2.utils import ftl_pcl2numpy, ftl_numpy2pcl, make_vector3, make_quaternion

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import json
import numpy as np
import cv2
import cv_bridge

import pkg_resources
import time
import tf2_py as tf
import tf2_ros
import transforms3d as tf3

from ctypes import *  # convert float to uint32
from numba import jit
from crow_control.utils import ParamClient


class Locator(Node):
    MIN_X = -0.2
    MAX_X = 1.25
    MIN_Y = -0.4
    MAX_Y = 1
    MIN_Z = -0.01
    MAX_Z = 0.4

    def __init__(self, node_name="locator", min_points_pcl=2, depth_range=(0.3, 3.5)):
        """
        @arg min_points_plc : >0, default 500, In the segmented pointcloud, minimum number for points (xyz) to be a (reasonable) cloud.
        @arg depth_range: tuple (int,int), (min, max) range for estimated depth [in mm], default 10cm .. 1m. Points in cloud
            outside this range are dropped. Sometimes camera fails to measure depth and inputs 0.0m as depth, this is to filter out those values.
        """
        super().__init__(node_name)
        # Wait for calibrator
        calib_client = self.create_client(GetParameters, '/calibrator/get_parameters')
        self.get_logger().info("Waiting for calibrator to setup cameras")
        calib_client.wait_for_service()
        # Retreive camera information
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_extrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_extrinsics", "camera_frames"]).values]
        while len(self.cameras) == 0: #wait for cams to come online
            self.get_logger().warn("No cams detected, waiting 2s.")
            time.sleep(2)
            self.image_topics, self.cameras, self.camera_instrinsics, self.camera_extrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_extrinsics", "camera_frames"]).values]

        # get global transform
        self.robot2global_tf = np.reshape([p.double_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["robot2global_tf"]).values], (4, 4))
        self.global_frame_id = call_get_parameters(node=self, node_name="/calibrator", parameter_names=["global_frame_id"]).values[0].string_value

        self.pclient = ParamClient()
        self.pclient.define("locator_alive", True)

        # Set camera parameters and topics
        self.camera_instrinsics = [json.loads(cintr) for cintr in self.camera_instrinsics]
        self.camera_extrinsics = [json.loads(cextr) for cextr in self.camera_extrinsics]
        self.mask_topics = [cam + "/detections/masks" for cam in self.cameras] #input masks from 2D rgb (from our detector.py)
        self.pcl_topics = [cam + "/depth/color/points" for cam in self.cameras] #input pcl data (from RS camera)

        self.depth_min, self.depth_max = depth_range
        assert self.depth_min < self.depth_max and self.depth_min > 0.0

        # create publisher for located objects (segmented PCLs)
        qos = QoSProfile(depth=50, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.pubPCL = self.create_publisher(SegmentedPointcloud, '/detections/segmented_pointcloud', qos_profile=qos)

        # For now, every camera published segmented hand data PCL.
        self.pubPCL_avatar = self.create_publisher(SegmentedPointcloud, '/detections/segmented_pointcloud_avatar', qos_profile=qos)

        self.cvb = cv_bridge.CvBridge()
        self.mask_dtype = {'names':['f{}'.format(i) for i in range(2)], 'formats':2 * [np.int32]}

        #create listeners, synchronized for each camera (Mask + PCL)
        for i, (cam, pclTopic, maskTopic, camera_instrinsics, camera_extrinsics) in enumerate(zip(self.cameras, self.pcl_topics, self.mask_topics, self.camera_instrinsics, self.camera_extrinsics)):
            # convert camera data to numpy
            self.camera_instrinsics[i]["camera_matrix"] = np.array(camera_instrinsics["camera_matrix"], dtype=np.float32)
            self.camera_instrinsics[i]["distortion_coefficients"] = np.array(camera_instrinsics["distortion_coefficients"], dtype=np.float32)

            self.camera_extrinsics[i]["dtc_tf"] = np.array(camera_extrinsics["dtc_tf"], dtype=np.float32)  # dtc = depth to color sensor transform
            self.camera_extrinsics[i]["ctg_tf"] = np.array(camera_extrinsics["ctg_tf"], dtype=np.float32)  # ctg = color/camera to global coord system transform

            # create approx syncro callbacks
            cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
            subPCL = message_filters.Subscriber(self, PointCloud2, pclTopic, qos_profile=qos, callback_group=cb_group) #listener for pointcloud data from RealSense camera
            subMasks = message_filters.Subscriber(self, DetectionMask, maskTopic, qos_profile=qos, callback_group=cb_group) #listener for masks from detector node
            sync = message_filters.ApproximateTimeSynchronizer([subPCL, subMasks], 60, slop=0.05) #create and register callback for syncing these 2 message streams, slop=tolerance [sec]
            sync.registerCallback(lambda pcl_msg, mask_msg, cam=cam: self.detection_callback(pcl_msg, mask_msg, cam))

        self.min_points_pcl = min_points_pcl
        self.avatar_data_classes = ["leftWrist", "rightWrist", "leftElbow", "rightElbow", "leftShoulder", "rightShoulder", "head"]

    @staticmethod
    @jit(nopython=True, parallel=False)
    def project(camera_matrix, point_cloud):
        # converts pcl (shape 3,N) of [x,y,z] (3D) into image space (with cam_projection matrix) -> [u,v,w] -> [u/w, v/w] which is in 2D
        imspace = np.dot(camera_matrix, point_cloud)
        # [u,v,w] -> [u/w, v/w, w/w] -> [u',v'] = 2D
        return imspace[:2, :] / imspace[2, :]

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compareMasksPCL_fast(idxs, masks):
        idxs1d = idxs[1, :] + idxs[0, :] * masks[0].shape[1]
        wheres = []
        masks_raveled = masks.reshape(masks.shape[0], -1)  # ravel all masks
        # set (0, 0) to 0 - "wrong" pixels are projected to this
        masks_raveled[:, 0] = 0
        for mask in masks_raveled:
            wheres.append(np.nonzero(mask[idxs1d])[0])
        return wheres

    def detection_callback(self, pcl_msg, mask_msg, camera):
        self.pclient.locator_alive = True
        if not mask_msg.masks:
            # self.get_logger().info("no masks, no party. Quitting early.")
            return  # no mask detections (for some reason)

        cameraData = self.getCameraData(camera)
        masks = np.array([self.cvb.imgmsg_to_cv2(mask, "mono8") for mask in mask_msg.masks])
        object_ids, class_names, scores = mask_msg.object_ids, mask_msg.class_names, mask_msg.scores

        point_cloud, point_rgb, rgb_raw = ftl_pcl2numpy(pcl_msg)
        point_cloud = point_cloud.T
        # tf_mat = tf3.affines.compose(t, tf3.quaternions.quat2mat(q[-1:] + q[:-1]), np.ones(3))
        tf_mat = cameraData["dtc_tf"]
        point_cloud = np.dot(tf_mat, np.pad(point_cloud, ((0, 1), (0, 0)), mode="constant", constant_values=1))[:3, :]
        point_cloud = point_cloud.astype(np.float32)  # secret hack to make sure this works...
        ##process pcd
        # 1. convert 3d pcl to 2d image-space
        imspace = self.project(cameraData["camera_matrix"], point_cloud) # converts pcl (shape 3,N) of [x,y,z] (3D) into image space (with cam_projection matrix) -> [u,v,w] -> [u/w, v/w] which is in 2D
        imspace[np.isnan(imspace)] = -1 #marking as -1 results in deletion (omission) of these points in 3D, as it's impossible to match to -1
        # dist = np.linalg.norm(point_cloud, axis=0)
        bad_points = np.logical_or(point_cloud[2, :] < self.depth_min, point_cloud[2, :] > self.depth_max)
        # bad_points = np.logical_or(np.logical_or(point_cloud[2, :] < self.depth_min, point_cloud[2, :] > self.depth_max), point_cloud[0, :] < -0.65)
        # bad_points = np.logical_or(np.logical_or(dist < self.depth_min, dist > self.depth_max), point_cloud[0, :] < -0.65)
        imspace[:, bad_points] = -1
        # assert np.isnan(imspace).any() == False, 'must not have NaN element'  # sorry, but this is expensive (half a ms) #optimizationfreak
        imspace = imspace.astype(np.int32)
        mshape = masks[0].shape
        imspace[:, (imspace[0] < 0) | (imspace[1] < 0) | (
            imspace[1] >= mshape[0]) | (imspace[0] >= mshape[1])] = 0

        wheres = self.compareMasksPCL_fast(imspace[[1, 0], :], masks)
        ctg_tf_mat = cameraData["ctg_tf"]
        for where, object_id, class_name, score in zip(wheres, object_ids, class_names, scores):
            # 2. segment PCL & compute median
            # skip pointclouds with too few datapoints to be useful
            # if class_name in self.avatar_data_classes:
            #     self.get_logger().error(f"{class_name}")
            if len(where) < self.min_points_pcl:
            #     self.get_logger().info(
            #         "Cam {}: Skipping pcl {} for '{}' mask_score: {} -- too few datapoints. ".format(camera, len(where), class_name, score))
                continue

            seg_pcd = np.dot(ctg_tf_mat, np.pad(point_cloud[:, where], ((0, 1), (0, 0)), mode="constant", constant_values=1))
            seg_pcd = seg_pcd[:3, :] / seg_pcd[3, :]
            # filter points outside the main work area
            m = np.mean(seg_pcd, axis=1)
            if m[0] < self.MIN_X or m[0] > self.MAX_X or m[1] < self.MIN_Y or m[1] > self.MAX_Y or m[2] < self.MIN_Z or m[2] > self.MAX_Z:
                continue

            seg_color = rgb_raw[where]

            # output: create back a pcl from seg_pcd and publish it as ROS PointCloud2
            segmented_pcl = ftl_numpy2pcl(seg_pcd, pcl_msg.header, seg_color)
            segmented_pcl.header.frame_id = self.global_frame_id
            segmented_pcl.header.stamp = pcl_msg.header.stamp
            # wrap together PointCloud2 + label + score => SegmentedPointcloud
            seg_pcl_msg = SegmentedPointcloud()
            seg_pcl_msg.header = segmented_pcl.header
            seg_pcl_msg.pcl = segmented_pcl
            seg_pcl_msg.object_id = object_id
            seg_pcl_msg.label = str(class_name)
            seg_pcl_msg.confidence = float(score)

            # Data about hand position is published on different topic

            if class_name in self.avatar_data_classes:
                self.pubPCL_avatar.publish(seg_pcl_msg)
                self.get_logger().error(f"{class_name} = {np.mean(seg_pcd, 1)}")
            else:
                self.pubPCL.publish(seg_pcl_msg)

    # def compareMaskPCL(self, mask_array, projected_points):  # OLD, SLOW version
    #     a = mask_array.T.astype(np.int32).copy()
    #     b = projected_points.T.copy()
    #     self.mask_dtype = {'names':['f{}'.format(i) for i in range(2)], 'formats':2 * [np.int32]}
    #     result = np.intersect1d(a.view(self.mask_dtype), b.view(self.mask_dtype), return_indices=True)
    #     return result[2]

    def getCameraData(self, camera):
        idx = self.cameras.index(camera)
        return {
            "camera": camera,
            "image_topic": self.image_topics[idx],
            "camera_matrix": self.camera_instrinsics[idx]["camera_matrix"],
            "distortion_coefficients": self.camera_instrinsics[idx]["distortion_coefficients"],
            "dtc_tf": self.camera_extrinsics[idx]["dtc_tf"],
            # "ctg_tf": self.camera_extrinsics[idx]["ctg_tf"],
            # "ctg_tf": (robot_2_global @ realsense_2_robot @ self.camera_extrinsics[idx]["ctg_tf"]).astype(np.float32),
            "ctg_tf": (self.robot2global_tf @ self.camera_extrinsics[idx]["ctg_tf"]).astype(np.float32),
            "optical_frame": self.camera_frames[idx],
            "mask_topic": self.mask_topics[idx],
            "pcl_topic": self.pcl_topics[idx],
        }

def main():
    rclpy.init()
    locator = Locator()
    try:
        n_threads = len(locator.cameras)
        mte = MultiThreadedExecutor(num_threads=n_threads, context=rclpy.get_default_context())
        rclpy.spin(locator, executor=mte)
    except KeyboardInterrupt:
        print("User requested shutdown.")
    finally:
        locator.destroy_node()


if __name__ == "__main__":
    main()
