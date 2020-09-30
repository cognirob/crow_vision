import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

#Pointcloud
from crow_msgs.msg import DetectionMask, SegmentedPointcloud
from sensor_msgs.msg import PointCloud2, PointField

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import json
import numpy as np
import cv2
import cv_bridge

#TF
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import TransformStamped
from crow_vision_ros2.utils import make_vector3, ftl_pcl2numpy

import pkg_resources
#import open3d as o3d #we don't use o3d, as it's too slow
import time 
from ctypes import * # convert float to uint32
from time import sleep



class Locator(Node):

    def __init__(self, node_name="locator", min_points_pcl=500, depth_range=(100, 1000)):
        """
        @arg min_points_plc : >0, default 500, In the segmented pointcloud, minimum number for points (xyz) to be a (reasonable) cloud.
        @arg depth_range: tuple (int,int), (min, max) range for estimated depth [in mm], default 10cm .. 1m. Points in cloud
            outside this range are dropped. Sometimes camera fails to measure depth and inputs 0.0m as depth, this is to filter out those values.
        """
        super().__init__(node_name)
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]
        while len(self.cameras) == 0: #wait for cams to come online
            self.get_logger().warn("No cams detected, waiting 2s.")
            time.sleep(2)
            self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]

        self.camera_instrinsics = [json.loads(cintr) for cintr in self.camera_instrinsics]
        self.mask_topics = [cam + "/detections/masks" for cam in self.cameras] #input masks from 2D rgb (from our detector.py)
        self.pcl_topics = [cam + "/depth/color/points" for cam in self.cameras] #input pcl data (from RS camera)

        self.depth_min, self.depth_max = depth_range
        assert self.depth_min < self.depth_max and self.depth_min > 0.0

        # create output topic and publisher dynamically for each cam
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.pubPCL = {} #output: segmented pcl sent as SegmentedPointcloud, separate publisher for each camera, indexed by 'cam', topic: "<cam>/detections/segmented_pointcloud"
        self.pubPCLdebug = {} #output: segmented pcl sent as PointCloud2, so we can directly visualize it in rviz2. Not needed, only for debug to avoid custom msgs above.
        for cam in self.cameras:
            out_pcl_topic = cam + "/" + "detections/segmented_pointcloud"
            out_pcl_publisher = self.create_publisher(SegmentedPointcloud, out_pcl_topic, qos_profile=qos)
            self.pubPCL[cam] = out_pcl_publisher
            self.get_logger().info("Created publisher for topic {}".format(out_pcl_topic))
            self.pubPCLdebug[cam] = self.create_publisher(PointCloud2, out_pcl_topic+"_debug", qos_profile=qos)

        self.cvb = cv_bridge.CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.mask_dtype = {'names':['f{}'.format(i) for i in range(2)], 'formats':2 * [np.int32]}

        #create listeners (synchronized)
        for i, (cam, pclTopic, maskTopic, camera_instrinsics) in enumerate(zip(self.cameras, self.pcl_topics, self.mask_topics, self.camera_instrinsics)):
            # convert camera data to numpy
            self.camera_instrinsics[i]["camera_matrix"] = np.array(camera_instrinsics["camera_matrix"], dtype=np.float32)
            self.camera_instrinsics[i]["distortion_coefficients"] = np.array(camera_instrinsics["distortion_coefficients"], dtype=np.float32)

            # create approx syncro callbacks
            subPCL = message_filters.Subscriber(self, PointCloud2, pclTopic, qos_profile=qos) #listener for pointcloud data from RealSense camera
            subMasks = message_filters.Subscriber(self, DetectionMask, maskTopic, qos_profile=qos) #listener for masks from detector node
            sync = message_filters.ApproximateTimeSynchronizer([subPCL, subMasks], 20, slop=0.03) #create and register callback for syncing these 2 message streams, slop=tolerance [sec]
            sync.registerCallback(lambda pcl_msg, mask_msg, cam=cam: self.detection_callback(pcl_msg, mask_msg, cam))

        self.min_points_pcl = min_points_pcl
        sleep(10)



    def detection_callback(self, pcl_msg, mask_msg, camera):
        # print(self.getCameraData(camera))
        if not mask_msg.masks:
            self.get_logger().info("no masks, no party. Quitting early.")
            return  # no mask detections (for some reason)

        cameraData = self.getCameraData(camera)
        masks = [self.cvb.imgmsg_to_cv2(mask, "mono8") for mask in mask_msg.masks]
        class_names, scores = mask_msg.class_names, mask_msg.scores

        point_cloud, point_rgb = ftl_pcl2numpy(pcl_msg)
        point_cloud = point_cloud.T
        ## get pointcloud data from ROS2 msg to open3d format
        # pcd = convertCloudFromRosTo.astype(np.float32)

        ##process pcd
        # 1. convert 3d pcl to 2d image-space
        start = time.time()
        camera_matrix = cameraData["camera_matrix"]
        imspace = np.dot(camera_matrix, point_cloud) # converts pcl (shape 3,N) of [x,y,z] (3D) into image space (with cam_projection matrix) -> [u,v,w] -> [u/w, v/w] which is in 2D
        imspace = imspace[:2, :] / imspace[2, :] # [u,v,w] -> [u/w, v/w, w/w] -> [u',v'] = 2D
        imspace[np.isnan(imspace)] = -1 #marking as -1 results in deletion (ommision) of these points in 3D, as it's impossible to match to -1
        imspace[imspace > self.depth_max] = -1 # drop points with depth outside of range
        imspace[imspace < self.depth_min] = -1 # drop points with depth outside of range
        # assert np.isnan(imspace).any() == False, 'must not have NaN element'  # sorry, but this is expensive (half a ms) #optimizationfreak
        imspace = imspace.astype(np.int32)
        end = time.time()
        #print("Transform: ", end - start)

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
            # 2. segment PCL & compute median
            start = time.time()
            where = self.compareMaskPCL(np.array(np.where(mask.T)), imspace)
            # skip pointclouds with too few datapoints to be useful
            if len(where) < self.min_points_pcl:
                self.get_logger().info("Skipping pcl {} for '{}' mask_score: {} -- too few datapoints. ".format(len(where), class_name, score))
                continue

            # create segmented pcl
            seg_pcd = point_cloud[:, where]

            mean = np.median(seg_pcd, axis=1)
            #self.get_logger().info("Object {}: {} Centroid: {} accuracy: {}".format(i, class_name, mean, score))
            assert len(mean) == 3, 'incorrect mean dim'
            self.sendPosition(cameraData["optical_frame"], class_name + f"_{i}", mask_msg.header.stamp, mean)
            end = time.time()
            #print("Filter: ", end - start)

            # output: create back a pcl from seg_pcd and publish it as ROS PointCloud2
            itemsize = np.dtype(np.float32).itemsize
            fields = [PointField(name=n, offset=i*itemsize, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
            #fill PointCloud2 correctly according to https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0#file-dragon_pointcloud-py-L32
            segmented_pcl = PointCloud2(
                     header=pcl_msg.header,
                     height=1,
                     width=seg_pcd.shape[1],
                     fields=fields,
                     point_step=(itemsize*3), #3=xyz
                     row_step=(itemsize*3*seg_pcd.shape[1]),
                     data=seg_pcd.tobytes()
                     )
            segmented_pcl.header.stamp = mask_msg.header.stamp
            assert segmented_pcl.header.stamp == mask_msg.header.stamp, "timestamps for mask and segmented_pointcloud must be synchronized!"

            # wrap together PointCloud2 + label + score => SegmentedPointcloud
            seg_pcl_msg = SegmentedPointcloud()
            seg_pcl_msg.header = segmented_pcl.header
            seg_pcl_msg.pcl = segmented_pcl
            seg_pcl_msg.label = str(class_name)
            seg_pcl_msg.confidence = float(score)

            self.pubPCL[camera].publish(seg_pcl_msg)
            self.pubPCLdebug[camera].publish(segmented_pcl) #for debug visualization only, can be removed.


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
            "camera_matrix": self.camera_instrinsics[idx]["camera_matrix"],
            # "camera_matrix": np.array([383.591, 0, 318.739, 0, 383.591, 237.591, 0, 0, 1]).reshape(3, 3),
            "distortion_coefficients": self.camera_instrinsics[idx]["distortion_coefficients"],
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
