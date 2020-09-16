import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters
from crow_msgs.msg import DetectionMask
from sensor_msgs.msg import PointCloud2, PointField
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
import numpy as np

import pkg_resources
from time import time


class Match3D(Node):
    """
    Match our objects (from .STL) to the segmented point-cloud ("<cam>/detections/segmented_pointcloud") 
    based on segmentation from 2D RGB masks (mask_msg) from YOLACT. 
    """

    def __init__(self, node_name="match3d"):
        super().__init__(node_name)
        self.cameras = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_nodes"]).values]
        self.mask_topics = [cam + "/" + "detections/masks" for cam in self.cameras] #input masks from 2D rgb
        self.seg_pcl_topics = [cam + "/" + "detections/segmented_pointcloud" for cam in self.cameras] #input segmented pcl data
        #TODO merge (segmented) pcls before this node
        #TODO create output publisher (what exactly to publish?)

        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        for i, (cam, pclTopic, maskTopic) in enumerate(zip(self.cameras, self.seg_pcl_topics, self.mask_topics)):
            # create approx syncro callbacks
            self.subPCL = message_filters.Subscriber(self, PointCloud2, pclTopic, qos_profile=10)
            self.subMasks = message_filters.Subscriber(self, DetectionMask, maskTopic, qos_profile=10)
            self.get_logger().info("Created synced subscriber for masks: \"{}\" & segmented_pcl \"{}\"".format(maskTopic, pclTopic))
            self.sync = message_filters.ApproximateTimeSynchronizer([self.subPCL, self.subMasks], 20, 0.005)
            self.sync.registerCallback(lambda pcl_msg, mask_msg, cam=cam: self.detection_callback(pcl_msg, mask_msg, cam))

        self.objects = {} # map str:label -> .stl model #TODO load STL


    def detection_callback(self, pcl_msg, mask_msg, camera):
        #print(self.getCameraData(camera))
        if not mask_msg.masks:
            print("no masks, no party. Quitting early.")
            return  # no mask detections (for some reason)

        # labels & score from detections masks
        class_names, scores = mask_msg.class_names, mask_msg.scores
        # pointcloud in numpy from pcl_msg
        point_cloud = np.array(pcl_msg.data).view(np.float32).reshape(-1, 8)[:, :3].T

        for i, (class_name, score) in enumerate(zip(class_names, scores)):
            icp_score = 0.0 #TODO ICP match self.objects["class_name"] to point_cloud
            print("ICP Matching pcl: {} to {} (confidence {}) with confidence: {}".format(np.shape(pcl_msg.data), class_name, score, icp_score))




def main():
    rclpy.init()

    matcher = Match3D()

    rclpy.spin(matcher)
    matcher.destroy_node()


if __name__ == "__main__":
    main()
