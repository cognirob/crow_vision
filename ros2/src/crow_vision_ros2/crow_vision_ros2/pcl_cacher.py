import rclpy
from rclpy.node import Node
import message_filters

#PointCloud2
from crow_vision_ros2.utils import make_vector3, ftl_pcl2numpy, ftl_numpy2pcl

#TF
import tf2_py as tf

# msgs
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import MultiArrayDimension, Float32MultiArray
from crow_msgs.msg import FilteredPose, PclDimensions, ObjectPointcloud
from trio3_ros2_interfaces.msg import Units
from trio3_ros2_interfaces.srv import GetMaskedPointCloud

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import numpy as np

import time

class PCLItem():

    def __init__(self, uuid, stamp, pcl) -> None:
        self.uuid = uuid
        self.stamp = stamp
        self.pcl = pcl


class PCLCacher(Node):
    PCL_MEMORY_SIZE = 5
    PCL_GETTER_SERVICE_NAME = "get_masked_point_cloud_rs"
    MAX_ALLOWED_DISTANCE = 0.2  # in meters

    def __init__(self, node_name="pcl_cacher"):
        super().__init__(node_name)
        # Get existing cameras from and topics from the calibrator
        qos = QoSProfile(
            depth=self.PCL_MEMORY_SIZE,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        sub = message_filters.Subscriber(self, ObjectPointcloud, '/filtered_pcls', qos_profile=qos)
        self.cache = message_filters.Cache(sub, self.PCL_MEMORY_SIZE, allow_headerless=False)
        self.srv = self.create_service(GetMaskedPointCloud, self.PCL_GETTER_SERVICE_NAME, self.get_pcl)
        self.objects = {}

    def refresh_pcls(self):
        for stamp, msg in zip(self.cache.cache_times, self.cache.cache_msgs):
            for uid, pcl in zip(msg.uuid, msg.pcl):
                if uid not in self.objects:  # object is not yet in the database
                    self.objects[uid] = PCLItem(uid, stamp, pcl)
                else:
                    obj = self.objects[uid]
                    if obj.stamp != stamp:  # object is in the DB but with an old PCL
                        obj.stamp = stamp
                        obj.pcl = pcl
        # cleanup way too old PCLs
        latest_allowed_time = self.get_clock().now().to_msg() # TODO - RCLPY.time(some time limit)
        for uid, obj in self.objects:
            if obj.stamp < latest_allowed_time:
                del self.objects[uid]


    def get_pcl(self, request, response):
        request_pose = np.r_["0,2,0", [getattr(request.expected_position.position, a) for a in "xyz"]]
        # print(request_pose)
        # print(request_pose.shape)
        if request.request_units.unit_type == Units.MILIMETERS:
            request_pose / 1000
        response.response_units.unit_type = Units.METERS
        self.refresh_pcls()
        # TODO: get PCL somehow
        # response.masked_point_cloud
        self.get_logger().info(f"Got request for a segmented PCL near location {str(request_pose.tolist())}")
        return response


def main():
    rclpy.init()
    pclCacher = PCLCacher()
    try:
        rclpy.spin(pclCacher)
    except KeyboardInterrupt:
        print("User requested shutdown.")
    finally:
        pclCacher.destroy_node()


if __name__ == "__main__":
    main()
