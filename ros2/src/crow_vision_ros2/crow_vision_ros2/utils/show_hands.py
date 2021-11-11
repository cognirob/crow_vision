from time import time
import rclpy
from rclpy import executors
from rclpy.node import Node
from rclpy.time import Duration, Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import message_filters
import traceback as tb
from crow_msgs.msg import SegmentedPointcloud
from crow_vision_ros2.utils import make_vector3, ftl_pcl2numpy, ftl_numpy2pcl
from crow_vision_ros2.filters import ParticleFilter

# Tracker
from crow_vision_ros2.tracker import Tracker

#TF
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import MultiArrayDimension, Float32MultiArray
from crow_msgs.msg import FilteredPose, PclDimensions, ObjectPointcloud
from crow_ontology.crowracle_client import CrowtologyClient
from crow_control.utils.profiling import StatTimer

from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import numpy as np
from crow_control.utils import ParamClient

from crow_vision_ros2.tracker.tracker_avatar import Avatar


class HandShow(Node):

    def __init__(self, node_name="hand_show"):
        super().__init__(node_name)
        qos = QoSProfile(
            depth=30,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.create_subscription(SegmentedPointcloud, '/detections/segmented_pointcloud_avatar', callback=self.avatar_callback, qos_profile=qos, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info("Online...")

    def avatar_callback(self, spcl_msg):
        # print(self.getCameraData(camera))
        if not spcl_msg.pcl:
            self.get_logger().info("no avatar data. Quitting early.")
            return  # no mask detections (for some reason)

        np_pcl, _, c = ftl_pcl2numpy(spcl_msg.pcl)
        label = str(spcl_msg.label)

        np_pcl_center = np.median(np_pcl, axis=0).reshape(1, 3)

        self.get_logger().info(f"{label}: {np_pcl_center}")


def main():
    rclpy.init()
    hs = HandShow()
    try:
        rclpy.spin(hs)
    except KeyboardInterrupt:
        print("User requested shutdown.")
    except BaseException as e:
        print(f"Some error had occured: {e}")
        tb.print_exc()        
    finally:
        hs.destroy_node()


if __name__ == "__main__":
    main()
