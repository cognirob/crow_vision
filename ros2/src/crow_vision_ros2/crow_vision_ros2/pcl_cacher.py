import time

import message_filters
import numpy as np
import rclpy
# msgs
from crow_msgs.msg import ObjectPointcloud
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.duration import Duration
from trio3_ros2_interfaces.msg import Units
from trio3_ros2_interfaces.srv import GetMaskedPointCloud

#PointCloud2
from crow_vision_ros2.utils import ftl_pcl2numpy


class PCLItem():

    def __init__(self, uuid, stamp, pcl) -> None:
        self._uuid = uuid
        self.update(stamp, pcl)

    def update(self, stamp, pcl):
        self._stamp = stamp
        self._pcl = pcl
        self._pcl_numpy, _, _ = ftl_pcl2numpy(pcl)  # convert from PointCloud2.msg to numpy array
        self._center = np.mean(self._pcl_numpy, 0)

    def compute_distance(self, point):
        # TODO: mabye include box?
        return np.linalg.norm(point - self._center)

    @property
    def stamp(self):
        return self._stamp

    @property
    def uuid(self):
        return self._uuid

    @property
    def pcl(self):
        return self._pcl


class PCLCacher(Node):
    PCL_MEMORY_SIZE = 5
    PCL_GETTER_SERVICE_NAME = "get_masked_point_cloud_rs"
    MAX_ALLOWED_DISTANCE = 0.2  # in meters
    KEEP_ALIVE_DURATION = 10

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
        self.keep_alive_duration = Duration(seconds=self.KEEP_ALIVE_DURATION)

    def refresh_pcls(self):
        for stamp, msg in zip(self.cache.cache_times, self.cache.cache_msgs):
            for uid, pcl in zip(msg.uuid, msg.pcl):
                if uid not in self.objects:  # object is not yet in the database
                    self.objects[uid] = PCLItem(uid, stamp, pcl)
                else:
                    obj = self.objects[uid]
                    if obj.stamp != stamp:  # object is in the DB but with an old PCL
                        obj.update(stamp, pcl)
        # cleanup way too old PCLs
        latest_allowed_time = self.get_clock().now() - self.keep_alive_duration
        stale_uuids = []
        for uid, obj in self.objects.items():
            if obj.stamp < latest_allowed_time:
                stale_uuids.append(uid)
        for uid in stale_uuids:
            del self.objects[uid]


    def get_pcl(self, request, response):
        request_pose = np.r_["0,1,0", [getattr(request.expected_position.position, a) for a in "xyz"]]
        if request.request_units.unit_type == Units.MILIMETERS:
            request_pose / 1000
        self.get_logger().info(f"Got a request for a segmented PCL near location {str(request_pose.tolist())}")
        response.response_units.unit_type = Units.METERS
        self.refresh_pcls()
        if len(self.objects) == 0:
            self.get_logger().error("Error requesting a segmented PCL: There are no PCLs in the cacher!")
            return response

        dist_objs = np.r_["0,2,0", [[o.compute_distance(request_pose), uid] for uid, o in self.objects.items()]]
        distances = dist_objs[:, 0].astype(float)
        min_idx = np.argmin(distances)
        min_value = distances[min_idx]
        if min_value > self.MAX_ALLOWED_DISTANCE:
            self.get_logger().error(f"Error requesting a segmented PCL: No PCL is close enough to the requested position!\n\trequest: {str(request_pose)}\n\tdistances: {str(distances)}")
            return response

        closest_object = self.objects[dist_objs[min_idx, 1]]
        response.masked_point_cloud = closest_object.pcl
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
