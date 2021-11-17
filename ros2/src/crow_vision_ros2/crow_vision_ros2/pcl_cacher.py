import time

import message_filters
import numpy as np
import rclpy
# msgs
from crow_msgs.msg import ObjectPointcloud
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.duration import Duration
from trio3_ros2_interfaces.msg import Units, RobotActionType, ObjectType
from trio3_ros2_interfaces.srv import GetMaskedPointCloud
from crow_ontology.crowracle_client import CrowtologyClient

#PointCloud2
from crow_vision_ros2.utils import ftl_pcl2numpy
import open3d as o3d
import sys


class PCLItem():

    def __init__(self, uuid, stamp, pcl, label) -> None:
        self._uuid = uuid
        self.update(stamp, pcl, label)

    def update(self, stamp, pcl, label):
        self._stamp = stamp
        self._pcl = pcl
        self._label = label
        self._pcl_numpy, _, _ = ftl_pcl2numpy(pcl)  # convert from PointCloud2.msg to numpy array
        self._center = np.mean(self._pcl_numpy, 0)

    def compute_distance(self, point):
        # TODO: mabye include box?
        return np.linalg.norm(point - self._center)

    @property
    def stamp(self):
        return self._stamp

    @property
    def label(self):
        return self._label

    @property
    def uuid(self):
        return self._uuid

    @property
    def pcl(self):
        return self._pcl

    @property
    def pcl_numpy(self):
        return self._pcl_numpy

    @property
    def center(self):
        return self._center


class PCLCacher(Node):
    PCL_MEMORY_SIZE = 30
    PCL_GETTER_SERVICE_NAME = "get_masked_point_cloud_rs"
    MAX_ALLOWED_DISTANCE = 0.2  # in meters
    MAX_AFF_ALLOWED_DISTANCE = 0.2
    KEEP_ALIVE_DURATION = 20
    DEBUG = False

    def __init__(self, node_name="pcl_cacher"):
        super().__init__(node_name)
        # Get existing cameras from and topics from the calibrator
        qos = QoSProfile(
            depth=self.PCL_MEMORY_SIZE,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        sub = message_filters.Subscriber(self, ObjectPointcloud, '/filtered_pcls', qos_profile=qos)
        self.crowracle = CrowtologyClient(node=self)
        self.cache = message_filters.Cache(sub, self.PCL_MEMORY_SIZE, allow_headerless=False)
        self.srv = self.create_service(GetMaskedPointCloud, self.PCL_GETTER_SERVICE_NAME, self.get_pcl)
        self.objects = {}
        self.aff_objects = {}
        self.aff_classes = ["hammer_handle", "hammer_head", "pliers_handle", "pliers_head",
                           "screw_round_thread", "screw_round_head", "screwdriver_handle",
                           "screwdriver_head", "wrench_handle", "wrench_open", "wrench_ring"]
        self.action_to_aff_type = {RobotActionType.PICK_AND_HOME: "_handle",
                                   RobotActionType.PICK_AND_PLACE: "_handle",
                                   RobotActionType.PICK_AND_PASS: "_head"}
        self.keep_alive_duration = Duration(seconds=self.KEEP_ALIVE_DURATION)
        self.create_timer(5, self.print_n_ojs)
        self.get_logger().info("PCL cacher ready.")

    def print_n_ojs(self):
        self.refresh_pcls()
        self.get_logger().info(f"# of objs = {len(self.objects)}, # of aff objs = {len(self.aff_objects)}")
        # for k, v in self.objects.items():
        #     self.get_logger().info(f"{k} = {np.mean(v.pcl, axis=0)}")

    def remove_stale_pcls(self, objects):
        latest_allowed_time = self.get_clock().now() - self.keep_alive_duration
        stale_uuids = []
        for uid, obj in objects.items():
            if obj.stamp < latest_allowed_time:
                stale_uuids.append(uid)
        for uid in stale_uuids:
            del objects[uid]


    def refresh_pcls(self):
        for stamp, msg in zip(self.cache.cache_times, self.cache.cache_msgs):
           # print(len(msg.pcl) == len(msg.uuid) == len(msg.labels))
            for uid, pcl, label in zip(msg.uuid, msg.pcl, msg.labels):
                if uid not in self.objects and uid not in self.aff_objects:  # object is not yet in the database
                    if label in self.aff_classes:
                        self.aff_objects[uid] = PCLItem(uid, stamp, pcl, label)
                    else:
                        self.objects[uid] = PCLItem(uid, stamp, pcl, label)
                else:
                    if uid in self.objects:
                        obj = self.objects[uid]
                    else:
                        obj = self.aff_objects[uid]
                    if obj.stamp != stamp:  # object is in the DB but with an old PCL or has a new label
                        # if label != obj.label:
                        #     print("Inconistency", label, obj.label)
                        obj.update(stamp, pcl, label)
                # self.get_logger().info(f"PCL size = {np.mean(self.objects[uid].pcl_numpy, axis=0)}")
        # cleanup way too old PCLs
        self.remove_stale_pcls(self.objects)
        self.remove_stale_pcls(self.aff_objects)

    def get_pcl(self, request, response):
        try:
            request_pose = np.r_["0,1,0", [getattr(request.expected_position.position, a) for a in "xyz"]]
            request_action = request.robot_action_type
            request_object = request.object_type
            if request.request_units.unit_type == Units.MILIMETERS:
                request_pose /= 1000
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

            # Construct response based on the request action and object type
            if request_action == RobotActionType.POINT:
                self.get_logger().info(f"Responding with a PCL @ {str(closest_object.center)} located {min_value}m away from the requested location.")
                response.masked_point_cloud = closest_object.pcl
            elif request_action in self.action_to_aff_type:
                request_affordance = request_object + self.action_to_aff_type[request_action]
                dist_affs = np.r_["0,2,0", [[o.compute_distance(request_pose), uid] for uid, o in self.aff_objects.items()
                                                                                     if o.label == request_affordance]]
                if dist_affs:
                    distances = dist_affs[:, 0].astype(float)
                    min_idx = np.argmin(distances)
                    min_value = distances[min_idx]
                    if min_value > self.MAX_AFF_ALLOWED_DISTANCE:
                        self.get_logger().error(f"Error requesting a segmented PCL: No PCL is close enough to the requested position!\n\trequest: {str(request_pose)}\n\tdistances: {str(distances)}")
                        return response
                    closest_aff = self.aff_objects[dist_affs[min_idx, 1]]
                    self.get_logger().info(f"Responding with a PCL @ {str(closest_aff.center)} located {min_value}m away from the requested location.")
                    response.masked_point_cloud = closest_aff.pcl
                else:
                    self.get_logger().error(f"Error requesting a segmented PCL: No PCL is close enough to the requested position!\n\trequest: {str(request_pose)}\n\tdistances: {str(distances)}")
                    return response

            if self.DEBUG:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(closest_object.pcl_numpy.astype(np.float32))
                o3d.visualization.draw_geometries([pcd])
        except BaseException as e:
            self.get_logger().fatal(f"Fatal error requesting a segmented PCL:\n{e}")
        finally:
            return response


def main():
    rclpy.init()
    if "-d" in sys.argv:
        PCLCacher.DEBUG = True

    pclCacher = PCLCacher()
    try:
        rclpy.spin(pclCacher)
    except KeyboardInterrupt:
        print("User requested shutdown.")
    finally:
        pclCacher.destroy_node()


if __name__ == "__main__":
    main()
