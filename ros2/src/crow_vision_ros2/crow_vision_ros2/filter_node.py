import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

#PointCloud2
from crow_msgs.msg import SegmentedPointcloud
import open3d as o3d
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

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import numpy as np

import pkg_resources
import time


class ParticleFilterNode(Node):
    UPDATE_INTERVAL = 0.05
    VISUALIZE_PARTICLES = True

    def __init__(self, node_name="particle_filter"):
        super().__init__(node_name)
        self.crowracle = CrowtologyClient(node=self)
        # Get existing cameras from and topics from the calibrator
        self.cameras = []
        while(len(self.cameras) == 0):
            try:
                self.cameras , self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]
            except:
                self.get_logger().error("getting cameras failed. Retrying in 2s")
                time.sleep(2)
        assert len(self.cameras) > 0
        # create necessary topics to get detected PCLs
        self.seg_pcl_topics = [cam + "/" + "detections/segmented_pointcloud" for cam in self.cameras] #input segmented pcl data

        #time.sleep(5)
        self.object_properties = self.crowracle.get_filter_object_properties()
        self.particle_filter = ParticleFilter(self.object_properties)  # the main component

        qos = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.get_logger().info("Created subscriber for segmented_pcl '/detections/segmented_pointcloud'")
        # self.create_subscription(SegmentedPointcloud, pclTopic, self.detection_cb, qos_profile=qos)
        sub = message_filters.Subscriber(self, SegmentedPointcloud, '/detections/segmented_pointcloud', qos_profile=qos)
        self.cache = message_filters.Cache(sub, 15, allow_headerless=False)
        self.pubPCLdebug = self.create_publisher(PointCloud2, "/detections/segmented_pointcloud_debug", qos_profile=qos)
            # self.cache.registerCallback(self.cache_cb)
        # for i, (cam, pclTopic) in enumerate(zip(self.cameras, self.seg_pcl_topics)):
        #     self.get_logger().info("Created subscriber for segmented_pcl \"{}\"".format(pclTopic))
        #     # self.create_subscription(SegmentedPointcloud, pclTopic, self.detection_cb, qos_profile=qos)
        #     sub = message_filters.Subscriber(self, SegmentedPointcloud, pclTopic, qos_profile=qos)
        #     self.cache = message_filters.Cache(sub, 15, allow_headerless=True)
        #     # self.cache.registerCallback(self.cache_cb)

        self.lastFilterUpdate = self.get_clock().now()  # when was the filter last update (called pf.update())
        self.lastMeasurement = self.get_clock().now()  # timestamp of the last measurement (last segmented PCL message processed)
        self.updateWindowDuration = rclpy.time.Duration(seconds=0.05)
        self.timeSlipWindow = rclpy.time.Duration(seconds=1.5)
        self.measurementTolerance = rclpy.time.Duration(seconds=0.00001)
        self.lastUpdateMeasurementDDiff = rclpy.time.Duration(seconds=2)
        # Publisher for the output of the filter
        self.filtered_publisher = self.create_publisher(FilteredPose, "/filtered_poses", qos)
        self.pcl_publisher = self.create_publisher(ObjectPointcloud, "/filtered_pcls", qos)
        self.timer = self.create_timer(self.UPDATE_INTERVAL, self.filter_update) # this callback is called periodically to handle everyhing

        # Tracker initialization
        self.tracker = Tracker()
        self.avatar_data_classes = ["leftWrist", "rightWrist", "leftElbow", "rightElbow", "leftShoulder", "rightShoulder", "head"]
        # create approx syncro callbacks
        cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        subSPcl = message_filters.Subscriber(self, SegmentedPointcloud, '/detections/segmented_pointcloud_avatar', qos_profile=qos)
        sync = message_filters.ApproximateTimeSynchronizer([subSPcl], 40, slop=0.5) #create and register callback for syncing these 2 message streams, slop=tolerance [sec]
        sync.registerCallback(lambda spcl_msg: self.avatar_callback(spcl_msg))

    def add_and_process(self, messages):
        if type(messages) is not list:
            messages = [messages]

        if len(messages) == 0:
            return

        self.get_logger().info(f"Adding {len(messages)} measurements to the filter")
        latest = self.lastMeasurement
        for pcl_msg in messages:
            self.frame_id = pcl_msg.header.frame_id
            if latest < rclpy.time.Time.from_msg(pcl_msg.header.stamp):
                latest = rclpy.time.Time.from_msg(pcl_msg.header.stamp)
            label = pcl_msg.label
            score = pcl_msg.confidence

            if label not in self.avatar_data_classes:
                ###########################################################################################################################
                ###########################################################################################################################
                ### FIX UNKNOWN OBJECTS - l/r...Wrist, l/r...Shoulder, l/r...Elbow, head ###
                try:
                    class_id = next((k for k, v in self.object_properties.items() if label == v["name"]))
                except StopIteration as e:
                    class_id = -1
                ############################################################################

                pcl, _, c = ftl_pcl2numpy(pcl_msg.pcl)

                self.particle_filter.add_measurement((pcl, class_id, score))



        now = self.get_clock().now()
        self.lastMeasurement = latest
        self.lastUpdateMeasurementDDiff = now - self.lastMeasurement
        self.update(now)

    def update(self, now=None):
        self.particle_filter.update()
        if now is not None:
            self.lastFilterUpdate = now
        else:
            self.lastFilterUpdate = self.get_clock().now()

        if self.particle_filter.n_models > 0:
            estimates = self.particle_filter.getEstimates()

            ####
            # - repair objects internally in the filter
            # - update
            # - on return "0" delete object on that position
            # Format input data
            poses_formatted, class_names_formatted, dimensions_formatted, uuids_formatted = ([],[],[],[])
            for pose, label, dims, uuid in estimates:
                poses_formatted.append(pose.tolist())
                class_names_formatted.append(label)
                dimensions_formatted.append(dims)
                uuids_formatted.append(uuid)


            last_uuid, latest_uuid = self.tracker.track_and_get_uuids( centroid_positions=poses_formatted, dimensions=dimensions_formatted, class_names=class_names_formatted, uuids=uuids_formatted)
            print(f"*** last_uuid: {last_uuid}")
            print(f"*** latest_uuid: {latest_uuid}")
            self.particle_filter._correct_model_uuids(last_uuids=last_uuid, latest_uuids=latest_uuid)


            #self.get_logger().info(str(estimates))
            poses = []
            dimensions = []
            labels = []
            uuids = []
            for pose, label, dims, uuid in estimates:
                pose_msg = Pose()
                pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = pose.tolist()
                poses.append(pose_msg)
                dim_msg = PclDimensions(dimensions=dims)
                dimensions.append(dim_msg)
                labels.append(label)
                uuids.append(uuid)
            self.get_logger().info('Publishing objects:' + str(labels))
            pose_array_msg = FilteredPose(poses=poses)
            pose_array_msg.size = dimensions
            pose_array_msg.label = labels
            pose_array_msg.uuid = uuids
            pose_array_msg.header.stamp = self.get_clock().now().to_msg()
            pose_array_msg.header.frame_id = self.frame_id

            if self.VISUALIZE_PARTICLES:
                particles_msg = []
                particles = self.particle_filter.get_model_particles()
                for model_particles in particles:
                    model_particles_msg = Float32MultiArray()
                    dims = model_particles.shape
                    model_particles_msg.layout.dim.append(MultiArrayDimension())
                    model_particles_msg.layout.dim[0].label = 'num_points'
                    model_particles_msg.layout.dim[0].size = dims[0]
                    model_particles_msg.layout.dim[0].stride = dims[0]*dims[1]
                    model_particles_msg.layout.dim.append(MultiArrayDimension())
                    model_particles_msg.layout.dim[1].label = 'xyz'
                    model_particles_msg.layout.dim[1].size = dims[1]
                    model_particles_msg.layout.dim[1].stride = dims[0]
                    data = np.frombuffer(model_particles.tobytes(),'float32')
                    model_particles_msg.data = data.tolist()
                    particles_msg.append(model_particles_msg)
                pose_array_msg.particles = particles_msg

            self.filtered_publisher.publish(pose_array_msg)

            pcl_uuids, pcl_points = self.particle_filter.getPclsEstimates()
            pcl_msg = ObjectPointcloud()
            pcl_msg.header.stamp = self.get_clock().now().to_msg()
            pcl_msg.header.frame_id = self.frame_id
            aggregate_pcl = []
            for obj in pcl_points:
                if len(obj) > 1:
                    aggregate_pcl.append(np.concatenate(obj, axis=0))
                else:
                    aggregate_pcl.append(obj[0])
            pcls = []
            for i, np_pcl in enumerate(aggregate_pcl):
                if len(np_pcl) > 0:
                    pcls.append(ftl_numpy2pcl(np_pcl.astype(np.float32).T, pcl_msg.header))
                else:
                    pcl_uuids.pop(i)
            if len(pcl_uuids) > 0:
                # self.pubPCLdebug.publish(pcls[-1])
                pcl_msg.uuid = pcl_uuids
                pcl_msg.pcl = pcls
                self.pcl_publisher.publish(pcl_msg)

    def filter_update(self):
        """Main function, periodically called by rclpy.Timer
        """
        latest_time = self.cache.getLastestTime()  # get the time of the last message received
        if latest_time is not None:  # None -> there are no messages
            # Find timestamp of the oldest message that wasn't processed, yet
            oldest_time = self.cache.getOldestTime()
            # orig_oldest = oldest_time.seconds_nanoseconds
            # if oldest_time <= (self.lastFilterUpdate - self.timeSlipWindow):
            #     oldest_time = self.lastFilterUpdate - self.timeSlipWindow
            # # if oldest_time <= (self.lastFilterUpdate - self.lastUpdateMeasurementDDiff):
            # #     oldest_time = self.lastFilterUpdate - self.lastUpdateMeasurementDDiff
            if oldest_time < self.lastMeasurement:
                oldest_time = self.lastMeasurement
                # oldest_time += self.measurementTolerance

            anyupdate = False  # helper var to see if there was some update
            while oldest_time < latest_time:
                next_time = oldest_time + self.updateWindowDuration
                messages = self.cache.getInterval(oldest_time, next_time)
                oldest_time = next_time
                if len(messages) == 0:
                    continue
                self.add_and_process(messages)
                anyupdate = True

            # if anyupdate:
            #     self.lastMeasurement += self.measurementTolerance
        else:
            self.update()

    def cache_cb(self, *args):
        pass

    def detection_cb(self, pcl_msg):
        self.get_logger().info("got some pcl")
        print(pcl_msg.label)
        self.frame_id = pcl_msg.header.frame_id
        label = pcl_msg.label
        try:
            class_id = next((k for k, v in self.object_properties.items() if label in v["name"]))
        except StopIteration as e:
            class_id = -1

        pcl, _, _ = ftl_pcl2numpy(pcl_msg.pcl)

        print(f"*** pcl: {pcl}")
        self.particle_filter.add_measurement((pcl, class_id, score))

    # Get Avatar PCL and update his parts in <class:Tracker>
    def avatar_callback(self, spcl_msg):
        # print(self.getCameraData(camera))
        if not spcl_msg.pcl:
            self.get_logger().info("no avatar data. Quitting early.")
            return  # no mask detections (for some reason)

        header = spcl_msg.header
        np_pcl, _, c = ftl_pcl2numpy(spcl_msg.pcl)
        object_id = spcl_msg.object_id
        label = str(spcl_msg.label)
        confidence = float(spcl_msg.confidence)

        np_pcl_center = np.median(np_pcl, axis=0).reshape(1, 3)
        np_pcl_dimension = (np.max(np_pcl, axis=0) - np.min(np_pcl, axis=0)).reshape(1, 3)

        # print(f"self.tracked_objects: {self.tracker.tracked_objects}")
        # print(f"self.tracker.avatar: {self.tracker.avatar}")
        self.tracker.avatar.update_avatar_object(avatar_object_name=label, np_position=np_pcl_center, np_dimensions=np_pcl_dimension)
        self.tracker.avatar.dump_info()
        # print("")
        print(f"~~~~~~~ header: {header}")
        # print(f"~~~~~~~ np_pcl: {np_pcl}")
        print(f"~~~~~~~ np_pcl_center: {np_pcl_center}")
        print(f"~~~~~~~ np_pcl_dimension: {np_pcl_dimension}")
        print(f"~~~~~~~ object_id: {object_id}")
        print(f"~~~~~~~ label: {label}")
        print(f"~~~~~~~ confidence: {confidence}")

def main():
    rclpy.init()
    try:
        pfilter = ParticleFilterNode()
        rclpy.spin(pfilter)
    finally:
        pfilter.destroy_node()


if __name__ == "__main__":
    main()
