import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

#PointCloud2
from crow_msgs.msg import SegmentedPointcloud
import open3d as o3d
from crow_vision_ros2.utils import make_vector3, ftl_pcl2numpy, ftl_numpy2pcl
from crow_vision_ros2.filters import ParticleFilter, object_properties

#TF
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import MultiArrayDimension, Float32MultiArray
from crow_msgs.msg import FilteredPose, PclDimensions

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import numpy as np

import pkg_resources
import time


class ParticleFilterNode(Node):
    UPDATE_INTERVAL = 0.05
    VISUALIZE_PARTICLES = True
    INVERSE_OBJ_MAP = {v["name"]: i for i, v in enumerate(object_properties.values())}

    def __init__(self, node_name="particle_filter"):
        super().__init__(node_name)
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
        self.particle_filter = ParticleFilter()  # the main component

        qos = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        for i, (cam, pclTopic) in enumerate(zip(self.cameras, self.seg_pcl_topics)):
            self.get_logger().info("Created subscriber for segmented_pcl \"{}\"".format(pclTopic))
            # self.create_subscription(SegmentedPointcloud, pclTopic, self.detection_cb, qos_profile=qos)
            sub = message_filters.Subscriber(self, SegmentedPointcloud, pclTopic, qos_profile=qos)
            self.cache = message_filters.Cache(sub, 15, allow_headerless=True)
            # self.cache.registerCallback(self.cache_cb)

        self.lastFilterUpdate = self.get_clock().now()  # when was the filter last update (called pf.update())
        self.lastMeasurement = self.get_clock().now()  # timestamp of the last measurement (last segmented PCL message processed)
        self.updateWindowDuration = rclpy.time.Duration(seconds=0.05)
        self.timeSlipWindow = rclpy.time.Duration(seconds=1.5)
        self.measurementTolerance = rclpy.time.Duration(seconds=0.00001)
        self.lastUpdateMeasurementDDiff = rclpy.time.Duration(seconds=2)
        # Publisher for the output of the filter
        self.filtered_publisher = self.create_publisher(FilteredPose, "/filtered_poses", qos)
        self.timer = self.create_timer(self.UPDATE_INTERVAL, self.filter_update) # this callback is called periodically to handle everyhing
        if self.VISUALIZE_PARTICLES:
            # Initialize visualization properties
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            # geometry for the particles
            self.particle_cloud = o3d.geometry.PointCloud()
            # some initial random particles necessary automaticall set the viewpoint
            self.particle_cloud.points = o3d.utility.Vector3dVector(np.random.randn(10, 3)*2)
            self.vis.add_geometry(self.particle_cloud)
            # geometry for the axis
            self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
            self.vis.add_geometry(self.axis)
        else:
            self.vis = None

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
            try:
                class_id = next((k for k, v in object_properties.items() if label == v["name"]))
            except StopIteration as e:
                class_id = -1

            pcl, _, _ = ftl_pcl2numpy(pcl_msg.pcl)
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

                # Visualization
                if np.size(particles) > 0:
                    # clear the geometries
                    self.particle_cloud.clear()
                    self.axis.clear()
                    for pts, (pose, label, dims, uuid) in zip(particles, estimates):
                        # for each model, add its particles and pose as axis
                        tmp_pcl = o3d.geometry.PointCloud()
                        tmp_pcl.points = o3d.utility.Vector3dVector(pts)
                        c = self._get_obj_color(label)  # get model color according to the label
                        tmp_pcl.paint_uniform_color(c)
                        self.particle_cloud += tmp_pcl
                        self.axis += o3d.geometry.TriangleMesh.create_coordinate_frame(0.2, pose.tolist())
                    # o3d.visualization.draw_geometries([self.particle_cloud])
                    self.vis.update_geometry(self.particle_cloud)
                    self.vis.update_geometry(self.axis)
                # if there are no models, only update the rendering window (otherwise panning/zooming won't work)
                self.vis.poll_events()
                self.vis.update_renderer()

            self.filtered_publisher.publish(pose_array_msg)

    def _get_obj_color(self, obj_name):
        return object_properties[self.INVERSE_OBJ_MAP[obj_name]]["color"]

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
            class_id = next((k for k, v in object_properties.items() if label in v["name"]))
        except StopIteration as e:
            class_id = -1

        pcl, _, _ = ftl_pcl2numpy(pcl_msg.pcl)
        self.particle_filter.add_measurement((pcl, class_id, score))


def main():
    rclpy.init()
    try:
        pfilter = ParticleFilterNode()
        rclpy.spin(pfilter)
    finally:
        if pfilter.vis is not None:
            pfilter.vis.destroy_window()
        pfilter.destroy_node()


if __name__ == "__main__":
    main()
