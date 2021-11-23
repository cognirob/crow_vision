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
from crow_msgs.srv import ResetScene
from crow_vision_ros2.utils import make_vector3, ftl_pcl2numpy, ftl_numpy2pcl
from crow_vision_ros2.filters import ParticleFilter

# Tracker
from crow_vision_ros2.tracker import Tracker
from crow_vision_ros2.tracker.tracker_config import DETECTIONS_FOR_SETUP_NEEDED

#TF
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import MultiArrayDimension, Float32MultiArray
from crow_msgs.msg import FilteredPose, PclDimensions, ObjectPointcloud, AssemblyObjectProbability
from crow_ontology.crowracle_client import CrowtologyClient
from crow_control.utils.profiling import StatTimer

from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import numpy as np
from crow_control.utils import ParamClient

from crow_vision_ros2.tracker.tracker_avatar import Avatar


class ParticleFilterNode(Node):
    UPDATE_INTERVAL = 0.3
    # UPDATE_WINDOW_DURATION = 0.2  # NOT USED! Time window size from which messages are aggregated (should normally be the same as UPDATE_INTERVAL)
    VISUALIZE_PARTICLES = True  # whether to publish filter particles via ROS message
    VISUALIZE_POSES = True  # whether to publish separate PoseArray for tracked objects
    SEGMENTED_PCL_TOPIC = "/detections/segmented_pointcloud"
    FILTERED_POSES_TOPIC = "/filtered_poses"
    FILTERED_PCL_TOPIC = "/filtered_pcls"
    ASSEMBLY_OBJECT_TOPIC = '/assembly_object'
    RESET_OBJECTS_SERVICE = "/reset_tracker_objects"

    def __init__(self, node_name="particle_filter"):
        super().__init__(node_name)
        # get a dictionary with object properties from onto
        self.crowracle = CrowtologyClient(node=self)
        self.object_properties = self.crowracle.get_filter_object_properties()
        # create an instance of PF
        self.particle_filter = ParticleFilter(self.object_properties)  # the main component

        # message counting vars
        self.received_msg = 0
        self.messages_processed = 0

        # create message cache for segmented PCLs
        qos = QoSProfile(
            depth=30,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.get_logger().info("Created subscriber for segmented_pcl '{self.SEGMENTED_PCL_TOPIC}'")
        sub = message_filters.Subscriber(self, SegmentedPointcloud, self.SEGMENTED_PCL_TOPIC, qos_profile=qos, callback_group=MutuallyExclusiveCallbackGroup())
        self.cache = message_filters.Cache(sub, 50, allow_headerless=False)

        sub2 = message_filters.Subscriber(self, SegmentedPointcloud, self.SEGMENTED_PCL_TOPIC, qos_profile=qos, callback_group=MutuallyExclusiveCallbackGroup())
        # sub2.registerCallback(self.message_counter_callback)

        # self.pubPCLdebug = self.create_publisher(PointCloud2, self.SEGMENTED_PCL_TOPIC + "_debug", qos_profile=qos)  # not used currently

        # setup time variables
        self.lastFilterUpdate = self.get_clock().now()  # when was the filter last update (called pf.update())
        self.lastMeasurement = self.get_clock().now()  # timestamp of the last measurement (last segmented PCL message processed)
        # self.updateWindowDuration = Duration(seconds=self.UPDATE_WINDOW_DURATION)
        self.timeSlipWindow = Duration(seconds=1.5)
        self.measurementTolerance = Duration(nanoseconds=1)

        # Publisher for the output of the filter
        self.filtered_publisher = self.create_publisher(FilteredPose, self.FILTERED_POSES_TOPIC, qos)
        self.debug_pose_publisher = self.create_publisher(PoseArray, self.FILTERED_POSES_TOPIC + "_debug", qos)
        self.pcl_publisher = self.create_publisher(ObjectPointcloud, self.FILTERED_PCL_TOPIC, qos)
        self.timer = self.create_timer(self.UPDATE_INTERVAL, self.filter_update) # this callback is called periodically to handle everyhing
        StatTimer.init()

        self.pclient = ParamClient()
        self.pclient.define("filter_alive", True)

        # Tracker initialization
        self.tracker = Tracker(crowracle=self.crowracle, freezeing_cb=self.freezing_cb)
        self.avatar_data_classes = Avatar.AVATAR_PARTS
        # create avatar callback
        self.create_subscription(SegmentedPointcloud, '/detections/segmented_pointcloud_avatar', callback=self.avatar_callback, qos_profile=qos, callback_group=MutuallyExclusiveCallbackGroup())

        # create assembly object publisher
        self.object_pub = self.create_publisher(AssemblyObjectProbability, self.ASSEMBLY_OBJECT_TOPIC, 10)
        self.assembly_object_types = [getattr(AssemblyObjectProbability, o) for o in sorted(dir(AssemblyObjectProbability)) if o.startswith("O_")]

        self.start_build_srv = self.create_service(ResetScene, self.RESET_OBJECTS_SERVICE, self.reset_tracker_objects)#, callback_group=rclpy.callback_groups.ReentrantCallbackGroup())

        self.get_logger().info("Filter is up")

    def message_counter_callback(self, msg):
        print(f'received msgs: {self.received_msg}')
        print(f'processed messages: {self.messages_processed}')

        self.received_msg += 1
        print(f'dropped {(1-(self.messages_processed / self.received_msg)) *100}% of messages')

        oldest_time = self.cache.getOldestTime()
        latest_time = self.cache.getLastestTime()

        print(f'times {[t.seconds_nanoseconds() for t in self.cache.cache_times]}')

    def freezing_cb(self, class_name, obj_uuid):
        # get classes probability and publish object added to workspace message
        n = len(self.assembly_object_types)
        probs = np.zeros(n)
        probs += np.random.rand(n) * 0.01
        if class_name not in self.assembly_object_types:
            self.get_logger().warn(f"Tracker wanted to freeze {class_name} but it isn't in assembly object types!")
            return
        probs[self.assembly_object_types.index(class_name)] = 1
        probs /= probs.sum()
        aop = AssemblyObjectProbability(probabilities=probs)
        self.object_pub.publish(aop)
        self.get_logger().info(f"Published assembly object probabilities: {aop}")

    def reset_tracker_objects(self, request, response):
        """Resets the tracker scene setup (i.e. the tracked objects)"""
        self.get_logger().info("Got request to reset the scene.")
        n_updates = request.n_updates
        if n_updates > 0:
            self.tracker.DETECTIONS_FOR_SETUP_NEEDED = n_updates
        else:
            self.tracker.DETECTIONS_FOR_SETUP_NEEDED = DETECTIONS_FOR_SETUP_NEEDED

        self.tracker.reset_setup()
        response.success = True
        return response

    def add_and_process(self, messages):
        if type(messages) is not list:  # make sure messages is a list for consistency
            messages = [messages]

        if len(messages) == 0:  # if there are no messages to process, exit
            return

        # self.get_logger().info(f"Adding {len(messages)} measurements to the filter")
        # latest = self.lastMeasurement
        for pcl_msg in messages:
            self.frame_id = pcl_msg.header.frame_id
            # if latest < Time.from_msg(pcl_msg.header.stamp):
                # latest = Time.from_msg(pcl_msg.header.stamp)
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
            mdelay = (now - rclpy.time.Time.from_msg(pcl_msg.header.stamp)).nanoseconds * 1e-9
            self.get_logger().error(f"Processed pcl in filter, delay {mdelay:0.3f}")

        now = self.get_clock().now()
        # self.lastMeasurement = latest_time + self.measurementTolerance
        self.update(now)

    def update(self, now=None):
        StatTimer.enter("Filter node update loop")
        self.pclient.filter_alive = time()
        self.particle_filter.update()
        if now is not None:
            self.lastFilterUpdate = now
        else:
            self.lastFilterUpdate = self.get_clock().now()

        if self.particle_filter.n_models > 0:
            StatTimer.enter("Filter publishing")
            estimates = self.particle_filter.getEstimates()
            if len(estimates) == 0:
                self.get_logger().info("Got no estimates from filter, doing nothing.")
                return

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


            # print(f"<filter_node>: Before tracker")
            StatTimer.enter("tracking")
            last_uuids, original_uuids = self.tracker.track_and_get_uuids(centroid_positions=poses_formatted, dimensions=dimensions_formatted, class_names=class_names_formatted, uuids=uuids_formatted)
            # print(f"*** last_uuid: {last_uuid}")
            # print(f"*** latest_uuid: {latest_uuid}")
            StatTimer.exit("tracking")
            StatTimer.enter("correcting uuids")
            self.particle_filter.correct_model_uuids(last_uuids=last_uuids, original_uuids=original_uuids)
            StatTimer.exit("correcting uuids")

            # self.tracker.dump_tracked_objects_info()

            #self.get_logger().info(str(estimates))
            poses = []
            dimensions = []
            tracked = []
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
            pose_array_msg = FilteredPose(poses=poses)
            if self.VISUALIZE_POSES:
                pmsg = PoseArray(poses=poses)
                pmsg.header.stamp = self.get_clock().now().to_msg()
                pmsg.header.frame_id = self.frame_id
                self.debug_pose_publisher.publish(pmsg)
            pose_array_msg.size = dimensions
            # Differentiate between tracked and non-tracked objects
            for uid in uuids_formatted:  # FIXME: check if this is correct (last/latest?)
                if uid in latest_uuid:
                    tracked.append(True)
                else:
                    tracked.append(False)
            # if (len(last_uuid) != 0) and (len(latest_uuid) != 0):
            #     for idx in range(len(latest_uuid)):
            #         if latest_uuid[idx] == -1:
            #             tracked.append(False)
            #         else:
            #             tracked.append(True)
            pose_array_msg.tracked = tracked
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

            self.get_logger().info('Publishing objects:' + str(labels))
            self.get_logger().info('Tracked:' + str(pose_array_msg.tracked))
            self.filtered_publisher.publish(pose_array_msg)

            StatTimer.enter("Filter PCL publish")
            # get PCL for each model
            pcl_uuids, pcl_points, pcl_labels = self.particle_filter.getPclsEstimates()
            pcl_msg = ObjectPointcloud()
            pcl_msg.header.stamp = self.get_clock().now().to_msg()
            pcl_msg.header.frame_id = self.frame_id
            aggregate_pcl = []
            # print("////////////////////")
            # print(class_names_formatted)
            for obj in pcl_points:
                if len(obj) > 1:  # if there are more PCLs for this model
                    aggregate_pcl.append(np.concatenate(obj, axis=0))  # aggregate them
                else:
                    aggregate_pcl.append(obj[0])
            pcls = []
            for i, np_pcl in enumerate(aggregate_pcl):  # clear empty PCLs
                if len(np_pcl) > 0:
                    pcls.append(ftl_numpy2pcl(np_pcl.astype(np.float32).T, pcl_msg.header))
                else:
                    pcl_uuids.pop(i)
                    class_names_formatted.pop(i)
            # print(len(pcls), len(pcl_uuids), class_names_formatted)
            if len(pcl_uuids) > 0:
                # self.pubPCLdebug.publish(pcls[-1])
                pcl_msg.uuid = pcl_uuids
                pcl_msg.pcl = pcls
                pcl_msg.labels = pcl_labels # TODO: class_names_formatted
                self.pcl_publisher.publish(pcl_msg)
                StatTimer.exit("Filter PCL publish")
            StatTimer.exit("Filter publishing")
        StatTimer.exit("Filter node update loop")

    def filter_update(self):
        """Main function, periodically called by rclpy.Timer
        """
        messages = self.cache.cache_msgs
        self.cache.cache_msgs = []
        self.cache.cache_times = []
        if len(messages) > 0:
            self.add_and_process(messages)
            self.messages_processed += len(messages)
        # # latest_time = self.cache.getLastestTime()  # get the time of the last message received
        # latest_time = max(self.cache.cache_times)  # get the time of the last message received
        # if latest_time is not None:  # None -> there are no messages
        #     # Find timestamp of the oldest message that wasn't processed, yet
        #     # oldest_time = self.cache.getOldestTime()  # this is BROKEN
        #     oldest_time =  min(self.cache.cache_times)
        #     if oldest_time < self.lastMeasurement:
        #         oldest_time = self.lastMeasurement

        #     # anyupdate = False  # helper var to see if there was some update
        #     # while oldest_time < latest_time:  # this is old - it uses update window
        #     #     next_time = oldest_time + self.updateWindowDuration
        #     #     messages = self.cache.getInterval(oldest_time, next_time)
        #     #     oldest_time = next_time
        #     #     if len(messages) == 0:
        #     #         continue
        #     #     self.add_and_process(messages)
        #     #     anyupdate = True
        #     # print(f"oldest-latest: {oldest_time.seconds_nanoseconds()} - {latest_time.seconds_nanoseconds()}")
        #     try:
        #         messages = self.cache.getInterval(oldest_time, latest_time)
        #     except:
        #         print(oldest_time)
        #         print(latest_time)
        #         print([t.seconds_nanoseconds() for t in self.cache.cache_times])
        #         exit(-1)
        #     # print(f'message_size: {len(messages)}')
        #     self.messages_processed += len(messages)
        #     self.add_and_process(messages, latest_time)
        #     # if anyupdate:
        #     #     self.lastMeasurement += self.measurementTolerance
        else:
            self.update()

    # Get Avatar PCL and update his parts
    def avatar_callback(self, spcl_msg):
        # print(self.getCameraData(camera))
        if not spcl_msg.pcl:
            self.get_logger().info("no avatar data. Quitting early.")
            return  # no mask detections (for some reason)

        np_pcl, _, c = ftl_pcl2numpy(spcl_msg.pcl)
        # object_id = spcl_msg.object_id
        label = str(spcl_msg.label)
        # confidence = float(spcl_msg.confidence)

        np_pcl_center = np.median(np_pcl, axis=0).reshape(1, 3)
        np_pcl_dimension = (np.max(np_pcl, axis=0) - np.min(np_pcl, axis=0)).reshape(1, 3)

        # print(f"self.tracked_objects: {self.tracker.tracked_objects}")
        # print(f"self.tracker.avatar: {self.tracker.avatar}")
        self.tracker.avatar.update_avatar_object(avatar_object_name=label, np_position=np_pcl_center, np_dimensions=np_pcl_dimension)
        self.tracker.avatar.dump_info()

def main():
    rclpy.init()
    pfilter = ParticleFilterNode()
    try:
        # n_threads = 2
        # mte = MultiThreadedExecutor(num_threads=n_threads, context=rclpy.get_default_context())
        # rclpy.spin(pfilter, executor=mte)
        rclpy.spin(pfilter)
    except KeyboardInterrupt:
        print("User requested shutdown.")
    except BaseException as e:
        print(f"Some error had occured: {e}")
        tb.print_exc()
    finally:
        pfilter.destroy_node()


if __name__ == "__main__":
    main()
