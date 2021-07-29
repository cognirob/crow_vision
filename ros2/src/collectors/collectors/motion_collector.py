from torch import where
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

#Pointcloud
from sensor_msgs.msg import PointCloud2, Image
from crow_vision_ros2.utils import ftl_pcl2numpy, MicroTimer
from crow_msgs.msg import ObjectPointcloud, FilteredPose

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from datetime import datetime
import json
import numpy as np
import cv2
import cv_bridge

import pkg_resources
import time
#from numba import jit
#from torchvision.utils import save_image
#from datetime import datetime
import os
import yaml
import transforms3d as tf3


class PCLPose():

    def __init__(self, uuid, stamp, label, pcl, pose, size) -> None:
        self._uuid = uuid
        self._stamp = stamp
        self._pcl = pcl
        self._pcl_numpy, _, _ = ftl_pcl2numpy(pcl)  # convert from PointCloud2.msg to numpy array
        self._label = label
        self._pose = pose
        self._position, self._quat, self._orientation = self._extract_pose(pose)
        self._size = size
        self._center = np.mean(self._pcl_numpy, 0)

    def _extract_pose(self, pose):
        trans = np.r_["0,2,0", [getattr(pose.position, a) for a in "xyz"]]
        quat = [getattr(pose.orientation, a) for a in "xyzw"]
        rot = tf3.quaternions.quat2mat(quat[3:] + quat[:3])
        return trans, quat, rot

    def serialize(self):
        return {
            "uuid": self._uuid,
            # "stamp": f"{self._stamp.sec}.{self._stamp.nanosec}",
            "label": self._label,
            "position": self._position.ravel().tolist(),
            "quaternion": self._quat,
            "orientation": self._orientation.ravel().tolist(),
            "size": self._size.dimensions.tolist(),
            "center": self._center.tolist(),  # center of PCL
        }

    @property
    def pcl_numpy(self):
        return self._pcl_numpy

    @property
    def stamp(self):
        return self._stamp

    @property
    def uuid(self):
        return self._uuid


class ImageCollector(Node):
    cv_win_name = "image"
    SCREEN_HEIGHT = 1000
    SCREEN_WIDTH = 1900
    IMAGES_PER_ROW = 2
    GUI_UPDATE_INTERVAL = 0.05
    FILT_PCL_TOPIC = "/filtered_pcls"
    FILT_POSE_TOPIC = "/filtered_poses"

    def __init__(self, node_name="image_collector"):
        super().__init__(node_name)
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(
            node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]

        while len(self.cameras) == 0:  # wait for cams to come online
            self.get_logger().warn("No cams detected, waiting 2s.")
            time.sleep(2)
            self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(
                node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]

        # camera intrinsics
        self.camera_instrinsics = [json.loads(cintr) for cintr in self.camera_instrinsics]
        # rgb, depth, and pcl topics
        self.image_topics = [cam + "/color/image_raw" for cam in self.cameras]
        self.annot_topics = [cam + "/detections/image_annot" for cam in self.cameras]
        self.depth_topics = [cam + "/depth/image_rect_raw" for cam in self.cameras]
        self.pcl_topics = [cam + "/depth/color/points" for cam in self.cameras]

        # create output topic and publisher dynamically for each cam
        qos = QoSProfile(depth=60, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        qos_pcl = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        self.cvb = cv_bridge.CvBridge()

        subs = []
        n_cams = len(self.cameras)
        #create listeners (synchronized)
        for i, (cam, imageTopic, annotTopic, depthTopic, pclTopic, camera_instrinsics) in enumerate(zip(self.cameras, self.image_topics, self.annot_topics, self.depth_topics, self.pcl_topics, self.camera_instrinsics)):
            # convert camera data to numpy
            self.camera_instrinsics[i]["camera_matrix"] = np.array(camera_instrinsics["camera_matrix"], dtype=np.float32)
            self.camera_instrinsics[i]["distortion_coefficients"] = np.array(camera_instrinsics["distortion_coefficients"], dtype=np.float32)

            # create approx syncro callbacks
            subs.append(message_filters.Subscriber(self, Image, annotTopic, qos_profile=qos))
            # subs.append(message_filters.Subscriber(self, Image, imageTopic, qos_profile=qos))
            self.get_logger().info(f"Connected to {cam}")

        # create and register callback for syncing these message streams, slop=tolerance [sec]
        subs.append(message_filters.Subscriber(self, ObjectPointcloud, self.FILT_PCL_TOPIC, qos_profile=qos_pcl))
        subs.append(message_filters.Subscriber(self, FilteredPose, self.FILT_POSE_TOPIC, qos_profile=qos_pcl))
        sync = message_filters.ApproximateTimeSynchronizer(subs, 30 * (n_cams + 1), slop=0.05)
        sync.registerCallback(self.message_cb)

        self.entry_count = 0
        self.image_width = int(self.SCREEN_WIDTH / self.IMAGES_PER_ROW)
        intrscs = self.camera_instrinsics[0]
        factor = intrscs["width"] / self.image_width
        self.image_height = int(intrscs["height"] / factor)
        # create named window for GUI
        cv2.namedWindow(self.cv_win_name)

        self.lastImageList = None
        self.pauseAcquisition = False

        # Connect to onto
        # self.crowracle = CrowtologyClient(node=self)
        # self.onto = self.crowracle.onto

        self.main_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # Create folder for this recording session
        print(os.getcwd())
        print(self.main_dir)
        os.makedirs(self.main_dir)
        os.makedirs(os.path.join(self.main_dir, "images"))
        os.makedirs(os.path.join(self.main_dir, "pcls"))
        for nc in range(1, n_cams + 1):
            os.makedirs(os.path.join(self.main_dir, "images", f"camera_{nc}"))

        # setup some vars
        # self.create_timer(self.GUI_UPDATE_INTERVAL, self.gui_cb)

    def process_pcls(self, pcl_msg, pose_msg):
        poses = pose_msg.poses
        labels = pose_msg.label
        sizes = pose_msg.size
        current_stamp = pcl_msg.header.stamp
        pose_uuids = pose_msg.uuid

        objects = {}
        for uid, pcl in zip(pcl_msg.uuid, pcl_msg.pcl):
            if uid in objects:  # probably an error?
                self.get_logger().error(f"Object with uuid {uid} appeared multiple times in the PCL message.")
            try:
                where_in_poses = pose_uuids.index(uid)
            except ValueError as e:
                self.get_logger().error(f"Object with uuid {uid} could not be found in poses!\nError: {e}")
                continue
            pose = poses[where_in_poses]
            label = labels[where_in_poses]
            size = sizes[where_in_poses]
            objects[uid] = PCLPose(uid, current_stamp, label, pcl, pose, size)
        return objects, pcl_msg.header.stamp, pose_msg.header.stamp

    def message_cb(self, *msgs):
        """
        ROS message callback. Received synchronized messages from all recorded topics (from one camera)
        """
        if not self.pauseAcquisition:
            self.lastImageList = [self.cvb.imgmsg_to_cv2(img_msg, "rgb8") for img_msg in msgs[:-2]]
            imageStamps = [img_msg.header.stamp for img_msg in msgs[:-2]]
            objects, pcl_stamp, pose_stamp = self.process_pcls(msgs[-2], msgs[-1])
            if len(objects) > 0:
                self.storeData(self.lastImageList, imageStamps, objects, pcl_stamp, pose_stamp)
                self.get_logger().info(f"Got image(s) @ {imageStamps[0].sec} with {len(objects)} object(s).")

    def gui_cb(self):
        """
        GUI callback. It is called once every "self.GUI_UPDATE_INTERVAL" seconds.
        """
        image_stack = np.zeros((100, 100))
        # get images from all buffers
        if self.lastImageList is not None:
            images = [cv2.resize(img, (self.image_width, self.image_height), cv2.INTER_NEAREST) for img in self.lastImageList]
            image_stack = np.hstack(images)  # stack images to a single row
        # # Optionally, append an info bar at the bottom
        # image_stack = np.vstack((
        #     image_stack,
        #     np.zeros((30, image_stack.shape[1], 3), dtype=np.uint8)
        # ))

        cv2.imshow(self.cv_win_name, image_stack)
        key = cv2.waitKey(10) & 0xFF

        # *** Keyboard input handling ***
        if key == ord("q"):  # QUIT
            self.get_logger().info("Quitting.")
            cv2.destroyAllWindows()
            rclpy.try_shutdown()
            return
        elif key == ord(" "):  # START/STOP RECORDING
            self.pauseAcquisition = not self.pauseAcquisition
        elif key != 255:
            print(key)

    def storeData(self, imageList, imageStamps, objects, pcl_stamp, pose_stamp):
        image_names = []
        for i, (stamp, image) in enumerate(zip(imageStamps, imageList)):
            image_name = self.save_image(image, os.path.join(self.main_dir, "images", f"camera_{str(i + 1)}"), self.entry_count)
            image_names.append({
                "stamp": f"{stamp.sec}.{stamp.nanosec}",
                "image": image_name
            })

        object_list = {}
        if len(objects) > 0:
            pcls = {}
            for uid, obj in objects.items():
                pcls[obj.uuid] = obj.pcl_numpy
                object_list[uid] = obj.serialize()

            with open(os.path.join(self.main_dir, "pcls", f"pcl_{self.entry_count:05d}.npz"), "wb") as f:
                np.savez_compressed(f, **pcls)

        json_data = {
            "frame": self.entry_count,
            "pcl_stamp": f"{pcl_stamp.sec}.{pcl_stamp.nanosec}",
            "pose_stamp": f"{pose_stamp.sec}.{pose_stamp.nanosec}",
            "images": image_names,
            "objects": object_list
        }
        with open(os.path.join(self.main_dir, f"frame_{self.entry_count:05d}.json"), "w") as f:
            json.dump(json_data, f, indent=4)

        self.entry_count += 1

    def save_image(self, image, path, img_id):
        """
        Save data for camera
        """
        file_name = f"image_{img_id:03d}.png"
        file_path = os.path.join(path, f"image_{img_id:05d}.png")
        cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return file_name

    def getCameraData(self, camera):
        idx = self.cameras.index(camera)
        return {
            "camera": camera,
            "image_topic": self.image_topics[idx],
            "camera_matrix": self.camera_instrinsics[idx]["camera_matrix"].tolist(),
            # "camera_matrix": np.array([383.591, 0, 318.739, 0, 383.591, 237.591, 0, 0, 1]).reshape(3, 3),
            "distortion_coefficients": self.camera_instrinsics[idx]["distortion_coefficients"].tolist(),
            "optical_frame": self.camera_frames[idx],
        }


def main():
    rclpy.init()

    ic = ImageCollector()

    rclpy.spin(ic)
    ic.destroy_node()
    cv2.destroyAllWindows()
    exit(0)


if __name__ == "__main__":
    main()
