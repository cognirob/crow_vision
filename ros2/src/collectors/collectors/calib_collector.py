import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

#Pointcloud
from sensor_msgs.msg import PointCloud2, Image
from crow_vision_ros2.utils import ftl_pcl2numpy, MicroTimer

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


class ImageCollector(Node):
    cv_win_name = "image"
    SCREEN_HEIGHT = 1000
    SCREEN_WIDTH = 1900
    IMAGES_PER_ROW = 3
    GUI_UPDATE_INTERVAL = 0.05

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
        self.depth_topics = [cam + "/depth/image_rect_raw" for cam in self.cameras]
        self.pcl_topics = [cam + "/depth/color/points" for cam in self.cameras]

        # create output topic and publisher dynamically for each cam
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        self.cvb = cv_bridge.CvBridge()

        subs = []
        #create listeners (synchronized)
        for i, (cam, imageTopic, depthTopic, pclTopic, camera_instrinsics) in enumerate(zip(self.cameras, self.image_topics, self.depth_topics, self.pcl_topics, self.camera_instrinsics)):
            # convert camera data to numpy
            self.camera_instrinsics[i]["camera_matrix"] = np.array(camera_instrinsics["camera_matrix"], dtype=np.float32)
            self.camera_instrinsics[i]["distortion_coefficients"] = np.array(camera_instrinsics["distortion_coefficients"], dtype=np.float32)

            # create approx syncro callbacks
            subs.append(message_filters.Subscriber(self, Image, imageTopic, qos_profile=qos))
            self.get_logger().info(f"Connected to {cam}")

        # create and register callback for syncing these message streams, slop=tolerance [sec]
        sync = message_filters.ApproximateTimeSynchronizer(subs, 10 * (i + 1), slop=0.03)
        sync.registerCallback(self.message_cb)

        self.lastImageList = None
        self.pauseAcquisition = False
        self.main_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # Create folder for this recording session
        os.makedirs(self.main_dir)
        os.makedirs(os.path.join(self.main_dir, "_camera1"))
        os.makedirs(os.path.join(self.main_dir, "_camera1", "camera1"))
        os.makedirs(os.path.join(self.main_dir, "_camera2"))
        os.makedirs(os.path.join(self.main_dir, "_camera2", "camera2"))
        os.makedirs(os.path.join(self.main_dir, "_camera1_camera2"))
        os.makedirs(os.path.join(self.main_dir, "_camera1_camera2", "camera1"))
        os.makedirs(os.path.join(self.main_dir, "_camera1_camera2", "camera2"))

        n_cams = len(subs)
        self.image_counts = np.zeros(n_cams + 1, dtype=np.int)  # image_counts[0] = all cams
        self.image_width = int(self.SCREEN_WIDTH / self.IMAGES_PER_ROW)
        intrscs = self.camera_instrinsics[0]
        factor = intrscs["width"] / self.image_width
        self.image_height = int(intrscs["height"] / factor)
        # create named window for GUI
        cv2.namedWindow(self.cv_win_name)



        # setup some vars
        self.create_timer(self.GUI_UPDATE_INTERVAL, self.gui_cb)

    def message_cb(self, *img_msgs):
        """
        ROS message callback. Received synchronized messages from all recorded topics (from one camera)
        """
        if not self.pauseAcquisition:
            self.lastImageList = [self.cvb.imgmsg_to_cv2(img_msg, "rgb8") for img_msg in img_msgs]

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
        elif key == ord("s"):  # SAVE DATA
            tmp = self.pauseAcquisition
            if tmp:
                self.pauseAcquisition = False
            self.save_data(self.lastImageList[0], os.path.join(self.main_dir, "_camera1_camera2", "camera1"), self.image_counts[0])
            self.save_data(self.lastImageList[1], os.path.join(self.main_dir, "_camera1_camera2", "camera2"), self.image_counts[0])
            self.image_counts[0] += 1
            self.pauseAcquisition = tmp
        elif key == ord("a"):  # save cam 1
            self.store_single_cam(1)
        elif key == ord("d"):  # save cam 2
            self.store_single_cam(2)
        elif key == ord(" "):  # START/STOP RECORDING
            self.pauseAcquisition = not self.pauseAcquisition
        elif key != 255:
            print(key)

    def store_single_cam(self, cam_id):
        self.save_data(self.lastImageList[cam_id - 1], os.path.join(self.main_dir, f"_camera{cam_id}", f"camera{cam_id}"), self.image_counts[cam_id])
        self.image_counts[cam_id] += 1

    def save_data(self, image, path, img_id):
        """
        Save data for every camera
        """
        file_path = os.path.join(path, f"image_{img_id:03d}.png")
        cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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
