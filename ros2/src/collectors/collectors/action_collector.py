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

import json
import numpy as np
import cv2
import cv_bridge

import pkg_resources
import time
from numba import jit
import torch
from torchvision.utils import save_image
from datetime import datetime
import os
import yaml


class RollingNDBuffer():

    def __init__(self, length, shape, dtype):
        self._length = length
        self._shape = shape
        self._dtype = dtype
        self._buffer = None
        self.clear()

    def _init_buffer(self):
        raise NotImplementedError()

    def push(self, item):
        """Insert an element to the "top" of the buffer.
        """
        if self._input_index + 1 == self._output_index:
            if self._input_index == self._length - 2:
                self._output_index = 0
            else:
                self._output_index += 1
        elif self._input_index < 0:
            self._output_index = 0
        self._input_index += 1
        if self._input_index == self._length:
            self._input_index = 0
            if self._output_index == 0:
                self._output_index = 1

        self._insert_item(item)

    def _insert_item(self, item):
        raise NotImplementedError()

    def pop(self):
        """Return the oldest (first) added element (i.e. from the buffer bottom) and remove it from the buffer.
        """
        if not self.empty:
            item = self._buffer[self._output_index, ...]
            self._output_index += 1
            if self._output_index + 1 == self._input_index:  # buffer was emptied
                self.clear()
            elif self._output_index == self._length:
                self._output_index = 0

            return item

    def clear(self):
        self._input_index = -1
        self._output_index = -1
        self._was_filled = False
        self._init_buffer()

    def get_data(self):
        if not self.empty:
            if self._input_index < self._output_index:
                return torch.cat([
                    self._buffer[self._output_index:, ...],
                    self._buffer[:self._input_index + 1, ...]
                ])
            else:
                return self._buffer[self._output_index:self._input_index + 1, ...]

    @property
    def top(self):
        """Return the latest added element.
        """
        if not self.empty:
            return self._buffer[self._input_index, ...]

    @property
    def bottom(self):
        """Return the oldest added element. Similar to "pop" but without removing the element.
        """
        if not self.empty:
            return self._buffer[self._output_index, ...]

    @property
    def empty(self):
        return self._output_index == -1

    @property
    def full(self):
        return self.size == self._length

    @property
    def size(self):
        s = self._input_index - self._output_index
        return 0 if self.empty else ((self._length + s if s < 0 else s) + 1)


class TorchRollingNDBuffer(RollingNDBuffer):

    def _init_buffer(self):
        self._buffer = torch.zeros((self._length, ) + self._shape, dtype=self._dtype)

    def _insert_item(self, item):
        self._buffer[self._input_index, ...] = torch.tensor(item)


class ActionCollector(Node):
    cv_win_name = "Cameras"
    bufferLength = 180
    INCLUDE_PCL = False
    IMAGE_HEIGHT = 424
    IMAGE_WIDTH = 240
    AUTO_STOP = True

    def __init__(self, node_name="action_collector"):
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

        self.imageBuffer = {}
        self.depthBuffer = {}
        #create listeners (synchronized)
        for i, (cam, imageTopic, depthTopic, pclTopic, camera_instrinsics) in enumerate(zip(self.cameras, self.image_topics, self.depth_topics, self.pcl_topics, self.camera_instrinsics)):
            # convert camera data to numpy
            self.camera_instrinsics[i]["camera_matrix"] = np.array(camera_instrinsics["camera_matrix"], dtype=np.float32)
            self.camera_instrinsics[i]["distortion_coefficients"] = np.array(camera_instrinsics["distortion_coefficients"], dtype=np.float32)

            # create approx syncro callbacks
            subs = [
                message_filters.Subscriber(self, Image, imageTopic, qos_profile=qos),
                message_filters.Subscriber(self, Image, depthTopic, qos_profile=qos)
            ]
            if self.INCLUDE_PCL:
                subs.append(message_filters.Subscriber(self, PointCloud2, pclTopic, qos_profile=qos))

            # create and register callback for syncing these message streams, slop=tolerance [sec]
            sync = message_filters.ApproximateTimeSynchronizer(subs, 10, slop=0.03)
            sync.registerCallback(lambda img_msg, depth_msg, pcl_msg=None, cam=cam: self.message_cb(img_msg, depth_msg, pcl_msg, cam))
            self.get_logger().info(f"Connected to {cam}")

            self.imageBuffer[cam] = TorchRollingNDBuffer(self.bufferLength, (480, 848, 3), torch.uint8)
            self.depthBuffer[cam] = TorchRollingNDBuffer(self.bufferLength, (480, 848), torch.int32)

        cv2.namedWindow(self.cv_win_name)
        self.is_recording = False
        self.last_recording_saved = True
        self.create_timer(0.05, self.gui_cb)

        # OpenCV window initialization

    def message_cb(self, img_msg, depth_msg, pcl_msg, camera):
        image = self.cvb.imgmsg_to_cv2(img_msg, "rgb8")
        depth = self.cvb.imgmsg_to_cv2(depth_msg, "passthrough")
        if pcl_msg is not None:
            pcl, pcl_colors, _ = ftl_pcl2numpy(pcl_msg)

        if not(self.AUTO_STOP and self.is_recording and self.imageBuffer[camera].full):
            self.imageBuffer[camera].push(image)
            self.depthBuffer[camera].push(depth.astype(np.int32))
        # self.get_logger().info("\t".join([str(image.shape), str(depth.shape), str(pcl.shape)]))

    def gui_cb(self):
        images = [cv2.resize(self.imageBuffer[cam].top.numpy(), (self.IMAGE_HEIGHT, self.IMAGE_WIDTH), cv2.INTER_NEAREST) for cam in self.cameras if not self.imageBuffer[cam].empty]

        if len(images) == 0:
            return

        image_row = np.hstack(images)
        image_stack = np.vstack((
            image_row,
            np.zeros((30, image_row.shape[1], 3), dtype=np.uint8)
        ))
        if self.is_recording:
            if self.AUTO_STOP and all([self.imageBuffer[cam].full for cam in self.cameras]):
                self.is_recording = False
                cv2.putText(image_stack, "stopped", (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image_stack, "recording", (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(self.cv_win_name, image_stack)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):  # QUIT
            self.get_logger().info("Quitting.")
            cv2.destroyAllWindows()
            rclpy.try_shutdown()
            return
        elif key == ord("s"):  # SAVE DATA
            self.save_data()
        elif key == 8:  # DELETE RECORDING BUFFER
            if not self.is_recording:
                self.clear_buffers()
                self.last_recording_saved = True
                self.get_logger().info("Cleared buffers.")
        elif key == ord(" "):  # START/STOP RECORDING
            if self.is_recording:  # was recording -> stop and save data
                self.is_recording = False
                self.last_recording_saved = False
            else:  # wasn't recording -> start recording
                if self.last_recording_saved:
                    self.get_logger().info("Started recording.")
                    self.is_recording = True
                    self.last_recording_saved = False
                    dt = datetime.now()
                    self.rec_session = dt.strftime("%Y-%m-%d_%H_%M_%S")
                    self.clear_buffers()
                else:
                    self.get_logger().warn("Last recording not save! Save it or delete it (backspace) before starting recording.")
        elif key != 255:
            print(key)

    def clear_buffers(self):
        for cam in self.cameras:
            self.imageBuffer[cam].clear()
            self.depthBuffer[cam].clear()

    def save_data(self):
        self.get_logger().info(f"Saving data to folder {self.rec_session}.")
        # save data
        os.mkdir(self.rec_session)
        os.chdir(self.rec_session)

        for cam in self.cameras:
            # make & change to camera folder
            os.mkdir(cam[1:])
            os.chdir(cam[1:])

            # save camera info
            with open("camera_info.yaml", "w+") as f:
                yaml.dump(self.getCameraData(cam), f)

            # save images
            image_data = self.imageBuffer[cam].get_data()
            os.mkdir("images")
            os.chdir("images")
            for i, image in enumerate(image_data):
                cv2.imwrite(f"image_{i}.png", cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2RGB))

            os.chdir("..")
            os.mkdir("depth")
            os.chdir("depth")

            depth_data = self.depthBuffer[cam].get_data()
            for i, depth in enumerate(depth_data):
                cv2.imwrite(f"depth_{i}.png", depth.numpy().astype(np.int16))
            os.chdir("../..")

        os.chdir("..")
        self.last_recording_saved = True

    def getCameraData(self, camera):
        idx = self.cameras.index(camera)
        return {
            "camera": camera,
            "image_topic": self.image_topics[idx],
            "camera_matrix": self.camera_instrinsics[idx]["camera_matrix"],
            # "camera_matrix": np.array([383.591, 0, 318.739, 0, 383.591, 237.591, 0, 0, 1]).reshape(3, 3),
            "distortion_coefficients": self.camera_instrinsics[idx]["distortion_coefficients"],
            "optical_frame": self.camera_frames[idx],
        }


def main():
    rclpy.init()

    ac = ActionCollector()

    rclpy.spin(ac)
    ac.destroy_node()
    cv2.destroyAllWindows()
    exit(0)


if __name__ == "__main__":
    main()
