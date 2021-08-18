import rclpy #add to package.xml deps
from rclpy.node import Node
from ros2param.api import call_get_parameters

import sensor_msgs

from crow_msgs.msg import DetectionMask
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import cv2
import numpy as np
import torch

import commentjson as json
import pkg_resources
import time
import copy

# Posenet imports
import sys, os
# Import arm detection - POSENET
POSENET_REPO='~/crow_vision_posenet/' # Existing POSENET
POSENET_REPO_full = '~/crow2/src/crow_vision_posenet/'
sys.path.append(os.path.abspath(os.path.expanduser(POSENET_REPO_full)))
from posenet_tool import Posenet

print(f"Running PyTorch:")
print(f"\tver: {torch.__version__}")
print(f"\tfile: {torch.__file__}")
qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

class CrowVisionPose(Node):
    """
    ROS2 node for CNN used in Crow.

    Run as
    `ros2 run crow_vision_ros2 detector [path_to_weights [config to use [input topic name]]]`

    This node listens on network for topic "/camera1/color/image_raw",
    obtains raw RGB image, processes it in neural net, and publishes results in form of (for given camera):
    - "/detections/masks"
    """
    def __init__(self, config='config.json'):
        super().__init__('crow_detector')
        #parse config
        CONFIG_DEFAULT = pkg_resources.resource_filename("crow_vision_ros2", config)
        with open(CONFIG_DEFAULT) as configFile:
            self.config = json.load(configFile)
            print(self.config)

        ## handle multiple inputs (cameras).
        # store the ROS Listeners,Publishers in a dict{}, keys by topic.
        self.cameras, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]
        while len(self.cameras) == 0:
            self.get_logger().warn("Waiting for any cameras!")
            time.sleep(2)
            self.cameras, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]

        self.ros = {}
        for cam in self.cameras:
            camera_topic=cam+"/color/image_raw"
            # create INput listener with raw images
            listener = self.create_subscription(msg_type=sensor_msgs.msg.Image,
                                                topic=camera_topic,
                                                # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                                callback=lambda msg, topic=camera_topic: self.input_callback(msg, topic),
                                                callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
                                                # callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
                                                qos_profile=1) #the listener QoS has to be =1, "keep last only".
            self.get_logger().info('Input listener created on topic: "%s"' % camera_topic)
            self.ros[camera_topic] = {} # camera_topic is used as an ID for this input, all I/O listeners,publishers will be based under that id.

            # there are multiple publishers (for each input/camera topic).
            # the results are separated into different (optional) subtopics the clients can subscribe to (eg 'labels', 'masks')
            # If an output topic is empty (""), we skip publishing on that stream, it is disabled. Use to save computation resources.
            if self.config["outputs"]["masks"]:
                # Posenet masks
                topic = cam + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["masks"]
                self.get_logger().info('Output publisher created for topic: "%s"' % topic)
                self.ros[camera_topic]["pub_masks"] = self.create_publisher(DetectionMask, topic, qos_profile=qos) #publishes the processed (annotated,detected) pose

        self.noMessagesYet = True
        self.cvb_ = CvBridge()

        ## Posenet setup
        self.declare_parameter("posenet_model", self.config["posenet_model"])
        model = self.get_parameter("posenet_model").get_parameter_value().string_value
        model_abs = os.path.join(os.path.abspath(os.path.expanduser(POSENET_REPO_full)),str(model))
        assert os.path.exists(model_abs), "Provided path to posenet model weights does not exist! {}".format(model_abs)
        self.posenet = Posenet(model_abs_path=model_abs)

        print('Hi from crow_vision_ros2 posenet.')

    def input_callback(self, msg, topic):
        """
        @param msg - ROS msg (Image data) to be processed. From camera
        @param topic - str, from camera/input on given topic.
        @return nothing, but send new message(s) via output Publishers.
        """
        if self.noMessagesYet:
            self.get_logger().info("Image received from camera! (will not report on next image callbacks)")
            self.noMessagesYet = False

        # self.get_logger().info("I heard: {} for topic {}".format(str(msg.height), topic))
        # self.get_logger().error(str(dir(self.ros[topic]["pub_masks"])))
        # assert topic in self.ros, "We don't have registered listener for the topic {} !".format(topic)

        img_raw = self.cvb_.imgmsg_to_cv2(msg, "bgr8")
        #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # Pose mask
        if "pub_masks" in self.ros[topic]:
            object_ids, classes, class_names, scores, masks = self.posenet.inference(img=img_raw)
            classes = list(map(int, classes))
            scores = list(map(float, scores)) # Remove this -> get errors
            # if len(classes) == 0:
            #     self.get_logger().info("No poses detected, skipping.")
            #     return
            msg_mask = DetectionMask()
            m_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            msg_mask.masks = []
            for mask in masks:
                new_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, m_kernel)
                new_smth = self.cvb_.cv2_to_imgmsg(new_mask, encoding="mono8")
                msg_mask.masks.append(new_smth)
            msg_mask.header.stamp = msg.header.stamp
            for mask in msg_mask.masks:
                mask.header.stamp = msg.header.stamp
            msg_mask.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
            msg_mask.object_ids = object_ids
            msg_mask.classes = classes
            msg_mask.class_names = class_names
            msg_mask.scores = scores
            self.ros[topic]["pub_masks"].publish(msg_mask)

def main(args=None):
    rclpy.init(args=args)
    try:
        cnn = CrowVisionPose()
        n_threads = len(cnn.cameras)
        mte = rclpy.executors.MultiThreadedExecutor(num_threads=n_threads, context=rclpy.get_default_context())
        rclpy.spin(cnn, executor=mte)
        # rclpy.spin(cnn)
        cnn.destroy_node()
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
