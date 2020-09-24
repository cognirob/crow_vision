import rclpy #add to package.xml deps
from rclpy.node import Node
from ros2param.api import call_get_parameters

import sensor_msgs
import std_msgs

from crow_msgs.msg import DetectionMask, DetectionBBox, BBox
from cv_bridge import CvBridge

import cv2
import torch
import numpy as np

import commentjson as json
import pkg_resources
import argparse
import time


class CrowVision(Node):
  """
  ROS2 node for CNN used in Crow.

  Run as
  `ros2 run crow_vision_ros2 detector [path_to_weights [config to use [input topic name]]]`

  This node listens on network for topic "/camera1/color/image_raw",
  obtains raw RGB image, processes it in neural net, and publishes results in form of (for given camera):
  - "/detections/image" - processed image with visualized masks, bboxes and labels
  - "/detections/masks"
  - "/detections/labels"
  - "/detections/bboxes"
  - "/detections/confidences", etc. TODO
  """


  def __init__(self,
               config='config.json',
               ):
    super().__init__('crow_detector')

    #parse config
    CONFIG_DEFAULT = pkg_resources.resource_filename("crow_vision_ros2", config)
    with open(CONFIG_DEFAULT) as configFile:
      self.config = json.load(configFile)
      print(self.config)

    # specific imports based on YOLACT / Detectron2
    if self.config["type"] == "YOLACT":
        # import CNN - YOLACT
        YOLACT_REPO='~/crow_vision_yolact/' #use your existing yolact setup
        import sys; import os; sys.path.append(os.path.abspath(os.path.expanduser(YOLACT_REPO)))
        from inference_tool import InfTool
        from yolact import Yolact
        from data import set_cfg
    elif self.config["type"] == "Detectron2":
        import detectron2
    else:
        raise Exception("Supported types only: 'Detectron2', 'YOLACT'. Set in config.type. ")

    ## handle multiple inputs (cameras).
    # store the ROS Listeners,Publishers in a dict{}, keys by topic.
    self.cameras, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_nodes", "camera_frames"]).values]
    while len(self.cameras) == 0:
        self.get_logger().warn("Waiting for any cameras!")
        time.sleep(2)
    for i, cam in enumerate(self.cameras):
        cam = cam[0:-7] #FIXME cameras contain '/camera1/camera' while the RS actually publishes only on `/camera1`
        self.get_logger().info(cam)
        self.cameras[i] = cam

    self.ros = {}
    for cam in self.cameras:
      camera_topic=cam+"/color/image_raw"
      # create INput listener with raw images
      listener = self.create_subscription(msg_type=sensor_msgs.msg.Image,
                                          topic=camera_topic,
                                          # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                          callback=lambda msg, topic=camera_topic: self.input_callback(msg, topic),
                                          qos_profile=1) #the listener QoS has to be =1, "keep last only".
      self.get_logger().info('Input listener created on topic: "%s"' % camera_topic)
      self.ros[camera_topic] = {} # camera_topic is used as an ID for this input, all I/O listeners,publishers will be based under that id.


      # there are multiple publishers (for each input/camera topic).
      # the results are separated into different (optional) subtopics the clients can subscribe to (eg 'labels', 'masks')
      # If an output topic is empty (""), we skip publishing on that stream, it is disabled. Use to save computation resources.
      if self.config["outputs"]["image_annotated"]:
        topic = cam + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["image_annotated"]
        publisher_img = self.create_publisher(sensor_msgs.msg.Image, topic, 10) #publishes the processed (annotated,detected) image
        self.get_logger().info('Output publisher created for topic: "%s"' % topic)
        self.ros[camera_topic]["pub_img"] = publisher_img
      if self.config["outputs"]["masks"]:
        topic = cam + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["masks"]
        self.get_logger().info('Output publisher created for topic: "%s"' % topic)
        self.ros[camera_topic]["pub_masks"] = self.create_publisher(DetectionMask, topic, 10) #publishes the processed (annotated,detected) image
      if self.config["outputs"]["bboxes"]:
        topic = cam + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["bboxes"]
        self.get_logger().info('Output publisher created for topic: "%s"' % topic)
        self.ros[camera_topic]["pub_bboxes"] = self.create_publisher(DetectionBBox, topic, 10) #publishes the processed (annotated,detected) image


      #TODO others publishers
    self.noMessagesYet = True
    self.cvb_ = CvBridge()

    ## YOLACT setup
    # setup additional args
    self.declare_parameter("config", self.config["config"])
    cfg = self.get_parameter("config").get_parameter_value().string_value

    # load model weights
    self.declare_parameter("weights", self.config["weights"])
    model = self.get_parameter("weights").get_parameter_value().string_value

    if ".obj" in cfg:
      cfg = os.path.join(os.path.abspath(
          os.path.expanduser(YOLACT_REPO)), cfg)
    elif "none" in cfg.lower():
      cfg = None

    print("Using config '{}'.".format(cfg))
    print("Using weights from file '{}'.".format(model))

    model_abs = os.path.join(
        os.path.abspath(os.path.expanduser(YOLACT_REPO)),
        str(model)
    )
    assert os.path.exists(model_abs), "Provided path to model weights does not exist! {}".format(model_abs)
    self.cnn = InfTool(weights=model_abs, top_k=self.config["top_k"], score_threshold=self.config["threshold"], config=cfg)
    print('Hi from crow_vision_ros2.')



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
    assert topic in self.ros, "We don't have registered listener for the topic {} !".format(topic)

    img_raw = self.cvb_.imgmsg_to_cv2(msg, "bgr8")
    #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    preds, frame = self.cnn.process_batch(img_raw)
    if preds[0]["detection"] is None:
        return  # do not publish if nothing was detected

    #the input callback triggers the publishers here.
    if "pub_img" in self.ros[topic]: # labeled image publisher. (Use "" to disable)
      img_labeled = self.cnn.label_image(img_raw, preds, frame)
      img_labeled = cv2.cvtColor(img_labeled, cv2.COLOR_BGR2RGB)

      if img_labeled.ndim == 3:
        batch,w,h,c = 1, *img_labeled.shape
      else:
        batch,w,h,c = img_labeled.shape
        img_labeled = img_labeled[0]
      assert batch==1,"Batch mode not supported in ROS yet"

      msg_img = self.cvb_.cv2_to_imgmsg(img_labeled, encoding="rgb8")
      # parse time from incoming msg, pass to outgoing msg
      msg_img.header.stamp.nanosec = msg.header.stamp.nanosec
      msg_img.header.stamp.sec  = msg.header.stamp.sec
    #   self.get_logger().info("Publishing as Image {} x {}".format(msg_img.width, msg_img.height))
      self.ros[topic]["pub_img"].publish(msg_img)

    if "pub_masks" in self.ros[topic] or "pub_bboxes" in self.ros[topic]:
      classes, class_names, scores, bboxes, masks, centroids = self.cnn.raw_inference(img_raw, preds)
      classes = classes.astype(int).tolist()
      scores = scores.astype(float).tolist()
      if len(classes) == 0: 
          self.get_logger().info("No objects detected, skipping.")
          return

      if "pub_masks" in self.ros[topic]:
        msg_mask = DetectionMask()
        msg_mask.masks = [self.cvb_.cv2_to_imgmsg(mask, encoding="mono8") for mask in masks.astype(np.uint8)]
        # parse time from incoming msg, pass to outgoing msg
        msg_mask.header.stamp = msg.header.stamp
        msg_mask.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
        msg_mask.classes = classes
        msg_mask.class_names = class_names
        msg_mask.scores = scores
        #self.get_logger().info("Publishing as String {} at time {} ".format(msg_mask.class_names, msg_mask.header.stamp.sec))
        self.ros[topic]["pub_masks"].publish(msg_mask)
      if "pub_bboxes" in self.ros[topic]:
        msg_bbox = DetectionBBox()
        msg_bbox.bboxes = [BBox(bbox=bbox) for bbox in bboxes]
        # parse time from incoming msg, pass to outgoing msg
        msg_bbox.header.stamp = msg.header.stamp
        msg_bbox.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
        msg_bbox.classes = classes
        msg_bbox.class_names = class_names
        msg_bbox.scores = scores
        # self.get_logger().info("Publishing as String {} at time {} ".format(msg_mask.data, msg_mask.header.stamp.sec))
        self.ros[topic]["pub_bboxes"].publish(msg_bbox)


def main(args=None):
  rclpy.init(args=args)
  try:
    cnn = CrowVision()
    rclpy.spin(cnn)
    cnn.destroy_node()
  finally:
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
