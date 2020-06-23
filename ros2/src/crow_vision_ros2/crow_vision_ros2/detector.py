import rclpy #add to package.xml deps
from rclpy.node import Node
import sensor_msgs
import std_msgs
from cv_bridge import CvBridge

import cv2
import torch

import commentjson as json
import pkg_resources

# import CNN - YOLACT
YOLACT_REPO='~/crow_vision_yolact/' #use your existing yolact setup
import sys; import os; sys.path.append(os.path.abspath(os.path.expanduser(YOLACT_REPO)))
from inference_tool import InfTool
from yolact import Yolact
from data import set_cfg

class CrowVision(Node):
  """
  ROS2 node for CNN used in Crow.

  Run as
  `ros2 run crow_vision_ros2 detector [path_to_weights [config to use [input topic name]]]`

  This node listens on network for topic "/crow/camX/image",
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
    super().__init__('CrowVision')

    #parse config
    CONFIG_DEFAULT = pkg_resources.resource_filename("crow_vision_ros2", config)
    with open(CONFIG_DEFAULT) as configFile:
      self.config = json.load(configFile)
      print(self.config)

    ## handle multiple inputs (cameras).
    # store the ROS Listeners,Publishers in a dict{}, keys by topic.
    self.ros = {}
    assert len(self.config["inputs"]) >= 1
    for inp in self.config["inputs"]:
      prefix = inp["camera"]

      # create INput listener with raw images
      topic = inp["topic"]
      camera_topic = prefix + "/" + topic
      listener = self.create_subscription(msg_type=sensor_msgs.msg.Image,
                                          topic=camera_topic,
                                          # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                          callback=lambda msg, topic=camera_topic: self.input_callback(msg, topic),
                                          qos_profile=1) #the listener QoS has to be =1, "keep last only".
      self.get_logger().info('Input listener created on topic: "%s"' % camera_topic)
      self.ros[camera_topic] = {} # camera_topic is used as an ID for this input, all I/O listeners,publishers will be based under that id.
      self.ros[camera_topic]["listener"] = listener


      # there are multiple publishers (for each input/camera topic).
      # the results are separated into different (optional) subtopics the clients can subscribe to (eg 'labels', 'masks')
      # If an output topic is empty (""), we skip publishing on that stream, it is disabled. Use to save computation resources.
      if self.config["outputs"]["image_annotated"]:
        topic = prefix + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["image_annotated"]
        publisher_img = self.create_publisher(sensor_msgs.msg.Image, topic, 1024) #publishes the processed (annotated,detected) image
        self.get_logger().info('Output publisher created for topic: "%s"' % topic)
        self.ros[camera_topic]["pub_img"] = publisher_img

      #TODO others publishers

    self.cvb_ = CvBridge()

    ## YOLACT setup
    # setup additional args
    if len(sys.argv) >= 3:
      print("Using config {} from command-line (2nd argument).".format(sys.argv[2]))
      cfg = sys.argv[2]
    else:
      cfg = self.config["config"]

    # load model weights
    if len(sys.argv) >= 2:
      print("Using weights from file {} (1st argument).".format(sys.argv[1]))
      model = sys.argv[1]
    else:
      model = self.config["weights"]

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
    self.get_logger().info("I heard: {} for topic {}".format(str(msg.height), topic))
    assert topic in self.ros, "We don't have registered listener for the topic {} !".format(topic)

    img_raw = self.cvb_.imgmsg_to_cv2(msg)

    preds, frame = self.cnn.process_batch(img_raw)

    #the input callback triggers the publishers here.
    if self.ros[topic]["pub_img"]: # labeled image publisher. (Use "" to disable)
      img_labeled = self.cnn.label_image(img_raw, preds, frame)

      msg_img = self.cvb_.cv2_to_imgmsg(img_labeled, encoding="rgb8")
      # parse time from incoming msg, pass to outgoing msg
      msg_img.header.stamp.nanosec = msg.header.stamp.nanosec
      msg_img.header.stamp.sec  = msg.header.stamp.sec
      self.get_logger().info("Publishing as Image {} x {}".format(msg_img.width, msg_img.height))
      self.ros[topic]["pub_img"].publish(msg_img)

    if False and self.ros[topic]["pub_masks"]:  # TODO: fix
      classes, scores, bboxes, masks = self.cnn.raw_inference(img_raw, preds)

      msg_mask = std_msgs.msg.String()
      msg_mask.data = str(masks)
      # parse time from incoming msg, pass to outgoing msg
      msg_mask.header.stamp.nsec = msg.header.stamp.nsec
      msg_mask.header.stamp.sec  = msg.header.stamp.sec
      self.get_logger().info("Publishing as String {} at time {} ".format(msg_mask.data, msg_mask.header.stamp.sec))
      self.ros[topic]["pub_masks"].publish(msg_mask)


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
