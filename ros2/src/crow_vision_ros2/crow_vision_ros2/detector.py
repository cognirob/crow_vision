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
from yolact import Yolact
from data import set_cfg
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
from eval import prep_display
from data.config import Config

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
    # setup yolact args
    global args
    args=Config({})
    args.top_k = self.config["top_k"]
    args.score_threshold = self.config["threshold"]
    # set here everything that would have been set by parsing arguments in yolact/eval.py:
    args.display_lincomb = False
    args.crop = False
    args.display_fps = False
    args.display_text = True
    args.display_bboxes = True
    args.display_masks =True
    args.display_scores = True

    # CUDA setup for yolact
    torch.backends.cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if len(sys.argv) >= 3:
      print("Using config {} from command-line (2nd argument).".format(sys.argv[2]))
      cfg = sys.argv[2]
    else:
      cfg = self.config["config"]
    set_cfg(cfg)

    self.net_ = Yolact().cuda()

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

    self.net_.load_weights(model_abs)
    self.net_.eval()
    self.net_.detect.use_fast_nms = True
    self.net_.detect.use_cross_class_nms = False

    print('Hi from crow_vision_ros2.')


  def net_process_predictions_(self, img):
    """
    call Yolact on img, get preds.
    Used in label_image, raw_inference.
    """
    if isinstance(self.net_, Yolact):
      frame = torch.from_numpy(img).cuda().float()
      batch = FastBaseTransform()(frame.unsqueeze(0))
      preds = self.net_(batch)
      return preds
    else:
      assert "Currently only Yolact is supported."


  def label_image(self, preds):
    """
    Visualize detections and display as an image. Apply CNN inference.
    """
    if isinstance(self.net_, Yolact):
      global args
      processed = prep_display(preds, frame, h=None, w=None, undo_transform=False, args=args)
      return processed
    else:
      assert "Currently only Yolact is supported."

  def raw_inference(self, preds):
    """
    Inference, detections by YOLACT but without visualizations. 
    Should be fast and all that is needed.

    @return list of lists: [classes, scores, boxes, masks] 
    """
    if isinstance(self.net_, Yolact):
      global args
      [classes, scores, boxes, masks] = postprocess(preds, w=None, h=None, batch_idx=0, interpolation_mode='bilinear', 
                                                    visualize_lincomb=False, crop_masks=True, score_threshold=args.score_threshold)
      return [classes, scores, boxes, masks]
    else:
      assert "Currently only Yolact is supported."



  def input_callback(self, msg, topic):
    """
    @param msg - ROS msg (Image data) to be processed. From camera
    @param topic - str, from camera/input on given topic.
    @return nothing, but send new message(s) via output Publishers. 
    """
    self.get_logger().info("I heard: {} for topic {}".format(str(msg.height), topic))
    assert topic in self.ros, "We don't have registered listener for the topic {} !".format(topic)

    img_raw = self.cvb_.imgmsg_to_cv2(msg)

    preds = self.net_process_predictions_(img_raw)

    #the input callback triggers the publishers here.
    if self.ros[topic]["pub_img"]: # labeled image publisher. (Use "" to disable)
      img_labeled = self.label_image(preds)

      msg_img = self.cvb_.cv2_to_imgmsg(img_labeled, encoding="rgb8")
      # parse time from incoming msg, pass to outgoing msg
      msg_img.header.stamp.nsec = msg.header.stamp.nsec
      msg_img.header.stamp.sec  = msg.header.stamp.sec
      self.get_logger().info("Publishing as Image {} x {}".format(msg_img.width, msg_img.height))
      self.ros[topic]["pub_img"].publish(msg_img)

    if self.ros[topic]["pub_masks"]:
      classes, scores, bboxes, masks = self.raw_inference(preds)

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
