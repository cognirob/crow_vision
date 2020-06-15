import rclpy #add to package.xml deps
from rclpy.node import Node
import sensor_msgs
import std_msgs
from cv_bridge import CvBridge

import cv2
import torch

import commentjson
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
      self.config = commentjson.load(configFile)

    # there is 1 listener with raw images: #TODO create N listeners
    assert len(self.config["inputs"]) >= 1
    in_topic = config["inputs"][0]["camera"] + "/" + config["inputs"][0]["topic"]
    self.listener_ = self.create_subscription(sensor_msgs.msg.Image, in_topic, self.input_callback, 1) #the listener QoS has to be =1, "keep last only".
    self.get_logger().info('Input listener created on topic: "%s"' % in_topic)

    # there are multiple publishers. We publish all the info for a single detection step (a single image)
    # but optionally the results are separated into different subtopics the clients can subscribe (eg 'labels', 'masks')
    # If a topic_out_* is None, we skip publishing on that stream, it is disabled.
    if config["output"]["image_annotated"] is not None:
      topic = config["inputs"][0]["camera"] + "/" + config["output"]["image_annotated"] 
      self.publisher_img = self.create_publisher(sensor_msgs.msg.Image, topic, 1024) #publishes the processed (annotated,detected) image
      self.get_logger().info('Output publisher created for topic: "%s"' % topic)
    else:
      self.publisher_img = None

    self.publisher_masks = None
    #TODO others publishers

    self.cvb_ = CvBridge()

    ## YOLACT setup
    # setup yolact args
    global args
    args=Config({})
    args.top_k = top_k
    args.score_threshold = threshold
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
      cfg = config["config"]
    set_cfg(config)

    self.net_ = Yolact().cuda()

    # load model weights
    if len(sys.argv) >= 2:
      print("Using weights from file {} (1st argument).".format(sys.argv[1]))
      model = sys.argv[1]
    else:
      model = config["weights"]

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


  def label_image(self, img):
    """
    Visualize detections and display as an image. Apply CNN inference.
    """
    if isinstance(self.net_, Yolact):
      frame = torch.from_numpy(img).cuda().float()
      batch = FastBaseTransform()(frame.unsqueeze(0))
      preds = self.net_(batch)
      global args
      processed = prep_display(preds, frame, h=None, w=None, undo_transform=False, args=args)
      return processed
    else:
      assert "Currently only Yolact is supported."


  def input_callback(self, msg):
    self.get_logger().info('I heard: "%s"' % str(msg.height))
    img_raw = self.cvb_.imgmsg_to_cv2(msg)

    masks = "TODO" #TODO process from cnn
    #the input callback triggers the publishers here.
    if self.publisher_img is not None:
      img_labeled = self.label_image(img_raw)
      msg = self.cvb_.cv2_to_imgmsg(img_labeled, encoding="rgb8")
      self.get_logger().info("Publishing as Image {} x {}".format(msg.width, msg.height))
      self.publisher_img.publish(msg)

    if self.publisher_masks is not None:
      message = std_msgs.msg.String()
      message.data = str(masks)
      self.publisher_masks.publish(message)


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
