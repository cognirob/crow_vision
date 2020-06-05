import rclpy #add to package.xml deps
from rclpy.node import Node
import sensor_msgs
import std_msgs
from cv_bridge import CvBridge

import cv2
import torch

# import CNN - YOLACT
YOLACT_REPO='~/crow_vision_yolact/' #use your existing yolact setup
import sys; import os; sys.path.append(os.path.abspath(os.path.expanduser(YOLACT_REPO)))
from yolact import Yolact
from data import set_cfg
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
from eval import prep_display
from data.config import Config

import matplotlib.pyplot as plt

class CrowVision(Node):
  """
  ROS2 node for CNN used in Crow.

  This node listens on network for topic "/crow/camX/image", 
  obtains raw RGB image, processes it in neural net, and publishes results in form of (for given camera):
  - "/detections/image" - processed image with visualized masks, bboxes and labels
  - "/detections/masks" 
  - "/detections/labels"
  - "/detections/bboxes"
  - "/detections/confidences", etc. TODO
  """

  def __init__(self, 
               camera='/crow/cam1',
               topic_in="/raw", #TODO match with default RGB topic of RS camera node
               topic_out_img="/detections/image",
               topic_out_masks="/detections/masks",
               top_k = 15,
               threshold=0.51,
               model='./data/yolact/weights/yolact_base_54_800000.pth', #relative to YOLACT_REPO
               ):
    super().__init__('CrowVision')
    self.cam = camera

    # there is 1 listener with raw images:
    self.listener_ = self.create_subscription(sensor_msgs.msg.Image, camera+topic_in, self.input_callback, 1024)
    
    # there are multiple publishers. We publish all the info for a single detection step (a single image)
    # but optionally the results are separated into different subtopics the clients can subscribe (eg 'labels', 'masks')
    # If a topic_out_* is None, we skip publishing on that stream, it is disabled.
    if topic_out_img is not None:
      self.publisher_img = self.create_publisher(sensor_msgs.msg.Image, camera+topic_out_img, 1024) #publishes the processed (annotated,detected) image
    else:
      self.publisher_img = None
    if topic_out_masks is not None:
      self.publisher_masks = self.create_publisher(std_msgs.msg.String, camera+topic_out_masks, 1024) #TODO change to correct dtype, not string
    else:
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
    set_cfg('yolact_base_config') #TODO add and use crow_base_config

    self.net_ = Yolact().cuda()

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
      msg = self.cvb_.cv2_to_imgmsg(img_labeled)
      self.get_logger().info("Publishing as Image {} x {}".format(msg.width, msg.height))
      self.publisher_img.publish(msg)
      cv2.imshow('ros', img_labeled)
      cv2.waitKey(500)
      cv2.destroyAllWindows()
      #plt.imshow(img_labeled)
      #plt.title('ROS')
      #plt.show()

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
    rclpy.shutdown()


if __name__ == '__main__':
    main()
