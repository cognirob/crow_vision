import rclpy #add to package.xml deps
from rclpy.node import Node
import sensor_msgs
import std_msgs
from cv_bridge import CvBridge
import cv2

try:
  from yolact_inference import YolactInfTool
except:
  assert False, "Failed to import YOLACT. ROS2 does not work with conda, make sure to have Yolact importable from python3"

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
               topic_in="/raw",
               topic_out_img="/image",
               topic_out_masks="/masks",
               top_k = 15,
               threshold=0.15,
               model='./data/yolact/weights/weights_yolact_kuka_13/crow_base_59_400000.pth',
               ):
    super().__init__('CrowVision')
    self.cam = camera

    # there is 1 listener with raw images:
    self.listener_ = self.create_subscription(sensor_msgs.msg.Image, camera+topic_in, self.input_callback, 10)
    
    # there are multiple publishers. We publish all the info for a single detection step (a single image)
    # but optionally the results are separated into different subtopics the clients can subscribe (eg 'labels', 'masks')
    # If a topic_out_* is None, we skip publishing on that stream, it is disabled.
    if topic_out_img is not None:
      self.publisher_img = self.create_publisher(sensor_msgs.msg.Image, camera+topic_out_img, 10) #publishes the processed (annotated,detected) image
    else:
      self.publisher_img = None
    if topic_out_masks is not None:
      self.publisher_masks = self.create_publisher(std_msgs.msg.String, camera+topic_out_masks, 10) #TODO change to correct dtype, not string
    else:
      self.publisher_masks = None
    #TODO others publishers

    self.cvb_ = cv_bridge.CvBridge()
    self.yolact_ = YolactInfTool(trained_model=model, top_k=top_k, score_threshold=threshold)
    print('Hi from crow_vision_ros2.')


  def input_callback(self, msg):
    self.get_logger().info('I heard: "%s"' % str(msg.height))
    img_raw = self.cvb_.imgmsg_to_cv2(msg)
    masks = "TODO" #TODO process from cnn
    #the input callback triggers the publishers here.
    if publisher_img is not None:
      img_labeled = self.yolact_.visualized_inference(img_raw)
      self.publisher_img.publish(cvb_.cv_to_imgmsg(img_labeled))
    if publisher_masks is not None:
      self.publisher_masks.publish(masks)


def main(args=None):
  rclpy.init(args=args)
  try:
    cnn = CrowVision()
    rclpy.spin(cnn)
  finally:
    cnn.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
