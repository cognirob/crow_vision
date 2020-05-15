import rclpy #add to package.xml deps
from rclpy.node import Node
import sensor_msgs
import std_msgs
from cv_bridge import CvBridge
import cv2
#TODO import yolact 

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
               topic_in='/crow/cam1/image', 
               topic_out_img='/crow/cam1/detections/image',
               topic_out_mask='/crow/cam1/detections/masks'):
    super().__init__('CrowVision')
    
    # there is 1 listener with raw images:
    self.listener_ = self.create_subscription(sensor_msgs.msg.Image, topic_in, self.input_callback, 10)
    
    # there are multiple publishers. We publish all the info for a single detection step (a single image)
    # but optionally the results are separated into different subtopics the clients can subscribe (eg 'labels', 'masks')
    # If a topic_out_* is None, we skip publishing on that stream, it is disabled.
    if topic_out_img is not None:
      self.publisher_img = self.create_publisher(sensor_msgs.msg.Image, topic_out_img, 10) #publishes the processed (annotated,detected) image
    else:
      self.publisher_img = None
    if topic_out_masks is not None:
      self.publisher_masks = self.create_publisher(std_msgs.msg.String, topic_out_masks, 10) #TODO change to correct dtype, not string
    else:
      self.publisher_masks = None
    #TODO others publishers

    self.cvb_ = cv_bridge.CvBridge()
    print('Hi from crow_vision_ros2.')


  def input_callback(self, msg):
    self.get_logger().info('I heard: "%s"' % str(msg.height))
    img_raw = self.cvb_.imgmsg_to_cv2(msg)
    img_labeled = img_raw #TODO process by CNN
    masks = "TODO" #TODO process from cnn
    #the input callback triggers the publishers here.
    if publisher_img is not None:
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
