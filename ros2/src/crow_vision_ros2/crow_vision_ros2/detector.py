import rclpy #add to package.xml deps
from rclpy.node import Node
import sensor_msgs
from cv_bridge import CvBridge
import cv2

class CrowVision(Node):
  """
  ROS2 node for CNN used in Crow. 
  """

  def __init__(self, topic_in='/crow/cam1/image', topic_out_img='/crow/cam1/detections/image'):
    super().__init__('CrowVision')
    self.listener_ = self.create_subscription(sensor_msgs.msg.Image, topic_in, self.input_callback, 10)
    print('Hi from crow_vision_ros2.')

  def input_callback(self, msg):
    self.get_logger().info('I heard: "%s"' % str(msg.height))

def main():
  cnn = CrowVision()


if __name__ == '__main__':
    main()
