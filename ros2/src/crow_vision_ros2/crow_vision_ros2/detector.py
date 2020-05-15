import rclpy #add to package.xml deps
from rclpy.node import Node
import sensor_msgs
from cv_bridge import CvBridge
import cv2

class CrowVision(Node):
  """
  ROS2 node for CNN used in Crow. 
  """

  def __init__(self):
    print('Hi from crow_vision_ros2.')


def main():
  cnn = CrowVision()


if __name__ == '__main__':
    main()
