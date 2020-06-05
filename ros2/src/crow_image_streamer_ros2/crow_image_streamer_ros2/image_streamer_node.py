import rclpy #add to package.xml deps
from rclpy.node import Node
import sensor_msgs
from cv_bridge import CvBridge
import cv2
import os
import sys
import glob

class ImageFolderPublisher(Node):
    """
  ROS2 publisher node. 
  Streams content of a folder with images over a ROS topic. 
  Serves as cv2 <-> msg.Image convertor. 

  @param: expected command-line argument (str) path to folder with images (recursive). 
  @return streamed ROS msg with image data at topic with give framerate. 

  @except Behavior when changing folder's content while streaming is undefined. Intended
    use is being able to add images while being streamed. But the images must appear after (alphabetically)
    the current msg. 
    """

    def __init__(self, path, framerate=5, topic="/crow/cam1/raw", imgExt='.png', loop:bool=True):
        super().__init__('crow_image_streamer_ros2')
        self.publisher_ = self.create_publisher(sensor_msgs.msg.Image, topic, 1024)
        assert os.path.isdir(path),"path must be a string pointing to an existing directory with images"
        self.path_ = str(path)
        self.loop_ = loop
        self.i_ = int(0) #ith image to handle
        assert framerate > 0, "Framerate [Hz] must be > 0"
        self.timer_ = self.create_timer(1/framerate, self.timer_callback)
        self.ext_ = str(imgExt)
        self.cvb_ = CvBridge()


    def timer_callback(self):
        files = glob.glob(str(self.path_) + '/**/*' + str(self.ext_), recursive=True)
        if self.i_ < len(files):
          imgfile = files[self.i_]
          self.i_ += 1

          # load image, convert to msg
          im = cv2.imread(imgfile)
          msg = self.cvb_.cv2_to_imgmsg(im)

          self.publisher_.publish(msg)
          self.get_logger().info('Publishing: "%s"' % imgfile)
        else:
          if self.loop_:
            self.i_ = 0
            self.get_logger().info('Looping.')
            return
          self.get_logger().info('Finished.')
          self.destroy_node()
          rclpy.shutdown()



def main(args=sys.argv):
    rclpy.init(args=args)
    assert len(args) > 1, "Must provide 1 argument - path to image folder!"+str(args)
    node = ImageFolderPublisher(path=args[1])

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
