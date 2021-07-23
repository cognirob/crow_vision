import rclpy
from rcl_interfaces.msg import ParameterType
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import sys
from glob import glob


class ImageFolderPublisher(Node):
    """
    ROS2 publisher node.
    Streams content of a folder with images over a ROS topic.
    Serves as cv2 <-> msg.CompressedImage convertor.

    @param: expected command-line argument (str) path to folder with images (recursive).
    @return streamed ROS msg with image data at topic with give framerate.

    @except Behavior when changing folder's content while streaming is undefined. Intended
    use is being able to add images while being streamed. But the images must appear after (alphabetically)
    the current msg.
    """

    FPS = 15
    BASE_CAMERA_FOLDER = os.path.expanduser("~/Videos/ros_cameras")
    IMG_EXT = "png"
    LOOP = True
    SORT = True
    COLOR_IMAGE_TOPIC = "color/image_raw"

    def __init__(self):
        super().__init__('calibrator')

        # Find camera image folders
        camera_dirs = os.listdir(self.BASE_CAMERA_FOLDER)

        # Setup cameras
        self.n_cams = 0
        self.pubs = {}  # publishers per camera
        self.images = {}  # image lists per camera
        self.n_images = {}  # number of images in image list per camera
        self.idxs = {}  # index to image list per camera
        self.cameras = []
        for cd in camera_dirs:
            full_path = os.path.join(self.BASE_CAMERA_FOLDER, cd)
            if not os.path.isdir(full_path):
                continue
            files = glob(os.path.join(full_path, f"*.{self.IMG_EXT}"))
            n_images = len(files)
            if n_images == 0:
                continue
            if self.SORT:
                files = sorted(files)
            topic = f"/{cd}/{self.COLOR_IMAGE_TOPIC}"
            self.get_logger().info(f"Found {n_images} images in {full_path} folder. Creating an image topic at {topic}.")
            self.n_cams += 1
            self.images[cd] = files
            self.n_images[cd] = n_images
            self.cameras.append(cd)
            self.idxs[cd] = 0
            self.pubs[cd] = self.create_publisher(Image, topic, 1)

        if self.n_cams == 0:
            self.get_logger().fatal("No camera folders or images found!!!")
            sys.exit(-1)

        self.get_logger().info(f"Found {self.n_cams} cameras.")
        self.timer_ = self.create_timer(1 / self.FPS, self.timer_callback)
        self.cvb_ = CvBridge()

        # calibrator witchcraft

        # DECLARE PARAMS
        imageTopicsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of available image topics')
        self.declare_parameter("image_topics", value=[f"/{c}/{self.COLOR_IMAGE_TOPIC}" for c in self.cameras], descriptor=imageTopicsParamDesc)
        infoTopicsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera_info topics for the available cameras')
        self.declare_parameter("info_topics", value=[f"/{c}/color/camera_info" for c in self.cameras], descriptor=infoTopicsParamDesc)
        cameraNodesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of available camera nodes (e.g. "/camera<X>/camera")')
        self.declare_parameter("camera_nodes", value=[f"/{c}/camera" for c in self.cameras], descriptor=cameraNodesParamDesc)
        cameraNamespacesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of namespaces (e.g. "/camera<X>") of available cameras')
        self.declare_parameter("camera_namespaces", value=[f"/{c}" for c in self.cameras], descriptor=cameraNamespacesParamDesc)
        cameraFramesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera coordinate frames for available cameras')
        self.declare_parameter("camera_frames", value=[f"{c}_color_optical_frame" for c in self.cameras], descriptor=cameraFramesParamDesc)
        # TODO: the rest of the parameters
        # cameraMatricesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera intrinsic parameters for available cameras')
        # self.declare_parameter("camera_intrinsics", value=[], descriptor=cameraMatricesParamDesc)
        # extrinsicsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='Depth to color and color to global transformations for cameras.')
        # self.declare_parameter("camera_extrinsics", value=[], descriptor=extrinsicsParamDesc)
        # globalFrameParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The global workspace frame name.')
        # self.declare_parameter("global_frame_id", value=self.GLOBAL_FRAME_NAME, descriptor=globalFrameParamDesc)

        # TODO: tf listener to get TF frames
        # self.tf_buffer = tf2_ros.Buffer()
        # tf2_ros.TransformListener(buffer=self.tf_buffer, node=self)
        # self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)


    def timer_callback(self):
        any_published = False
        try:
            for cam, pub in self.pubs.items():
                if self.idxs[cam] == self.n_images[cam]:
                    if self.LOOP:
                        self.idxs[cam] = 0
                    else:
                        continue

                img = cv2.imread(self.images[cam][self.idxs[cam]])
                if img is None:
                    self.get_logger().warn(f"Could not load image {self.images[cam]} for camera {cam}, skipping.")
                else:
                    msg = self.cvb_.cv2_to_imgmsg(img, encoding="bgr8")
                    msg.header.stamp = self.get_clock().now().to_msg()
                    pub.publish(msg)
                    any_published = True

                self.idxs[cam] += 1
        except BaseException as e:
            self.get_logger().error(f"An exception occurred:\n{e}")

        if not any_published:
            self.get_logger().info('Finished.')
            self.destroy_node()
            rclpy.shutdown()


def main(args=sys.argv):
    rclpy.init(args=args)
    node = ImageFolderPublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
