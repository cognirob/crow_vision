import rclpy
from rclpy.node import Node
from rclpy.time_source import ROSClock
from ros2node import api
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
import cv_bridge
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from crow_vision_ros2.utils import make_vector3, make_quaternion

import transforms3d as tf3


class Calibrator(Node):

    markerLength = 0.036  # mm
    squareLength = 0.042  # mm
    squareMarkerLengthRate = squareLength / markerLength
    dictionary = cv2.aruco.Dictionary_create(48, 4, 65536)
    color_K = np.r_[609.6669921875, 0.0, 327.0022277832031, 0.0, 609.7865600585938, 244.9646453857422, 0.0, 0.0, 1.0].reshape((3, 3))
    distCoeffs = np.r_[0, 0, 0, 0, 0]

    def __init__(self, node_name="calibrator"):
        super().__init__(node_name)

        self.bridge = cv_bridge.CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.cameras = [(name, namespace) for name, namespace in self.get_node_names_and_namespaces() if "camera" in name]
        topic_list = [(self.get_publisher_names_and_types_by_node(node_name, namespace), namespace) for node_name, namespace in self.cameras]
        self.cam_info_topics = [(topic_name, namespace) for sublist, namespace in topic_list for topic_name, topic_type in sublist if ("/color/" in topic_name and "sensor_msgs/msg/CameraInfo" in topic_type[0])]
        self.color_image_topics = {namespace: topic_name for sublist, namespace in topic_list for topic_name, topic_type in sublist if ("/color/" in topic_name and "sensor_msgs/msg/Image" in topic_type[0])}

        self.get_logger().info(f"Found {len(self.color_image_topics)} cameras.")

        self.optical_frames = dict()

        for topic, camera_ns in self.cam_info_topics:
            # camera = next(ns + "/" + cam for cam, ns in self.cameras if ns in topic)
            self.create_subscription(CameraInfo, topic, lambda msg, cam=camera_ns: self.camera_info_cb(msg, cam), 1)
            self.get_logger().info(f"Connected to '{topic}' camera info topic for camera '{camera_ns}'. Waiting for camera info to start the image subscriber.")

            optical_frame = f"{camera_ns[1:]}_color_optical_frame"
            self.optical_frames[camera_ns] = optical_frame

            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = f"{camera_ns[1:]}_link"
            tf_msg.child_frame_id = optical_frame
            tf_msg.transform.translation = make_vector3([-0.00030501, 0.015123, 0.0])
            tf_msg.transform.rotation = make_quaternion([-0.5, 0.5, -0.5, 0.5])
            # print(tf_msg)

            self.tf_static_broadcaster.sendTransform(tf_msg)

    def camera_info_cb(self, msg, camera_ns):
        print(msg.k)
        self.create_subscription(Image, self.color_image_topics[camera_ns], lambda msg, cam_frame=f"{camera_ns[1:]}_link", opt_frame=self.optical_frames[camera_ns]: self.image_cb(msg, cam_frame, opt_frame), 1)
        self.get_logger().info(f"Connected to '{self.color_image_topics[camera_ns]}' image topic for camera '{camera_ns}' and '{self.optical_frames[camera_ns]}' frame.")
        self.destroy_subscription(next(subscrip for subscrip in self.subscriptions if camera_ns in subscrip.topic))

    def image_cb(self, msg, camera_frame, optical_frame):
        # TODO: get optical_frame -> base link transform and set the output position to world -> base_link
        image = self.bridge.imgmsg_to_cv2(msg)
        start = self.get_clock().now()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        markerCorners, markerIds, rejectedPts = cv2.aruco.detectMarkers(image, self.dictionary, cameraMatrix=self.color_K)
        # print(np.shape(markerCorners))

        if len(markerCorners) > 3:
            try:
                # print(np.shape(markerCorners))
                diamondCorners, diamondIds = cv2.aruco.detectCharucoDiamond(image, markerCorners, markerIds, self.squareMarkerLengthRate, cameraMatrix=self.color_K)
                # print(diamondIds)
                # print(diamondCorners)

                if diamondIds is not None and len(diamondIds) > 0:
                    img_out = cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)
                    img_out = cv2.aruco.drawDetectedDiamonds(img_out, diamondCorners, diamondIds)
                    rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(np.reshape(diamondCorners, (-1, 4, 2)), self.squareLength, self.color_K, self.distCoeffs, )
                    rmat = cv2.Rodrigues(rvec)[0]
                    # transform = tf3.affines.compose(tvec.ravel(), rmat, [1, 1, 1])
                    # transform_inv = np.linalg.inv(transform)
                    # quat = tf3.quaternions.mat2quat(transform_inv[:3, :3])
                    # tvec = transform_inv[:3, 3]
                    quat = tf3.quaternions.mat2quat(rmat)

                    end = self.get_clock().now()
                    print(f"TF computed in {(end - start).nanoseconds / 1e9} seconds.")

                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = self.get_clock().now().to_msg()
                    tf_msg.header.frame_id = "world"
                    tf_msg.child_frame_id = optical_frame
                    tf_msg.transform.translation = make_vector3(tvec)
                    tf_msg.transform.rotation = make_quaternion(quat, order="wxyz")
                    # print(tf_msg)

                    self.tf_broadcaster.sendTransform(tf_msg)

                    # cv2.imshow("out", img_out)
                    # cv2.waitKey(1)
                    # cv2.waitKey(1000)
            except Exception as e:
                print(e)
                # pass


def main(args=[]):
    rclpy.init()

    calibrator = Calibrator()

    rclpy.spin(calibrator)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()