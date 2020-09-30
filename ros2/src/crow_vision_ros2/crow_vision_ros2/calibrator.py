from sys import argv
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
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
from crow_vision_ros2.filters import CameraPoser
from time import sleep
import transforms3d as tf3
import json
import argparse
import sys


class Calibrator(Node):

    markerLength = 0.050  # mm
    squareLength = 0.060  # mm
    squareMarkerLengthRate = squareLength / markerLength
    try:
      dictionary = cv2.aruco.Dictionary_create(48, 4, 65536)
    except:
      dictionary = cv2.aruco.Dictionary_create(48, 4) #TODO hack for (old?) cv2 version, fallback to API with 2 args only

    # distCoeffs = np.r_[0, 0, 0, 0, 0]

    def __init__(self, node_name="calibrator"):
        super().__init__(node_name)

        self.bridge = cv_bridge.CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # DECLARE PARAMS
        haltCalibrationDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_BOOL, description='If True, node will not run camera pose estimation')
        self.declare_parameter("halt_calibration", value=False, descriptor=haltCalibrationDesc)
        imageTopicsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of available image topics')
        self.declare_parameter("image_topics", value=[], descriptor=imageTopicsParamDesc)
        infoTopicsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera_info topics for the available cameras')
        self.declare_parameter("info_topics", value=[], descriptor=infoTopicsParamDesc)
        cameraNodesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of available camera nodes (e.g. "/camera<X>/camera")')
        self.declare_parameter("camera_nodes", value=[], descriptor=cameraNodesParamDesc)
        cameraNamespacesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of namespaces (e.g. "/camera<X>") of available cameras')
        self.declare_parameter("camera_namespaces", value=[], descriptor=cameraNamespacesParamDesc)
        cameraMatricesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera intrinsic parameters for available cameras')
        self.declare_parameter("camera_intrinsics", value=[], descriptor=cameraMatricesParamDesc)
        cameraFramesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera coordinate frames for available cameras')
        self.declare_parameter("camera_frames", value=[], descriptor=cameraFramesParamDesc)

        # parse args
        parser = argparse.ArgumentParser()
        parser.add_argument("--camera_namespaces", nargs="*")
        args = parser.parse_known_args(sys.argv[1:])[0]

        # setup vars
        self.optical_frames = dict()
        self.intrinsics = dict()
        self.distCoeffs = dict()
        self.camMarkers = {}
        self.cameras = []
        self.cam_info_topics = []
        self.color_image_topics = {}

        # Get cameras
        self.camera_namespaces = args.camera_namespaces
        if len(self.camera_namespaces) == 0:  # the all way: wait a little and try to detect all cameras
            self.get_logger().info(f"Sleeping to allow the cameras to come online.")
            sleep(15)
            self.cameras = [(name, namespace) for name, namespace in self.get_node_names_and_namespaces() if "camera" in name]
            topic_list = [(self.get_publisher_names_and_types_by_node(node_name, namespace), namespace) for node_name, namespace in self.cameras]
            self.cam_info_topics = [(topic_name, namespace) for sublist, namespace in topic_list for topic_name, topic_type in sublist if ("/color/" in topic_name and "sensor_msgs/msg/CameraInfo" in topic_type[0])]
            self.color_image_topics = {namespace: topic_name for sublist, namespace in topic_list for topic_name, topic_type in sublist if ("/color/" in topic_name and "sensor_msgs/msg/Image" in topic_type[0])}

            self.get_logger().info(f"Found {len(self.color_image_topics)} cameras.")

            for topic, camera_ns in self.cam_info_topics:
                # camera = next(ns + "/" + cam for cam, ns in self.cameras if ns in topic)
                self.create_subscription(CameraInfo, topic, lambda msg, cam=camera_ns: self.camera_info_cb(msg, cam), 1)
                self.get_logger().info(f"Connected to '{topic}' camera info topic for camera '{camera_ns}'. Waiting for camera info to start the image subscriber.")

                # TODO: get camera color optical frame from /tf
                optical_frame = f"{camera_ns[1:]}_color_optical_frame"
                self.optical_frames[camera_ns] = optical_frame
        else:  # the novel way: periodically check if cameras are online and add cameras one by one
            sleep(2)
            self.timer = self.create_timer(2, self.timer_cb)

    def timer_cb(self):
        online_cameras = [(name, namespace) for name, namespace in self.get_node_names_and_namespaces() if "camera" in name]

        missing_cameras = [c for c in online_cameras if c not in self.cameras]

        for node_name, namespace in missing_cameras:
            try:
                topic_list = self.get_publisher_names_and_types_by_node(node_name, namespace)
                camera_info_topic = next(topic_name for topic_name, topic_type in topic_list if "/color/" in topic_name and "sensor_msgs/msg/CameraInfo" in topic_type[0])
                color_image_topic = next(topic_name for topic_name, topic_type in topic_list if "/color/" in topic_name and "sensor_msgs/msg/Image" in topic_type[0])
            except Exception as e:
                self.get_logger().debug(f"Failed to add camera {namespace}/{node_name}. The camera topics are probably not up, yet.")
                continue

            self.cameras.append((node_name, namespace))
            self.cam_info_topics.append((camera_info_topic, namespace))
            self.color_image_topics[namespace] = color_image_topic

            self.get_logger().info(f"Found camera {namespace}/{node_name}.")

            self.create_subscription(CameraInfo, camera_info_topic, lambda msg, cam=namespace: self.camera_info_cb(msg, cam), 1)
            self.get_logger().info(f"Connected to '{camera_info_topic}' camera info topic for camera '{namespace}'. Waiting for camera info to start the image subscriber.")

            # TODO: get camera color optical frame from /tf
            optical_frame = f"{namespace[1:]}_color_optical_frame"
            self.optical_frames[namespace] = optical_frame


    def camera_info_cb(self, msg, camera_ns):
        self.intrinsics[self.optical_frames[camera_ns]] = msg.k.reshape((3, 3))
        self.distCoeffs[self.optical_frames[camera_ns]] = np.r_[msg.d]
        image_topic = self.color_image_topics[camera_ns]
        optical_frame = self.optical_frames[camera_ns]

        # create subscription
        self.create_subscription(Image, image_topic, lambda msg, cam_frame=f"{camera_ns[1:]}_link", opt_frame=optical_frame: self.image_cb(msg, cam_frame, opt_frame), 1)
        self.camMarkers[optical_frame] = CameraPoser(optical_frame)

        self.get_logger().info(f"Connected to '{image_topic}' image topic for camera '{camera_ns}' and '{optical_frame}' frame.")

        # Image topics parameter
        paramImageTopics = self.get_parameter("image_topics").get_parameter_value().string_array_value
        paramImageTopics.append(image_topic)
        # Camera Info topics parameter
        paramInfoTopics = self.get_parameter("info_topics").get_parameter_value().string_array_value
        info_topic, ns = next(filter(lambda item: camera_ns in item, self.cam_info_topics))
        paramInfoTopics.append(info_topic)
        # Camera node address paramter
        paramCameraNodes = self.get_parameter("camera_nodes").get_parameter_value().string_array_value
        cname, ns = next(filter(lambda item: camera_ns in item, self.cameras))
        paramCameraNodes.append(ns + "/" + cname)
        paramCameraNamespaces = self.get_parameter("camera_namespaces").get_parameter_value().string_array_value
        paramCameraNamespaces.append(ns)
        # Camera intrinsics parameter
        paramCameraIntrinsics = self.get_parameter("camera_intrinsics").get_parameter_value().string_array_value
        paramCameraIntrinsics.append(json.dumps({"camera_matrix": self.intrinsics[self.optical_frames[camera_ns]].astype(np.float32).tolist(), "distortion_coefficients": self.distCoeffs[self.optical_frames[camera_ns]].astype(np.float32).tolist()}))
        # Camera coordinate frame name parameter
        paramCameraFrames = self.get_parameter("camera_frames").get_parameter_value().string_array_value
        paramCameraFrames.append(optical_frame)

        # Update the parameters
        self.set_parameters([
            rclpy.parameter.Parameter(
                "image_topics",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramImageTopics
            ),
            rclpy.parameter.Parameter(
                "info_topics",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramInfoTopics
            ),
            rclpy.parameter.Parameter(
                "camera_nodes",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramCameraNodes
            ),
            rclpy.parameter.Parameter(
                "camera_namespaces",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramCameraNamespaces
            ),
            rclpy.parameter.Parameter(
                "camera_intrinsics",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramCameraIntrinsics
            ),
            rclpy.parameter.Parameter(
                "camera_frames",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramCameraFrames
            )
        ])
        self.destroy_subscription(next(subscrip for subscrip in self.subscriptions if camera_ns in subscrip.topic))

    def image_cb(self, msg, camera_frame, optical_frame):
        if self.get_parameter("halt_calibration").get_parameter_value().bool_value:
            return

        # TODO: get optical_frame -> base link transform and set the output position to world -> base_link
        image = self.bridge.imgmsg_to_cv2(msg)
        start = self.get_clock().now()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_K = self.intrinsics[optical_frame]
        distCoeffs = self.distCoeffs[optical_frame]
        markerCorners, markerIds, rejectedPts = cv2.aruco.detectMarkers(image, self.dictionary, cameraMatrix=color_K)

        cameraMarkers = self.camMarkers[optical_frame]
        # print(np.shape(markerCorners))

        if len(markerCorners) > 0:
            for m_idx, (markerId, mrkCorners) in enumerate(zip(markerIds, markerCorners)):
                cameraMarkers.updateMarker(markerId, mrkCorners)

            markerCorners, markerIds = cameraMarkers.getMarkers()

        if len(markerCorners) > 3 and cameraMarkers.markersReady:
            try:
                # print(np.shape(markerCorners))
                diamondCorners, diamondIds = cv2.aruco.detectCharucoDiamond(image, markerCorners, markerIds, self.squareMarkerLengthRate, cameraMatrix=color_K)
                markerImage = cv2.aruco.drawDetectedDiamonds(image, diamondCorners, diamondIds)
                # cv2.imshow("computed markers", markerImage)
                # cv2.waitKey(1)
                # print(diamondIds)
                # print(diamondCorners)

                if diamondIds is not None and len(diamondIds) > 0:
                    img_out = cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)
                    img_out = cv2.aruco.drawDetectedDiamonds(img_out, diamondCorners, diamondIds)
                    rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(np.reshape(diamondCorners, (-1, 4, 2)), self.squareLength, color_K, distCoeffs)
                    rmat = cv2.Rodrigues(rvec)[0]
                    # transform = tf3.affines.compose(tvec.ravel(), rmat, [1, 1, 1])
                    # transform_inv = np.linalg.inv(transform)
                    # quat = tf3.quaternions.mat2quat(transform_inv[:3, :3])
                    # tvec = transform_inv[:3, 3]
                    quat = tf3.quaternions.mat2quat(rmat)

                    end = self.get_clock().now()
                    self.get_logger().info(f"TF computed in {(end - start).nanoseconds / 1e9} seconds.")

                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = self.get_clock().now().to_msg()
                    tf_msg.header.frame_id = "world"
                    tf_msg.child_frame_id = optical_frame
                    tf_msg.transform.translation = make_vector3(tvec)
                    tf_msg.transform.rotation = make_quaternion(quat, order="wxyz")
                    # print(tf_msg)

                    self.tf_static_broadcaster.sendTransform(tf_msg)
                    self.get_logger().info("TF published")

                    self.set_parameters([
                        rclpy.parameter.Parameter(
                            "halt_calibration",
                            rclpy.Parameter.Type.BOOL,
                            True
                        )
                    ])

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
