import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from sensor_msgs.msg import CameraInfo
import numpy as np
import cv2
import tf2_ros
from geometry_msgs.msg import TransformStamped
from crow_vision_ros2.utils import make_vector3, make_quaternion, getTransformFromTF
from ament_index_python import get_package_share_directory
from time import sleep
import transforms3d as tf3
import json
import argparse
import sys
import os
from crow_vision_ros2.utils.get_camera_transformation import CameraGlobalTFGetter


class Calibrator(Node):
    GLOBAL_FRAME_NAME = "workspace_frame"

    def __init__(self, node_name="calibrator"):
        super().__init__(node_name)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # get robot 2 global transform from config
        config_file = os.path.join(
            get_package_share_directory('crow_vision_ros2'),
            'config',
            'camera_transformation_data.yaml'
        )
        cgtfg = CameraGlobalTFGetter(config_file)
        g2r_tf = np.array(cgtfg.get_camera_transformation("global_2_robot").split()).astype(np.float64).tolist()
        t_global, q_global = g2r_tf[:3], g2r_tf[3:]
        g2r_mat = self.trans_quat2tf_mat(t_global, q_global)
        r2g_mat = np.linalg.inv(g2r_mat)
        self.get_logger().info(f"robot_frame (camera_global) to global_frame transform:\n{str(r2g_mat)}")

        # DECLARE PARAMS
        imageTopicsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of available image topics')
        self.declare_parameter("image_topics", value=[], descriptor=imageTopicsParamDesc)
        infoTopicsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera_info topics for the available cameras')
        self.declare_parameter("info_topics", value=[], descriptor=infoTopicsParamDesc)
        cameraNodesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of available camera nodes (e.g. "/camera<X>/camera")')
        self.declare_parameter("camera_nodes", value=[], descriptor=cameraNodesParamDesc)
        cameraSerialsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of serial numbers of available cameras')
        self.declare_parameter("camera_serials", value=[], descriptor=cameraSerialsParamDesc)
        cameraNamespacesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of namespaces (e.g. "/camera<X>") of available cameras')
        self.declare_parameter("camera_namespaces", value=[], descriptor=cameraNamespacesParamDesc)
        cameraMatricesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera intrinsic parameters for available cameras')
        self.declare_parameter("camera_intrinsics", value=[], descriptor=cameraMatricesParamDesc)
        extrinsicsParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='Depth to color and color to global transformations for cameras.')
        self.declare_parameter("camera_extrinsics", value=[], descriptor=extrinsicsParamDesc)
        cameraFramesParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='List of camera coordinate frames for available cameras')
        self.declare_parameter("camera_frames", value=[], descriptor=cameraFramesParamDesc)
        globalFrameParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The global workspace frame name.')
        self.declare_parameter("global_frame_id", value=self.GLOBAL_FRAME_NAME, descriptor=globalFrameParamDesc)
        globalTFParamDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY, description='Transformation from robot to global frame.')
        self.declare_parameter("robot2global_tf", value=r2g_mat.ravel().tolist(), descriptor=globalTFParamDesc)

        # parse args
        parser = argparse.ArgumentParser()
        parser.add_argument("--camera_namespaces")
        parser.add_argument("--camera_serials")
        parser.add_argument("--camera_transforms")
        parser.add_argument("--color_fps")
        args = parser.parse_known_args(sys.argv[1:])[0]

        # setup vars
        self.optical_frames = dict()
        self.tf_depth_to_color = dict()
        self.tf_color_to_global = dict()
        self.intrinsics = dict()
        self.distCoeffs = dict()
        self.camMarkers = {}
        self.cameras = []
        self.cam_info_topics = []
        self.color_image_topics = {}

        # tf listener to get TF frames
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(buffer=self.tf_buffer, node=self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Get cameras
        self.camera_namespaces = args.camera_namespaces.split(" ")
        self.camera_serials = args.camera_serials.split(" ")
        self.color_fps = [float(f) for f in args.color_fps.split(" ")]
        self.camera_global_transforms = args.camera_transforms.split(" | ")

        fpsDesc = rclpy.node.ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY, description='The color camera FPS.')
        self.declare_parameter("color_fps", value=self.color_fps, descriptor=fpsDesc)

        if len(self.camera_namespaces) == 0:  # the old way: wait a little and try to detect all cameras
            self.get_logger().info(f"Sleeping to allow the cameras to come online.")
            sleep(1)
            self.cameras = [(name, namespace) for name, namespace in self.get_node_names_and_namespaces() if "camera" in name]
            topic_list = [(self.get_publisher_names_and_types_by_node(node_name, namespace), namespace) for node_name, namespace in self.cameras]
            self.cam_info_topics = [(topic_name, namespace) for sublist, namespace in topic_list for topic_name, topic_type in sublist if ("/color/" in topic_name and "sensor_msgs/msg/CameraInfo" in topic_type[0])]
            self.color_image_topics = {namespace: topic_name for sublist, namespace in topic_list for topic_name, topic_type in sublist if ("/color/" in topic_name and "sensor_msgs/msg/Image" in topic_type[0])}

            self.get_logger().info(f"Found {len(self.color_image_topics)} cameras.")

            for topic, camera_ns in self.cam_info_topics:
                # camera = next(ns + "/" + cam for cam, ns in self.cameras if ns in topic)
                self.get_logger().fatal("===")
                self.create_subscription(CameraInfo, topic, lambda msg, cam=camera_ns: self.camera_info_cb(msg, cam), 1)
                self.get_logger().info(f"Connected to '{topic}' camera info topic for camera '{camera_ns}'. Waiting for camera info to start the image subscriber.")

                # TODO: get camera color optical frame from /tf
                optical_frame = f"{camera_ns[1:]}_color_optical_frame"
                self.optical_frames[camera_ns] = optical_frame
        else:  # the novel way: periodically check if cameras are online and add cameras one by one
            sleep(20)
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
            depth_frame = f"{namespace[1:]}_depth_optical_frame"
            link_frame = f"{namespace[1:]}_link"

            # retrieve depth to color transformation
            future = self.tf_buffer.wait_for_transform_async(optical_frame, depth_frame, self.get_clock().now())
            future.add_done_callback(lambda future, camera_ns=namespace: self.found_dtc_transform_cb(future, camera_ns))
            # retrieve color to link transformation
            future = self.tf_buffer.wait_for_transform_async(link_frame, optical_frame, self.get_clock().now())
            future.add_done_callback(lambda future, camera_ns=namespace: self.found_ctl_transform_cb(future, camera_ns))
            # # if transform from camera to global frame exists, retrieve it
            # if self.tf_buffer.can_transform(self.GLOBAL_FRAME_NAME, optical_frame, self.get_clock().now(), rclpy.duration.Duration(seconds=5)):
            #     future = self.tf_buffer.wait_for_transform_async(self.GLOBAL_FRAME_NAME, optical_frame, self.get_clock().now())
            #     future.add_done_callback(lambda future, camera_ns=namespace: self.found_ctg_transform_cb(future, camera_ns))
            # else:
            #     self.tf_color_to_global[namespace] = np.eye(4)  # if TF is not defined, use identity

    def trans_quat2tf_mat(self, trans, quat) -> np.ndarray:
        return tf3.affines.compose(trans, tf3.quaternions.quat2mat(quat[-1:] + quat[:-1]), np.ones(3))

    def makeTransformMatrix(self, transform_msg):
        translation = [getattr(transform_msg.transform.translation, a) for a in "xyz"]
        quat = [getattr(transform_msg.transform.rotation, a) for a in "xyzw"]
        return self.trans_quat2tf_mat(translation, quat)

    def found_dtc_transform_cb(self, future, camera_ns):  # depth to color
        if not future.result():
            self.get_logger().fatal(f"Could not get DTC transform for {camera_ns}!")
            return
        transform_msg = self.tf_buffer.lookup_transform(self.optical_frames[camera_ns], f"{camera_ns[1:]}_depth_optical_frame", self.get_clock().now(), rclpy.duration.Duration(seconds=5))
        self.tf_depth_to_color[camera_ns] = self.makeTransformMatrix(transform_msg)

    def found_ctl_transform_cb(self, future, camera_ns):  # color to baselink
        if not future.result():
            self.get_logger().fatal(f"Could not get CTL transform for {camera_ns}!")
            return
        link_frame = f"{camera_ns[1:]}_link"
        transform_msg = self.tf_buffer.lookup_transform(self.optical_frames[camera_ns], link_frame, self.get_clock().now(), rclpy.duration.Duration(seconds=5))
        t_ctl, r_ctl = getTransformFromTF(transform_msg)
        t_ctl = t_ctl.ravel()

        transform = TransformStamped()
        d = np.array(self.camera_global_transforms[self.camera_namespaces.index(camera_ns)].split()).astype(np.float64).tolist()
        t_global, r_global = d[:3], tf3.quaternions.quat2mat(d[-1:] + d[3:-1])
        # import time
        # time.sleep(5)
        tf_global = tf3.affines.compose(t_global, r_global, np.ones(3))
        tf_global = np.linalg.inv(tf_global)
        tf_ctl = tf3.affines.compose(t_ctl, r_ctl, np.ones(3))

        tf = np.dot(tf_global, tf_ctl)

        t, r, _, _ = tf3.affines.decompose(tf)
        wrong_quat = tf3.quaternions.mat2quat(r).tolist()
        quat = wrong_quat[1:] + wrong_quat[:1]
        transform.transform.translation = make_vector3(t)
        transform.transform.rotation = make_quaternion(quat)

        # print(self.makeTransformMatrix(transform))

        transform.header.frame_id = self.GLOBAL_FRAME_NAME
        transform.child_frame_id = link_frame
        transform.header.stamp = self.get_clock().now().to_msg()

        self.tf_static_broadcaster.sendTransform(transform)

        # self.tf_color_to_global[camera_ns] = tf_global  # FIXME: this might be good?
        # self.tf_color_to_global[camera_ns] = np.linalg.inv(tf3.affines.compose(t_global, np.linalg.inv(r_global), np.ones(3)))
        self.tf_color_to_global[camera_ns] = np.linalg.inv(tf3.affines.compose(t_global, r_global, np.ones(3)))
        # self.get_logger().error(str(tf_global.tolist()))

        # self.tf_color_to_global[camera_ns] = np.linalg.inv(tf_global)
        # if transform from camera to global frame exists, retrieve it
        # if self.tf_buffer.can_transform(self.GLOBAL_FRAME_NAME, self.optical_frames[camera_ns], self.get_clock().now(), rclpy.duration.Duration(seconds=15)):
        #     print("Yasssss")
        #     future = self.tf_buffer.wait_for_transform_async(self.GLOBAL_FRAME_NAME, optical_frame, self.get_clock().now())
        #     future.add_done_callback(lambda future, camera_ns=camera_ns: self.found_ctg_transform_cb(future, camera_ns))
        # else:
        #     print("Nyeeee")
        #     self.tf_color_to_global[camera_ns] = np.eye(4)  # if TF is not defined, use identity

    def found_ctg_transform_cb(self, future, camera_ns):  # color to global
        if not future.result():
            self.get_logger().fatal(f"Could not get GTC transform for {camera_ns}!")
            return
        transform_msg = self.tf_buffer.lookup_transform(self.GLOBAL_FRAME_NAME, self.optical_frames[camera_ns], self.get_clock().now(), rclpy.duration.Duration(seconds=5))
        self.tf_color_to_global[camera_ns] = self.makeTransformMatrix(transform_msg)
        self.get_logger().warn(f"{self.tf_color_to_global[camera_ns]}!")

    def camera_info_cb(self, msg, camera_ns):
        if camera_ns not in self.tf_depth_to_color or camera_ns not in self.tf_color_to_global:  # wating to get the transforms first
            return
        self.intrinsics[self.optical_frames[camera_ns]] = msg.k.reshape((3, 3))
        self.distCoeffs[self.optical_frames[camera_ns]] = np.r_[msg.d]
        image_topic = self.color_image_topics[camera_ns]
        optical_frame = self.optical_frames[camera_ns]

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
        paramCameraSerials = self.get_parameter("camera_serials").get_parameter_value().string_array_value
        paramCameraSerials.append(self.camera_serials[self.camera_namespaces.index(camera_ns)])
        # Camera intrinsics parameter
        paramCameraIntrinsics = self.get_parameter("camera_intrinsics").get_parameter_value().string_array_value
        paramCameraIntrinsics.append(json.dumps({
            "camera_matrix": self.intrinsics[self.optical_frames[camera_ns]].astype(np.float32).tolist(),
            "distortion_coefficients": self.distCoeffs[self.optical_frames[camera_ns]].astype(np.float32).tolist(),
            "width": msg.width,
            "height": msg.height
        }))
        paramCameraExtrinsics = self.get_parameter("camera_extrinsics").get_parameter_value().string_array_value
        paramCameraExtrinsics.append(json.dumps({
            "dtc_tf": self.tf_depth_to_color[camera_ns].astype(np.float32).tolist(),
            "ctg_tf": self.tf_color_to_global[camera_ns].astype(np.float32).tolist(),
        }))
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
                "camera_serials",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramCameraSerials
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
                "camera_extrinsics",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramCameraExtrinsics
            ),
            rclpy.parameter.Parameter(
                "camera_frames",
                rclpy.Parameter.Type.STRING_ARRAY,
                paramCameraFrames
            )
        ])
        self.destroy_subscription(next(subscrip for subscrip in self.subscriptions if camera_ns in subscrip.topic))
        self.get_logger().info(f"Parameters for camera {camera_ns} are set.")


def main(args=[]):
    rclpy.init()

    calibrator = Calibrator()

    rclpy.spin(calibrator)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
