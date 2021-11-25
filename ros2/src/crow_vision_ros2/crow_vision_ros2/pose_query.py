import time
import numpy as np
import json
import rclpy
from rcl_interfaces.srv import GetParameters
from ros2param.api import call_get_parameters
import traceback as tb
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from crow_ontology.crowracle_client import CrowtologyClient
from crow_control.utils.service_client import ServiceClient
from sensor_msgs.msg import Image
from time import sleep
import transforms3d as tf3
import cv2
from datetime import datetime as dt


cosy2yolact = {
    "obj_000023": "peg_screw"
}

def maybeMakeFullMatrix(trans, quat) -> np.ndarray:
    translation = [trans[a] for a in "xyz"]
    rotation = [quat[a] for a in "xyzw"]
    return tf3.affines.compose(translation, tf3.quaternions.quat2mat(rotation[-1:] + rotation[:-1]), np.ones(3))

def composeFullMatrix(translation, rotation) -> np.ndarray:
    return tf3.affines.compose(translation.ravel(), tf3.quaternions.quat2mat(rotation[-1:] + rotation[:-1]), np.ones(3))

def rotation_matrix2quat(rotation_matrx):
    wrong_quat = tf3.quaternions.mat2quat(rotation_matrx).tolist()
    quat = wrong_quat[1:] + wrong_quat[:1]
    return quat


class PoseQuery(Node):
    SERVER_IP = '10.35.16.77'
    SERVER_PORT = 242424

    def __init__(self, node_name="pose_query"):
        super().__init__(node_name)
        # Get existing cameras from and topics from the calibrator
        self.crowracle = CrowtologyClient(node=self)
        self.pose_service = ServiceClient(port=self.SERVER_PORT, addr=self.SERVER_IP)

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.cvb = CvBridge()

        calib_client = self.create_client(GetParameters, '/calibrator/get_parameters')
        self.get_logger().info("Waiting for calibrator to setup cameras")
        calib_client.wait_for_service()
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_extrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_extrinsics", "camera_frames"]).values]
        while len(self.cameras) == 0: #wait for cams to come online
            self.get_logger().warn("No cams detected, waiting 2s.")
            time.sleep(2)
            self.image_topics, self.cameras, self.camera_instrinsics, self.camera_extrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_extrinsics", "camera_frames"]).values]

        # get global transform
        self.robot2global_tf = np.reshape([p.double_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["robot2global_tf"]).values], (4, 4))
        self.global_frame_id = call_get_parameters(node=self, node_name="/calibrator", parameter_names=["global_frame_id"]).values[0].string_value

        # Set camera parameters and topics
        self.camera_instrinsics = [json.loads(cintr) for cintr in self.camera_instrinsics]
        self.camera_extrinsics = [json.loads(cextr) for cextr in self.camera_extrinsics]

        self.last_images = {cam: None for cam in self.cameras}  # empty dict for last camera images

        for cam in self.cameras:
            self.get_logger().warn(f"Setting up subscription to camera {cam}.")
            camera_topic = cam + "/color/image_raw"
            # create input listener for raw images
            self.create_subscription(msg_type=Image,
                                                topic=camera_topic,
                                                # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                                callback=lambda msg, camera=cam: self.input_callback(msg, camera),
                                                callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
                                                # callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
                                                qos_profile=qos) #the listener QoS has to be =1, "keep last only".
            self.get_logger().info('Input listener created on topic: "%s"' % camera_topic)

        self.last_camera_index = 0

        self.get_logger().info("Pose query ready.")

    def update_next(self):
        self.last_camera_index += 1
        if self.last_camera_index >= len(self.cameras):
            self.last_camera_index = 0
        # if self.last_camera_index != 3:
        #     return
        self.current_camera = self.cameras[self.last_camera_index]
        if self.current_camera not in self.last_images:
            self.get_logger().error(f"Something is broken, camera {self.current_camera} is not in the initial list of cameras!")
            return
        image = self.last_images[self.current_camera]
        if image is None:
            self.get_logger().warn(f"No images from camera {self.current_camera}.")
            return
        self.last_images[self.current_camera] = None
        camera_data = self.getCameraData(self.current_camera)
        handle = self.sendRequest(image, camera_data['camera_matrix'])
        return handle

    def sendRequest(self, image, camera_matrix):
        # print(camera_matrix)
        try:
            request = {
                "image": image,
                "camera_matrix": camera_matrix.ravel()
            }
            self.get_logger().info(f"Sending request with image {type(request['image'])} [{request['image'].dtype}] and camera matrix {type(request['camera_matrix'])} [{request['camera_matrix'].dtype}]")
            handle = self.pose_service.call_async(request)
        except BaseException as e:
            print(f"An error had occured when trying to send pose request: {e}")
            tb.print_exc()
            return
        return handle

    def store_poses(self, camera, poses):
        # print(poses)
        if len(poses) == 0:
            self.get_logger().warn(f"No poses detected in camera {camera}")
            return
        existing_objects = self.crowracle.getTangibleObjectsPositions()
        existing_positions = np.array(list(map(lambda x: x["absolute_location"], existing_objects)))
        existing_classes = list(map(lambda x: x["class"], existing_objects))
        existing_dnames = list(map(lambda x: x["detector_name"], existing_objects))
        existing_uris = list(map(lambda x: x["uri"], existing_objects))
        if len(existing_objects) == 0:
            self.get_logger().warn(f"Poses detected in camera {camera} but there are no objects in the database!")
            return
        self.get_logger().info(f"Number of existing objects = {len(existing_objects)}")

        camera_data = self.getCameraData(camera)
        ctg_tf_mat = camera_data["ctg_tf"]

        stamp = dt.fromtimestamp(self.get_clock().now().nanoseconds * 1e-9).strftime('%Y-%m-%dT%H:%M:%SZ')
        pose_updates = []
        for obj, detection in poses.items():
            # print(detection)
            # pose_camera = maybeMakeFullMatrix(detection["position"], detection["orientation"])
            pose_camera = composeFullMatrix(detection[0], detection[1])
            pose_global = np.dot(ctg_tf_mat, pose_camera)
            # self.get_logger().info(f"Global object {obj} position:\n{pose_global}")
            g_position, g_orientation, zoom, shear = tf3.affines.decompose44(pose_global)
            euler = np.rad2deg(tf3.euler.mat2euler(g_orientation))
            g_quat = rotation_matrix2quat(g_orientation)
            self.get_logger().info(f"Pose of object {obj}:\n\tposition = {g_position}\n\torientation = {euler} (quat = {g_quat})")
            
            # last_image = self.last_images[camera]
            # if last_image is not None:
            #     m = ctg_tf_mat
            #     tvec, rvec = m[:3, 3], m[:3, :3]
            #     # print(">>>>>", cv2.Rodrigues(rvec))
            #     print(cv2.projectPoints(g_position[np.newaxis], cv2.Rodrigues(rvec)[0], tvec, camera_data["camera_matrix"], camera_data["distortion_coefficients"])[0])
            nearest_idx = np.argmin(np.linalg.norm(existing_positions - g_position, axis=1))
            print(existing_classes[nearest_idx], existing_dnames[nearest_idx])
            if obj in cosy2yolact:
                print(">>>")
                yobj = cosy2yolact[obj]
                # print(existing_dnames[nearest_idx])
                if yobj in existing_dnames[nearest_idx]:
                    print("BINGO!")
                    pose_updates.append((existing_uris[nearest_idx], stamp, g_quat))

        if len(pose_updates) > 0:
            try:
                self.crowracle.update_6dof_batch(pose_updates)
            except BaseException as e:
                self.get_logger().error(f"Succefully updating pose(s)")
                tb.print_exc()
            else:
                self.get_logger().info(f"Succefully updated pose(s) of {len(pose_updates)} objects.")
    
    def input_callback(self, msg, camera):
        if camera not in self.last_images:
            self.get_logger().error(f"Camera {camera} is not in the initial list of cameras, but I got an image for it!")
            return
        self.last_images[camera] = self.cvb.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # self.last_images[camera] = msg

    def getCameraData(self, camera):
        idx = self.cameras.index(camera)
        return {
            "camera": camera,
            "image_topic": self.image_topics[idx],
            "camera_matrix": np.array(self.camera_instrinsics[idx]["camera_matrix"]),
            "distortion_coefficients": np.array(self.camera_instrinsics[idx]["distortion_coefficients"]),
            "dtc_tf": self.camera_extrinsics[idx]["dtc_tf"],
            "ctg_tf": (self.robot2global_tf @ self.camera_extrinsics[idx]["ctg_tf"]).astype(np.float32),
            "optical_frame": self.camera_frames[idx],
        }


def main():
    rclpy.init()
    pq = PoseQuery()
    fhandle = None
    try:
        rate = pq.create_rate(10)
        while rclpy.ok():
            rclpy.spin_once(pq)
            if fhandle is None:  # no pending service call, try next camera
                fhandle = pq.update_next()  # if handle is None, update failed, go to next camera
            else:  # there is a pending service call
                if fhandle.done:  # service call finished
                    result = fhandle.get_result()
                    if result is None:  # service call failed
                        print("Service call failed")
                    else:  # service call succeeded
                        if fhandle.success: # service call succeeded
                            print("Got a result!")
                            pq.store_poses(pq.current_camera, result)
                        else: # service call failed
                            print("Service call was not successful!")
                    fhandle = None
                else:  # service call pending
                    # print("Still waiting for response...")
                    sleep(0.1)  # wait a bit

    except KeyboardInterrupt:
        print("User requested shutdown.")
    except BaseException as e:
        print(f"Some error had occured: {e}")
        tb.print_exc()
    finally:
        pq.destroy_node()
    print("Terminated")

if __name__ == "__main__":
    main()
