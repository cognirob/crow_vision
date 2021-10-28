import operator
import os
import time

import commentjson as json
import cv2
import numpy as np
import pkg_resources
import rclpy
import sensor_msgs
import torch
from crow_msgs.msg import DetectionMask
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from ros2param.api import call_get_parameters

import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

print(f"Running PyTorch:")
print(f"\tver: {torch.__version__}")
print(f"\tfile: {torch.__file__}")
qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

def parse_named_output(image, pose_data, object_counts, objects, normalized_peaks):

    results = {}

    height = image.shape[0]
    width = image.shape[1]

    count = int(object_counts[0])
    for i in range(count):
        obj = objects[0][i]
        C = obj.shape[0]
        for j in range(C):
            name = pose_data["keypoints"][j]
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * width)
                y = round(float(peak[0]) * height)
                results[name] = (x, y)

    return results

class CrowVisionNvidiaPose(Node):

    def __init__(self, config='config.json'):
        super().__init__('crow_nvidia_pose_detector')

        #parse config
        CONFIG_DEFAULT = pkg_resources.resource_filename("crow_vision_ros2", config)
        with open(CONFIG_DEFAULT) as configFile:
            self.config = json.load(configFile)
            print(self.config)

        ## handle multiple inputs (cameras).
        # store the ROS Listeners,Publishers in a dict{}, keys by topic.
        self.cameras, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]
        while len(self.cameras) == 0:
            self.get_logger().warn("Waiting for any cameras!")
            time.sleep(2)
            self.cameras, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]

        self.ros = {}
        for cam in self.cameras:
            camera_topic=cam+"/color/image_raw"
            # create INput listener with raw images
            listener = self.create_subscription(msg_type=sensor_msgs.msg.Image,
                                                topic=camera_topic,
                                                # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                                callback=lambda msg, topic=camera_topic: self.input_callback(msg, topic),
                                                callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
                                                # callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
                                                qos_profile=1) #the listener QoS has to be =1, "keep last only".
            self.get_logger().info('Input listener created on topic: "%s"' % camera_topic)
            self.ros[camera_topic] = {} # camera_topic is used as an ID for this input, all I/O listeners,publishers will be based under that id.

            # there are multiple publishers (for each input/camera topic).
            # the results are separated into different (optional) subtopics the clients can subscribe to (eg 'labels', 'masks')
            # If an output topic is empty (""), we skip publishing on that stream, it is disabled. Use to save computation resources.
            if self.config["outputs"]["masks"]:
                # Posenet masks
                topic = cam + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["masks"]
                self.get_logger().info('Output publisher created for topic: "%s"' % topic)
                self.ros[camera_topic]["pub_masks"] = self.create_publisher(DetectionMask, topic, qos_profile=qos) #publishes the processed (annotated,detected) pose

        self.noMessagesYet = True
        self.cvb_ = CvBridge()

        ## Posenet setupwith open('human_pose.json', 'r') as f:

        script_loc = os.path.dirname(os.path.realpath(__file__))
        human_pose_json_path = os.path.join(script_loc, 'nvidia_pose/human_pose.json')

        with open(human_pose_json_path, 'r') as f:
            self.human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)

        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])

        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

        cluster_path = os.environ["CIIRC_CLUSTER"]
        model_path = os.path.join(cluster_path, "nfs/projects/crow/data/trt_pose/", "resnet18_baseline_att_224x224_A_epoch_249.pth")
        model.load_state_dict(torch.load(model_path))

        self.parse_objects = ParseObjects(topology)
        self.draw_objects = DrawObjects(topology)

        self.model = model

        self.device = torch.device('cuda')

        try:
            import pydevd_pycharm
            pydevd_pycharm.settrace('172.18.0.1', port=25565, stdoutToServer=True, stderrToServer=True)
        except Exception:
            print("Skipping debugger")

    @staticmethod
    def cropND(img, bounding):
        start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]

    def input_callback(self, msg, topic):
        """
        @param msg - ROS msg (Image data) to be processed. From camera
        @param topic - str, from camera/input on given topic.
        @return nothing, but send new message(s) via output Publishers.
        """

        img_raw = self.cvb_.imgmsg_to_cv2(msg, "bgr8")
        img_raw = CrowVisionNvidiaPose.cropND(img_raw, (224, 224))

        # Pose mask
        if "pub_masks" in self.ros[topic]:
            img_raw = np.reshape(img_raw, (1, 3, 224, 224))

            data = torch.from_numpy(img_raw).to(self.device)
            data = data.float()

            cmap, paf = self.model(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)

            detections = parse_named_output(img_raw, self.human_pose, counts, objects, peaks)
            # detections is an dictionary of the following format:
            # {'left_ear': (126, 59),
            #  'left_elbow': (185, 203),
            #  'left_eye': (111, 58),
            #  'left_shoulder': (169, 107),
            #  'left_wrist': (121, 221),
            #  'neck': (116, 110),
            #  'nose': (101, 70),
            #  'right_ear': (77, 62),
            #  'right_elbow': (64, 194),
            #  'right_eye': (90, 60),
            #  'right_shoulder': (65, 114),
            #  'right_wrist': (82, 220)}
            # where keys are body parts and values are x and y coordinate

            msg_mask = DetectionMask()
            msg_mask.header.frame_id = msg.header.frame_id

            # TODO convert detected data to DetectionMask


def main(args=None):
    rclpy.init(args=args)
    try:
        cnn = CrowVisionNvidiaPose()
        n_threads = len(cnn.cameras)
        mte = rclpy.executors.MultiThreadedExecutor(num_threads=n_threads, context=rclpy.get_default_context())
        rclpy.spin(cnn, executor=mte)
        # rclpy.spin(cnn)
        cnn.destroy_node()
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
