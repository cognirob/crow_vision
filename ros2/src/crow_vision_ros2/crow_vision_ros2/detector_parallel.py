import rclpy #add to package.xml deps
from rclpy.node import Node
from ros2param.api import call_get_parameters

import sensor_msgs

from crow_msgs.msg import DetectionMask, DetectionBBox, BBox
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import cv2
import numpy as np
import torch

import commentjson as json
import pkg_resources
import time
import copy

from rclpy.qos import qos_profile_sensor_data
from multiprocessing import Queue
from multiprocessing.pool import  ThreadPool

print(f"Running PyTorch:")
print(f"\tver: {torch.__version__}")
print(f"\tfile: {torch.__file__}")
qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

class CrowVision(Node):
    """
    ROS2 node for CNN used in Crow.

    Run as
    `ros2 run crow_vision_ros2 detector [path_to_weights [config to use [input topic name]]]`

    This node listens on network for topic "/camera1/color/image_raw",
    obtains raw RGB image, processes it in neural net, and publishes results in form of (for given camera):
    - "/detections/image" - processed image with visualized masks, bboxes and labels
    - "/detections/masks"
    - "/detections/labels"
    - "/detections/bboxes"
    - "/detections/confidences", etc. TODO
    """
    def __init__(self, config='config.json'):
        super().__init__('crow_detector')

        #parse config
        CONFIG_DEFAULT = pkg_resources.resource_filename("crow_vision_ros2", config)
        with open(CONFIG_DEFAULT) as configFile:
            self.config = json.load(configFile)
            print(self.config)

        # specific imports based on YOLACT / Detectron2
        if self.config["type"] == "YOLACT":
            # import CNN - YOLACT
            YOLACT_REPO='~/crow_vision_yolact/' #use your existing yolact setup
            import sys; import os; sys.path.append(os.path.abspath(os.path.expanduser(YOLACT_REPO)))
            from inference_tool import InfTool
            from yolact import Yolact
            from eval import prep_display
            from data import set_cfg
        elif self.config["type"] == "Detectron2":
            import detectron2
        else:
            raise Exception("Supported types only: 'Detectron2', 'YOLACT'. Set in config.type. ")

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
                                                qos_profile=qos_profile_sensor_data) #the listener QoS has to be =1, "keep last only".
            self.get_logger().info('Input listener created on topic: "%s"' % camera_topic)
            self.ros[camera_topic] = {} # camera_topic is used as an ID for this input, all I/O listeners,publishers will be based under that id.

            # there are multiple publishers (for each input/camera topic).
            # the results are separated into different (optional) subtopics the clients can subscribe to (eg 'labels', 'masks')
            # If an output topic is empty (""), we skip publishing on that stream, it is disabled. Use to save computation resources.
            if self.config["outputs"]["image_annotated"]:
                topic = cam + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["image_annotated"]
                publisher_img = self.create_publisher(sensor_msgs.msg.Image, topic, qos_profile=qos) #publishes the processed (annotated,detected) image
                self.get_logger().info('Output publisher created for topic: "%s"' % topic)
                self.ros[camera_topic]["pub_img"] = publisher_img
            if self.config["outputs"]["masks"]:
                topic = cam + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["masks"]
                self.get_logger().info('Output publisher created for topic: "%s"' % topic)
                self.ros[camera_topic]["pub_masks"] = self.create_publisher(DetectionMask, topic, qos_profile=qos) #publishes the processed (annotated,detected) masks
            if self.config["outputs"]["bboxes"]:
                topic = cam + "/" + self.config["outputs"]["prefix"] + "/" + self.config["outputs"]["bboxes"]
                self.get_logger().info('Output publisher created for topic: "%s"' % topic)
                self.ros[camera_topic]["pub_bboxes"] = self.create_publisher(DetectionBBox, topic, qos_profile=qos) #publishes the processed (annotated,detected) bboxes

        # self.get_logger().error(str(self.ros))

        self.noMessagesYet = True
        self.cvb_ = CvBridge()

        ## YOLACT setup
        # setup additional args
        self.declare_parameter("config", self.config["config"])
        cfg = self.get_parameter("config").get_parameter_value().string_value

        # load model weights
        self.declare_parameter("weights", self.config["weights"])
        model = self.get_parameter("weights").get_parameter_value().string_value

        if ".obj" in cfg:
            cfg = os.path.join(os.path.abspath(os.path.expanduser(YOLACT_REPO)), cfg)
        elif "none" in cfg.lower():
            cfg = None

        self.get_logger().info("Using config '{}'.".format(cfg))
        self.get_logger().info("Using weights from file '{}'.".format(model))

        model_abs = os.path.join(
            os.path.abspath(os.path.expanduser(YOLACT_REPO)),
            str(model)
        )
        assert os.path.exists(model_abs), "Provided path to model weights does not exist! {}".format(model_abs)
        self.cnn = InfTool(weights=model_abs, top_k=self.config["top_k"], score_threshold=self.config["threshold"], config=cfg, parallel=True)
        self.prep_display = prep_display
        self.frame_buffer = Queue()
        self.finished_frames = Queue()
        self.sequence = [self.prep_frame, self.eval_network, self.transform_frame]
        self.multiframe = 8
        self.pool = ThreadPool(processes=len(self.sequence) + self.multiframe + 2)
        self.started = False
        self.start_time = time.time()
        self.frames_processed = 0
        print('Hi from crow_vision_ros2.')

    def get_next_frame(self):
    #    print(123)
        frames = []
        topics = []
        for idx in range(self.multiframe):
            frame, topic = self.frame_buffer.get()
            if frame is None:
                return frames
            frames.append(frame)
            topics.append(topic)
    #    print("get next end")
        return frames, topics

    def transform_frame(self, frames):
    #    print("Transform")
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, self.cnn.transform(torch.stack(frames, 0))

    def eval_network(self, inp):
        #print("Eval")
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < self.multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = self.cnn.net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
        #    print("end eval")
            return frames, out

    def prep_frame(self, inp, label_on):
        #print("Prep frame")
        with torch.no_grad():
            result = {'raw_output': [], 'labeled_image': []}
            frame, preds = inp
            # result = prep_display(preds, frame, None, None, undo_transform=False, class_color=True)
            #print(frame.cpu().numpy())
            result['raw_output'] = self.cnn.raw_inference(frame.cpu().numpy(), preds)
        #    print("Prep processing")
            if label_on:
                result['labeled_image'] = self.prep_display(preds, frame, None, None, undo_transform=False, class_color=True)
        #    print("End prep frame")
            return result

    def process_frames(self):
        #print(1000)
        extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])
        frame, topic = self.get_next_frame()
        first_batch = self.eval_network(self.transform_frame(frame))
        active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0, 'topic': topic} for i in range(len(first_batch[0]))]
        #print(active_frames)


        try:
            while True:
                #print(1000)

                # Start loading the next frames from the disk
                next_frames = self.pool.apply_async(self.get_next_frame)
            #    print(234)
                #print(active_frames)
                if len(active_frames):
                    print(len(active_frames))
                    # For each frame in our active processing queue, dispatch a job
                    # for that frame using the current function in the sequence
                    for frame in active_frames:
                        _args =  [frame['value']]
                    #    print(567)
                    #    print(frame['idx'])
                        if frame['idx'] == 0:
                            _args.append(True)
                        frame['value'] = self.pool.apply_async(self.sequence[frame['idx']], args=_args)
                    #    print(89)


                    for frame in active_frames:
                        if frame['idx'] == 0:
                            #print(frame['value'].get()['raw_output'])
                            finished_frame = frame
                            frame_topic = frame["topic"][0]
                        #    print("Topic:")
                        #    print(frame_topic)
                        #    print(finished_frame)
                            if "pub_img" in self.ros[frame_topic]: # labeled image publisher. (Use "" to disable)
                            #    print("Start publish")
                                #img_labeled = self.cnn.label_image(img_raw, copy.deepcopy(preds), copy.deepcopy(frame))
                                img_labeled = cv2.cvtColor(finished_frame['value'].get()['labeled_image'], cv2.COLOR_BGR2RGB)
                                #print(img_labeled)
                            #    print("Still publishing")

                                if img_labeled.ndim == 3:
                                    batch,w,h,c = 1, *img_labeled.shape
                                else:
                                    batch,w,h,c = img_labeled.shape
                                    img_labeled = img_labeled[0]
                                assert batch==1,"Batch mode not supported in ROS yet"

                                msg_img = self.cvb_.cv2_to_imgmsg(img_labeled, encoding="rgb8")
                                #print("Image converted")
                                # parse time from incoming msg, pass to outgoing msg
                                #msg_img.header.stamp = msg.header.stamp #we always inherit timestamp from the original "time taken", ie stamp from camera topic
                                #msg_img.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
                                #   self.get_logger().info("Publishing as Image {} x {}".format(msg_img.width, msg_img.height))
                                #print("Before publish")
                                self.ros[frame_topic]["pub_img"].publish(msg_img)
                                #print("Published")

                            if "pub_masks" in self.ros[frame_topic] or "pub_bboxes" in self.ros[frame_topic]:
                                classes, class_names, scores, bboxes, masks, centroids, _ = finished_frame['value'].get()['raw_output']
                                classes = classes.astype(int).tolist()
                                scores = scores.astype(float).tolist()
                                if len(classes) == 0:
                                    self.get_logger().info("No objects detected, skipping.")
                                    return

                                if "pub_masks" in self.ros[frame_topic]:
                                    msg_mask = DetectionMask()
                                    m_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                                    msg_mask.masks = [self.cvb_.cv2_to_imgmsg(cv2.morphologyEx(mask, cv2.MORPH_OPEN, m_kernel), encoding="mono8") for mask in masks.astype(np.uint8)]
                                    #cv2.imshow("aa", cv2.morphologyEx(masks[0], cv2.MORPH_OPEN, m_kernel)); cv2.waitKey(4)
                                    # parse time from incoming msg, pass to outgoing msg
                                    # msg_mask.header.stamp = msg.header.stamp
                                    # for mask in msg_mask.masks:
                                    #     mask.header.stamp = msg.header.stamp
                                    # msg_mask.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
                                    msg_mask.classes = classes
                                    msg_mask.class_names = class_names
                                    msg_mask.scores = scores
                                    #self.get_logger().info("Publishing as String {} at time {} ".format(msg_mask.class_names, msg_mask.header.stamp.sec))
                                    self.ros[frame_topic]["pub_masks"].publish(msg_mask)
                                if "pub_bboxes" in self.ros[frame_topic]:
                                    msg_bbox = DetectionBBox()
                                    msg_bbox.bboxes = [BBox(bbox=bbox) for bbox in bboxes]
                                    # parse time from incoming msg, pass to outgoing msg
                                    # msg_bbox.header.stamp = msg.header.stamp
                                    # msg_bbox.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
                                    msg_bbox.classes = classes
                                    msg_bbox.class_names = class_names
                                    msg_bbox.scores = scores
                                    # self.get_logger().info("Publishing as String {} at time {} ".format(msg_mask.data, msg_mask.header.stamp.sec))
                                    self.ros[frame_topic]["pub_bboxes"].publish(msg_bbox)
                            self.frames_processed += 1
                            print("Detector fps: {}".format(self.frames_processed / (time.time()-self.start_time)))



                    # Remove the finished frames from the processing queue
                    active_frames = [x for x in active_frames if x['idx'] > 0]

                    # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                    for frame in list(reversed(active_frames)):
                        frame['value'] = frame['value'].get()
                        frame['idx'] -= 1
                        #print(frame['idx'])

                        if frame['idx'] == 0:
                            #print(234)
                            # Split this up into individual threads for prep_frame since it doesn't support batch size
                            active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0, 'topic': frame['topic']} for i in range(1, len(frame['value'][0]))]
                            frame['value'] = extract_frame(frame['value'], 0)

                # Finish loading in the next frames and add them to the processing queue
                #print(next_frames)
                if next_frames is not None:
                    frames, topics = next_frames.get()
                    if len(frames) != 0:
                        active_frames.append({'value': frames, 'idx': len(self.sequence)-1, 'topic': topics})

        except KeyboardInterrupt:
            print('\nStopping...')

        self.pool.terminate()

    def input_callback(self, msg, topic):
        """
        @param msg - ROS msg (Image data) to be processed. From camera
        @param topic - str, from camera/input on given topic.
        @return nothing, but send new message(s) via output Publishers.
        """
        if self.noMessagesYet:
            self.get_logger().info("Image received from camera! (will not report on next image callbacks)")
            self.noMessagesYet = False

        # self.get_logger().info("I heard: {} for topic {}".format(str(msg.height), topic))
        # self.get_logger().error(str(dir(self.ros[topic]["pub_masks"])))
        # assert topic in self.ros, "We don't have registered listener for the topic {} !".format(topic)
        img_raw = self.cvb_.imgmsg_to_cv2(msg, "bgr8")
        self.frame_buffer.put((img_raw, topic))
        print(self.frame_buffer.qsize())
        #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # preds, frame = self.cnn.process_batch(img_raw)
        # if preds[0]["detection"] is None:
        #     return  # do not publish if nothing was detected

        #the input callback triggers the publishers here.

        # while not self.finished_frames.empty():
            # finished_frame = self.finished_frames.get()
            # frame_topic = finished_frame["topic"][0]
            # print("Topic:")
            # print(frame_topic)
            # if "pub_img" in self.ros[frame_topic]: # labeled image publisher. (Use "" to disable)
            #     #img_labeled = self.cnn.label_image(img_raw, copy.deepcopy(preds), copy.deepcopy(frame))
            #     img_labeled = cv2.cvtColor(finished_frame['value'].get()['labeled_image'], cv2.COLOR_BGR2RGB)
            #     #print(img_labeled)
            #
            #     if img_labeled.ndim == 3:
            #         batch,w,h,c = 1, *img_labeled.shape
            #     else:
            #         batch,w,h,c = img_labeled.shape
            #         img_labeled = img_labeled[0]
            #     assert batch==1,"Batch mode not supported in ROS yet"
            #
            #     msg_img = self.cvb_.cv2_to_imgmsg(img_labeled, encoding="rgb8")
            #     # parse time from incoming msg, pass to outgoing msg
            #     msg_img.header.stamp = msg.header.stamp #we always inherit timestamp from the original "time taken", ie stamp from camera topic
            #     msg_img.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
            #     #   self.get_logger().info("Publishing as Image {} x {}".format(msg_img.width, msg_img.height))
            #     self.ros[frame_topic]["pub_img"].publish(msg_img)
            #
            # if "pub_masks" in self.ros[frame_topic] or "pub_bboxes" in self.ros[frame_topic]:
            #     classes, class_names, scores, bboxes, masks, centroids, _ = finished_frame['value'].get()['raw_output']
            #     classes = classes.astype(int).tolist()
            #     scores = scores.astype(float).tolist()
            #     if len(classes) == 0:
            #         self.get_logger().info("No objects detected, skipping.")
            #         return
            #
            #     if "pub_masks" in self.ros[frame_topic]:
            #         msg_mask = DetectionMask()
            #         m_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            #         msg_mask.masks = [self.cvb_.cv2_to_imgmsg(cv2.morphologyEx(mask, cv2.MORPH_OPEN, m_kernel), encoding="mono8") for mask in masks.astype(np.uint8)]
            #         #cv2.imshow("aa", cv2.morphologyEx(masks[0], cv2.MORPH_OPEN, m_kernel)); cv2.waitKey(4)
            #         # parse time from incoming msg, pass to outgoing msg
            #         msg_mask.header.stamp = msg.header.stamp
            #         for mask in msg_mask.masks:
            #             mask.header.stamp = msg.header.stamp
            #         msg_mask.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
            #         msg_mask.classes = classes
            #         msg_mask.class_names = class_names
            #         msg_mask.scores = scores
            #         #self.get_logger().info("Publishing as String {} at time {} ".format(msg_mask.class_names, msg_mask.header.stamp.sec))
            #         self.ros[frame_topic]["pub_masks"].publish(msg_mask)
            #     if "pub_bboxes" in self.ros[frame_topic]:
            #         msg_bbox = DetectionBBox()
            #         msg_bbox.bboxes = [BBox(bbox=bbox) for bbox in bboxes]
            #         # parse time from incoming msg, pass to outgoing msg
            #         msg_bbox.header.stamp = msg.header.stamp
            #         msg_bbox.header.frame_id = msg.header.frame_id  # TODO: fix frame name because stupid Intel RS has only one frame for all cameras
            #         msg_bbox.classes = classes
            #         msg_bbox.class_names = class_names
            #         msg_bbox.scores = scores
            #         # self.get_logger().info("Publishing as String {} at time {} ".format(msg_mask.data, msg_mask.header.stamp.sec))
            #         self.ros[frame_topic]["pub_bboxes"].publish(msg_bbox)
            # self.frames_processed += 1
            # print("Detector fps: {}".format(self.frames_processed / (time.time()-self.start_time)))

        if not self.started:
            print("Start")
            self.pool.apply_async(self.process_frames)
            self.started = True


def main(args=None):
    rclpy.init(args=args)
    try:
        cnn = CrowVision()
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
