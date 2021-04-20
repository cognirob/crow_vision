import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from rclpy.exceptions import ParameterNotDeclaredException
from ros2param.api import call_get_parameters
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image as msg_image
from crow_msgs.msg import FilteredPose, NlpStatus
from geometry_msgs.msg import PoseArray
from crow_vision_ros2.filters import object_properties
from cv_bridge import CvBridge
import message_filters
from crow_ontology.crowracle_client import CrowtologyClient

import cv2
import numpy as np
import time
import open3d as o3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
from unicodedata import normalize
import subprocess


class Visualizator(Node):
    INVERSE_OBJ_MAP = {v["name"]: i for i, v in enumerate(object_properties.values())}
    TIMER_FREQ = .5 # seconds
    VISUALIZE_PARTICLES = False #@TODO: communicate with ParticleFilter about this param!
    LANGUAGE = 'CZ' #language of the visualization
    COLOR_GRAY = (128, 128, 128)

    def __init__(self, node_name="visualizator"):
        super().__init__(node_name)

        self.processor_state_srv = self.create_client(GetParameters, '/sentence_processor/get_parameters')
        self.crowracle = CrowtologyClient(node=self)

        calib_client = self.create_client(GetParameters, '/calibrator/get_parameters')
        self.get_logger().info("Waiting for calibrator to setup cameras")
        calib_client.wait_for_service()
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]
        while len(self.cameras) == 0: #wait for cams to come online
            self.get_logger().warn("No cams detected, waiting 2s.")
            time.sleep(2)
            self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]
        self.mask_topics = [cam + "/detections/image_annot" for cam in self.cameras] #input masks from 2D rgb (from our detector.py)
        self.filter_topics = ["filtered_poses"] #input masks from 2D rgb (from our detector.py)
        self.nlp_topics = ["/nlp/status"] #nlp status (from our sentence_processor.py)
        self.cvb_ = CvBridge()
        self.cv_image = {} # initialize dict of images, each one for one camera
        # response = str(subprocess.check_output("ros2 param get /sentence_processor halt_nlp".split()))
        self.NLP_HALTED = False  #"False" in response

        #create timer for nlp params check - periodically check and update params in the annot figure
        #self.create_timer(self.TIMER_FREQ, self.check_nlp_params_timer)

        self.infoPanel = None
        #create listeners
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        for i, (cam, maskTopic) in enumerate(zip(self.cameras, self.mask_topics)):
            listener = self.create_subscription(msg_type=msg_image,
                                          topic=maskTopic,
                                          # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                          callback=lambda img_array_msg, cam=cam: self.input_detector_callback(img_array_msg, cam),
                                          qos_profile=qos) #the listener QoS has to be =1, "keep last only".

            self.get_logger().info('Input listener created on topic: "%s"' % maskTopic)

            # Initialize cv2 annotated image visualization with descriptions
            self.cv_image['{}'.format(cam)] = np.zeros((128, 128, 3))

            cv2.namedWindow('Detekce{}'.format(cam))
            cv2.setMouseCallback('Detekce{}'.format(cam), self.window_click)

        self.create_subscription(msg_type=NlpStatus,
                                            topic=self.nlp_topics[0],
                                            # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                            callback=lambda status_array_msg: self.input_nlp_callback(status_array_msg),
                                            qos_profile=qos) #the listener QoS has to be =1, "keep last only".
                
        self.get_logger().info('Input listener created on topic: "%s"' % self.nlp_topics[0])

        if self.VISUALIZE_PARTICLES:
            self.create_subscription(msg_type=FilteredPose,
                                            topic=self.filter_topics[0],
                                            # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                            callback=lambda pose_array_msg: self.input_filter_callback(pose_array_msg),
                                            qos_profile=qos) #the listener QoS has to be =1, "keep last only".
                
            self.get_logger().info('Input listener created on topic: "%s"' % self.filter_topics[0])

            # Initialize visualization properties
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            # geometry for the particles
            self.particle_cloud = o3d.geometry.PointCloud()
            # some initial random particles necessary automaticall set the viewpoint
            self.particle_cloud.points = o3d.utility.Vector3dVector(np.random.randn(10, 3)*2)
            self.vis.add_geometry(self.particle_cloud)
            # geometry for the axis
            self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
            self.vis.add_geometry(self.axis)
            # geometry for the texts
            self.text_cloud = o3d.geometry.PointCloud()
            # some initial random particles necessary automaticall set the viewpoint
            self.text_cloud.points = o3d.utility.Vector3dVector(np.random.randn(10, 3)*2)
            self.vis.add_geometry(self.text_cloud)

        # Initialize nlp params for info bellow image
        self.params = {'det_obj': '-', 'det_command': '-', 'det_obj_name': '-', 'det_obj_in_ws': '-', 'status': '-'}

    def window_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            response = str(subprocess.check_output("ros2 param get /sentence_processor halt_nlp".split()))
            if "False" in response:
                subprocess.run("ros2 param set /sentence_processor halt_nlp True".split())
                self.NLP_HALTED = True
            else:
                subprocess.run("ros2 param set /sentence_processor halt_nlp False".split())
                self.NLP_HALTED = False
            # print("Response === " + str(response.stdout))

    def input_filter_callback(self, pose_array_msg):
        if not pose_array_msg.particles:
            self.get_logger().info("No particles. Quitting early.")
            return  # no particles received (for some reason)
        self.visualize_particles(pose_array_msg.particles, pose_array_msg.label, pose_array_msg.poses)

    def input_detector_callback(self, img_array_msg, cam):
        if not img_array_msg.data:
            self.get_logger().info("No image. Quitting early.")
            return  # no annotated image received (for some reason)
        self.cv_image['{}'.format(cam)] = self.cvb_.imgmsg_to_cv2(img_array_msg, desired_encoding='bgr8')
        self.update_annot_image()

    def input_nlp_callback(self, status_array_msg):
        if not status_array_msg.det_obj:
            self.get_logger().info("No nlp detections. Quitting early.")
            return  # no nlp detections received (for some reason)

        obj_str = status_array_msg.det_obj
        obj_uri = self.crowracle.get_uri_from_str(obj_str)
        nlp_name = self.crowracle.get_nlp_from_uri(obj_uri)
        if len(nlp_name) > 0:
            nlp_name = nlp_name[0]
        else:
            nlp_name = '-'

        if status_array_msg.found_in_ws:
            obj_in_ws = 'ano'
        else:
            obj_in_ws = 'ne'

        self.params['det_obj'] = status_array_msg.det_obj
        self.params['det_command'] = status_array_msg.det_command
        self.params['det_obj_name'] = nlp_name
        self.params['det_obj_in_ws'] = obj_in_ws
        self.params['status'] = status_array_msg.status
        self.update_annot_image()

    def check_nlp_params_timer(self):
        print('in check_nlp_params')
        req = GetParameters.Request()
        req.names = ['template_detected','object_detected','found_in_workspace']
        # print(self.processor_state_srv.service_is_ready())
        future = self.processor_state_srv.call_async(req)
        
        # future.add_done_callback(self.nlp_params_callback)
        
        rclpy.spin_until_future_complete(self, future, timeout_sec=1)
        if future.done():
            self.nlp_params_callback(future)
        # else:
        #     return
        # self.update_annot_image()
        
        # while not future.done():
        #     rclpy.spin_once(self)
        # self.nlp_params_callback(future)
        #     #TODO: check if node/service is alive        
        # print(future.result().values[0].bool_value)
        # return future.result().values[0].bool_value

    def nlp_params_callback(self, future):
        print('nlp_params_callback')
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().warn("Nlp service call failed %r" % (e,))
        else:
            self.params = []
            for param in result.values:
                print("Got global param: {}".format(param.string_value))
            self.params.append(result.values[0].string_value)
            self.params.append(result.values[1].string_value)
            self.params.append(result.values[2].bool_value)
            obj_str = result.values[1].string_value
            obj_uri = self.crowracle.get_uri_from_str(obj_str)
            nlp_name = self.crowracle.get_nlp_from_uri(obj_uri, language=self.LANGUAGE)
            self.params.append(nlp_name)
            self.update_annot_image()

    def __putText(self, image, text, origin, size=1, color=(255, 255, 255), thickness=2):
        """ Prints text into an image. Uses the original OpenCV functions
        but simplifies some things. The main difference betwenn OpenCV and this function
        is that in this function, the origin is the center of the text.
        Arguments:
            image {ndarray} -- the image where the text is to be printed into
            text {str} -- the text itself
            origin {tuple} -- position in the image where the text should be centered around
        Keyword Arguments:
            size {int} -- size/scale of the font (default: {1})
            color {tuple} -- color of the text (default: {(255, 255, 255)})
            thickness {int} -- line thickness of the text (default: {2})
        Returns:
            ndarray -- the original image with the text printed into it
        """
        text = normalize('NFKD', text).encode('ascii', 'ignore').decode("utf-8")
        offset = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0] * np.r_[-1, 1] * 0 # / 2
        return cv2.putText(image, text, tuple(np.int32(origin + offset).tolist()), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

    def _get_obj_color(self, obj_name):
        return object_properties[self.INVERSE_OBJ_MAP[obj_name]]["color"]

    def update_annot_image(self):
        xp = 5
        yp = 20
        scHeight = 128
        im_shape = next(iter(self.cv_image.values())).shape
        scoreScreen = np.zeros((scHeight, im_shape[1], 3), dtype=np.uint8)
        if self.infoPanel is None:
            self.infoPanel = np.zeros((im_shape[0] + scHeight, 256, 3), dtype=np.uint8)
            self.infoPanel = self.__putText(self.infoPanel, "Prikazy:", (xp, yp*1), color=self.COLOR_GRAY, size=0.5, thickness=1)
            self.infoPanel = self.__putText(self.infoPanel, "Ukaz na <OBJEKT>", (xp, yp*2), color=self.COLOR_GRAY, size=0.5, thickness=1)
            self.infoPanel = self.__putText(self.infoPanel, "Ukaz na <BARVA> <OBJEKT>", (xp, yp*3), color=self.COLOR_GRAY, size=0.5, thickness=1)
            self.infoPanel = self.__putText(self.infoPanel, "Seber <OBJEKT>", (xp, yp*4), color=self.COLOR_GRAY, size=0.5, thickness=1)

        if self.LANGUAGE == 'CZ':
            scoreScreen = self.__putText(scoreScreen, "Detekovany prikaz: {}".format(self.params['det_command']), (xp, yp*1), color=(255, 255, 255), size=0.5, thickness=1)
            scoreScreen = self.__putText(scoreScreen, "Detekovany objekt: {}".format(self.params['det_obj']), (xp, yp*2), color=self.COLOR_GRAY, size=0.5, thickness=1)
            scoreScreen = self.__putText(scoreScreen, "Detekovany objekt (jmeno): {}".format(self.params['det_obj_name']), (xp, yp*3), color=(255, 255, 255), size=0.5, thickness=1)
            scoreScreen = self.__putText(scoreScreen, "Objekt je na pracovisti: {}".format(self.params['det_obj_in_ws']), (xp, yp*4), color=(255, 255, 255), size=0.5, thickness=1)
            scoreScreen = self.__putText(scoreScreen, "Stav: {}".format(self.params['status']), (xp, yp*5 + 10), color=(255, 224, 200), size=0.7, thickness=2)
            if self.NLP_HALTED:
                scoreScreen = self.__putText(scoreScreen, "STOP", (im_shape[1] - 70, yp), color=(0, 0, 255), size=0.7, thickness=2)

            for cam, img in self.cv_image.items():
                up_image = np.hstack((self.infoPanel, np.vstack((img, scoreScreen))))
                cv2.imshow('Detekce{}'.format(cam), up_image)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("f"):
                    print(cv2.getWindowProperty('Detekce{}'.format(cam), cv2.WND_PROP_FULLSCREEN))
                    print(cv2.WINDOW_FULLSCREEN)
                    print(cv2.getWindowProperty('Detekce{}'.format(cam), cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN)
                    if cv2.getWindowProperty('Detekce{}'.format(cam), cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                        cv2.setWindowProperty('Detekce{}'.format(cam), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)
                    else:
                        cv2.setWindowProperty('Detekce{}'.format(cam), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                if key == ord(" "):
                    self.window_click(cv2.EVENT_LBUTTONDOWN, None, None, None, None)
                if key == ord("q"):
                    rclpy.utilities.try_shutdown()

        else:
            scoreScreen = self.__putText(scoreScreen, "Detected this command: {}".format(self.params['det_command']), (xp, yp*1), color=(255, 255, 255), size=0.5, thickness=1)
            scoreScreen = self.__putText(scoreScreen, "Detected this object: {}".format(self.params['det_obj']), (xp, yp*2), color=(255, 255, 255), size=0.5, thickness=1)
            scoreScreen = self.__putText(scoreScreen, "Detected object name: {}".format(self.params['det_obj_name']), (xp, yp*3), color=(255, 255, 255), size=0.5, thickness=1)
            scoreScreen = self.__putText(scoreScreen, "Object in the workspace: {}".format(self.params['det_obj_in_ws']), (xp, yp*4), color=(255, 255, 255), size=0.5, thickness=1)
            scoreScreen = self.__putText(scoreScreen, "Status: {}".format(self.params['status']), (xp, yp*5), color=(255, 255, 255), size=0.5, thickness=1)

            for cam, img in self.cv_image.items():
                up_image = np.hstack((self.infoPanel, np.vstack((img, scoreScreen))))
                cv2.imshow('Detections{}'.format(cam), up_image)
                cv2.waitKey(10)

    def text_3d(self, text, pos, direction=None, degree=-90, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=8):
        """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param direction: 3D normalized direction of where the text faces
        :param degree: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
        """
        if direction is None:
            direction = (0., 0., 1.)

        font_obj = ImageFont.truetype(font, font_size*density)
        font_dim = font_obj.getsize(text)

        img = Image.new('RGB', font_dim, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
        img = np.asarray(img)
        img_mask = img[:, :, 0] < 128
        indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
        pcd.points = o3d.utility.Vector3dVector(indices / 1000.0 / density)

        raxis = np.cross([0.0, 0.0, 1.0], direction)
        if np.linalg.norm(raxis) < 1e-6:
            raxis = (0.0, 0.0, 1.0)
        trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
                Quaternion(axis=direction, degrees=degree)).transformation_matrix
        trans[0:3, 3] = np.asarray(pos)
        pcd.transform(trans)
        return pcd

    def visualize_particles(self, particles, labels, poses):
        if np.size(particles) > 0:
            # clear the geometries
            self.particle_cloud.clear()
            self.axis.clear()
            self.text_cloud.clear()
            for label, pose, points in zip(labels, poses, particles):
                # for each model, add its particles and pose as axis
                pts = np.frombuffer(points.data, dtype=np.float32)
                pts = np.reshape(pts, newshape=(points.layout.dim[0].size, points.layout.dim[1].size))
                pose = [pose.position.x, pose.position.y, pose.position.z]
                label_3d = self.text_3d(label, [pose[0], pose[1], pose[2]+0.1])
                tmp_pcl = o3d.geometry.PointCloud()
                tmp_pcl.points = o3d.utility.Vector3dVector(pts)
                c = self._get_obj_color(label)  # get model color according to the label
                tmp_pcl.paint_uniform_color(c)
                
                self.particle_cloud += tmp_pcl
                self.axis += o3d.geometry.TriangleMesh.create_coordinate_frame(0.2, pose)
                self.text_cloud += label_3d
            # o3d.visualization.draw_geometries([self.particle_cloud])
            self.vis.update_geometry(self.particle_cloud)
            self.vis.update_geometry(self.axis)
            self.vis.update_geometry(self.text_cloud)
        # if there are no models, only update the rendering window (otherwise panning/zooming won't work)
        self.vis.poll_events()
        self.vis.update_renderer()

def main():
    rclpy.init()
    #time.sleep(5)
    visualizator = Visualizator()
    # while rclpy.ok():
    #     visualizator.check_nlp_params_timer()
    #     print("sadasd")
    #     rclpy.spin_once(visualizator)
    rclpy.spin(visualizator)
    cv2.destroyAllWindows()
    visualizator.destroy_node()

if __name__ == "__main__":
    main()