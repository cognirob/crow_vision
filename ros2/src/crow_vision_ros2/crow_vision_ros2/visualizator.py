import rclpy
from rclpy.node import Node
from ros2param.api import call_get_parameters
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image as msg_image
from crow_msgs.msg import DetectionMask, FilteredPose
from geometry_msgs.msg import PoseArray
from crow_vision_ros2.filters import object_properties
import message_filters

import numpy as np
import time
import open3d as o3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

class Visualizator(Node):
    INVERSE_OBJ_MAP = {v["name"]: i for i, v in enumerate(object_properties.values())}

    def __init__(self, node_name="visualizator"):
        super().__init__(node_name)

        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]
        while len(self.cameras) == 0: #wait for cams to come online
            self.get_logger().warn("No cams detected, waiting 2s.")
            time.sleep(2)
            self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]
        self.mask_topics = [cam + "/detections/image" for cam in self.cameras] #input masks from 2D rgb (from our detector.py)
        self.filter_topics = ["filtered_poses"] #input masks from 2D rgb (from our detector.py)

        #create listeners
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        for i, (cam, maskTopic) in enumerate(zip(self.cameras, self.mask_topics)):
            listener = self.create_subscription(msg_type=msg_image,
                                          topic=maskTopic,
                                          # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                          callback=lambda img_array_msg, cam=cam: self.input_detector_callback(img_array_msg, cam),
                                          qos_profile=qos) #the listener QoS has to be =1, "keep last only".

            self.get_logger().info('Input listener created on topic: "%s"' % maskTopic)

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

    def input_filter_callback(self, pose_array_msg):
        if not pose_array_msg.particles:
            self.get_logger().info("No particles. Quitting early.")
            return  # no particles received (for some reason)
        self.visualize_particles(pose_array_msg.particles, pose_array_msg.label, pose_array_msg.poses)

    def input_detector_callback(self,img_array_msg, cam):
        if not img_array_msg.image:
            self.get_logger().info("No image. Quitting early.")
            return  # no annotated image received (for some reason)
        print(cam)
        print(img_array_msg.data)
        print(img_array_msg.height)
        print(img_array_msg.width)

    def _get_obj_color(self, obj_name):
        return object_properties[self.INVERSE_OBJ_MAP[obj_name]]["color"]

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
    time.sleep(1)
    visualizator = Visualizator()
    rclpy.spin(visualizator)
    visualizator.destroy_node()

if __name__ == "__main__":
    main()