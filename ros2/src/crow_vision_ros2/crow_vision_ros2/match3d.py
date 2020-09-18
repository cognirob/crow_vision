import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters
from crow_msgs.msg import DetectionMask
from sensor_msgs.msg import PointCloud2, PointField
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
import numpy as np

import pkg_resources
from time import time

import open3d as o3d


class Match3D(Node):
    """
    Match our objects (from .STL) to the segmented point-cloud ("<cam>/detections/segmented_pointcloud") 
    based on segmentation from 2D RGB masks (mask_msg) from YOLACT. 
    """
    
    def load_models(self,
                    list_path_stl=["/home/imitrob/crow_simulation/crow_simulation/envs/objects/crow/stl/cube_holes.stl"],
                    list_labels=["cube_holes"],
                    num_points=5000):
        """
        Load our original model files (.stl) as pointclouds.
        @arg: list_path_stl : list of string paths to stl files, ie.: ["/home/data/models/chair.stl", "/hammer.stl"]
        @arg: list_labels: corresponding list of object labels: ["chair", "hammer"]
        @arg: num_points: how many points in resulting pcl (balance ICP speed vs precision)

        @return dict matching "label" -> PointCloud2
        """
        objects = {} #map class -> loaded pcl

        assert len(list_path_stl) == len(list_labels)
        self.get_logger().info("Loading models, please wait...")
        for i, (cls, path) in enumerate(zip(list_labels, list_path_stl)):
          #1. create Mesh
          mesh = o3d.io.read_triangle_mesh(path) #o3d is slow, but here it's only once in init, so it's OK.
          
          #2. o3d.PointCloud from mesh
          orig_pcd = o3d.geometry.PointCloud()
          orig_pcd.points = mesh.vertices
          orig_pcd.colors = mesh.vertex_colors
          orig_pcd.normals = mesh.vertex_normals

          #2.1 optimize pcd size (filter mesh) to reduce num points, for faster ICP matching
          # http://www.open3d.org/docs/release/tutorial/Basic/mesh.html#Mesh-filtering
          #pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, pcl=orig_pcd)
          #pcd = mesh.sample_points_uniformly(number_of_points=num_points)
          pcd = orig_pcd.voxel_down_sample(voxel_size=0.05)

          self.get_logger().info("Loading '{}' : mesh: {}\tPointCloud: {}\treduced pointcloud: {}".format(cls, mesh, orig_pcd, pcd))
          objects[cls] = pcd
        return objects



    def __init__(self, node_name="match3d"):
        super().__init__(node_name)
        #FIXME nefunguje?? pritom v locator.py jo!: 
        #FIXME self.cameras = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_nodes"]).values]
        self.cameras = ["/camera1/camera"]
        print(self.cameras)
        assert len(self.cameras) > 0

        self.mask_topics = [cam + "/" + "detections/masks" for cam in self.cameras] #input masks from 2D rgb
        self.seg_pcl_topics = [cam + "/" + "detections/segmented_pointcloud" for cam in self.cameras] #input segmented pcl data
        self.get_logger().info(str(self.seg_pcl_topics))
        self.get_logger().info(str(self.mask_topics))

        #TODO merge (segmented) pcls before this node
        #TODO create output publisher (what exactly to publish?)

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        for i, (cam, pclTopic, maskTopic) in enumerate(zip(self.cameras, self.seg_pcl_topics, self.mask_topics)):
            # create approx syncro callbacks
            subPCL = message_filters.Subscriber(self, PointCloud2, pclTopic, qos_profile=qos)
            subMasks = message_filters.Subscriber(self, DetectionMask, maskTopic, qos_profile=qos)
            self.get_logger().info("Created synced subscriber for masks: \"{}\" & segmented_pcl \"{}\"".format(maskTopic, pclTopic))
            sync = message_filters.ApproximateTimeSynchronizer([subPCL, subMasks], 20, 0.03)
            sync.registerCallback(lambda pcl_msg, mask_msg, cam=cam: self.detection_callback(pcl_msg, mask_msg, cam))

        # map str:label -> o3d.PointCloud model
        MODEL_PATH=str(pkg_resources.resource_filename("crow_simulation", 'envs/objects/crow/stl/'))
        #MODEL_PATH="/home/imitrob/crow_simulation/crow_simulation/envs/objects/crow/stl/"
        self.get_logger().info(MODEL_PATH)
        #self.objects = self.load_models()
        self.objects = self.load_models(list_path_stl=[
            MODEL_PATH+"car_roof.stl", 
            MODEL_PATH+"pliers.stl", 
            MODEL_PATH+"cube_holes.stl", 
            MODEL_PATH+"screw_round.stl", 
            MODEL_PATH+"ex_bucket.stl", 
            MODEL_PATH+"screwdriver.stl", 
            MODEL_PATH+"hammer.stl", 
            MODEL_PATH+"sphere_holes.stl", 
            MODEL_PATH+"wafer.stl", 
            MODEL_PATH+"nut.stl", 
            MODEL_PATH+"wheel.stl", 
            MODEL_PATH+"peg_screw.stl", 
            MODEL_PATH+"wrench.stl", 
            MODEL_PATH+"peg_simple.stl" 
            ], 
            list_labels=["car_roof", "pliers", "cube_holes", "screw_round", "ex_bucket", "screwdriver", "hammer", 
                "sphere_holes", "wafer", "nut", "wheel", "peg_screw", "wrench", "peg_simple"
                ], 
            num_points=1000) 



    def detection_callback(self, pcl_msg, mask_msg, camera):
        if not mask_msg.masks:
            print("no masks, no party. Quitting early.")
            return  # no mask detections (for some reason)

        # labels & score from detections masks
        class_names, scores = mask_msg.class_names, mask_msg.scores
        # pointcloud in numpy from pcl_msg
        point_cloud = np.array(pcl_msg.data).view(np.float32).reshape(-1, 3).T # 3 as we have x,y,z only in the pcl

        for i, (class_name, score) in enumerate(zip(class_names, scores)):
            icp_score = 0.0 #TODO ICP match self.objects["class_name"] to point_cloud
            self.get_logger().info("ICP Matching pcl {}  to \"{}\" (mask confidence {}) with match confidence: {}".format(np.shape(point_cloud), class_name, score, icp_score))




def main():
    rclpy.init()

    matcher = Match3D()

    rclpy.spin(matcher)
    matcher.destroy_node()


if __name__ == "__main__":
    main()
