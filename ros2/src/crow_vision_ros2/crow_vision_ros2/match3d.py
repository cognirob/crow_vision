import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

from crow_msgs.msg import DetectionMask, SegmentedPointcloud
from sensor_msgs.msg import PointCloud2, PointField

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import numpy as np

import pkg_resources
from time import time
import copy

import open3d as o3d


class Match3D(Node):
    """
    Match our objects (from .STL) to the segmented point-cloud ("<cam>/detections/segmented_pointcloud") 
    based on segmentation from 2D RGB masks (mask_msg) from YOLACT. 
    
    Publish the complete matched object as SegmentedPointcloud.
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

        @return dict matching "label" -> o3d.PointCloud
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
          #pcd = orig_pcd.voxel_down_sample(voxel_size=0.005) # 5mm ?
          pcd = orig_pcd

          self.get_logger().info("Loading '{}' : mesh: {}\tPointCloud: {}\treduced pointcloud: {}".format(cls, mesh, orig_pcd, pcd))
          objects[cls] = pcd
        return objects



    def __init__(self, node_name="match3d"):
        super().__init__(node_name)
        self.cameras = []
        while(len(self.cameras) == 0):
            try:
                self.cameras = call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_nodes"]).values[0].string_array_value
            except:
                print("getting cameras failed. Retrying in 2s")
                time.sleep(2)
        #self.cameras = ["/camera1/camera"]
        assert len(self.cameras) > 0

        self.seg_pcl_topics = [cam + "/" + "detections/segmented_pointcloud" for cam in self.cameras] #input segmented pcl data

        #TODO merge (segmented) pcls before this node
        #TODO create output publisher (what exactly to publish?)

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        for i, (cam, pclTopic) in enumerate(zip(self.cameras, self.seg_pcl_topics)):
            self.get_logger().info("Created subscriber for segmented_pcl \"{}\"".format(pclTopic))
            self.create_subscription(SegmentedPointcloud, pclTopic, lambda pcl_msg, cam=cam: self.detection_callback(pcl_msg, cam), qos_profile=qos)

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


    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)

        o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482 ,0.1556])


    def preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh


    def prepare_dataset(self, source, target, voxel_size):
        print(":: Load two point clouds and disturb initial pose.")
        #source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        #target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        #source.transform(trans_init)
        #self.draw_registration_result(source, target, np.identity(4))

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh



    def execute_fast_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
        result = o3d.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result





    def detection_callback(self, msg, camera):

        # model (as a pointcloud) to be matched to a real-world position
        model_pcl = None
        try:
            model_pcl = self.objects[msg.label]
        except:
            self.get_logger().info("Unknown model for detected label {}. Skipping.".format(msg.label))
            return

        
        # pointcloud from msg PointCloud2 -> numpy -> o3d.PointCloud
        start = time()
        pcd = np.array(msg.pcl.data).view(np.float32).reshape(-1, 3) # 3 as we have x,y,z only in the pcl
        real_pcl = o3d.geometry.PointCloud()
        #real_pcl.points = o3d.utility.Vector3dVector(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg.pcl)) #using ros2_numpy 
        real_pcl.points = o3d.utility.Vector3dVector(pcd)
        #real_pcl = real_pcl.voxel_down_sample(voxel_size=0.001) #TODO should we also downsample the incoming pcl? (slower conversion to o3d, but might do faster ICP)
        end = time()
        self.get_logger().info("Convert to o3d: {}".format(end-start)) # ~0.003s



        # fit global RANSAC
        start = time()
        voxel_size = 1 # means 5mm for this dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(model_pcl, real_pcl, voxel_size)
        result_ransac = self.execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        end = time()
        print("RANSAC {}".format(end-start))
        print(result_ransac)
        #self.draw_registration_result(source_down, target_down, result_ransac.transformation)
        
        # fit model to real pcl: ICP
        # http://www.open3d.org/docs/release/tutorial/Basic/icp_registration.html#Point-to-point-ICP
        print("Apply point-to-point ICP")
        start = time()
        reg_p2p = o3d.registration.registration_icp(
            source=model_pcl, 
            target=real_pcl, 
            max_correspondence_distance=0.1, 
            #init=trans_init, #TODO bundle the TF from locator to SegmentedPointcloud and a) transform model_pcl to the real_pcl's close location; or b) provide the init as 4x4 float64 initial transform estimation (better?)
            estimation_method=o3d.registration.TransformationEstimationPointToPoint())
        end = time()
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        self.get_logger().info("ICP Matching pcl {}  to \"{}\" (mask confidence {}) with fit success: {} in {} sec.".format(real_pcl, msg.label, msg.confidence, reg_p2p.fitness, (end-start)))




def main():
    rclpy.init()

    matcher = Match3D()

    rclpy.spin(matcher)
    matcher.destroy_node()


if __name__ == "__main__":
    main()
