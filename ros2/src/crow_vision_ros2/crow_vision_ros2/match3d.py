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
                    num_points=5000,
                    voxel_size=0.0005):
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
          objects[cls] = {}

          #1. create Mesh
          mesh = o3d.io.read_triangle_mesh(path) #o3d is slow, but here it's only once in init, so it's OK.
          
          #2. o3d.PointCloud from mesh
          orig_pcd = o3d.geometry.PointCloud()
          orig_pcd.points = mesh.vertices
          orig_pcd.colors = mesh.vertex_colors
          orig_pcd.normals = mesh.vertex_normals
          orig_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=300))
          objects[cls]["orig"] = orig_pcd
          #_, orig_fpfh = self.preprocess_point_cloud(orig_pcd, voxel_size=0.00001)
          #objects[cls]["orig_fpfh"] = orig_fpfh #features for RANSAC

          #2.1 optimize pcd size (filter mesh) to reduce num points, for faster ICP matching
          down, down_fpfh = self.preprocess_point_cloud(orig_pcd, voxel_size)
          #objects[cls]["down"] = down
          #objects[cls]["down_fpfh"] = down_fpfh #features for RANSAC

          self.get_logger().info("Loading '{}' : mesh: {}\tPointCloud: {}\treduced pointcloud: {}".format(cls, mesh, orig_pcd, down))
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


    def preprocess_point_cloud(self, pcd, voxel_size=0.001):
        #print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = copy.deepcopy(pcd)
        downsampled = pcd.voxel_down_sample(voxel_size)
        if len(downsampled.points) > 1000:
          pcd_down = downsampled #actually downsampled data

        radius_normal = voxel_size * 2
        #print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh


    def execute_fast_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        #print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
        result = o3d.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result





    def detection_callback(self, msg, camera):

        # model (as a pointcloud) to be matched to a real-world position
        model_pcl = None
        try:
            model_pcl = self.objects[msg.label]["orig"]
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


        # 0/ compute (approx) initial transform; just moves the clouds "nearby"
        transform_initial = np.identity(4) #initial affine transform, moves clouds "close together" (by mean xyz of the pcls)
        fromm = np.asarray(model_pcl.points)
        to    = np.asarray(real_pcl.points)
        fromm_xyz = np.mean(fromm, axis=0, keepdims=True)
        to_xyz = np.mean(to, axis=0, keepdims=True)
        move = (to_xyz - fromm_xyz)[0]
        transform_initial[0:3,3] = move #write the "move"=translation (x,y,z) to the translation part of the affine matrix
        #print("BEFORE: Mean proto: {} \tMean pcl: {}".format(np.mean(fromm, axis=0, keepdims=True), np.mean(to, axis=0, keepdims=True)))
        model_pcl.transform(transform_initial) #move the model approx where the real pcl is.
        #print("AFTER: Mean proto: {} \tMean pcl: {}".format(np.mean(fromm, axis=0, keepdims=True), np.mean(to, axis=0, keepdims=True)))
        assert np.abs(np.mean(fromm, axis=0, keepdims=True) - np.mean(to, axis=0, keepdims=True)).sum() < 0.0001, "Initial transform should move cloud centers together!"


        # 1/ (optional) fit global RANSAC
        doGlobalApprox = True
        if doGlobalApprox:
          start = time()
          voxel_size = 0.5 # means 5mm for this dataset #TODO what is voxel size related to mm IRL?
          source_down, source_fpfh = self.preprocess_point_cloud(real_pcl, voxel_size)
          target_down, target_fpfh = self.preprocess_point_cloud(model_pcl,voxel_size)
          #target_down = self.objects[msg.label]["down"]
          #target_fpfh = self.objects[msg.label]["down_fpfh"]
          result_ransac = self.execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
          end = time()
          print("RANSAC {}".format(end-start))
          print(result_ransac)
          model_pcl.transform(result_ransac.transformation)
        

        # 1.2/ global RANSAC more precise:
        doGlobalPrecise = False
        if doGlobalPrecise:
          voxel_size = 0.01 # means 5mm for this dataset
          source_down, source_fpfh = self.preprocess_point_cloud(real_pcl, voxel_size)
          target_down, target_fpfh = self.preprocess_point_cloud(model_pcl,voxel_size)
          result_ransac2 = self.execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
          print(result_ransac2)
          #self.draw_registration_result(source_down, target_down, result_ransac.transformation)
        
        # 2/ local registration - ICP
        # http://www.open3d.org/docs/release/tutorial/Basic/icp_registration.html#Point-to-point-ICP
        doLocal = False
        if doLocal:
          print("Apply point-to-point ICP")
          start = time()
          reg_p2p = o3d.registration.registration_icp(
            source=model_pcl, 
            target=real_pcl, 
            max_correspondence_distance=0.01, 
            #init=result_ransac.transformation, #TODO bundle the TF from locator to SegmentedPointcloud and a) transform model_pcl to the real_pcl's close location; or b) provide the init as 4x4 float64 initial transform estimation (better?)
            estimation_method=o3d.registration.TransformationEstimationPointToPoint())
          end = time()
          print(reg_p2p)
          self.get_logger().info("ICP Matching pcl {}  to \"{}\" (mask confidence {}) with fit success: {} in {} sec.".format(real_pcl, msg.label, msg.confidence, reg_p2p.fitness, (end-start)))




def main():
    rclpy.init()

    matcher = Match3D()

    rclpy.spin(matcher)
    matcher.destroy_node()


if __name__ == "__main__":
    main()
