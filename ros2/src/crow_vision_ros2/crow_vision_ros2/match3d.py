import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters

#PointCloud2
from crow_msgs.msg import DetectionMask, SegmentedPointcloud
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
from crow_vision_ros2.utils import make_vector3, ftl_pcl2numpy, ftl_numpy2pcl

#TF
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import TransformStamped

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import numpy as np

import pkg_resources
import time
import copy


class Match3D(Node):
    """
    Match our objects (from .STL) to the segmented point-cloud ("<cam>/detections/segmented_pointcloud") 
    based on segmentation from 2D RGB masks (mask_msg) from YOLACT. 
    
    Publish the complete matched object as SegmentedPointcloud.
    """
    
    def load_models(self,
                    list_path_stl=["/home/imitrob/crow_simulation/crow_simulation/envs/objects/crow/stl/cube_holes.stl"],
                    list_labels=["cube_holes"]):
        """
        Load our original model files (.stl) as pointclouds.
        @arg: list_path_stl : list of string paths to stl files, ie.: ["/home/data/models/chair.stl", "/hammer.stl"]
        @arg: list_labels: corresponding list of object labels: ["chair", "hammer"]
        @arg voxel_size: corresponding size of voxel in real world (in mm), default 1voxel=1mm

        @return dict matching "label" -> o3d.PointCloud
        """
        objects = {} #map class -> loaded pcl

        assert len(list_path_stl) == len(list_labels)
        self.get_logger().info("Loading models, please wait... (Precisions: orig: {}, down: {}, fine: {} [mm]".format(
            self.voxel_size*1000, self.approx_precision*1000, self.fine_precision*1000))

        for i, (cls, path) in enumerate(zip(list_labels, list_path_stl)):
          objects[cls] = {}

          #1. create Mesh
          mesh = o3d.io.read_triangle_mesh(path) #o3d is slow, but here it's only once in init, so it's OK.
          
          #2. o3d.PointCloud from mesh
          orig_pcd = o3d.geometry.PointCloud()
          orig_pcd.points = mesh.vertices
          orig_pcd.colors = mesh.vertex_colors
          orig_pcd.normals = mesh.vertex_normals

          orig_pcd, orig_fpfh = self.preprocess_point_cloud(orig_pcd, voxel_size=self.voxel_size, min_support=self.min_support, label="proto"+cls)
          objects[cls]["orig"] = orig_pcd
          objects[cls]["orig_fpfh"] = orig_fpfh #features for RANSAC

          self.get_logger().info("Loading '{}' : mesh: {}\tPointCloud: {}".format(cls, mesh, orig_pcd))
        return objects



    def __init__(self, node_name="match3d", voxel_size=0.001, approx_precision=0.005, fine_precision=0.001, min_support=200): #TODO make a) finer STLs (for some objects), b) merge pcl to get bigger clouds, so support can be much higher (~1000 ideally)
        """
        @arg voxel_size: size [in m] of voxels used for sampling from our ground-truth models. Default 0.001 = 1mm in real world precision. 
            Defines overall precision-possibilities of this module. 
        @arg approx_precision: precision in voxel_size (= in m) for global registration method (RANSAC). This is used to get a rough estimate of the object's
            position. Default 0.050 is 5cm. Use 0.0 to disable this method.
        @arg fine_precision: same as approx_precision but for local registration method (ICP). Default 0.005 = 5mm, set to 0.0 to disable.
        @arg min_support: min number of downsampled points in pointclouds (both "proto" models and "incoming" segmented pcl). Also used as threshold for acceptance in registration's
            result (the higher the support, the more sound the match is).
        """
        super().__init__(node_name)
        self.approx_precision = approx_precision
        self.voxel_size = voxel_size
        assert voxel_size > 0
        if approx_precision > 0:
            assert voxel_size < approx_precision
        if fine_precision > 0:
            assert voxel_size <= fine_precision
        self.fine_precision = fine_precision
        self.min_support = min_support

        self.cameras = []
        while(len(self.cameras) == 0):
            try:
                self.cameras , self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["camera_namespaces", "camera_frames"]).values]
            except:
                self.get_logger().error("getting cameras failed. Retrying in 2s")
                time.sleep(2)
        assert len(self.cameras) > 0

        self.seg_pcl_topics = [cam + "/" + "detections/segmented_pointcloud" for cam in self.cameras] #input segmented pcl data

        #TODO merge (segmented) pcls before this node

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) #publishing transformations (ie centroids etc)
        self.pubMatched = {} #dict for publishers mapped by cam key
        self.pubMatchedDebug = {} #ditto, only publishes PointCloud2 for rviz. Optional, debug only.
        for i, (cam, pclTopic) in enumerate(zip(self.cameras, self.seg_pcl_topics)):
            self.get_logger().info("Created subscriber for segmented_pcl \"{}\"".format(pclTopic))
            self.create_subscription(SegmentedPointcloud, pclTopic, lambda pcl_msg, cam=cam: self.detection_callback(pcl_msg, cam), qos_profile=qos)
            #create output publisher - publish pointcloud of the model (complete 3D, unlike the segmented pcl from camera) transformed to real-world position
            self.pubMatched[cam] = self.create_publisher(SegmentedPointcloud, cam+"/detections/matched_pointcloud", qos)
            self.pubMatchedDebug[cam] = self.create_publisher(PointCloud2, cam+"/detections/matched_pointcloud_debug", qos)


        # map str:label -> o3d.PointCloud model
        #MODEL_PATH=str(pkg_resources.resource_filename("crow_simulation", 'envs/objects/crow/stl/'))
        MODEL_PATH="/home/imitrob/crow_simulation/crow_simulation/envs/objects/crow/stl/"
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
       #     MODEL_PATH+"peg_simple.stl" #has too few points (36), making our min_support useless
            ], 
            list_labels=["car_roof", "pliers", "cube_holes", "screw_round", "ex_bucket", "screwdriver", "hammer", 
                "sphere_holes", "wafer", "nut", "wheel", "peg_screw", "wrench"
                #, "peg_simple"
            ] 
            )


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


    def preprocess_point_cloud(self, pcd, voxel_size, min_support, label="n/a"): 
        """
        @arg pcd: o3d.PointCloud data
        @arg voxel_size: size of voxel meant for downsampling (->precision)
        @arg min_support: min threshold on number of points after the downsampling (->trustfullness of the match)
        """
        #print(":: Downsample with a voxel size %.3f." % voxel_size)
        assert voxel_size >= self.voxel_size

        pcd_down = copy.deepcopy(pcd)
        downsampled = pcd.voxel_down_sample(voxel_size)
        assert len(downsampled.points) > min_support, "Did not manage to downsample '{}' -  {} to voxel_size {} with support {}, only has {} support points.".format(
                label, pcd, voxel_size, min_support, len(downsampled.points))
        pcd_down = downsampled #actually downsampled data

        radius_normal = voxel_size * 2 #TODO should we tune these *X numbers?
        #print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
        self.get_logger().info("Downsampled {} voxel {} from {} to {}".format(label, voxel_size, pcd, pcd_down))
        return pcd_down, pcd_fpfh


    def execute_fast_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, match_tolerance):
        assert match_tolerance > self.voxel_size
        #print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
        result = o3d.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=match_tolerance))
        return result




    def detection_callback(self, msg, camera):

        # model (as a pointcloud) to be matched to a real-world position
        model_pcl = None
        try:
            model_pcl = self.objects[msg.label]["orig"]
        except:
            self.get_logger().error("Unknown model for detected label {}. Skipping.".format(msg.label))
            return

        
        # pointcloud from msg PointCloud2 -> numpy -> o3d.PointCloud
        start = time.time()
        #pcd = np.array(msg.pcl.data).view(np.float32).reshape(-1, 3) # 3 as we have x,y,z only in the pcl
        pcd, point_rgb, rgb_raw = ftl_pcl2numpy(msg.pcl)
        real_pcl = o3d.geometry.PointCloud()
        #real_pcl.points = o3d.utility.Vector3dVector(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg.pcl)) #using ros2_numpy 
        real_pcl.points = o3d.utility.Vector3dVector(pcd)
        # downsample to self.voxel_size resolution
        real_pcl, _ = self.preprocess_point_cloud(real_pcl, voxel_size=self.voxel_size, min_support=0, label="incoming-"+msg.label)
        end = time.time()
        if len(real_pcl.points) < self.min_support:
            self.get_logger().warn("Skipping [{}] incoming pointcloud {} as it has too few points (min_support={}).".format(
                msg.label, real_pcl, self.min_support))
            return
        #self.get_logger().info("Convert to o3d: {}".format(end-start)) # ~0.003s


        # 0/ compute (approx) initial transform; just moves the clouds "nearby"
        transform_initial = np.identity(4) #initial affine transform, moves clouds "close together" (by mean xyz of the pcls)
        fromm = np.asarray(model_pcl.points)
        to    = np.asarray(real_pcl.points)
        fromm_xyz = np.mean(fromm, axis=0, keepdims=True) #FIXME use median
        to_xyz = np.mean(to, axis=0, keepdims=True)
        move = (to_xyz - fromm_xyz)[0]
        transform_initial[0:3,3] = move #write the "move"=translation (x,y,z) to the translation part of the affine matrix
        #print("BEFORE: Mean proto: {} \tMean pcl: {}".format(np.mean(fromm, axis=0, keepdims=True), np.mean(to, axis=0, keepdims=True)))
        model_pcl.transform(transform_initial) #move the model approx where the real pcl is.
        #print("AFTER: Mean proto: {} \tMean pcl: {}".format(np.mean(fromm, axis=0, keepdims=True), np.mean(to, axis=0, keepdims=True)))
        assert np.abs(np.mean(fromm, axis=0, keepdims=True) - np.mean(to, axis=0, keepdims=True)).sum() < 0.0001, "Initial transform should move cloud centers together!"


        # 1/ (optional) fit global RANSAC
        result = None
        matched = False

        if self.approx_precision > 0:
          start = time.time()
          # compute pcl & features for incoming segmented pcl (this is the target position)
          target_down, target_fpfh = self.preprocess_point_cloud(real_pcl, voxel_size=self.approx_precision, min_support=self.min_support, label="real-approx-"+msg.label)
          # a) re-compute pcl & features for the model (which moved in step 0)
          source_down, source_fpfh = self.preprocess_point_cloud(model_pcl,voxel_size=self.approx_precision, min_support=self.min_support, label="proto-approx-"+msg.label)
          
          # b) use cached downsampled model & features #TODO can we do that? or model_fpfh depends on absolute coords, which changed in step 0/ default transform?
          #source_down = self.objects[msg.label]["down"]
          #source_fpfh = self.objects[msg.label]["down_fpfh"]
          result = self.execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, self.approx_precision*10)
          end = time.time()

          #apply the transform only if: 1) cprrespondence_set is atleast 50% of the segmented pcl (real_pcl) & fitness > 0.1
          #print("diff {}\tlen orig: {}\tlen match: {}".format(float(len(result.correspondence_set) / len(target_down.points)), len(target_down.points), len(result.correspondence_set)))
          applyit = (float(len(result.correspondence_set)) / len(target_down.points)) > 0.25 and len(result.correspondence_set) > self.min_support and result.fitness > 0.01
          self.get_logger().info("RANSAC [{}]: {}\t in {}sec - {}".format(msg.label, result, (end-start),  "APPLIED" if applyit else "SKIPPED"))
          if applyit:
              model_pcl.transform(result.transformation)
              matched = True
              #TODO assert the transform is in the correct direction, ie the model is moving closer. 
          else:
              #unsuccessful registration (why?), skip
              pass #TODO probably should not happen, we should retry global reg. with a larger lookup tolerance?

        
        # 2/ local registration - ICP
        # http://www.open3d.org/docs/release/tutorial/Basic/icp_registration.html#Point-to-point-ICP
        if self.fine_precision > 0:
          start = time.time()
          # compute pcl & features for incoming segmented pcl (this is the target position)
          target_fine, target_fpfh = self.preprocess_point_cloud(real_pcl, voxel_size=self.voxel_size, min_support=self.min_support, label="real-fine-"+msg.label) #TODO if we wanted to alter min_support based on object's label (-> "size"). As hammer "is bigger" than "nut"
          # a) re-compute pcl & features for the model (which moved in step 0)
          source_fine, source_fpfh = self.preprocess_point_cloud(model_pcl,voxel_size=self.voxel_size, min_support=self.min_support, label="proto-fine-"+msg.label)
          # b) use cached downsampled model & features #TODO can we do that? or model_fpfh depends on absolute coords, which changed in step 0/ default transform?
          #source_fine = self.objects[msg.label]["fine"]
          #source_fpfh = self.objects[msg.label]["fine_fpfh"]
          #TODO c) or use the unchanged originals here in the final step? - real_pcl, model_pcl 
          # source_fine = model_pcl
          # target_fine = real_pcl

          result = o3d.registration.registration_icp(
              source= source_fine,
              target= target_fine, 
              max_correspondence_distance=self.fine_precision, #match tolerance in mm
              #init=result_ransac.transformation, #TODO bundle the TF from locator to SegmentedPointcloud and a) transform model_pcl to the real_pcl's close location; or b) provide the init as 4x4 float64 initial transform estimation (better?) #TODO2 not needed now as we move the model ourselves?
              estimation_method=o3d.registration.TransformationEstimationPointToPoint())
          end = time.time()

          #apply the transform only if: 1) correspondence_set is atleast 50% of the segmented pcl (real_pcl) & fitness > 0.1
          applyit = (float(len(result.correspondence_set)) / len(target_fine.points)) > 0.1 and len(result.correspondence_set) > self.min_support and result.fitness > 0.1
          self.get_logger().info("ICP [{}]: {}\t in {}sec - {}".format(msg.label, result, (end-start),  "APPLIED" if applyit else "SKIPPED"))
          if applyit:
              matched = True
              model_pcl.transform(result.transformation)
          else:
              #unsuccessful registration (why?), skip
              pass #TODO probably should not happen, we should retry global reg. with a larger lookup tolerance?

        # 3/ publish the matched complete model (as PointCloud2 moved to the true 3D world position)
        if matched:
            #fill PointCloud2 correctly according to https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0#file-dragon_pointcloud-py-L32
            model_pcd = np.asarray(model_pcl.points).reshape(3, -1).astype(np.float32)
            model = ftl_numpy2pcl(model_pcd, msg.pcl.header, None)
            model.header.stamp = msg.header.stamp
            assert model.header.stamp == msg.header.stamp, "timestamps for segmented_pointcloud & new matched_pointcloud must be synchronized!"

            matched_pcl_msg = SegmentedPointcloud()
            matched_pcl_msg.header = model.header
            matched_pcl_msg.pcl = model
            matched_pcl_msg.label = msg.label
            matched_pcl_confidence = float(result.fitness)

            self.pubMatched[camera].publish(matched_pcl_msg)
            self.get_logger().info("Publishing matched {} with confidence {}.".format(msg.label, result.fitness))
            self.pubMatchedDebug[camera].publish(model) #debug, can be removed

        # 4/ publish TF of centroids
        mean = np.median(pcd.reshape(3,-1).astype(np.float32), axis=1) #TODO when it works, publish center of matched pcl, not the segmented part only
        #self.get_logger().info("Object {}: {} Centroid: {} accuracy: {}".format(msg.label, mean, msg.score))
        assert len(mean) == 3, 'incorrect mean dim'+str(mean.shape)
        self.sendPosition(self.camera_frames[self.cameras.index(camera)], msg.label, msg.header.stamp, mean)


    def sendPosition(self, camera_frame, object_frame, time, xyz):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = time
        tf_msg.header.frame_id = camera_frame
        tf_msg.child_frame_id = object_frame
        tf_msg.transform.translation = make_vector3(xyz)

        self.tf_broadcaster.sendTransform(tf_msg)


def main():
    rclpy.init()

    matcher = Match3D()

    rclpy.spin(matcher)
    matcher.destroy_node()


if __name__ == "__main__":
    main()
