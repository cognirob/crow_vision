import rclpy #add to package.xml deps
from rclpy.node import Node
from ros2param.api import call_get_parameters

import sensor_msgs
#from std_msgs import TransformStamped
from crow_ontology.crowracle_client import CrowtologyClient
from crow_vision_ros2.filters import CameraPoser
from crow_msgs.msg import MarkerMsg
from crow_control.utils import ParamClient

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.linalg import lstsq
import json
import cv2
import cv_bridge
import numpy as np
import transforms3d as tf3
import pkg_resources
import time
import subprocess

qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

global_2_robot = np.array(
    [0.7071068, 0.7071068, 0, 0,
     -0.7071068, 0.7071068, 0, 0,
     0, 0, 1, 0.233,
     0, 0, 0, 1]
).reshape(4, 4)
robot_2_global = np.linalg.inv(global_2_robot)
realsense_2_robot = np.array(  # new from 16.7. 2021
    [5.2478784e-01,  4.7426718e-01, -7.0687222e-01,  1.3974695,
     8.4990799e-01, -3.3824810e-01,  4.0403539e-01, -0.32303824,
    -4.7477469e-02, -8.1280923e-01, -5.8059198e-01,  0.65096106,
    0,  0,  0,  1]
 ).reshape(4, 4)


class MarkerDetector(Node):
    """
    ROS2 node for marker detection.

    TODO
    """
    VISUALIZE = False

    def __init__(self, node_name='marker_detector'):
        super().__init__('marker_detector')
        self.crowracle = CrowtologyClient(node=self)
        self.pclient = ParamClient()
        self.pclient.declare("robot_done", True)
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_extrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_extrinsics", "camera_frames"]).values]
        while len(self.cameras) == 0: #wait for cams to come online
            self.get_logger().warn("No cams detected, waiting 2s.")
            time.sleep(2)
            self.image_topics, self.cameras, self.camera_instrinsics, self.camera_extrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_extrinsics", "camera_frames"]).values]
        self.detect_markers_flag = {}
        self.define_flag = None
        for (topic, cam, intrinsic, extrinsic) in zip(self.image_topics, self.cameras, self.camera_instrinsics, self.camera_extrinsics):
            # create INput listener with raw images
            listener = self.create_subscription(msg_type=sensor_msgs.msg.Image,
                                                topic=topic,
                                                # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                                callback=lambda msg, cam=cam, intrinsic=intrinsic, extrinsic=extrinsic: self.image_callback(msg, cam, intrinsic, extrinsic),
                                                qos_profile=1) #the listener QoS has to be =1, "keep last only".
            self.get_logger().info('Input listener created on topic: "%s"' % topic)
            self.detect_markers_flag[cam] = False
        topics = ["/new_storage", "/new_position"]
        for topic in topics:
            listener = self.create_subscription(msg_type=MarkerMsg,
                                                topic=topic,
                                                # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                                callback=lambda msg: self.control_callback(msg, topic),
                                                qos_profile=1) #the listener QoS has to be =1, "keep last only".
            self.get_logger().info('Input listener created on topic: "%s"' % topic)
        self.bridge = cv_bridge.CvBridge()
        self.pose_markers = []

    def control_callback(self, msg, topic):
        if 'new_storage' in topic:
            self.define_flag = 'storage'
        elif 'new_position' in topic:
            self.define_flag = 'position'
        else:
            self.define_flag = None
            return
        marker_group_info = self.crowracle.getMarkerGroupProps(msg.group_name, language='CZ')
        #marker_group_info = self.crowracle.getMarkerGroupProps('blue')
        if marker_group_info.get('dict_num', False):
            self.aruco_dict = cv2.aruco.Dictionary_create(marker_group_info['dict_num'], marker_group_info['size'], marker_group_info['seed'])
            self.define_name = msg.define_name
            #self.define_name = 'blue_strorage'
            self.marker_group_ids = marker_group_info['id']
            self.square_length = marker_group_info['square_len']
            for cam in self.cameras:
                self.detect_markers_flag[cam] = True

    def image_callback(self, msg, cam, intrinsic, extrinsic):
        if self.detect_markers_flag[cam]:
            intrinsic = json.loads(intrinsic)
            extrinsic = json.loads(extrinsic)
            image = self.bridge.imgmsg_to_cv2(msg)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color_K = np.asarray(intrinsic['camera_matrix'])
            distCoeffs = np.asarray(intrinsic['distortion_coefficients'])
            ctg_tf_mat = (robot_2_global @ realsense_2_robot @ extrinsic["ctg_tf"]).astype(np.float32)

            markerCorners, markerIds, rejectedPts = cv2.aruco.detectMarkers(image, self.aruco_dict, cameraMatrix=color_K)
            markerCornersKeep = []
            if len(markerCorners) > 0:
                for (mrkCrn, mrkId) in zip(markerCorners, markerIds):
                    if mrkId in self.marker_group_ids:
                        markerCornersKeep.append(mrkCrn)

            # filter for stabilization - not tested, but not needed now
            # cameraMarkers = CameraPoser(cam)
            # if len(markerCorners) > 0:
            #     for m_idx, (markerId, mrkCorners) in enumerate(zip(markerIds, markerCorners)):
            #         if markerId in self.marker_group_ids:
            #             cameraMarkers.updateMarker(markerId, mrkCorners)

            #     markerCorners, markerIds = cameraMarkers.getMarkers()

            if len(markerCornersKeep) > 0: # and cameraMarkers.markersReady:
                try:
                    if self.VISUALIZE:
                        img_out1 = cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)
                        cv2.imshow(f'marker detections {cam}', img_out1)
                        cv2.waitKey(1)
                    rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(np.reshape(markerCornersKeep, (-1, 4, 2)), self.square_length, color_K, distCoeffs)
                    rmat = [cv2.Rodrigues(rvec_i)[0] for rvec_i in rvec]
                    mtc_tf_mat = [tf3.affines.compose(tvec_i[0], rmat_i, np.ones(3)) for (tvec_i, rmat_i) in zip(tvec, rmat)]
                    for mtc_tf_mat_i in mtc_tf_mat:
                        pose = np.matmul(np.matmul(ctg_tf_mat, mtc_tf_mat_i), np.array([0, 0, 0, 1]))
                        self.pose_markers.append(pose[:-1])
                    if self.VISUALIZE:
                        for i in range(len(rvec)):
                            img_out2 = cv2.aruco.drawAxis(image, color_K, distCoeffs, rvec[i], tvec[i], 0.1)
                        cv2.imshow(f'group marker poses, {cam}', img_out2)
                        cv2.waitKey(1)
                except Exception as e:
                    print(e)
                    # pass
            self.detect_markers_flag[cam] = False
            if True not in self.detect_markers_flag.values():
                if len(self.pose_markers) > 0:
                    self.merge_markers(self.pose_markers)
                else:
                    self.get_logger().info('No markers detected, cannot create ' + self.define_flag)
                    self.pclient.robot_done = True

    def merge_markers(self, poses):
        #poses = [np.asarray([0, 0, 0]), np.asarray([1, 1, 1]), np.asarray([2, 2, 2]), np.asarray([3, 3, 3]), np.asarray([0.001, 0.001, 0])]
        if self.define_flag == 'storage' and len(marker_poses) > 2:
            poses_linkage = linkage(poses, method='complete', metric='euclidean')
            clusters = fcluster(poses_linkage, self.square_length, criterion='distance')
            marker_poses = []
            for i in range(1, max(clusters)+1):
                marker_poses.append(np.mean(np.asarray(poses)[np.where(clusters==i)], axis=0).tolist())
            self.process_storage_markers(marker_poses, self.define_name)
        elif self.define_flag == 'position':
            self.process_position_markers(poses, self.define_name)
        else:
            self.get_logger().info('Not enough markers detected, cannot create ' + self.define_flag)
            self.pclient.robot_done = True

    def process_storage_markers(self, points, name, height = 2):
        fit = self.get_plane_from_points(points)
        polygon3d = self.project_points_to_plane(points, fit)
        polygon2d = self.project_3d_to_2d(polygon3d)
        centroid = self.get_centroid_from_points(points)
        area = self.get_area2d(polygon2d)
        volume = self.get_volume(polygon2d, height)
        moved_points = self.move_points_by(polygon3d, fit, height)
        polyhedron = polygon3d.tolist()
        for point in moved_points:
            polyhedron.append(point)

        self.crowracle.add_storage_space(name, polygon3d, polyhedron, area, volume, centroid)
        self.pclient.robot_done = True

    def process_position_markers(self, points, name):
        centroid = self.get_centroid_from_points(points)

        self.crowracle.add_position(name, centroid)
        self.pclient.robot_done = True

    def get_plane_from_points(self, points):
        """Fit plane equation ax + by + cz = d to measured points and find normal vector
        Args:
            points (list of lists): location of markerks

        Returns:
            normal (array): normal vector of fitted plane
        """
        # [x1 y1 z1]*[a/d] = [1]
        # [x2 y2 z2] [b/d]   [1]
        # [x3 y3 z3] [c/d]   [1]
        # [x4 y4 z4]         [1]
        A = np.matrix(points)
        b = np.ones((len(points),1))
        fit, residual, rnk, s = lstsq(A, b)
        fit.reshape(3,)
        fit = np.vstack((fit, -1))
        normal = fit/np.linalg.norm(fit)
        return normal

    def project_points_to_plane(self, points, fit):
        """Project points p to the fitted plane p' = p - (n ⋅ p + d) * n
        Args:
            points (list of lists): location of markers (p)
            fit (array): (a,b,c,d) parameters of the plane (normal n=(a,b,c))
        Returns:
            new_points (array): projected points (p')
        """
        fit = fit.reshape(4,)
        new_points = []
        for point in points:
            new_point = point - (np.dot(fit[:-1], point) + fit[-1]) * fit[:-1]
            new_points.append(new_point)
        return np.asarray(new_points)

    def project_3d_to_2d(self, points):
        """Project 3d points in plane to 2d (parameterized by the plane)
        Args:
            points (list of lists): 3d location of markers projected to fitted plane
        Returns:
            local_points (array): projected 2d points
        """
        loc0 = points[0]                       # local origin
        locx = points[1] - loc0                # local X axis
        normal = np.cross(locx, points[2] - loc0) # vector orthogonal to polygon plane
        locy = np.cross(normal, locx)             # local Y axis
        locx /= np.linalg.norm(locx)
        locy /= np.linalg.norm(locy)
        local_points = [(np.dot(p - loc0, locx),  # local X coordinate
                         np.dot(p - loc0, locy))  # local Y coordinate
                                    for p in points]
        return np.asarray(local_points)

    def move_points_by(self, points, fit, distance):
        """Extrude 3d points in plane (polygon) to new 3d points (polygon) by given distance
        Args:
            points (list of lists): 3d location of markers projected to fitted plane
            fit (array): (a,b,c,d) parameters of the plane (normal n=(a,b,c))
            distance (float): distance between original and new polygon (height of extruded polyhedron)
        Returns:
            new_points (list of lists): extruded 3d points
        """
        new_points = []
        fit.reshape(4,)
        move = fit[:-1,0] * distance
        for point in points:
            new_points.append((point + move).tolist())
        return new_points

    def get_centroid_from_points(self, points):
        """Get mean of given 3d points
        Args:
            points (list of lists): 3d location of markers
        Returns:
            centroid (array): mean of points
        """
        centroid = np.mean(points, axis=0)
        # points = pyny.Polygon.make_ccw(np.asarray(points))
        # polygon = pyny.Polygon(np.asarray(points))
        # polygon.plot('b')
        # centroid = polygon.get_centroid()
        return centroid

    def get_area(self, corners):
        """Get area of given corners (3d or 2d points) using cross products
        Args:
            corners (list of lists): 3d or 2d points
        Returns:
            area (float): calculated area
        """
        #corners = self.sort_polygon(corners)
        n = len(corners)
        area = []
        for i in range(1,n-1):
            area.append(0.5 * np.cross(corners[i] - corners[0], corners[(i+1)%n] - corners[0]))
        area = np.asarray(area).sum(axis = 0)
        area = np.linalg.norm(area)
        return area

    def sort_polygon(self, corners):
        """Sort corners (2d points) of polygon to be in ccw order
        Args:
            corners (list of lists): 2d points
        Returns:
            sorted corners (list of lists): 2d points
        """
        n = len(corners)
        cx = float(sum(x[0] for x in corners)) / n
        cy = float(sum(x[1] for x in corners)) / n
        cornersWithAngles = []
        for x in corners:
            an = (np.arctan2(x[1] - cy, x[0] - cx) + 2.0 * np.pi) % (2.0 * np.pi)
            cornersWithAngles.append((x[0], x[1], an))
        cornersWithAngles.sort(key = lambda tup: tup[2])
        return [np.asarray(corner[:-1]) for corner in cornersWithAngles]

    def get_area2d(self, corners):
        """Get area of given corners (2d points) using shoelace formula
        Args:
            corners (list of lists): 2d points
        Returns:
            area (float): calculated area
        """
        corners = self.sort_polygon(corners)
        n = len(corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0
        return area

    def get_volume(self, points2d, height):
        """Get volume of polyhedron given corners (2d points) of the base and its height
        Args:
            points2d (list of lists): 2d points defining the base polygon
        Returns:
            volume (float): calculated volume
        """
        area = self.get_area2d(points2d)
        volume = area * height
        return volume

def main():
    rclpy.init()
    time.sleep(1)
    marker_detector = MarkerDetector()

    rclpy.spin(marker_detector)
    marker_detector.destroy_node()

if __name__ == "__main__":
    main()


