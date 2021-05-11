import rclpy #add to package.xml deps
from rclpy.node import Node
from ros2param.api import call_get_parameters

import sensor_msgs
from std_msgs import TransformStamped
from crow_ontology.crowracle_client import CrowtologyClient
from crow_vision_ros2.filters import CameraPoser

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

import cv2
import cv_bridge
import numpy as np
from scipy.linalg import lstsq

import pkg_resources
import time

qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

class MarkerDetector(Node):
    """
    ROS2 node for marker detection.

    TODO
    """
    
    def __init__(self, node_name='marker_detector'):
        super().__init__('marker_detector')
        self.crowracle = CrowtologyClient(node=self)
        self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]
        while len(self.cameras) == 0: #wait for cams to come online
            self.get_logger().warn("No cams detected, waiting 2s.")
            time.sleep(2)
            self.image_topics, self.cameras, self.camera_instrinsics, self.camera_frames = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_namespaces", "camera_intrinsics", "camera_frames"]).values]
        for (topic, cam, intrinsic) in zip(self.image_topics, self.cameras, self.camera_instrinsics):
            # create INput listener with raw images
            listener = self.create_subscription(msg_type=sensor_msgs.msg.Image,
                                                topic=topic,
                                                # we're using the lambda here to pass additional(topic) arg to the listner. Which then calls a different Publisher for relevant topic.
                                                callback=lambda msg, cam=cam, intrinsic=intrinsic: self.input_callback(msg, cam, intrinsic),
                                                qos_profile=1) #the listener QoS has to be =1, "keep last only".
            self.get_logger().info('Input listener created on topic: "%s"' % topic)
        self.bridge = cv_bridge.CvBridge()
        self.detect_markers_flag = False

    def nlp_callback(self, msg):
        marker_group_info = self.crowracle.getMarkerGroupProps(msg.group_name)
        self.aruco_dict = cv2.aruco.Dictionary_create(marker_group_info['dict_num'], marker_group_info['size'], marker_group_info['seed'])
        self.storage_name = msg.storage_name
        self.marker_group_ids = marker_group_info['ids']
        self.detect_markers_flag = True

    def image_callback(self, msg, cam, intrinsic):
        if self.detect_markers_flag:
            intrinsic = json.loads(intrinsic)
            image = self.bridge.imgmsg_to_cv2(msg)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color_K = intrinsic['camera_matrix']
            distCoeffs = intrinsic['distortion_coefficients']
            
            markerCorners, markerIds, rejectedPts = cv2.aruco.detectMarkers(image, self.aruco_dict, cameraMatrix=color_K)
            
            cameraMarkers = CameraPoser(cam)
            if len(markerCorners) > 0:
                for m_idx, (markerId, mrkCorners) in enumerate(zip(markerIds, markerCorners)):
                    if markerId in self.marker_group_ids:
                        cameraMarkers.updateMarker(markerId, mrkCorners)

                markerCorners, markerIds = cameraMarkers.getMarkers()

            if len(markerCorners) > 2 and cameraMarkers.markersReady:
                try:
                    #img_out = cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)
                    rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(np.reshape(markerCorners, (-1, 4, 2)), self.squareLength, color_K, distCoeffs)
                    rmat = cv2.Rodrigues(rvec)[0]
                    quat = tf3.quaternions.mat2quat(rmat)
                    #img_out = cv2.aruco.drawAxis(image, color_K, distCoeffs, rvec, tvec, 0.1)
                    print('rvec', rvec)
                    print('rmat', rmat)
                    print('quat', quat)
                    
                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = self.get_clock().now().to_msg()
                    tf_msg.header.frame_id = "world"
                    tf_msg.child_frame_id = optical_frame
                    tf_msg.transform.translation = make_vector3(tvec)
                    tf_msg.transform.rotation = make_quaternion(quat, order="wxyz")

                except Exception as e:
                    print(e)
                    # pass
        self.detect_markers_flag = False

    def markers_callback(self, marker_message_points, marker_message_name):
        points = [[1., 0., 2.001], [0.0000001, 0.0000001, 0.0000001], [1., 1., 2.], [0., 1., 0.], [2., 0., 4.], [2., 1., 4.]] # = marker_message_points
        height = 2
        name = 'storage1' # = marker_message_name

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
    
    marker_detector.markers_callback([],'')

    rclpy.spin(marker_detector)
    marker_detector.destroy_node()

if __name__ == "__main__":
    main()


