# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:51:00 2020

@author: rados
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from diamond_detector import FlatDiamondStack
from realsense_device_manager import post_process_depth_frame
import pylab
import hdbscan


markerLength = 0.036  # mm
squareLength = 0.042  # mm
squareMarkerLengthRate = squareLength / markerLength
fps = 30

# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
dictionary = cv2.aruco.Dictionary_create(48, 4, 65536)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, fps)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)

# Start streaming
pipeline.start(config)

pcl_rs = rs.pointcloud()

flatStack = FlatDiamondStack(8)


def draw_axis(img, R, t, K, axisLength=0.02):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32(
        [[axisLength, 0, 0], [0, axisLength, 0], [0, 0, axisLength], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[0].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[2].ravel()), (255, 0, 0), 3)
    return img


hdb = hdbscan.HDBSCAN()
cm = pylab.get_cmap('gist_rainbow')

try:
    streams = pipeline.get_active_profile().get_streams()
    color_stream = next(stream.as_video_stream_profile() for stream in streams if "color" in stream.stream_name().lower())
    depth_stream = next(stream.as_video_stream_profile() for stream in streams if "depth" in stream.stream_name().lower())
    color_intrinsics = color_stream.intrinsics
    color_K = np.r_["0,2,0", color_intrinsics.fy, 0, color_intrinsics.ppx,
                    0, color_intrinsics.fy, color_intrinsics.ppy,
                    0, 0, 1
                    ].reshape((3, 3))
    distCoeffs = np.r_[0, 0, 0, 0, 0]

    align_to = rs.stream.color
    align = rs.align(align_to)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='PCL View', width=1600, height=990, left=50, top=50)
    # vis.register_key_callback(ord("q"), self.esc_cb)
    vis_ctr = vis.get_view_control()
    vis_ctr.change_field_of_view(step=90)
    vis_ctr.set_up(np.r_[0, 0, 0.0])
    vis.get_render_option().show_coordinate_frame = True
    vis.get_render_option().line_width = 10.0
    points = [
        [0, 0, 0],
        [np.pi, 0, 0],
        [0, np.pi, 0],
        [0, 0, np.pi]
    ]

    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    axes.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(axes)
    pcl_corners3D = None
    pcl_verts = None

    # print(f"Color camera matrix: {color_K}")
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        gray = color_image
        markerCorners, markerIds, rejectedPts = cv2.aruco.detectMarkers(gray, dictionary, cameraMatrix=color_K)
        if len(markerCorners) > 0:
            # cv2.aruco.drawDetectedMarkers(gray, markerCorners, markerIds)
            diamondCorners, diamondIds = cv2.aruco.detectCharucoDiamond(
                gray, markerCorners, markerIds, squareMarkerLengthRate, cameraMatrix=color_K)
            # cv2.aruco.drawDetectedDiamonds(gray, diamondCorners, diamondIds)

            if diamondIds is not None and len(diamondIds) > 0:
                for dId, corners in zip(diamondIds, diamondCorners):
                    flatStack.pushDetection(dId, corners)

            foundDiamonds = flatStack.checkDiamonds()
            corners3D = []
            if len(foundDiamonds) >= 3:
                foundDiamondCorners = [fd.corners for fd in foundDiamonds]
                for fdc in foundDiamondCorners:
                    rv, tv, objPoints = cv2.aruco.estimatePoseSingleMarkers(np.reshape(fdc, (1, 4, 2)), squareLength, color_K, distCoeffs)
                    draw_axis(gray, rv, tv, color_K)
                    c_pts3D = np.dot(cv2.Rodrigues(rv)[0], objPoints.reshape((3, 4))) + tv.reshape(3, 1)
                    corners3D.append(c_pts3D.T)

                corners3D = np.stack(corners3D).reshape(-1, 3)

                # depth -> pcl
                mapped_frame, color_source = color_frame, color_image
                processed_depth = post_process_depth_frame(depth_frame, decimation_magnitude=2).as_depth_frame()
                points = pcl_rs.calculate(processed_depth)
                pcl_rs.map_to(mapped_frame)

                # Pointcloud data to arrays
                v, t = points.get_vertices(), points.get_texture_coordinates()
                # verts are the actual points in the point cloud
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
                # texcoords are just for rendering (to get the correct color)
                texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

                if pcl_corners3D is not None:
                    vis.remove_geometry(pcl_corners3D)
                # clusters = hdb.fit_predict(verts)
                # n_clusters = len(np.unique(clusters))
                # if n_clusters > 1:
                #     color_map = list(cm(1. * i / n_clusters) for i in range(n_clusters))
                #     colors = np.reshape([color_map[c][:3] for c in clusters], (-1, 3)).T
                #     draw_3d(verts, colors)
                if pcl_verts is not None:
                    vis.remove_geometry(pcl_verts)

                vis_ctr.set_lookat(corners3D.mean(axis=0))

                pcl_corners3D = o3d.geometry.PointCloud()
                pcl_corners3D.points = o3d.utility.Vector3dVector(corners3D)
                # pcl_corners3D.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(pcl_corners3D)

                pcl_verts = o3d.geometry.PointCloud()
                pcl_verts.points = o3d.utility.Vector3dVector(verts)
                # pcl_corners3D.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(pcl_verts)

                vis.poll_events()
                vis.update_renderer()

        # Stack both images horizontally
        # images = np.hstack((gray, depth_colormap))

        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', gray)
        # cv2.waitKey(1)

        # if len(res[0]) > 0:
        #     res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        #     if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 3 == 0:
        #         allCorners.append(res2[1])
        #         allIds.append(res2[2])

        #     cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

        # cv2.imshow('frame', gray)

        # decimator += 1

        # try:
        #     cal = cv2.aruco.calibrateCameraCharuco(
        #         allCorners, allIds, board, imsize, None, None)
        # except:
        #     cap.release()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()
