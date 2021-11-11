import queue as pyqueue
import datetime
from scipy import interpolate
from scipy.optimize import linear_sum_assignment
import numpy as np

# ROS2
from crow_vision_ros2.tracker.tracker_base import get_vector_length, random_alpha_numeric, Dimensions, Position, Color, ObjectsDistancePair
from crow_vision_ros2.tracker.tracker_config import DEFAULT_ALPHA_NUMERIC_LENGTH, TRAJECTORY_MEMORY_SIZE_SECONDS, WRIST_OBJECT_CLIP_DISTANCE_LIMIT

class Trajectory:
    """
    Class for storing information about trajectory of an object (AvatarObject)
    ARGS:
    - trajectory_memory_size_seconds: <float> How long should the trajectory be
    """

    def __init__(self, trajectory_memory_size_seconds):
        self.queue = pyqueue.Queue()
        self.trajectory_memory_size_seconds = trajectory_memory_size_seconds

    def pop_old_points(self):
        """
        Remove trajectory point older than self.trajectory_memory_size_seconds
        """

        time_now = datetime.datetime.now()

        while True:
            if self.queue.qsize(): # >0
                if (time_now - self.queue.queue[0].origin_date).total_seconds() > self.trajectory_memory_size_seconds:
                    # Remove the element, else break
                    self.queue.get()
                else:
                    break
            else:
                break

    def add_trajectory_point(self, position):
        """
        Add new position to the trajectory.
        ARGS:
        - position: <Class:Position>
        """

        self.pop_old_points()
        self.queue.put(position)
        return

    def get_chronological_position_list(self):
        """
        Get trajectory points with respect of time, older first?
        RETURN:
        - <list:Position>
        """

        return list(self.queue.queue)

    def get_average_trajectory_position(self):
        """
        Return average 3D position from trajectory.
        RETURN:
        - <list:Position>
        """

        position_data = self.get_chronological_position_list()
        num_pts = len(position_data)
        x_true, y_true, z_true = ([],[],[])
        for position in position_data:
            x_true.append(position.x)
            y_true.append(position.y)
            z_true.append(position.z)

        return Position(x=(sum(x_true)/len(x_true)), y=(sum(y_true)/len(y_true)), z=(sum(z_true)/len(z_true)) )

    def distance_3d(self, xyz_tup, x0y0z0_tup):
        """
        3D distance from point and a line
        RETURN:
        - <float>
        """

        dx = xyz_tup[0] - x0y0z0_tup[0]
        dy = xyz_tup[1] - x0y0z0_tup[1]
        dz = xyz_tup[2] - x0y0z0_tup[2]
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        return d

    def min_distance(self, x, y, z, P, precision=5):
        """
        Compute minimum/a distance/s between
        a point P[x0,y0,z0] and a curve (x,y,z)
        rounded at `precision`.
        ARGS:
        - x: <list:float>
        - y: <list:float>
        - z: <list:float>
        - P: <tuple:[float,float,float]>
        - precision: <int>
        RETURN:
        - (<int>,<float>) min indexes and distances array.
        """
        # compute distance
        d = self.distance_3d( xyz_tup=(x, y, z), x0y0z0_tup=(P[0], P[1], P[2]))
        d = np.round(d, precision)

        print(f"trajectory: {d}")
        # find the minima
        glob_min_idxs = np.argwhere(d==np.min(d)).ravel()
        
        print(f"trajectory_min_idx: {glob_min_idxs}")
        return glob_min_idxs, d

    def get_spline_xyz_data(self):
        """
        RETURN:
        - (<list:float>,<list:float>,<list:float>)
        """

        position_data = self.get_chronological_position_list()
        num_pts = len(position_data)
        x_true, y_true, z_true = ([],[],[])
        for position in position_data:
            x_true.append(position.x)
            y_true.append(position.y)
            z_true.append(position.z)

        tck, u = interpolate.splprep([x_true,y_true,z_true], s=1)
        x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
        u_fine = np.linspace(0,1, 3*num_pts)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        return (x_fine,y_fine,z_fine)

    def plot_trajectory_minimal_distance(self, object):
        """
        Get the position of an object and plot its distance to the trajectory
        spline.
        ARGS:
        - object: <Class:TrackedObject>
        """

        x_fine, y_fine, z_fine = self.get_spline_xyz_data()
        # a point to calculate distance to
        P = object.centroid_position.get_tuple()
        # 3d plot
        fig = plt.figure()

        min_idx, d = self.min_distance(x_fine, y_fine, z_fine, P)

        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        ax.plot(x_fine, y_fine, z_fine)
        ax.plot(P[0], P[1], P[2], 'or')
        ax.plot(x_fine[min_idx], y_fine[min_idx], z_fine[min_idx], 'ok')
        for idx in min_idx:
            ax.plot(
                [P[0], x_fine[idx]],
                [P[1], y_fine[idx]],
                [P[2], z_fine[idx]],
                'k--'
            )
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        print("distance:", d[min_idx])

    def get_trajectory_minimal_distance(self, avatar_object, object):
        """
        Return minimal distance from trajectory to the object.
        ARGS:
        - avatar_object: <Class:AvatarObject>
        - object: <Class:TrackedObject>
        RETURN:
        - <float>
        """

        if self.queue.qsize() > 3:
            x_fine, y_fine, z_fine = self.get_spline_xyz_data()
            # a point to calculate distance to
            P = object.centroid_position.get_tuple()

            min_idx, d = self.min_distance(x_fine, y_fine, z_fine, P)
            return d[min_idx][0]
        else:
            return self.distance_3d(xyz_tup=avatar_object.centroid_position.get_tuple() , x0y0z0_tup=object.centroid_position.get_tuple())

    #
    def get_spline_trajectory_positions(self):
        """
        Return list of positions for interpolated trajectory.
        RETURN:
        - <list:Class:Postion>
        """

        x_fine, y_fine, z_fine = self.get_spline_xyz_data()
        points = []
        for i in range(len(x_fine)):
            points.append(Position(x=x_fine[i], y=y_fine[i], z=z_fine[i]))
        return points
