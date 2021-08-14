# ROS2
from crow_vision_ros2.tracker.tracker_base import get_vector_length, random_alpha_numeric, Dimensions, Position, Color, ObjectsDistancePair
from crow_vision_ros2.tracker.tracker_config import DEFAULT_ALPHA_NUMERIC_LENGTH, TRAJECTORY_MEMORY_SIZE_SECONDS, WRIST_OBJECT_CLIP_DISTANCE_LIMIT
from crow_vision_ros2.tracker.tracker_trajectory import Trajectory

class AvatarObject:
    """
    Class object representing  wrists, elbows, shoulders, head.
    ARGS:
    - object_name: <string>
    """

    def __init__(self, object_name):
        self.object_name = object_name
        self.centroid_position = Position(x=0, y=0, z=0)
        self.dimensions = Dimensions(x=0, y=0, z=0)

        self.trajectory_memory = Trajectory(trajectory_memory_size_seconds=TRAJECTORY_MEMORY_SIZE_SECONDS)
        self.trajectory_memory.add_trajectory_point(position=self.centroid_position)

    def update_data(self, centroid_position, dimensions):
        """
        Update position and dimensions of this object.
        ARGS:
        - centroid_position: <Class:Position>
        - dimensions: <Class:Dimensions>
        """

        self.centroid_position = centroid_position
        self.dimensions = dimensions

        self.trajectory_memory.add_trajectory_point(position=centroid_position)

class Avatar:
    """
    Class containing AvatarObject, this class represents a "person" with its
    parts.
    """

    def __init__(self):
        self.avatar_objects = {
            "head": AvatarObject(object_name="head"),
            "leftWrist": AvatarObject(object_name="leftWrist"),
            "rightWrist": AvatarObject(object_name="rightWrist"),
            "leftElbow": AvatarObject(object_name="leftElbow"),
            "rightElbow": AvatarObject(object_name="rightElbow"),
            "leftShoulder": AvatarObject(object_name="leftShoulder"),
            "rightShoulder": AvatarObject(object_name="rightShoulder")}

    def update_avatar_object(self, avatar_object_name, np_position, np_dimensions):
        """
        Update position and dimensions of object with name avatar_object_name.
        ARGS:
        - avatar_object_name: <string>
        - np_position: <numpy_list:[x,y,z]:[float,float,float]>
        - np_dimensions: <numpy_list:[x,y,z]:[float,float,float]>
        """

        position = Position(x=np_position[0][0], y=np_position[0][1], z=np_position[0][2])
        dimensions = Dimensions(x=np_dimensions[0][0], y=np_dimensions[0][1], z=np_dimensions[0][2])

        self.avatar_objects[avatar_object_name].update_data(centroid_position=position, dimensions=dimensions)

    def dump_info(self):
        """
        Prints all the intormation about the Avatar's objects.
        """

        for key in self.avatar_objects:
            a_obj = self.avatar_objects[key]
            print(f"<Clsas:Avatar> object: {a_obj.object_name}, centroid_position: {a_obj.centroid_position}, dimensions: {a_obj.dimensions}")
