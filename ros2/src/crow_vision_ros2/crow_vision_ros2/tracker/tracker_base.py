import math, random, string
import datetime
import numpy as np

def get_vector_length(vector):
    """
    Returns length of a vector.
    ARGS:
    - vector: <tuple:(x,y,z):(float,float,float)>
    RETURN:
    - <float> Lenght of a vector
    """

    return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def random_alpha_numeric(len=16):
    """
    Returns random alpha numeric number of length len.
    ARGS:
    - len: <int>
    RETURN:
    - <string>
    """

    return "".join(random.choices(string.ascii_letters + string.digits, k=len))

class Dimensions:
    """
    Class representing object dimensions.
    ARGS:
    - x: <float>
    - y: <float>
    - z: <float>
    """

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def get_tuple(self):
        return (self.x, self.y, self.z)
    def get_xy_min(self):
        if self.x < self.y:
            return self.x
        else:
            return self.y
    def get_list(self):
        return [self.x, self.y, self.z]

    # Get return specified axis in that order
    def get_axis(self, axis):
        axis_to_return = []
        for ax in axis:
            if ax == "X":
                axis_to_return.append(self.x)
            elif ax == "Y":
                axis_to_return.append(self.y)
            elif ax == "Z":
                axis_to_return.append(self.z)
    def get_numpy_array(self):
        return np.array([[self.x, self.y, self.z]   ])

class Position:
    """
    Class representing a position in 3D space.
    ARGS:
    - x: <float>
    - y: <float>
    - z: <float>
    """

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

        self.origin_date = datetime.datetime.now()

    def get_tuple(self):
        return (self.x, self.y, self.z)
    def get_list(self):
        return [self.x, self.y, self.z]

    def get_distance_to(self, other_position):
        my_tuple = self.get_tuple()
        other_tuple = other_position.get_tuple()
        my_other_vector = tuple(map(lambda i, j: i - j,other_tuple, my_tuple))
        return get_vector_length(vector=my_other_vector)
    def get_numpy_array(self):
        return np.array([[self.x, self.y, self.z]])

class Color:
    """
    Class for saving color in RGB format.
    ARGS:
    - R: <int>
    - G: <int>
    - B: <int>
    """

    def __init__(self, R=255, G=255, B=255):
        self.R = R
        self.G = G
        self.B = B

    def get_tuple(self):
        return (self.R, self.G, self.B)
    def get_list(self):
        return [self.x, self.y, self.z]
    def get_numpy_array(self):
        return np.array([[self.R, self.G, self.B]])

class ObjectsDistancePair:
    """
    Encoding the distance between tracked object and detection.
    ARGS:
    - tracked_object: <Class:TrackedObject>
    - new_object: <Class:TrackedObject>
    """

    def __init__(self, tracked_object, new_object):
        self.tracked_object = tracked_object
        self.new_object = new_object
        self.distance = tracked_object.centroid_position.get_distance_to(other_position=new_object.centroid_position)

    def __lt__(self, other):
        return self.distance < other.distance
