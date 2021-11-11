import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters
from time import sleep

#PointCloud2
from crow_msgs.msg import SegmentedPointcloud
import open3d as o3d
from crow_vision_ros2.utils import make_vector3, ftl_pcl2numpy, ftl_numpy2pcl
from crow_vision_ros2.filters import ParticleFilter

import rdflib
from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD
from datetime import datetime
# Tracker
from crow_vision_ros2.tracker import Tracker

#TF
import tf2_py as tf
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import MultiArrayDimension, Float32MultiArray
from crow_msgs.msg import FilteredPose, PclDimensions, ObjectPointcloud
from crow_ontology.crowracle_client import CrowtologyClient

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

from rdflib import URIRef, BNode, Literal, Graph

import numpy as np

import pkg_resources
import time

ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
CROW = Namespace(f"{ONTO_IRI}#")

class TestDatabase(Node):
    def __init__(self, node_name="test_database"):
        super().__init__(node_name)
        self.crowracle = CrowtologyClient(node=self)
        self.onto = self.crowracle.onto

    def add_storage_space(self):
        name = "storage_a"
        polygon = [[0,0,0],[2,0,0],[2,2,0],[0,2,0]]
        polyhedron = [[0,0,0],[2,0,0],[2,2,0],[0,2,0], [0,0,2],[2,0,2],[2,2,2],[0,2,2]]
        area = (2**2) * 6
        volume = 2**3
        centroid = [1,1,1]
        self.crowracle.add_storage_space(name=name, polygon=polygon, polyhedron=polyhedron, area=area, volume=volume, centroid=centroid)

    def add_object(self):
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        prop_range = list(self.onto.objects(subject=CROW.hasDetectorName, predicate=RDFS.range))[0]
        corresponding_objects = list(self.onto.subjects(CROW.hasDetectorName, Literal("cube_holes", datatype=prop_range)))

        if len(corresponding_objects):
            self.crowracle.add_detected_object(object_name="cube_holes", location=[2.5, 2.5, 2.5], size=[0.1, 0.1, 0.1], uuid="12345678", timestamp=timestamp, template=corresponding_objects[0], adder_id=3)
            self.crowracle.add_detected_object(object_name="cube_holes", location=[0.5, 0.5, 0.5], size=[0.1, 0.1, 0.1], uuid="123456789", timestamp=timestamp, template=corresponding_objects[0], adder_id=4)
        else:
            print("ERROR: length of 'corresponding_objects' is 0!")

    def dump_objs_in_areas(self):
        print("")
        print("* All objects in all spaces:")
        all = self.onto.triples((None, CROW.insideOf, None))
        for bit in all:
            print(f"bit: {bit}")
        print("")




def main():
    rclpy.init()
    try:
        # Initialize database tester
        test_database = TestDatabase()
        space_areas = []

        i = 0
        while True:
            if i == 1:
                # Create new storage space
                test_database.add_storage_space()
            if i == 2:
                # Initialize memory of storage spaces
                space_areas = test_database.crowracle.getStoragesProps()
            if i > 2 and i < 10:
                # Add new object in the scene
                test_database.add_object()
                # Test all existing object in every save 'space_area'
                # and write the result in the memory
                areas_uris = test_database.crowracle.getStoragesProps()

                test_database.crowracle.pair_objects_to_areas_wq(info=True)
            if i > 10:
                # Test all existing object in every save 'space_area'
                # and write the result in the memory
                test_database.crowracle.pair_objects_to_areas_wq(info=True)

            i += 1
            sleep(0.5)
    finally:
        test_database.destroy_node()

if __name__ == "__main__":
    main()
