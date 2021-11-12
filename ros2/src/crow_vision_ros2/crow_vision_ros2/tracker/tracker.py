import datetime
from scipy import interpolate
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import queue as pyqueue
import random
from scipy.spatial import Delaunay

# ROS2
from crow_vision_ros2.tracker.tracker_base import get_vector_length, random_alpha_numeric, Dimensions, Position, Color, ObjectsDistancePair
from crow_vision_ros2.tracker.tracker_config import DEFAULT_ALPHA_NUMERIC_LENGTH, TRAJECTORY_MEMORY_SIZE_SECONDS, WRIST_OBJECT_CLIP_DISTANCE_LIMIT, DETECTIONS_FOR_SETUP_NEEDED, PERCENTAGE_NEEDED_TO_BE_APPROVED, TRACKER_ITERATION_HASH_LENGTH, SETUP_OVERLAP_DISTANCE_MERGE, SETUP_OVERLAP_DISTANCE_MAX
from crow_vision_ros2.tracker.tracker_avatar import Avatar, AvatarObject
from crow_vision_ros2.tracker.tracker_trajectory import Trajectory
from crow_ontology.crowracle_client import CrowtologyClient

class TrackedObject:
    """
    Every scene object which we are trying to track and is being recognized
    is stored in this class.
    ARGS:
    - class_name: <string>
    - centroid_position: <Class:Position>
    - dimensions: <Class:Dimensions>
    - original_order_index: <int> Position in the input array for tracking
    - last_uuid: <string>
    - latest_uuid: <string>

    last_uuid and latest_uuid (meaning of the names is wrong however – last_uuid is the uuid of the new detection and latest_uuid is the uuid of the TrackedObject which doesn’t change in time)
    """

    def __init__(self, class_name, centroid_position, dimensions, original_order_index, last_uuid, latest_uuid):
        self.class_name = class_name
        self.object_id = random_alpha_numeric(len=DEFAULT_ALPHA_NUMERIC_LENGTH)
        self.centroid_position = centroid_position
        self.dimensions = dimensions

        # Remembering original order for the return message to filter node
        self.original_order_index = original_order_index

        ### Tracking logic variables
        self.active = True # Do we have detection for this object
        # If object stops being active - this hash will represent the iteration when it was still active
        self.active_last_iteration_hash = random.getrandbits(TRACKER_ITERATION_HASH_LENGTH)
        self.just_updated = False
        self.position_history = [centroid_position] # List of past positions
        # Variable used in the duplication filter
        self.duplicate = False
        # Variable used in the intersection filter
        self.intersects = False
        self.used_for_update = False
        # Ros2 integration for uuid correction
        self.last_uuid = last_uuid
        self.latest_uuid = latest_uuid

        ## Save which hand has been close enough when object dissappeared
        self.close_hand_obj_memory = None
        # Wait till the hand trajectory is full
        self.close_hand_timer = datetime.datetime.now()
        # Average position sent to adder
        self.hand_was_near = False # If true -> wont be deleted
        self.sent_to_database = False
        # If object is freezed, ignore it but still have it in tracked objects
        self.freezed = False

    def update_by_object(self, object, iteration_hash):
        self.centroid_position = object.centroid_position
        self.dimensions = object.dimensions
        self.active = True
        self.active_last_iteration_hash = iteration_hash
        self.just_updated = True
        self.position_history.append(object.centroid_position)
        object.used_for_update = True
        self.original_order_index = object.original_order_index

        # Ros2 particle filter - uuid correction
        self.last_uuid = object.latest_uuid

        ### Reset hand adding logic data
        ## Save which hand has been close enough when object dissappeared
        self.close_hand_obj_memory = None
        # Wait till the hand trajectory is full
        self.close_hand_timer = datetime.datetime.now()
        # Average position sent to adder
        self.tracker_updated_to_average_position = False
        self.hand_was_near = False

        return

    # Used for output format (sent to filter) sorting
    def __lt__(self, other):
        return self.original_order_index < other.original_order_index

class Tracker:
    """
    Main class which takes care of tracking and hold all the
    <Class:TrackedObjects>
    ARGS:
    - crowracle: <Class:CrowtologyClient> filter_node passes database client
    """
    DEBUG = False

    def __init__(self, crowracle, freezeing_cb=None):
        # Current setup detection count
        self.setup_detection_count = 0
        # List of dictionaries - accesible through {"centroid_positions": [...], "dimensions": [...],
        # "class_names": [...], "uuids": [...]}
        self.setup_detection_history = []
        # Dictionary of lists - where class is the key. Each class name has its own list of <TrackedObject>'s
        self.tracked_objects = {}
        # Avatar implementation
        self.avatar = Avatar()

        # Database client passed as agument from filter node
        self.crowracle = crowracle
        assert(isinstance(self.crowracle, CrowtologyClient))

        self.freezeing_cb = freezeing_cb
        self.workspace_hull = None
        self.front_stage_hull = None
        self.trackable_classes = self.crowracle.getWorkpieceDetNames()

        # For the tracker to be able to distinguish different iterations
        # we can use hashes - will be changed every iteration
        self.current_iteration_hash = random.getrandbits(TRACKER_ITERATION_HASH_LENGTH)
        self.last_iteration_hash = random.getrandbits(TRACKER_ITERATION_HASH_LENGTH)

    def _get_area_hull(self, area_name):
        areas_uris = self.crowracle.getStoragesProps()
        for area_uri in areas_uris:
            if area_uri['name'] == area_name:
                area_poly = self.crowracle.get_polyhedron(area_uri['uri'])
                return Delaunay(area_poly)
        print(f"<tracker> Area of name '{area_name}' doesn't exist in ontology!")

    def check_position_in_workspace_area(self, xyz_list):
        """
        Check position xyz_list=[x,y,z] in area with name 'workspace'
        """
        if self.workspace_hull is None:
            self.workspace_hull = self._get_area_hull('workspace')
            if self.workspace_hull is None:
                return False
        res = self.workspace_hull.find_simplex(xyz_list)
        return res >= 0

    def check_position_in_front_stage(self, xyz_list):
        """
        Check position xyz_list=[x,y,z] in area with name 'front_stage'
        """
        if self.front_stage_hull is None:
            self.front_stage_hull = self._get_area_hull('front_stage')
            if self.front_stage_hull is None:
                return False
        res = self.front_stage_hull.find_simplex(xyz_list)
        return res >= 0

    def reset_flags(self):
        """
        Flags all objects of the tracker as 'fresh' i.e. not updated, not
        duplicate etc...
        """
        for class_object_group in self.tracked_objects.values():
            for object in class_object_group:
                object.active = False
                object.just_updated = False
                object.duplicate = False
                object.intersects = False

    def dump_list_of_objects(self):
        """
        Returns all objects of tracker in 1 list
        """
        dump_list = []
        for class_object_group in self.tracked_objects.values():
            for object in class_object_group:
                dump_list.append(object)
        return dump_list

    def class_duplication_filter(self, parsed_objects):
        """
        Filters out the objects which have distance between centroid points
        smaller than the length of their smallest dimension.
        -> "bigger object (smallest dimension is bigger) eats the smaller object"
        -> smaller removed
        ARGS:
        - parse_objects: <list:Class:TrackedObject>
        """

        smallest_dimensions = [] # Index-order sensitive with the class_state_dictionary lists
        for object in parsed_objects:
            smallest_dimensions.append(min(object.dimensions.get_tuple()))

        parsed_objects_cleaned = []
        for a in range(len(parsed_objects)):
            a_pos = parsed_objects[a].centroid_position
            for b in range(a+1, len(parsed_objects)):
                b_pos = parsed_objects[b].centroid_position
                a_b_distance = a_pos.get_distance_to(other_position=b_pos)

                if smallest_dimensions[a] > smallest_dimensions[b]:
                    if a_b_distance < smallest_dimensions[a]: # REMOVE B -> mark as duplicate
                        parsed_objects[b].duplicate = True
                        if self.DEBUG:
                            print("<class_duplication_filter> DUPLICATE DETECTED")
                else:
                    if a_b_distance < smallest_dimensions[b]: # REMOVE A -> mark as duplicate
                        parsed_objects[a].duplicate = True
                        if self.DEBUG:
                            print("<class_duplication_filter> DUPLICATE DETECTED")

        # Remove duplicates -> should take care of 3+ duplicates in the same space
        for object in parsed_objects:
            if not object.duplicate:
                parsed_objects_cleaned.append(object)
                object.duplicate = False
        return parsed_objects_cleaned

    def add_tracked_object_logic(self, class_name, parsed_object):
        """
        Add new tracked object to the tracker.
        ARGS:
        - class_name: <string>
        - parsed_object: <Class:TrackedObject>
        """
        if self.DEBUG:
            print(f"<tracker> ! Creating new object")
        parsed_object.just_updated = True
        if not class_name in self.tracked_objects:
            self.tracked_objects[class_name] = [parsed_object]
        else:
            self.tracked_objects[class_name].append(parsed_object)

    def add_tracked_object_base(self, centroid_position, dimension, class_name, uuid):
        """
        Create and add new tracked object to the tracker.
        ARGS:
        - centroid_position: <Class:Position>
        - dimension: <Class:Dimensions>
        - class_name: <string>
        - uuid: <string>
        """

        new_obj = TrackedObject(class_name=class_name, centroid_position=centroid_position, dimensions=dimension, original_order_index=-1, last_uuid=uuid, latest_uuid=uuid)
        self.add_tracked_object_logic(class_name=class_name, parsed_object=new_obj)



    def distance_ordering_algorithm(self, class_name, parsed_objects):
        """
        For objects which haven't already been updated with new detections ->
        do an algorithm which orders all possible distances within tracked
        objects and new detections so that we know
        <tracked object> <- DISTANCE -> <newly detected>,
        closest distances with are updated first, made faster with linear
        sum assignment optimization.
        ARGS:
        - class_name: <string>
        - parse_objects: <list:Class:TrackedObject>
        """

        # Create 2D list of distance pairs where every row is its own tracked object and
        # every column is the new detected object (1 cell is the "distance pair/distance" between them)
        distance_pairs_2d = []
        for tracked_object in self.tracked_objects[class_name]:
            tracked_object_row = []
            if not tracked_object.just_updated and not tracked_object.freezed:
                for detected_object in parsed_objects:
                    # print(f"detected_object.duplicate: {detected_object.duplicate}, detected_object.intersects: {detected_object.intersects}, detected_object.used_for_update: {detected_object.used_for_update}")
                    if (not detected_object.duplicate) and (not detected_object.intersects) and (not detected_object.used_for_update):
                        tracked_object_row.append(ObjectsDistancePair(tracked_object=tracked_object, new_object=detected_object))
                distance_pairs_2d.append(tracked_object_row)

        # Format input for the linear_sum_assignment algorithm
        if self.DEBUG:
            print(f"*** Linear sum assignment")
        if len(distance_pairs_2d):
            cost_matrix = np.zeros(shape=(len(distance_pairs_2d),len(distance_pairs_2d[0])))
            for row_i in range(len(cost_matrix)):
                for col_i in range(len(cost_matrix[0])):
                    cost_matrix[row_i][col_i] = distance_pairs_2d[row_i][col_i].distance
                    if self.DEBUG:
                        print(f"*** tracked_obj #{row_i} distance to detected_object #{col_i} : {distance_pairs_2d[row_i][col_i].distance}")

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            if self.DEBUG:
                print(f"row_ind: {row_ind}, col_ind: {col_ind}")

            for idx in range(len(row_ind)):
                if self.DEBUG:
                    print(f"*** *** Updating tracked_object #{row_ind[idx]} with detected_object #{col_ind[idx]}")
                distance_pair = distance_pairs_2d[row_ind[idx]][col_ind[idx]]
                distance_pair.tracked_object.update_by_object(object=distance_pair.new_object, iteration_hash=self.current_iteration_hash)
        return


    def evaluate_class_logic(self, class_name, parsed_objects):
        """
        Evaluates tracking for different classes separately.
        """

        # Remove smaller items which are in THIS FRAME overlapping with bigger items of same class
        parsed_objects_cleaned = self.class_duplication_filter(parsed_objects=parsed_objects)
        if self.DEBUG:
            print(f'parsed_objects_cleaned: {parsed_objects_cleaned}')

        if not class_name in self.tracked_objects:
            return
        self.distance_ordering_algorithm(class_name=class_name, parsed_objects=parsed_objects_cleaned)
        return

    def setup_scene_objects(self, centroid_positions, dimensions, class_names, uuids):
        """
        This function should be called at the beginning to load all the items in
        the scene to the tracker, or when you want to reset scene objects and
        all are visible. Returns True when you can continue and start tracking
        else False and you will have to wait for all iterations.
        ARGS:
        - centroid_positions: <list:[x,y,z]:[float:float:float]>
        - dimensions: <list:[x,y,z]:[float:float:float]>
        - class_names: <list:string>
        - uuids: <list:string>
        """

        if self.setup_detection_count < DETECTIONS_FOR_SETUP_NEEDED:
            self.setup_detection_count += 1
            # Save current data to dictionary
            curr_dict = {"centroid_positions": centroid_positions, "dimensions": dimensions, "class_names": class_names, "uuids": uuids}
            self.setup_detection_history.append(curr_dict)

            print(f"<tracker> Setup iteration #{self.setup_detection_count}/{DETECTIONS_FOR_SETUP_NEEDED} ...")

            # In last iteration - load objects
            if self.setup_detection_count >= DETECTIONS_FOR_SETUP_NEEDED:
                if self.DEBUG:
                    print(f"<tracker> self.setup_detection_history:")
                    for a in self.setup_detection_history:
                        print(f"self.setup_detection_history[i]: {a}")
                    print("<tracker> Loading setup objects")

                self.load_setup_objects()
            return False
        else:
            return True

    def load_setup_objects(self):
        """
        Count uuid's and load objects with occurrence more than x% (see config)
        (filter takes care of uuids). Each new uuid will have its own key in
        dictionary and its data as well as counter.
        """
        loader_dict = {}
        centroid_dict = {}  # to resolve multiple uuids in setup belonging to the same object
        uuid_remap_dict = {}
        for history_entry in self.setup_detection_history:
            for centroid_position, dimension, class_name, uid in zip(history_entry["centroid_positions"], history_entry["dimensions"], history_entry["class_names"], history_entry["uuids"]):
                if class_name not in self.trackable_classes:  # skip classes that are not workpieces
                    continue
                if uid in loader_dict:
                    if np.linalg.norm(np.array(loader_dict[uid]["centroid_positions"]) - np.array(centroid_position)) > SETUP_OVERLAP_DISTANCE_MAX:
                        # TODO: maybe ignore this update?
                        print(f"<tracker>: Object of class {class_name} with uuid {uid} found to be at two positions that are too far appart!")
                    loader_dict[uid]["counter"] += 1
                    loader_dict[uid]["class_names"] = class_name  # filter is computing the "mode" of class but it can change
                else:
                    loader_dict[uid] = {
                        "centroid_positions": centroid_position,
                        "dimensions": dimension,
                        "class_names": class_name,
                        "uuids": uid,
                        "counter": 1}
        # Iterate occurrence and save the "approved" objects
        centroid_positions, dimensions, class_names, uuids = ([],[],[],[])
        print(f"<tracker>: Loader dictionary: {loader_dict}")

        for key in loader_dict:
            if loader_dict[key]["counter"] > (PERCENTAGE_NEEDED_TO_BE_APPROVED * DETECTIONS_FOR_SETUP_NEEDED):
                centroid_positions.append(loader_dict[key]["centroid_positions"])
                dimensions.append(loader_dict[key]["dimensions"])
                class_names.append(loader_dict[key]["class_names"])
                uuids.append(loader_dict[key]["uuids"])
        # Create new objects from this data
        parsed_objects_dict = self.parse_objects(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names, uuids=uuids)

        print(f"<tracker>: parsed_objects_dict: {parsed_objects_dict}")

            # Check all inactive objects and check their distance to the trajectories
            # of wrists and check if the distance is smalle than the distance
            # WRIST_OBJECT_CLIP_DISTANCE_LIMIT, if so-> make their dissipation memory long
            # for tracked_object in self.tracked_objects[class_name]:
            #     if not tracked_object.active:
            #         distance, wrist_obj = self.get_closest_hand(object=tracked_object)
            #         if distance < WRIST_OBJECT_CLIP_DISTANCE_LIMIT:
            #             print(f"** Object: {tracked_object.class_name} was too close to the hand, before disappearing, lengthening memory loss")
            #             tracked_object.MEMORY_LOSS_SECONDS = OBJECT_CLOSE_TO_HAND_MEMORY_LOSS_SECONDS
        for class_name in parsed_objects_dict:
            for obj in parsed_objects_dict[class_name]:
                # Check if the object is in the workspace - if so - ignore it
                if not self.check_position_in_workspace_area(xyz_list=obj.centroid_position.get_list()):
                    self.add_tracked_object_logic(class_name=class_name, parsed_object=obj)
        return

    def reset_setup(self):
        """
        Reset scene objects setup. Use this function when you want to start
        setting up your objects - this also deletes all objects
        """

        # Reset the setup data
        self.tracked_objects = {}
        self.setup_detection_count = 0
        self.setup_detection_history = []

    def track_and_get_uuids(self, centroid_positions, dimensions, class_names, uuids):
        """
        Returns index-sensitive mapping of old uuids to new uuids ([a,b, ...],[c,b, ...])
        'b' should be 'a', 'c' should be 'a' ...
        ARGS:
        - centroid_positions: <list:[x,y,z]:[float:float:float]>
        - dimensions: <list:[x,y,z]:[float:float:float]>
        - class_names: <list:string>
        - uuids: <list:string>
        """

        # Flag all existing objects as not (updated) this frame
        self.last_iteration_hash = self.current_iteration_hash
        self.current_iteration_hash = random.getrandbits(TRACKER_ITERATION_HASH_LENGTH)

        self.reset_flags()

        # Setup scene objects - after 'DETECTIONS_FOR_SETUP_NEEDED' it will return True
        if not self.setup_scene_objects(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names, uuids=uuids):
            return ([], [])
        else:
            parsed_objects_dict = self.parse_objects(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names, uuids=uuids)

        for class_name in parsed_objects_dict:
            self.evaluate_class_logic(class_name=class_name, parsed_objects=parsed_objects_dict[class_name])

        self.check_inactive_objects_hand_logic()

        # Return id's with original order, if some detection are disregarded - return -1
        # Dump all object instances into 1 list
        objects_list = self.dump_list_of_objects()
        # Sort it by initial index order
        objects_list.sort()
        # Create new output list with the same length as original
        last_uuid, latest_uuid = ([],[])

        idx_ended = 0
        for idx in range(len(centroid_positions)):
            # Find first valid (just updated object) index and return its id
            appended = False
            for object_i in range(idx_ended, len(objects_list)):
                if self.DEBUG:
                    print(f"idx: {idx}  object.original_order_index: { objects_list[object_i].original_order_index}  object.just_updated: {objects_list[object_i].just_updated}")
                # print(f"object.original_order_index: {object.original_order_index}")
                if objects_list[object_i].just_updated and objects_list[object_i].original_order_index == idx:
                    last_uuid.append(objects_list[object_i].last_uuid)
                    latest_uuid.append(objects_list[object_i].latest_uuid)
                    appended = True
                    idx_ended = idx + 1
                    break
            if not appended:
                last_uuid.append(uuids[idx])
                latest_uuid.append(-1)

        return (last_uuid, latest_uuid)


    def parse_objects(self, centroid_positions, dimensions, class_names, uuids):
        """
        This function returns dictionary of classes with their objects
        parsed_objects = {"cube": [<TrackedObject>, <TrackedObject>],
        "ball": [<>...], ....}
        ARGS:
        - centroid_positions: <list:[x,y,z]:[float:float:float]>
        - dimensions: <list:[x,y,z]:[float:float:float]>
        - class_names: <list:string>
        - uuids: <list:string>
        """

        parsed_objects = {}
        # Initialize empty dicitonaries of unique class_names
        for class_name in class_names:
            if class_name in self.trackable_classes and class_name not in parsed_objects:
                parsed_objects[class_name] = []
        # Add objects to the dictionary
        for class_name_i, class_name in enumerate(class_names):
            if class_name not in self.trackable_classes:  # don't add non-trackable objects
                continue
            centroid_position = Position(x=centroid_positions[class_name_i][0], y=centroid_positions[class_name_i][1], z=centroid_positions[class_name_i][2])

            # Check if the position is in the workspace
            if not self.check_position_in_workspace_area(xyz_list=centroid_position.get_list()):
                dimension = Dimensions(x=dimensions[class_name_i][0], y=dimensions[class_name_i][1], z=dimensions[class_name_i][2])

                tracked_obj = TrackedObject(class_name=class_name, centroid_position=centroid_position, dimensions=dimension, original_order_index=class_name_i, last_uuid=uuids[class_name_i] ,latest_uuid=uuids[class_name_i])
                parsed_objects[class_name].append(tracked_obj)
        return parsed_objects

    def check_inactive_objects_hand_logic(self):
        """
        Check all inactive objects and check their distance to the trajectories
        of wrists and check if the distance is smaller than the distance
        WRIST_OBJECT_CLIP_DISTANCE_LIMIT - flag it - save the wrist object -
        if the wrist object crosses workspace -> move and freeze the object
        to the workspace
        """

        for class_object_group in self.tracked_objects.values():
            for tracked_object in class_object_group:
                if not tracked_object.freezed:
                    if not tracked_object.active:
                        # Check if he is already flagged as being "close to hand" and check
                        # that the object was active last iteration
                        if not tracked_object.hand_was_near: # and (tracked_object.active_last_iteration_hash == self.last_iteration_hash):
                            distance, wrist_obj = self.get_closest_hand(object=tracked_object)
                            if distance < WRIST_OBJECT_CLIP_DISTANCE_LIMIT:
                                # Mark tracked object as hand_was_near
                                tracked_object.hand_was_near = True
                                # Save avatar object ot the object
                                tracked_object.close_hand_obj_memory = wrist_obj
                                # Start the time in the object
                                tracked_object.close_hand_timer = datetime.datetime.now()

                                print(f"<tracker>: Object: {tracked_object.class_name} was too close to the hand before disappearing")

                if not tracked_object.active and tracked_object.hand_was_near and not tracked_object.freezed:
                    # Check last position of the hand object saved in the tracked object
                    # if it was in the 'workspace' environment.
                    print(f"<tracker>: Object: {tracked_object.class_name} inactive and hand was near.")

                    last_hand_pos_list = tracked_object.close_hand_obj_memory.centroid_position.get_list()
                    if self.check_position_in_workspace_area(xyz_list=last_hand_pos_list):
                        # Move object to the workspace (position of the hand in the workspace) and
                        # freeze it there
                        print(f"<tracker>: Object: {tracked_object.class_name} frozen in workspace: {last_hand_pos_list}")
                        self.move_and_freeze(xyz_list=last_hand_pos_list, uuid=tracked_object.last_uuid)
                    else:
                        print(f"<tracker>: Hand {tracked_object.close_hand_obj_memory.object_name} not (yet) in workspace: {last_hand_pos_list}")

    def get_closest_hand(self, object):
        """
        This function returns the distance and an wrist-object to which our
        "object" -argument- is closest to (if ther
        e is at least 1 wrist object
        of course), else returns (float('inf'), None)
        ARGS:
        - object: <Class:TrackedObject>
        """

        closest_wrist_object = None
        distance = float('inf')

        trajectory_objects = [self.avatar.avatar_objects["left_wrist"], self.avatar.avatar_objects["right_wrist"]]
        for trajectory_object in trajectory_objects:
            dist = trajectory_object.trajectory_memory.get_trajectory_minimal_distance(avatar_object=trajectory_object,object=object)
            # print(f"dist {trajectory_object.object_name} to {object.class_name}: {dist}")
            # print(f"previous dist: {distance}")
            if dist < distance:
                distance = dist
                closest_wrist_object = trajectory_object

        return distance, closest_wrist_object

    def dump_tracked_objects_info(self):
        """
        Print info about all the tracked objects
        """
        # if not self.DEBUG:
        #     return
        print("<tracker>: >>> all tracked objects >>>")
        count_objects = 0
        for class_name in self.tracked_objects:
            for tracked_object in self.tracked_objects[class_name]:
                count_objects += 1
                print(f"<tracker>: Class name: {class_name}, Position: {tracked_object.centroid_position.get_list()} , uuid: {tracked_object.last_uuid}, active: {tracked_object.active}, hand_was_near: {tracked_object.hand_was_near} freezed: {tracked_object.freezed}")
        print(f"<tracker>: All tracked objects count {count_objects} <<<")

    #
    def get_object_uuid(self, uuid):
        """
        Returns <class:TrackedObject> with this uuid which is saved in the
        tracker.
        ARGS:
        - uuid: <string>
        """
        for class_object_group in self.tracked_objects.values():
            for tracked_object in class_object_group:
                if tracked_object.last_uuid == uuid:
                    return tracked_object

    #
    def delete_object_uuid(self, uuid):
        """
        Delete object from the self.tracked_object
        ARGS:
        - uuid: <string>
        """

        # Iterate all classes and all objects and find the one - then delete it
        # from the list
        for class_name in self.tracked_objects:
            for tracked_object in self.tracked_objects[class_name]:
                if tracked_object.last_uuid == uuid:
                    # Remove object
                    self.tracked_objects[class_name].remove(tracked_object)
        return

    def freeze_object(self, tracked_object):
        tracked_object.active = False
        tracked_object.freezed = True

    def move_and_freeze(self, xyz_list, uuid):
        """
        This function moves existing object to a certain position and freezes
        it, which means that from now on, the object will still be saved but it
        will be ignored when iterating tracking (wont be updated) - robot
        function. Freezing an object is majorly a synonym with "being in
        workspace".
        ARGS:
        - xyz_list: <list:[x,y,z]:[float,float,float]>
        """

        object = self.get_object_uuid(uuid=uuid)
        new_centroid_position = Position(x=xyz_list[0], y=xyz_list[1], z=xyz_list[2])

        object.centroid_position = new_centroid_position
        self.freeze_object(tracked_object=object)
        # publish the newly frozen object
        if self.freezeing_cb is not None:
            self.freezeing_cb(object.class_name, object.latest_uuid)
        return

    def add_tracked_object_and_freeze(self, xyz_list, xyz_dimension, class_name, uuid):
        """
        Create tracked object and freeze it into the position xyz_list.
        ARGS:
        - xyz_list: <list:[x,y,z]:[float,float,float]>
        - xyz_dimensions: <list:[x,y,z]:[float,float,float]>
        - class_name: <string>
        - uuid: <string>
        """

        new_position = Position(x=xyz_list[0], y=xyz_list[1], z=xyz_list[2])
        new_dimension = Dimensions(x=xyz_dimension[0], y=xyz_dimension[1], z=xyz_dimension[2])

        new_obj = TrackedObject(class_name=class_name, centroid_position=centroid_position, dimensions=dimension, original_order_index=-1, last_uuid=uuid, latest_uuid=uuid)
        self.freeze_object(tracked_object=new_obj)
        self.add_tracked_object_logic(class_name=class_name, parsed_object=new_obj)

    # def intersection_filter(self, class_name, parsed_objects):
    #     """
    #     Checks intersection between existing tracked objects and new detection.
    #     * this function seems to be unnecesarry. Distance ordering seems to be
    #     enough.
    #     ARGS:
    #     - class_name: <string>
    #     - parsed_objects: <list:Class:TrackedObject>
    #     """
    #
    #     smallest_dimensions = [] # Index-order sensitive
    #     # print(f"len(self.tracked_objects[class_name]): {len(self.tracked_objects[class_name])}")
    #     for tracked_object in self.tracked_objects[class_name]:
    #         # print(f"tracked_object.dimensions.get_tuple(): {tracked_object.dimensions.get_tuple()}")
    #         smallest_dimensions.append(tracked_object.dimensions.get_xy_min())
    #         # print(f"smallest_dimensions: {smallest_dimensions}")
    #
    #     for tracked_object in self.tracked_objects[class_name]:
    #         closest_intersecting_with = None
    #         closest_intersecting_distance = float('inf')
    #         tracked_obj_pos = tracked_object.centroid_position
    #         for parsed_object in parsed_objects:
    #             parsed_object_pos = parsed_object.centroid_position
    #             # Check intersection ->(distance < smallest dimension)
    #             objects_distance = tracked_obj_pos.get_distance_to(other_position=parsed_object_pos)
    #             if objects_distance < smallest_dimensions[self.tracked_objects[class_name].index(tracked_object)]:
    #                 if not parsed_object.intersects:
    #                     if objects_distance < closest_intersecting_distance:
    #                         closest_intersecting_distance = objects_distance
    #                         closest_intersecting_with = parsed_object
    #
    #         # Update tracked object with closest intersecting object
    #         if closest_intersecting_with != None:
    #             tracked_object.update_by_object(object=closest_intersecting_with, iteration_hash=self.current_iteration_hash)
    #             print(f"<intersection filter>: UPDATING DISTANCE")
    #             # print(f"tracked_object.just_updated: {tracked_object.just_updated}")
    #             closest_intersecting_with.intersects = True
    #     return
