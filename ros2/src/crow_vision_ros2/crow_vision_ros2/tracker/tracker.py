import datetime
from scipy import interpolate
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import queue as pyqueue
import random

AZA = False
# ROS2
from crow_vision_ros2.tracker.tracker_base import get_vector_length, random_alpha_numeric, Dimensions, Position, Color, ObjectsDistancePair
from crow_vision_ros2.tracker.tracker_config import DEFAULT_ALPHA_NUMERIC_LENGTH, TRACKER_STAY_IN_PLACE_SECONDS, TRAJECTORY_MEMORY_SIZE_SECONDS, WRIST_OBJECT_CLIP_DISTANCE_LIMIT, OBJECT_CLOSE_TO_HAND_TRACKER_STAY_IN_PLACE_SECONDS, DETECTIONS_FOR_SETUP_NEEDED, PERCENTAGE_NEEDED_TO_BE_APPROVED, TRACKER_ITERATION_HASH_LENGTH
from crow_vision_ros2.tracker.tracker_avatar import Avatar, AvatarObject
from crow_vision_ros2.tracker.tracker_trajectory import Trajectory
# TESTING
# from tracker_base import get_vector_length, random_alpha_numeric, Dimensions, Position, Color, ObjectsDistancePair
# from tracker_config import DEFAULT_ALPHA_NUMERIC_LENGTH, TRACKER_STAY_IN_PLACE_SECONDS, TRAJECTORY_MEMORY_SIZE_SECONDS, WRIST_OBJECT_CLIP_DISTANCE_LIMIT, OBJECT_CLOSE_TO_HAND_TRACKER_STAY_IN_PLACE_SECONDS, DETECTIONS_FOR_SETUP_NEEDED, TRACKER_ITERATION_HASH_LENGTH
# from tracker_avatar import Avatar, AvatarObject
# from tracker_trajectory import Trajectory

class TrackedObject:
    def __init__(self, class_name, centroid_position, dimensions, original_order_index, last_uuid, latest_uuid):
        self.class_name = class_name
        self.object_id = random_alpha_numeric(len=DEFAULT_ALPHA_NUMERIC_LENGTH)
        self.centroid_position = centroid_position
        self.dimensions = dimensions

        # Remembering original order for the return message to filter node
        self.original_order_index = original_order_index

        ## Tracking logic variables
        self.active = True # Is the object currently in the "frame"
        # If object stops being active - this hash will represent the iteration when it was still active
        self.active_last_iteration_hash = random.getrandbits(TRACKER_ITERATION_HASH_LENGTH)
        self.just_updated = False
        self.position_history = [centroid_position] # List of past positions
        # Variable used in the duplication filter at the beginning
        self.duplicate = False
        # Variable used in the intersection filter
        self.intersects = False
        self.used_for_update = False

        # If there is a glitch detection (or item stopped existing),
        # tracker will forget/throw out the object after couple of frames
        self.detection_memory_loss = datetime.datetime.now()
        self.TRACKER_STAY_IN_PLACE_SECONDS = TRACKER_STAY_IN_PLACE_SECONDS

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
        self.last_uuid   = object.latest_uuid

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
    def __init__(self):
        # Current setup detection count
        self.setup_detection_count = 0
        # List of dictionaries - accesible through {"centroid_positions": [...], "dimensions": [...],
        # "class_names": [...], "uuids": [...]}
        self.setup_detection_history = []
        # Dictionary of lists - where class is the key. Each class name has its own list of <TrackedObject>'s
        self.tracked_objects = {}
        # Avatar implementation
        self.avatar = Avatar()

        # For the tracker to be able to distinguish different iterations
        # we can use hashes - will be changed every iteration
        self.current_iteration_hash = random.getrandbits(TRACKER_ITERATION_HASH_LENGTH)
        self.last_iteration_hash = random.getrandbits(TRACKER_ITERATION_HASH_LENGTH)

    # This function flags all the existing objects as not already updated and
    # used in the tracking loop
    def reset_flags(self):
        for class_name in self.tracked_objects:
            for object in self.tracked_objects[class_name]:
                object.active = False
                object.just_updated = False
                object.duplicate = False
                object.intersects = False

    # Get all objects in 1 list
    def dump_list_of_objects(self):
        dump_list = []
        for class_name in self.tracked_objects:
            for object in self.tracked_objects[class_name]:
                dump_list.append(object)
        return dump_list

    # Filters out the objects which have distance between centroid points smaller
    # than the length of their smallest dimension.
    # F.E. as well -> "bigger object (smallest dimension is bigger) eats the smaller object" -> smaller removed
    def class_duplication_filter(self, parsed_objects):
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
                        print("<class_duplication_filter> DUPLICATE DETECTED")
                else:
                    if a_b_distance < smallest_dimensions[b]: # REMOVE A -> mark as duplicate
                        parsed_objects[a].duplicate = True
                        print("<class_duplication_filter> DUPLICATE DETECTED")

        # Remove duplicates -> should take care of 3+ duplicates in the same space
        for object in parsed_objects:
            if not object.duplicate:
                parsed_objects_cleaned.append(object)
                object.duplicate = False
        return parsed_objects_cleaned

    def add_tracked_object_logic(self, class_name, parsed_object):
        print(f"<tracker> ! Creating new object")
        parsed_object.just_updated = True
        if not class_name in self.tracked_objects:
            self.tracked_objects[class_name] = [parsed_object]
        else:
            self.tracked_objects[class_name].append(parsed_object)

    def add_tracked_object_base(self, centroid_position, dimension, class_name, uuid):
        new_obj = TrackedObject(class_name=class_name, centroid_position=centroid_position, dimensions=dimension, original_order_index=-1, last_uuid=uuid, latest_uuid=uuid)
        self.add_tracked_object_logic(class_name=class_name, parsed_object=new_obj)

    # # Checks intersection with existing and new and update them if neccessary,
    # # returns objects which have not been yet assigned
    # def intersection_filter(self, class_name, parsed_objects):
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

    # Distance ordering and pairing - For objects which haven't already
    # been updated with new detections -> do an algorithm which orders
    # all possible distances within class and new detection so that we
    # know <tracked object> <- DISTANCE -> <newly detected>,
    # closest distances with are updated first, made faster with linear sum assignment
    # optimization
    def distance_ordering_algorithm(self, class_name, parsed_objects):
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
        print(f"*** Linear sum assignment")
        if len(distance_pairs_2d):
            cost_matrix = np.zeros(shape=(len(distance_pairs_2d),len(distance_pairs_2d[0])))
            for row_i in range(len(cost_matrix)):
                for col_i in range(len(cost_matrix[0])):
                    cost_matrix[row_i][col_i] = distance_pairs_2d[row_i][col_i].distance
                    print(f"*** tracked_obj #{row_i} distance to detected_object #{col_i} : {distance_pairs_2d[row_i][col_i].distance}")

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            print(f"row_ind: {row_ind}, col_ind: {col_ind}")

            for idx in range(len(row_ind)):
                print(f"*** *** Updating tracked_object #{row_ind[idx]} with detected_object #{col_ind[idx]}")
                distance_pair = distance_pairs_2d[row_ind[idx]][col_ind[idx]]
                distance_pair.tracked_object.update_by_object(object=distance_pair.new_object, iteration_hash=self.current_iteration_hash)
        return

    # This function evaluates existence based just on the class (works with list of objects),
    # results can later be evaluated in the next step as we know the object history
    # "which one is new"
    def evaluate_class_logic(self, class_name, parsed_objects):
        # Remove smaller items which are in THIS FRAME overlapping with bigger items of same class
        parsed_objects_cleaned = self.class_duplication_filter(parsed_objects=parsed_objects)

        if not class_name in self.tracked_objects:
            return
        else:
            # Distance ordering and pairing - For objects which haven't already
            # been updated with new detections -> do an algorithm which orders
            # all possible distances within class and new detection so that we
            # know <tracked object> <- DISTANCE -> <newly detected>,
            # closest distances with are updated first
            self.distance_ordering_algorithm(class_name=class_name, parsed_objects=parsed_objects_cleaned)

            ## Hand logic
            # Check all inactive objects and check their distance to the trajectories
            # of wrists and check if the distance is smaller than the distance
            # WRIST_OBJECT_CLIP_DISTANCE_LIMIT, if so-> make the object non-deletable,
            # -save the hand to the object - start its timer - if timer crosses
            # max trajectory time - calculate average point in 3D space for hand
            for tracked_object in self.tracked_objects[class_name]:
                if not tracked_object.freezed:
                    if not tracked_object.active:
                        # Check if he is already flagged as being "close to hand" and check
                        # that the object was active last iteration
                        if not tracked_object.hand_was_near and (tracked_object.active_last_iteration_hash == self.last_iteration_hash):
                            distance, wrist_obj = self.get_closest_hand(object=tracked_object)
                            if distance < WRIST_OBJECT_CLIP_DISTANCE_LIMIT:
                                # Mark tracked object as hand_was_near
                                tracked_object.hand_was_near = True
                                # Save avatar object ot the object
                                tracked_object.close_hand_obj_memory = wrist_obj
                                # Start the time in the object
                                tracked_object.close_hand_timer = datetime.datetime.now()

                                print(f"<tracker>: Object: {tracked_object.class_name} was too close to the hand before disappearing")

                    if not tracked_object.active and tracked_object.hand_was_near:
                        # FOREVER CHECK IF THE HAND ENTERED THE WORKSPACE #
                        # IF THE HAND ENTERED - FLAG OBJECT AS FROZEN AND SET THE POSITION
                        # TO THE WORKSPACE CENTROID - CREATE TRIPLET
                        pass
                        ############################################################################################################
                        ############################################################################################################

        return


    # This function should be called at the beginning to load all the items in the scene
    # to the tracker, or when you want to reset scene objects and all are visible
    # Returns True when you can continue and start tracking - else False and you
    # will have to wait
    def setup_scene_objects(self, centroid_positions, dimensions, class_names, uuids):
        if self.setup_detection_count < DETECTIONS_FOR_SETUP_NEEDED:
            self.setup_detection_count += 1
            # Save current data to dictionary
            curr_dict = {"centroid_positions": centroid_positions, "dimensions": dimensions, "class_names": class_names, "uuids": uuids}
            self.setup_detection_history.append(curr_dict)

            print(f"<tracker> Setup iteration #{self.setup_detection_count}/{DETECTIONS_FOR_SETUP_NEEDED} ...")

            # In last iteration - load objects
            if self.setup_detection_count >= DETECTIONS_FOR_SETUP_NEEDED:
                print(f"<tracker> self.setup_detection_history:")
                for a in self.setup_detection_history:
                    print(f"self.setup_detection_history[i]: {a}")
                print("<tracker> Loading setup objects")

                self.load_setup_objects()
            return False
        else:
            return True
    def load_setup_objects(self):
        # Count uuid's and load objects with occurence more thna 50% (filter takes care of uuids)
        # Each new uuid will have its own key in dicionary and its data as well as counter
        loader_dict = {}
        for history_i in range(len(self.setup_detection_history)):
            # Check if key (i.e. uuid) exists
            for key_i in range(len(self.setup_detection_history[history_i]["uuids"])):
                key = self.setup_detection_history[history_i]["uuids"][key_i]
                if self.setup_detection_history[history_i]["uuids"][key_i] in loader_dict:
                    loader_dict[key]["counter"] += 1
                else:
                    loader_dict[key] = {
                        "centroid_positions": self.setup_detection_history[history_i]["centroid_positions"][key_i],
                        "dimensions": self.setup_detection_history[history_i]["dimensions"][key_i],
                        "class_names": self.setup_detection_history[history_i]["class_names"][key_i],
                        "uuids": self.setup_detection_history[history_i]["uuids"][key_i],
                        "counter": 0}
        # Iterate occurence and save the "approved" objects
        centroid_positions, dimensions, class_names, uuids = ([],[],[],[])

        print(f"<tracker>: Loader dictionary: {loader_dict}")

        for key in loader_dict:
            if loader_dict[key]["counter"] > (PERCENTAGE_NEEDED_TO_BE_APPROVED*DETECTIONS_FOR_SETUP_NEEDED):
                centroid_positions.append(loader_dict[key]["centroid_positions"])
                dimensions.append(loader_dict[key]["dimensions"])
                class_names.append(loader_dict[key]["class_names"])
                uuids.append(loader_dict[key]["uuids"])
        # Create new objects from this data
        parsed_objects_dict = self.parse_objects(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names, uuids=uuids)

        print(f"<tracker>: parsed_objects_dict: {parsed_objects_dict}")

        for class_name in parsed_objects_dict:
            for obj in parsed_objects_dict[class_name]:
                self.add_tracked_object_logic(class_name=class_name, parsed_object=obj)
        return

    # Reset setup. Use this function when you want to start setting up your
    # objects - this also deletes all objects
    def reset_setup(self):
        # Reset the setup data
        self.tracked_objects = {}
        self.setup_detection_count = 0
        self.setup_detection_history = []

    # Returns index-sensitive mapping of old uuids to new uuids ([a,b, ...],[c,b, ...])
    # 'b' should be 'a', 'c' should be 'a'
    def track_and_get_uuids(self, centroid_positions, dimensions, class_names, uuids):
        # Flag all existing objects as not (updated) this frame
        print("<tracker>: 0")

        self.last_iteration_hash = self.current_iteration_hash
        self.current_iteration_hash = random.getrandbits(TRACKER_ITERATION_HASH_LENGTH)

        self.reset_flags()

        print("<tracker>: A")

        # Setup scene objects - after 'DETECTIONS_FOR_SETUP_NEEDED' it will return True
        if not self.setup_scene_objects(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names, uuids=uuids):
            return ([], [])
        else:
            parsed_objects_dict = self.parse_objects(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names, uuids=uuids)

        print("<tracker>: B")

        for class_name in parsed_objects_dict:
            self.evaluate_class_logic(class_name=class_name, parsed_objects=parsed_objects_dict[class_name])

        print("<tracker>: C")

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
                print(f"idx: {idx}  object.original_order_index: { objects_list[object_i].original_order_index}  object.just_updated: {objects_list[object_i].just_updated}")
                # print(f"object.original_order_index: {object.original_order_index}")
                if objects_list[object_i].just_updated and objects_list[object_i].original_order_index == idx:
                    last_uuid.append(objects_list[object_i].last_uuid)
                    latest_uuid.append(objects_list[object_i].latest_uuid)
                    appended = True
                    idx_ended = idx + 1
                    break
            if not appended:
                last_uuid.append(-1)
                latest_uuid.append(-1)

        self.reset_setup()

        return (last_uuid, latest_uuid)

    # This function returns dictionary of classes with their objects
    # parsed_objects = {"cube": [<obj1>, <obj2>], "ball": [<>...], ....}
    def parse_objects(self, centroid_positions, dimensions, class_names, uuids):
        parsed_objects = {}
        # Initialize empty dicitonaries of unique class_names
        for class_name in class_names:
            parsed_objects[class_name] = []
        # Add objects to the dictionary
        for class_name_i in range(len(class_names)):
            class_name = class_names[class_name_i]
            centroid_position = Position(x=centroid_positions[class_name_i][0], y=centroid_positions[class_name_i][1], z=centroid_positions[class_name_i][2])
            dimension = Dimensions(x=dimensions[class_name_i][0], y=dimensions[class_name_i][1], z=dimensions[class_name_i][2])

            tracked_obj = TrackedObject(class_name=class_name, centroid_position=centroid_position, dimensions=dimension, original_order_index=class_name_i, last_uuid=uuids[class_name_i] ,latest_uuid=uuids[class_name_i])
            parsed_objects[class_name].append(tracked_obj)
        return parsed_objects

    # This function returns the distance and an wrist-object to which our "object"
    # -argument- is closest to (if there is at least 1 wrist object of course)
    def get_closest_hand(self, object):
        closest_wrist_object = None
        distance = float('inf')

        trajectory_objects = [self.avatar.avatar_objects["leftWrist"], self.avatar.avatar_objects["rightWrist"] ]
        for trajectory_object in trajectory_objects:
            dist = trajectory_object.trajectory_memory.get_trajectory_minimal_distance(avatar_object=trajectory_object,object=object)
            if dist < distance:
                distance = dist
                closest_wrist_object = trajectory_object

        return distance, closest_wrist_object

    # Print info about all the tracked objects
    def dump_tracked_objects_info(self):
        print("<tracked>: next lines - all tracked objects")
        count_objects = 0
        for class_name in self.tracked_objects:
            for tracked_object in self.tracked_objects[class_name]:
                count_objects += 1
                print(f"<tracker>: Class name: {class_name}, Position: {tracked_object.centroid_position.get_list()} , uuid: {tracked_object.last_uuid}")
        print(f"<tracker>: All tracked objects count {count_objects}")

    # Returns object with this uuid
    def get_object_uuid(self, uuid):
        for class_name in self.tracked_objects:
            for tracked_object in self.tracked_objects[class_name]:
                if tracked_object.last_uuid == uuid:
                    return tracked_object

    # Delete object from the self.tracked_object
    def delete_object_uuid(self, uuid):
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
        tracked_object.freezed = True.

        ### CHECK IF HE IS IN THE WORKSPACE IF YES, FLAG IT and
        # add it to the workspace
        #####
        #
        #

    # This function moves existing object to a certain position and freezes it,
    # which means that from now on, the object will still be saved but it will
    # be ignored when iterating tracking (wont be updated) - robot function
    # ARGS: xyz_list (list of xyz coordinates to move object to [1.1, 2, 3])
    def move_and_freeze(self, xyz_list, uuid):
        object = self.get_object_uuid(uuid=uuid)
        new_centroid_position = Position(x=xyz_list[0], y=xyz_list[1], z=xyz_list[2])

        object.centroid_position = new_centroid_position
        self.freeze_object(tracked_object=object)
        return


    # Create tracked object and freeze it into the position xyz_list
    # ARGS:
    # - xyz_list: list of x,y,z coordinates [x,y,z]
    # - xyz_dimensions: list of x,y,z dimensions [x,y,z]
    # - class_name: name of the object
    # - uuid: uuid
    def add_tracked_object_and_freeze(self, xyz_list, xyz_dimension, class_name, uuid):
        new_position = Position(x=xyz_list[0], y=xyz_list[1], z=xyz_list[2])
        new_dimension = Dimensions(x=xyz_dimension[0], y=xyz_dimension[1], z=xyz_dimension[2])

        new_obj = TrackedObject(class_name=class_name, centroid_position=centroid_position, dimensions=dimension, original_order_index=-1, last_uuid=uuid, latest_uuid=uuid)
        self.freeze_object(tracked_object=new_obj)
        self.add_tracked_object_logic(class_name=class_name, parsed_object=new_obj)






    ############################################################################
    ############################################################################
    ### TESTING ENVIRONEMNT ONLY - REMOVE FROM ROS2 INTEGRATION ################
    # This function should be called at the beginning to load all the items in the scene
    # to the tracker, or when you want to reset scene objects and all are visible
    # Returns True when you can continue and start tracking - else False and you
    # will have to wait
    def setup_scene_objects_sim(self, centroid_positions, dimensions, class_names):
        if self.setup_detection_count == 0:
            self.setup_detection_count += 1
            # Create new objects
            parsed_objects_dict = self.parse_objects_sim(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names)
            for class_name in parsed_objects_dict:
                for obj in parsed_objects_dict[class_name]:
                    self.add_tracked_object_logic(class_name=class_name, parsed_object=obj)
            return False
        else:
            return True

    def track_and_get_ids(self, centroid_positions, dimensions, class_names):
        # Flag all existing objects as not (updated) this frame
        self.reset_flags()

        # Setup scene objects - after 'DETECTIONS_FOR_SETUP_NEEDED' it will return True
        if not self.setup_scene_objects_sim(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names):
            return False
        else:
            parsed_objects_dict = self.parse_objects_sim(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names)

        for class_name in parsed_objects_dict:
            self.evaluate_class_logic(class_name=class_name, parsed_objects=parsed_objects_dict[class_name])

        # Return id's with original order, if some detection is disregarded,
        # return -1
        # Dump all object instances into 1 list
        objects_list = self.dump_list_of_objects()
        # Sort it by initial index order
        objects_list.sort()
        # for object in objects_list:
            # print(f"object idx: {object.original_order_index}, object_name: {object.class_name}, object_id: {object.object_id}, just_updated: {object.just_updated}")
        # Create new output list with the same length as original
        output_list_of_ids = []
        for idx in range(len(centroid_positions)):
            # print(f"idx: {idx}")
            # Find first valid (just updated object) index and return its id
            appended = False
            for object in objects_list:
                # print(f"object.original_order_index: {object.original_order_index}")
                if object.just_updated and object.original_order_index == idx:
                    output_list_of_ids.append(object.object_id)
                    # print(f"appending object.object_id: {object.object_id}")
                    appended = True
                    break
            if not appended:
                output_list_of_ids.append(-1)
        return output_list_of_ids

    # This function returns dictionary of classes with their objects
    # parsed_objects = {"cube": [<obj1>, <obj2>], "ball": [<>...], ....}
    def parse_objects_sim(self, centroid_positions, dimensions, class_names):
        parsed_objects = {}
        # Initialize empty dicitonaries of unique class_names
        for class_name in class_names:
            parsed_objects[class_name] = []
        # Add objects to the dictionary
        for class_name_i in range(len(class_names)):
            class_name = class_names[class_name_i]
            centroid_position = Position(x=centroid_positions[class_name_i][0], y=centroid_positions[class_name_i][1], z=centroid_positions[class_name_i][2])
            dimension = Dimensions(x=dimensions[class_name_i][0], y=dimensions[class_name_i][1], z=dimensions[class_name_i][2])

            random_uuid = random_alpha_numeric(16)
            if class_name == "leftWrist" or class_name == "rightWrist":
                self.avatar.update_avatar_object(avatar_object_name=class_name, np_position=centroid_position.get_numpy_array(), np_dimensions=dimension.get_numpy_array())
            else:
                tracked_obj = TrackedObject(class_name=class_name, centroid_position=centroid_position, dimensions=dimension, original_order_index=class_name_i, last_uuid=random_uuid ,latest_uuid=random_uuid)
                parsed_objects[class_name].append(tracked_obj)
        return parsed_objects

    # Return list of lists, where single element is a <Position> and new row
    # is a new trajectory
    def get_hand_trajectories(self):
        trajectories = []
        trajectory_objects = [self.avatar.avatar_objects["leftWrist"], self.avatar.avatar_objects["rightWrist"] ]
        for trajectory_object in trajectory_objects:
            print(f"** CREATING NEW TRAJECTORY FOR: {trajectory_object.object_name}")
            trajectories.append( trajectory_object.trajectory_memory.get_chronological_position_list() )
        return trajectories

    # Return list of lists, where single element is a <Position> and new row
    # is a new trajectory
    def get_spline_trajectories_positions(self):
        trajectories = []

        trajectory_objects = []
        for key in self.avatar.avatar_objects:
            trajectory_objects.append(self.avatar.avatar_objects[key])

        for trajectory_object in trajectory_objects:
            if (trajectory_object.trajectory_memory.queue.qsize() > 4):
                print(f"** CREATING NEW TRAJECTORY FOR: {trajectory_object.object_name}")
                trajectories.append( trajectory_object.trajectory_memory.get_spline_trajectory_positions() )
        return trajectories

    ############################################################################
    ############################################################################
