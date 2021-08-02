import datetime
from scipy.optimize import linear_sum_assignment
import numpy as np
from crow_vision_ros2.tracker.tracker_base import get_vector_length, random_alpha_numeric, Dimensions, Position, Color, ObjectsDistancePair
from crow_vision_ros2.tracker.tracker_config import DEFAULT_ALPHA_NUMERIC_LENGTH, MEMORY_LOSS_SECONDS

class TrackedObject:
    def __init__(self, class_name, centroid_position, dimensions, original_order_index, last_uuid, latest_uuid):
        self.class_name = class_name
        self.object_id = random_alpha_numeric(len=DEFAULT_ALPHA_NUMERIC_LENGTH)
        self.centroid_position = centroid_position
        self.dimensions = dimensions

        # Remembering original order for the return message
        self.original_order_index = original_order_index

        # Tracking logic variables
        self.active = True # Is the object currently in the "frame"
        self.just_updated = False
        self.position_history = [centroid_position]
        # Variable used in the duplication filter at the beginning
        self.duplicate = False
        # Variable used in the intersection filter
        self.intersects = False
        self.used_for_update = False

        # If there is a glitch detection, tracker will forget/throw out the object
        # after couple of frames
        self.detection_memory_loss = datetime.datetime.now()

        # Ros2 partficle integration
        self.last_uuid = last_uuid
        self.latest_uuid = latest_uuid

        # Build some mapping that long position history -> we believe this object more
    def update_by_object(self, object):
        self.centroid_position = object.centroid_position
        self.dimensions = object.dimensions
        self.active = True
        self.just_updated = True
        self.position_history.append(object.centroid_position)
        object.used_for_update = True
        self.original_order_index = object.original_order_index
        # Reset memory loss
        self.detection_memory_loss = datetime.datetime.now()
        # Ros2 particle filter
        self.last_uuid = self.latest_uuid
        self.latest_uuid = object.latest_uuid
        return

    # Used for output sorting
    def __lt__(self, other):
        return self.original_order_index < other.original_order_index

class Tracker:
    def __init__(self):
        # Dictionary of lists. Each class name has its own list of <TrackedObject>'s
        self.tracked_objects = {}

    # This function flags all the existing objects as not already updated
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
                        print("<class_duplication_filter >DUPLICATE DETECTED")

        # Remove duplicates -> should take care of 3+ duplicates at the same space
        for object in parsed_objects:
            if not object.duplicate:
                parsed_objects_cleaned.append(object)
                object.duplicate = False
        return parsed_objects_cleaned

    def add_tracked_object_logic(self, class_name, parsed_object):
        print(f"<tracker> !!! Creating new object")
        parsed_object.just_updated = True
        if not class_name in self.tracked_objects:
            self.tracked_objects[class_name] = [parsed_object]
        else:
            self.tracked_objects[class_name].append(parsed_object)

    # Checks intersection with existing and new and update them if neccessary,
    # returns objects which have not been yet assigned
    def intersection_filter(self, class_name, parsed_objects):
        smallest_dimensions = [] # Index-order sensitive
        # print(f"len(self.tracked_objects[class_name]): {len(self.tracked_objects[class_name])}")
        for tracked_object in self.tracked_objects[class_name]:
            # print(f"tracked_object.dimensions.get_tuple(): {tracked_object.dimensions.get_tuple()}")
            smallest_dimensions.append(tracked_object.dimensions.get_xy_min())
            # print(f"smallest_dimensions: {smallest_dimensions}")

        for tracked_object in self.tracked_objects[class_name]:
            closest_intersecting_with = None
            closest_intersecting_distance = float('inf')
            tracked_obj_pos = tracked_object.centroid_position
            for parsed_object in parsed_objects:
                parsed_object_pos = parsed_object.centroid_position
                # Check intersection ->(distance < smallest dimension)
                objects_distance = tracked_obj_pos.get_distance_to(other_position=parsed_object_pos)
                if objects_distance < smallest_dimensions[self.tracked_objects[class_name].index(tracked_object)]:
                    if not parsed_object.intersects:
                        if objects_distance < closest_intersecting_distance:
                            closest_intersecting_distance = objects_distance
                            closest_intersecting_with = parsed_object

            # Update tracked object with closest intersecting object
            if closest_intersecting_with != None:
                tracked_object.update_by_object(object=closest_intersecting_with)
                print(f"<intersection filter>: UPDATING DISTANCE")
                # print(f"tracked_object.just_updated: {tracked_object.just_updated}")
                closest_intersecting_with.intersects = True
        return

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
            if not tracked_object.just_updated:
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
                distance_pair.tracked_object.update_by_object(object=distance_pair.new_object)
        return

    def dissipate_detection_memory(self):
        for class_name in self.tracked_objects:
            for tracked_object in self.tracked_objects[class_name]:
                if not tracked_object.active:
                    # Check if we crossed the time limit already, if so, delete the object
                    time = datetime.datetime.now()
                    difference_seconds = (time - tracked_object.detection_memory_loss).total_seconds()
                    if difference_seconds > MEMORY_LOSS_SECONDS:
                        # Forget the object ~ remove from the list
                        print(f"*** Removing object: {class_name}")
                        self.tracked_objects[class_name].remove(tracked_object)
                    else:
                        print(f"*** Not yet crossed MEMORY_LOSS_SECONDS limit-> difference_seconds: {difference_seconds}")

    # This function evaluates existence based just on the class (works with list of objects),
    # results can later be evaluated in the next step as we know the object history
    # "which one is new"
    def evaluate_class_logic(self, class_name, parsed_objects):

        # Remove smaller items which are in THIS FRAME overlapping with bigger items of same class
        parsed_objects_cleaned = self.class_duplication_filter(parsed_objects=parsed_objects)

        # Check if this class has any tracked object, if not -> create new for all
        if not class_name in self.tracked_objects:
            for parsed_object in parsed_objects_cleaned:
                self.add_tracked_object_logic(class_name=class_name, parsed_object=parsed_object)
            return
        else:
            # Do intersection filter ~ i.e update object's position if its too close
            # to the newly recognized objects (basically does similar thing as distance_ordering_algorithm
            # but its neccessarry because it gives priority to the thought that objects who are very close
            # are the same, assignment woulnt take that into consideration)
            self.intersection_filter(class_name=class_name, parsed_objects=parsed_objects_cleaned)

            count_intersects = 0
            for parsed_object in parsed_objects_cleaned:
                if parsed_object.intersects: count_intersects += 1
            print(f"{class_name}: has {count_intersects} intersects")

            # Distance ordering and pairing - For objects which haven't already
            # been updated with new detections -> do an algorithm which orders
            # all possible distances within class and new detection so that we
            # know <tracked object> <- DISTANCE -> <newly detected>,
            # closest distances with are updated first
            self.distance_ordering_algorithm(class_name=class_name, parsed_objects=parsed_objects_cleaned)

            # If there are still some new detection which havent been used in updating
            # and there are more detections than existing objects
            # create new track objects for them
            if len(parsed_objects_cleaned) > len(self.tracked_objects[class_name]):
                for parsed_object in parsed_objects_cleaned:
                    if not parsed_object.used_for_update:
                        self.add_tracked_object_logic(class_name=class_name, parsed_object=parsed_object)
        return


    # Returns index-sensitive mapping of old uuids to new uuids ([a,b, ...],[c,b, ...])
    # 'b' should be 'a', 'c' should be 'a'
    def track_and_get_uuids(self, centroid_positions, dimensions, class_names, uuids):
        # Flag all existing objects as not (updated) this frame
        self.reset_flags()

        # Get scene objects <Class:TrackedObject> divided into classes
        parsed_objects_dict = self.parse_objects(centroid_positions=centroid_positions, dimensions=dimensions, class_names=class_names, uuids=uuids)

        for class_name in parsed_objects_dict:
            self.evaluate_class_logic(class_name=class_name, parsed_objects=parsed_objects_dict[class_name])

            ### POSSIBLE PROBLEMS - interclass problems, the fact that we do not
            # take into consideration old and not active objects as a criteria
            # when updating -> new feature

        # Dissipate memory of glitch detections
        self.dissipate_detection_memory()

        # Return id's with original order, if some detection is disregarded,
        # return -1
        # Dump all object instances into 1 list
        objects_list = self.dump_list_of_objects()
        # Sort it by initial index order
        objects_list.sort()
        # for object in objects_list:
            # print(f"object idx: {object.original_order_index}, object_name: {object.class_name}, object_id: {object.object_id}, just_updated: {object.just_updated}")
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

        return (last_uuid, latest_uuid)
