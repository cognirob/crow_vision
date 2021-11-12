# Not important for ROS2 but used for development simulation
DEFAULT_ALPHA_NUMERIC_LENGTH = 4

# Size in seconds as to how long should the trajectory
TRAJECTORY_MEMORY_SIZE_SECONDS = 5

# If our object is closer to the wrist than this distance (before disappearing),
# set the dissipation memory of that object to
WRIST_OBJECT_CLIP_DISTANCE_LIMIT = 0.21

# How many detection needed for setting up the scene objects
DETECTIONS_FOR_SETUP_NEEDED = 50
PERCENTAGE_NEEDED_TO_BE_APPROVED = 0.1
SETUP_OVERLAP_DISTANCE_MAX = 0.1  # during setup, if two entries with the same uuid are further appart, it will rise error
SETUP_OVERLAP_DISTANCE_MERGE = 0.03  # merge objects with different uuids if they are closer then this distance

TRACKER_ITERATION_HASH_LENGTH = 64
