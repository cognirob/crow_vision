# Not important for ROS2 but used for development simulation
DEFAULT_ALPHA_NUMERIC_LENGTH = 4

# Size in seconds as to how long should the trajectory
TRAJECTORY_MEMORY_SIZE_SECONDS = 3

# If our object is closer to the wrist than this distance (before disappearing),
# set the dissipation memory of that object to
WRIST_OBJECT_CLIP_DISTANCE_LIMIT = 0.11

# How many detection needed for setting up the scene objects
DETECTIONS_FOR_SETUP_NEEDED = 50
PERCENTAGE_NEEDED_TO_BE_APPROVED = 0.3

TRACKER_ITERATION_HASH_LENGTH = 64
