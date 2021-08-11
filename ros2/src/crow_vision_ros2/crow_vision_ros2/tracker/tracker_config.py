DEFAULT_ALPHA_NUMERIC_LENGTH = 4
# If there is a glitch detection, it will delete it after some time/frames
# so the space doesn't become crowded with glitches
MEMORY_LOSS_SECONDS = 6 # After how many seconds should tracker delete
# its inactive objects

TRAJECTORY_MEMORY_SIZE_SECONDS = 4

# If our object is closer to the wrist than this distance (before disappearing),
# set the dissipation memory of that object to
WRIST_OBJECT_CLIP_DISTANCE_LIMIT = 500
OBJECT_CLOSE_TO_HAND_MEMORY_LOSS_SECONDS = 60
HAND_OBJECT_MEMORY_LOSS = 120
