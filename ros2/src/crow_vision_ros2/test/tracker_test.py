import numpy as np
from uuid import uuid4
from crow_vision_ros2.tracker import Tracker
from crow_ontology.crowracle_client import CrowtologyClient


CROW_OBJECTS = ["cube_holes", "sphere_holes", "wheel", "wafer"]
crw = CrowtologyClient()
tracker = Tracker(crw)
tracker.DETECTIONS_FOR_SETUP_NEEDED = 10

NO = 20
CROW_OBJECTS = ["cube_holes", "sphere_holes", "wheel", "wafer"]
centroids = []
positions = []
cls_names = []
uuids = []

for oi in range(NO):
    cent = np.random.randn(3) * 2
    pos = np.random.randn(3) * 0.1
    cln = CROW_OBJECTS[np.random.randint(0, len(CROW_OBJECTS))]
    uid = str(uuid4())
    centroids.append(cent)
    positions.append(pos)
    cls_names.append(cln)
    uuids.append(uid)

for i in range(10):
    last_uuids, original_uuids = tracker.track_and_get_uuids(centroids, positions, cls_names, uuids)

for i in range(10):
    centroids = []
    positions = []
    new_uuids = []
    for oi in range(NO):
        centroids.append(np.random.randn(3) * 2)
        positions.append(np.random.randn(3) * 0.1)
        if np.random.randn() > 0:
            uid = str(uuid4())
            new_uuids.append(uid)
        else:
            new_uuids.append(uuids[oi])

    last_uuids, original_uuids = tracker.track_and_get_uuids(centroids, positions, cls_names, new_uuids)
    print(last_uuids, original_uuids)