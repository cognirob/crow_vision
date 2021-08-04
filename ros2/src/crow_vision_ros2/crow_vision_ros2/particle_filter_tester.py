import random
import numpy as np
from filters.particle_filter import ParticleFilter
from tracker.tracker import Tracker

class Position:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z
    def get_tuple(self):
        return (self.x, self.y, self.z)
    def get_list(self):
        return [self.x, self.y, self.z]

class PclGenerator:
    def __init__(self, position, class_id, score, pcl_count, delta):
        self.class_id = class_id
        self.score = score
        self.position = position
        self.delta = delta

        self.pcl_count = pcl_count

    def generate_point(self):
        rnd_x = self.position.x + self.delta.x * ((random.random()-1)*2)
        rnd_y = self.position.y + self.delta.y * ((random.random()-1)*2)
        rnd_z = self.position.z + self.delta.z * ((random.random()-1)*2)
        return [rnd_x, rnd_y, rnd_z]
    def generate_pcl(self):
        pcl_points = []
        for i in range(self.pcl_count):
            pcl_points.append(self.generate_point())

        return pcl_points
    def get_particle_input_add_measurement(self):
        pcl = self.generate_pcl()
        return (np.array(pcl), self.class_id, self.score)

object_properties = {
    0: {
        "name": "car_roof",
        "sigma": 0.01,
        "color": np.r_[69, 39, 160] / 255
    },
    1: {
        "name": "cube_holes",
        "sigma": 0.01,
        "color": np.r_[0, 255, 0] / 255
    },
    2: {
        "name": "ex_bucket",
        "sigma": 0.01,
        "color": np.r_[221, 160, 221] / 255
    },
    3: {
        "name": "hammer",
        "sigma": 0.01,
        "color": np.r_[255, 0, 0] / 255
    },
    4: {
        "name": "nut",
        "sigma": 0.01,
        "color": np.r_[0, 153, 255] / 255
    },
    5: {
        "name": "peg_screw",
        "sigma": 0.01,
        "color": np.r_[128, 128, 128] / 255
    },
    6: {
        "name": "peg_simple",
        "sigma": 0.01,
        "color": np.r_[128, 128, 128] / 255
    },
    7: {
        "name": "pliers",
        "sigma": 0.01,
        "color": np.r_[255, 255, 0] / 255
    },
    8: {
        "name": "screw_round",
        "sigma": 0.01,
        "color": np.r_[0, 0, 255] / 255
    },
    9: {
        "name": "screwdriver",
        "sigma": 0.01,
        "color": np.r_[250, 128, 114] / 255
    },
    10: {
        "name": "sphere_holes",
        "sigma": 0.01,
        "color": np.r_[204, 153, 0] / 255
    },
    11: {
        "name": "wafer",
        "sigma": 0.01,
        "color": np.r_[255, 0, 255] / 255
    },
    12: {
        "name": "wheel",
        "sigma": 0.01,
        "color": np.r_[102, 0, 0] / 255
    },
    13: {
        "name": "wrench",
        "sigma": 0.01,
        "color": np.r_[215, 0, 215] / 255
    },
    14: {
        "name": "hand",
        "sigma": 0.01,
        "color": np.r_[1, 1, 1] / 255
    },
    15: {
        "name": "kuka",
        "sigma": 0.01,
        "color": np.r_[255, 160, 16] / 255
    },
    16: {
        "name": "hammer_handle",
        "sigma": 0.01,  # TODO,
        "color": np.r_[1, 1, 1] / 255
    },
    17: {
        "name": "hammer_head",
        "sigma": 0.01,
        "color": np.r_[255, 0, 0] / 255
    },
    18: {
        "name": "pliers_handle",
        "sigma": 0.01,  # TODO,
        "color": np.r_[1, 1, 1] / 255
    },
    19: {
        "name": "pliers_head",
        "sigma": 0.01,
        "color": np.r_[255, 255, 0] / 255
    },
    20: {
        "name": "screw_round_thread",
        "sigma": 0.01,  # TODO
        "color": np.r_[1, 1, 1] / 255
    },
    21: {
        "name": "screw_round_head",
        "sigma": 0.01,
        "color": np.r_[0, 0, 255] / 255
    },
    22: {
        "name": "screwdriver_handle",
        "sigma": 0.01,  # TODO
        "color": np.r_[1, 1, 1] / 255
    },
    23: {
        "name": "screwdriver_head",
        "sigma": 0.01,
        "color": np.r_[250, 128, 114] / 255
    },
    24: {
        "name": "wrench_handle",
        "sigma": 0.01,  # TODO
        "color": np.r_[1, 1, 1] / 255
    },
    25: {
        "name": "wrench_open",
        "sigma": 0.01,
        "color": np.r_[215, 0, 215] / 255
    },
    26: {
        "name": "wrench_ring",
        "sigma": 0.01,
        "color": np.r_[215, 0, 215] / 255
    },
    27: {
        "name": "peg_screw_shank",
        "sigma": 0.01,  # TODO
        "color": np.r_[1, 1, 1] / 255
    },
    28: {
        "name": "peg_screw_thread",
        "sigma": 0.01,
        "color": np.r_[128, 128, 128] / 255
    },
    29: {
        "name": "kuka_gripper",
        "sigma": 0.01,  # TODO
        "color": np.r_[1, 1, 1] / 255
    },
}
particle_filter = ParticleFilter(object_properties=object_properties)

generator_position = Position(x=0,y=0,z=0)
pcl_generator = PclGenerator(position=generator_position, class_id=0, score=1, pcl_count=300, delta=Position(x=0.1,y=0.1,z=0.1))

generator_position2 = Position(x=0,y=100,z=0)
pcl_generator2 = PclGenerator(position=generator_position2, class_id=1, score=1, pcl_count=300, delta=Position(x=0.1,y=0.1,z=0.1))


tracker = Tracker()

i = 0
while True:
    print("")
    print(f"*** ITERATION: #i : {i}")
    i += 1
    if i == 500:
        # Change position
        new_generator_position = Position(x=100,y=3,z=3)
        pcl_generator.position = new_generator_position

    if i == 1050:
        # Change position
        new_generator_position = Position(x=0,y=3,z=200)
        pcl_generator2.position = new_generator_position

    pcl, class_id, score = pcl_generator.get_particle_input_add_measurement()
    particle_filter.add_measurement((pcl,class_id, score))
    pcl2, class_id2, score2 = pcl_generator2.get_particle_input_add_measurement()
    particle_filter.add_measurement((pcl2,class_id2, score2))
    particle_filter.update()

    # particle_filter.update()

    estimates = particle_filter.getEstimates()

    poses_formatted, class_names_formatted, dimensions_formatted, uuids_formatted = ([],[],[],[])
    for pose, label, dims, uuid in estimates:

        poses_formatted.append(pose.tolist())
        print(f"* pose.tolist(): {pose.tolist()}")
        print(f"* uuid: {uuid}")
        class_names_formatted.append(label)
        dimensions_formatted.append(dims)
        uuids_formatted.append(uuid)

    print(f"*-* Poses formated: {poses_formatted}")

    last_uuid, latest_uuid = tracker.track_and_get_uuids( centroid_positions=poses_formatted, dimensions=dimensions_formatted, class_names=class_names_formatted, uuids=uuids_formatted)

    print("Returned from tracker: ")
    print(f"last_uuid: {last_uuid}")
    print(f"latest_uuid: {latest_uuid}")

    # Update by uuids
    particle_filter._correct_model_uuids(last_uuids=last_uuid, latest_uuids=latest_uuid)
