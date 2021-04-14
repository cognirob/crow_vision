import yaml
import os
from importlib.util import find_spec


class CameraGlobalTFGetter():

    def __init__(self, config_path='camera_transformation_data.yaml'):
        with open(config_path) as f:
            self.data = yaml.safe_load(f)
            print(self.data)

    def get_camera_transformation(self, serial_nr):
        if serial_nr[0] is "_":
            serial_nr = serial_nr[1:]
        if serial_nr in self.data:
            return self.data[serial_nr]
        else:
            return '0 0 0 0 0 0 1'
