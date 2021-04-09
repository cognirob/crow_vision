import yaml
import os
from importlib.util import find_spec

with open('camera_transformation_data.yaml') as f:
    data = yaml.safe_load(f)
    print(data)

def get_camera_transformation(serial_nr):
    if serial_nr in data:
        return data[serial_nr]
    else:
        return '0 0 0 0 0 0 1'
    