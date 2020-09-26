from setuptools import setup
from glob import glob
import os


package_name = 'crow_vision_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='syxtreme',
    maintainer_email='radoslav.skoviera@cvut.cz',
    description='ROS2 interface for CROW vision (Yolact)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = crow_vision_ros2.detector:main',
            'calibrator = crow_vision_ros2.calibrator:main',
            'locator = crow_vision_ros2.locator:main',
            'matcher = crow_vision_ros2.match3d:main',
            'merger = crow_vision_ros2.merger:main'
        ],
    },
)
