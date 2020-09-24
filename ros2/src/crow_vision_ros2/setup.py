from setuptools import setup

package_name = 'crow_vision_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/all_cameras.launch.py']),
        ('share/' + package_name, ['launch/cameras_detection.launch.py']),
        ('share/' + package_name, ['launch/dual_detection_cameras.launch.py']),
        ('share/' + package_name, ['launch/full_crow_object.launch.py']),
        ('share/' + package_name, ['launch/full_coco.launch.py']),
        ('share/' + package_name, ['launch/full_dual.launch.py']),
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
