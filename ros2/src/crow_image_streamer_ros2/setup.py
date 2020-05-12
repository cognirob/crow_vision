from setuptools import setup

package_name = 'crow_image_streamer_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='syxtreme',
    maintainer_email='radoslav.skoviera@cvut.cz',
    description='ROS2 publisher node. Streams images from a folder, alternative to a crow_camera_ros2 node. Used eg. for training neural nets for vision.',
    license='AGPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_streamer_node = crow_image_streamer_ros2.image_streamer_node:main'
        ],
    },
)
