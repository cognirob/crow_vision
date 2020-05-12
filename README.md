# crow_vision

This repo contains all computer vision code for the CROW project. 
We are using Detectron2 and Yolact as DNN vision frameworks. 

## Objective

Object recognition (on RGB camera, real-time), segmentation (bounding boxes/bboxes & pixel masks) -> conversion to 3D point-clouds/bboxes. 

### Components

- CNN (YOLACT, Detectron2)
- ROS2 (as communication framework)

## Installation

### YOLACT 

#### Prerequisities

- nvidia-driver(-418) (`apt install nvidia-driver-418`)
- [CUDA 10](https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork)
- you can also install `apt install nvidia-headless-440` and don't need CUDA! (the driver contains CUDA runtime)

#### Install YOLACT

##### Install env

Tested on:
- Ubuntu linux 18.04
- obtain [Anaconda python](https://www.anaconda.com/distribution/)
- install preset environment: `conda env create -f yolact.yml && conda activate --stack yolact-env`
- you can use YOLACT(++) following its [README](external/yolact/README.md)

##### Install YOLACT

TODO currently we're using yolact upstream from a separate repo. 

### ROS2

We use ROS2 (because we're cool, and also it works cross-platform). 
For installation and setup, see [our wiki on ROS2](https://gitlab.ciirc.cvut.cz/imitrob/project_crow/crow/-/wikis/ros-guide).

#### Packages

The following packages are exposed as ROS nodes: 

- RS camera: TODO
- image publisher: `ros2 run crow_image_streamer_ros2 image_streamer_node '/tmp/images2/'`
- CNN: TODO

#### Usage

See [YOLACT's original README](external/yolact/README.md)

##### Run on a single image
`python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.jpeg`

##### Obtain our custom model (weights)

`python eval.py --trained_model=weights/<custom weights>.pth --score_threshold=0.15 --top_k=15 --config=<custom config matching the new weights> --image=my_image.jpeg`

### Intel RealSense camera

TODO


