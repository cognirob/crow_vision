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

#### Install env

Tested on:
- Ubuntu linux 18.04
- obtain [Anaconda python](https://www.anaconda.com/distribution/)
- install preset environment: `conda env create -f environment.yml && conda activate ros2`

#### Install YOLACT

Make sure you have the repo `crow_vision_yolact` setup at path `~/crow_vision_yolact`.
If not: 
```sh
cd ~
git clone https://gitlab.ciirc.cvut.cz/imitrob/project_crow/crow_vision_yolact.git

# post-setup: 
# download weights as needed (see the repo's README
#TODO add script to auto fetch needed weights

#validate everything works
cd crow_vision_yolact
conda activate ros2
python eval.py --trained_model=weights/yolact_base_54_800000.pth --config=yolact_base_config --score_threshold=0.15 --top_k=15 --image=my_image.png
```

#### Usage

See [YOLACT's original README](external/yolact/README.md)

```sh
# Run on a single image
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.jpeg

# Obtain our custom model (weights)
python eval.py --trained_model=weights/<custom weights>.pth --score_threshold=0.15 --top_k=15 --config=<custom config matching the new weights> --image=my_image.jpeg
```



### ROS2

#### Install 
We use ROS2 (because we're cool, and also it works cross-platform). 
For installation and setup, see [our wiki on ROS2](https://gitlab.ciirc.cvut.cz/imitrob/project_crow/crow/-/wikis/ros-guide).

#### Build CROW ROS2 packages

```sh
# clone this repo:
cd ~
git clone https://gitlab.ciirc.cvut.cz/imitrob/project_crow/crow_vision.git
cd crow_vision

# source ROS2 paths
source /opt/ros/eloquent/setup.bash

# (optional) download other ROS packages' sources
mkdir -p crow_ws/src && cd crow_ws && rm -rf build/ install/ 
wstool update -t --from-path ../ros2/src/

# (re)build ROS2 packages from sources
rm -rf build/ && colcon build --symlink-install && . install/setup.sh
```

#### Packages & Run

The following packages are exposed as ROS nodes: 

- Intel RS camera: TODO
- CNN: `ros2 run crow_vision_ros2 detector &`
- image publisher: `ros2 run crow_image_streamer_ros2 image_streamer_node '/tmp/images2/'`



### Intel RealSense camera

We're using the `R450i` camera. 

#### Install

The external ROS package is automatically built, use `wstool update -t --from-path ../ros2/src/`  (from your workspace).
See [our ROS wiki how to build the Inter RealSense cam](https://gitlab.ciirc.cvut.cz/imitrob/project_crow/crow/-/wikis/ros-guide#ros-realsense). 

[Usage and documentation](https://github.com/intel/ros2_intel_realsense#usage-instructions).

