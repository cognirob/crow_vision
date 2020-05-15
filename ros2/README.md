# ROS2 packages for CROW vision

### External packages

We can automatically build and install external ROS packages from source. 
Specify them in `/ros2/src/.rosinstall` file, and then build in your workspace as:
```
# assume ./dev_ws :
mkdir -p dev_ws
cd dev_ws

wstool update -t --from-path ../ros2/src/
```

- Intel RealSense camera package is built this way.
