# vtr_bumblebee_xb3

Drivers for sensors used in VTR3.
Currently, only the BumblebeeXB3 stereo camera is implemented however, it should easily extend to other sensors.

For XB3, the dc1394 Ubuntu package is required.
The driver grabs images from the camera, converts them from Bayer, then rectifies and resizes them.
Finally, it publishes them as a custom ROS2 RigImages msg on the _images_ topic.
Parameters allow you to visualize the images.

Usage:

```bash
cd ~/ASRL/vtr3/ros2/
ros2 run vtr_bumblebee_xb3 BumblebeeDriver --ros-args --params-file src/vtr_bumblebee_xb3/param/XB3.yaml
```
