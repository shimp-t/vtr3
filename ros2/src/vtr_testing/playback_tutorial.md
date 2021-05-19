# ASRL Playback Tool Tutorial

## Launch the entire system

Remember to source the workspace.

Start the whole vtr system

```bash
cd ~/ASRL/vtr3/ros2/src/vtr_navigation/tmuxp
tmuxp load playback-vtr3.yaml
```

See the bottom panel for how to unpause the system and start teach.

Start a replay tool that keeps publishing images from a dataset. Run both calibration and replay in two terminals.

Calibration service:

```bash
source ~/ASRL/venv/bin/activate
source ~/ASRL/vtr3/ros2/install/setup.bash
ros2 run  vtr_sensors BumblebeeCalibration ~/ASRL/dataset/nov4_a front_xb3
```

Replay:

```bash
source ~/ASRL/venv/bin/activate
source ~/ASRL/vtr3/ros2/install/setup.bash
ros2 run  vtr_sensors BumblebeeReplay ~/ASRL/dataset/nov4_a front_xb3 false
```