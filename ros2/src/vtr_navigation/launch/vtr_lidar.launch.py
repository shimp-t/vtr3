import os

osp = os.path

import launch
import launch.actions
import launch.substitutions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
  vtr_navigation = get_package_share_directory('vtr_navigation')
  # base configs
  base_config = osp.join(vtr_navigation, 'config/lidar')

  return LaunchDescription([
      DeclareLaunchArgument('params', description='Run and data params'),
      Node(
          package='vtr_navigation',
          namespace='vtr',
          executable='vtr_navigation',
          output='screen',
          remappings=[("/cmd_vel", "/grizzly_velocity_controller/cmd_vel")],
          #   prefix=['xterm -e gdb --args'],
          parameters=[
              {
                  "data_dir": LaunchConfiguration("data_dir"),
                  #   "use_sim_time": LaunchConfiguration("use_sim_time"),
              },
              # configs
              PathJoinSubstitution((base_config, LaunchConfiguration("params")))
          ])
  ])
