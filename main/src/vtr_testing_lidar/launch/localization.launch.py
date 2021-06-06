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
  # configs
  base_config = osp.join(vtr_navigation, 'config/lidar/base/base.yaml')
  scenario_config = osp.join(vtr_navigation, 'config/lidar/scenario')

  return LaunchDescription([
      DeclareLaunchArgument('data_dir', description='Data directory'),
      DeclareLaunchArgument('scenario_params',
                            description='Run and data params'),
      DeclareLaunchArgument('clear_data_dir',
                            default_value='false',
                            description='Clear the data dir before launch VTR'),
      Node(
          package='vtr_testing_lidar',
          namespace='vtr',
          executable='vtr_testing_lidar_localization',
          output='screen',
          # prefix=['xterm -e gdb --args'],
          parameters=[
              {
                  "data_dir": LaunchConfiguration("data_dir"),
                  "clear_data_dir": LaunchConfiguration("clear_data_dir"),
                  #   "use_sim_time": LaunchConfiguration("use_sim_time"),
              },
              # base_config
              base_config,
              # scenario specific configs
              PathJoinSubstitution(
                  (scenario_config, LaunchConfiguration("scenario_params")))
          ])
  ])
