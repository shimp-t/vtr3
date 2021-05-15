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
  base_config = osp.join(vtr_navigation, 'config/camera/base')
  base_converter_config = list(
      map(lambda x: osp.join(base_config, "converter", x), [
          "extraction_surf.yaml",
          "image_triangulation.yaml",
      ]))
  base_odometry_config = list(
      map(lambda x: osp.join(base_config, "odometry", x), [
          "asrl_stereo_matcher.yaml",
          "stereo_ransac.yaml",
          "keyframe_opt.yaml",
          "vertex_test.yaml",
      ]))
  base_bundle_adjustment_config = list(
      map(lambda x: osp.join(base_config, "bundle_adjustment", x), [
          "window_opt.yaml",
      ]))
  base_localization_config = list(
      map(lambda x: osp.join(base_config, "localization", x), [
          "map_extraction.yaml",
          "time_of_day_recognition.yaml",
          "experience_triage.yaml",
          "mel_matcher.yaml",
          "stereo_ransac.yaml",
          "mel_opt.yaml",
      ]))
  # robot specific configs
  grizzly_config = osp.join(vtr_navigation, 'config/camera/grizzly')
  grizzly_converter_config = list(
      map(lambda x: osp.join(grizzly_config, "converter", x), [
          "extraction_surf.yaml",
          "image_triangulation.yaml",
      ]))
  grizzly_odometry_config = list(
      map(lambda x: osp.join(grizzly_config, "odometry", x), [
          "asrl_stereo_matcher.yaml",
          "stereo_ransac.yaml",
          "keyframe_opt.yaml",
          "vertex_test.yaml",
      ]))
  grizzly_bundle_adjustment_config = list(
      map(lambda x: osp.join(grizzly_config, "bundle_adjustment", x), [
          "window_opt.yaml",
      ]))
  grizzly_localization_config = list(
      map(lambda x: osp.join(grizzly_config, "localization", x), [
          "map_extraction.yaml",
          "time_of_day_recognition.yaml",
          "experience_triage.yaml",
          "mel_matcher.yaml",
          "stereo_ransac.yaml",
          "mel_opt.yaml",
      ]))
  # scenario specific configs
  scenario_config = osp.join(vtr_navigation, 'config/camera/scenario')

  return LaunchDescription([
      DeclareLaunchArgument('data_dir', description='Data directory'),
      DeclareLaunchArgument('scenario_params',
                            description='Run and data params'),
      Node(
          package='vtr_navigation',
          namespace='vtr',
          executable='vtr_navigation',
          # name='stereo_test',
          output='screen',
          #   prefix=['xterm -e gdb --args'],
          parameters=[
              {
                  "data_dir": LaunchConfiguration("data_dir"),
              },
              # base_configs
              *base_converter_config,
              *base_odometry_config,
              *base_bundle_adjustment_config,
              *base_localization_config,
              # robot specific configs
              *grizzly_converter_config,
              *grizzly_odometry_config,
              *grizzly_bundle_adjustment_config,
              *grizzly_localization_config,
              # scenario specific configs
              PathJoinSubstitution(
                  (scenario_config, LaunchConfiguration("scenario_params")))
          ]),
  ])