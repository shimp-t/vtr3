import os

import launch
import launch.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

osp = os.path


def generate_launch_description():

    vtr_grizzly = get_package_share_directory('vtr_grizzly')
    vtr_navigation = get_package_share_directory('vtr_navigation')
    vtr_testing = get_package_share_directory('vtr_testing_stereo')
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
    testing_config = osp.join(vtr_testing, 'config/localization.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('use_tdcp',
                              default_value='true',
                              description='Whether we want to read and publish TDCP msgs'),
        DeclareLaunchArgument('tdcp_data_dir',
                              default_value='${VTRDATA}/june16-tdcp',
                              description='Directory to find TDCP rosbag2'),
        DeclareLaunchArgument('tdcp_dataset',
                              default_value='june16b',
                              description='TDCP dataset/stream name'),
        Node(
            package='vtr_testing_stereo',
            namespace='vtr',
            executable='vtr_testing_stereo_localization_gps',
            # name='stereo_test',
            output='screen',
            # todo: figure out proper mapping/namespaces
            remappings=[("/vtr/tdcp", "/tdcp"), ("/vtr/query_trajectory", "/query_trajectory")],
            #   prefix=['xterm -e gdb --args'],
            parameters=[
                {
                    "use_tdcp": LaunchConfiguration("use_tdcp"),
                    "tdcp_data_dir": LaunchConfiguration("tdcp_data_dir"),
                    "tdcp_dataset": LaunchConfiguration("tdcp_dataset"),
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
                testing_config
            ]),
        # Launch grizzly description to get transformation matrices.
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                osp.join(vtr_grizzly,
                         "launch/grizzly_utias_description.launch.py"))),
    ])
