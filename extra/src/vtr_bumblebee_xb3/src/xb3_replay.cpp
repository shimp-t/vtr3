#include <vtr_bumblebee_xb3/xb3_replay.hpp>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace fs = std::filesystem;

Xb3Replay::Xb3Replay(const std::string &data_dir,
                     const std::string &stream_name, const std::string &topic,
                     const int qos)
    : Node("xb3_recorder"), reader_(data_dir, stream_name) {
  publisher_ = create_publisher<RigImageCalib>(topic, qos);
  clock_publisher_ = create_publisher<rosgraph_msgs::msg::Clock>("clock", 10);
}

/// @brief Replay XB3 stereo images from a rosbag2
int main(int argc, char *argv[]) {

  std::string path = argv[argc-1];
  std::cout << path << "\n";

  fs::path run_dir{path};

  fs::path data_dir{run_dir};
  std::string stream_name = "front_xb3";
  bool manual_scrub = false;

  bool use_original_timestamps = true;
  double framerate = 16.0;

  // temp
  double delay_scale = 1.0;  // make playblack slower
  double time_shift = 0;     // shift time stamp for repeat
  int start_index = 1;
  int stop_index = 9999999;

  rclcpp::init(argc, argv);
  std::cout << data_dir.string() << "\n";
  auto replay = Xb3Replay(data_dir.string(), stream_name, "images");

  replay.reader_.seekByIndex(start_index);

  int curr_index = start_index;


  float x_extr = -0.239946;
  std::vector<double> k_mat = {388.425, 0.0, 253.502, 0.0, 388.425, 196.822, 0.0, 0.0, 1.0};

  vtr_messages::msg::RigCalibration rig_calibration_message;
  vtr_messages::msg::CameraCalibration camera_calibration_message;
  vtr_messages::msg::Transform transform_message_left;
  vtr_messages::msg::Transform transform_message_right;
  vtr_messages::msg::Vec3 vec3_message_left;
  vtr_messages::msg::Vec3 vec3_message_right;
  vtr_messages::msg::AxisAngle axis_angle_message;
  
  vec3_message_left.x = 0.0;
  vec3_message_left.y = 0.0;
  vec3_message_left.z = 0.0;

  vec3_message_right.x = x_extr;
  vec3_message_right.y = 0.0;
  vec3_message_right.z = 0.0;

  axis_angle_message.x = 0.0;
  axis_angle_message.y = 0.0;
  axis_angle_message.z = 0.0;

  transform_message_left.translation = vec3_message_left;
  transform_message_left.orientation = axis_angle_message;

  transform_message_right.translation = vec3_message_right;
  transform_message_right.orientation = axis_angle_message;

  camera_calibration_message.k_mat = k_mat;

  rig_calibration_message.intrinsics.push_back(camera_calibration_message); // Same k_mat for left and right camera
  rig_calibration_message.intrinsics.push_back(camera_calibration_message);
  rig_calibration_message.extrinsics.push_back(transform_message_left);
  rig_calibration_message.extrinsics.push_back(transform_message_right);


  uint64_t prev_stamp = 0;
  while (true) {
    if (!rclcpp::ok()) break;

    if (curr_index == stop_index) break;
    curr_index++;

    auto message = replay.reader_.readNextFromSeek();
    if (!message.get()) break;

    auto image = message->template get<RigImages>();

    
    // \todo yuchen Add necessary info for vtr to run, but they should not be
    // here
    image.name = "front_xb3";
    image.vtr_header.sensor_time_stamp = image.channels[0].cameras[0].stamp;
    image.vtr_header.sensor_time_stamp.nanoseconds_since_epoch +=
        1e12 * time_shift;

    rosgraph_msgs::msg::Clock clock_msg;
    clock_msg.clock.sec =
        image.vtr_header.sensor_time_stamp.nanoseconds_since_epoch / 1e9;
    clock_msg.clock.nanosec =
        image.vtr_header.sensor_time_stamp.nanoseconds_since_epoch % (long)1e9;
    // Publish time to /clock for nodes using sim_time
    replay.clock_publisher_->publish(clock_msg);

    vtr_messages::msg::RigImageCalib rig_image_calib_message;
    rig_image_calib_message.rig_images = image;
    rig_image_calib_message.rig_calibration = rig_calibration_message;

    std::cout << "Publishing image with time stamp: "
              << image.vtr_header.sensor_time_stamp.nanoseconds_since_epoch
              << " and index is " << curr_index << std::endl;

    // Publish message for use with offline tools
    replay.publisher_->publish(rig_image_calib_message);
    // Add a delay so that the image publishes at roughly the true rate.
    // std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto left = image.channels[0].cameras[0];
    auto right = image.channels[0].cameras[1];

    // Replays images based on their timestamps. Converts nanoseconds to
    // milliseconds
    if (prev_stamp != 0) {
      double delay = use_original_timestamps
                         ? (left.stamp.nanoseconds_since_epoch - prev_stamp) *
                               std::pow(10, -6)
                         : 1000.0 / framerate;
      delay *= delay_scale;
      cv::waitKey(delay);
    }
    prev_stamp = left.stamp.nanoseconds_since_epoch;

  }
  rclcpp::shutdown();
}