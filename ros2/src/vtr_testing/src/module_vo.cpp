#include <filesystem>

#include "rclcpp/rclcpp.hpp"

#include <vtr_common/utils/filesystem.hpp>
#include <vtr_logging/logging_init.hpp>
#include <vtr_testing/module_vo.hpp>

#if 0
#include <tf/transform_listener.h>

#include <robochunk_msgs/MessageBase.pb.h>
#include <robochunk_msgs/XB3CalibrationRequest.pb.h>
#include <robochunk_msgs/XB3CalibrationResponse.pb.h>
#include <robochunk/base/DataStream.hpp>
#include <robochunk/util/fileUtils.hpp>

#include <vtr/vision/messages/bridge.h>
#include <asrl/common/timing/SimpleTimer.hpp>
#endif

using namespace vtr::common::utils;
using namespace vtr::logging;
using RigImages = vtr_messages::msg::RigImages;
using RigCalibration = vtr_messages::msg::RigCalibration;

int main(int argc, char** argv) {
  // easylogging++ configuration
  configureLogging();

  LOG(INFO) << "Starting Module VO, beep beep beep";

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("module_vo");
  auto data_dir_str =
      node->declare_parameter<std::string>("input_data_dir", "");
  auto results_dir_str =
      node->declare_parameter<std::string>("results_dir", "");
  auto sim_run_str = node->declare_parameter<std::string>("sim_run", "");
  auto stream_name = node->declare_parameter<std::string>("stream_name", "");

  fs::path data_dir{expand_user(data_dir_str)};
  fs::path results_dir{expand_user(results_dir_str)};
  fs::path sim_run{expand_user(sim_run_str)};

  auto start_index = node->declare_parameter<int>("start_index", 1);
  auto stop_index = node->declare_parameter<int>("stop_index", 20000);

  ModuleVO vo(node, results_dir);

  vtr::storage::DataStreamReader<RigImages, RigCalibration> stereo_stream(
      data_dir.string(), stream_name);
  vtr::vision::RigCalibration rig_calibration;

  try {
    auto calibration_msg =
        stereo_stream.fetchCalibration()->get<RigCalibration>();
    rig_calibration = vtr::messages::copyCalibration(calibration_msg);
  } catch (vtr::storage::NoBagExistsException& e) {
    LOG(ERROR) << "No calibration message recorded! URI: "
               << e.get_directory().string();
    return -1;
  }

  vo.setCalibration(
      std::make_shared<vtr::vision::RigCalibration>(rig_calibration));

  bool seek_success =
      stereo_stream.seekByIndex(static_cast<int32_t>(start_index));
  if (!seek_success) {
    LOG(ERROR) << "Seek failed!";
    return 0;
  }

  std::shared_ptr<vtr::storage::VTRMessage> storage_msg;
  int idx = 0;
  while (idx + start_index < stop_index && rclcpp::ok()) {
    storage_msg = stereo_stream.readNextFromSeek();
    if (!storage_msg) {
      LOG(ERROR) << "Storage msg is nullptr!";
      break;
    }
    auto rig_images = storage_msg->template get<RigImages>();
    // \todo current datasets didn't fill vtr_header so need this line
    rig_images.vtr_header.sensor_time_stamp.nanoseconds_since_epoch =
        rig_images.channels[0].cameras[0].stamp.nanoseconds_since_epoch;
    auto timestamp = rig_images.vtr_header.sensor_time_stamp;
    LOG(INFO) << "\nProcessing image: " << idx;
    vo.processImageData(std::make_shared<RigImages>(rig_images), timestamp);
    idx++;
  }
  LOG(INFO) << "Time to exit!";
}