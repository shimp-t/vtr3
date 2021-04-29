#include <filesystem>

#include <rclcpp/rclcpp.hpp>

#include <vtr_common/utils/filesystem.hpp>
#include <vtr_logging/logging_init.hpp>
#include <vtr_testing/module_vo.hpp>

using namespace vtr;
using RigImages = vtr_messages::msg::RigImages;
using RigCalibration = vtr_messages::msg::RigCalibration;

int main(int argc, char **argv) {
  // easylogging++ configuration
  logging::configureLogging();

  LOG(INFO) << "Starting Module VO with GPS, beep beep beep";

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("module_vo_gps");

  // image parameters and setup
  auto data_dir_str =
      node->declare_parameter<std::string>("input_data_dir", "");
  auto results_dir_str =
      node->declare_parameter<std::string>("results_dir", "");
  auto image_stream_name = node->declare_parameter<std::string>("stream_name", "");

  fs::path data_dir{common::utils::expand_user(data_dir_str)};
  fs::path results_dir{common::utils::expand_user(results_dir_str)};

  auto start_index = node->declare_parameter<int>("start_index", 1);
  auto stop_index = node->declare_parameter<int>("stop_index", 20000);

  storage::DataStreamReader<RigImages, RigCalibration> stereo_stream(
      data_dir.string(), image_stream_name);
  vision::RigCalibration rig_calibration;

  try {
    auto calibration_msg =
        stereo_stream.fetchCalibration()->get<RigCalibration>();
    rig_calibration = messages::copyCalibration(calibration_msg);
  } catch (storage::NoBagExistsException &e) {
    LOG(ERROR) << "No calibration message recorded! URI: "
               << e.get_directory().string();
    return -1;
  }

  // gps parameters and setup
  auto use_tdcp = node->declare_parameter<bool>("use_tdcp", false);
  auto tdcp_data_dir_str = node->declare_parameter<std::string>("tdcp_data_dir", "");
  auto tdcp_stream_name = node->declare_parameter<std::string>("tdcp_stream_name", "");
  auto use_gpgga = node->declare_parameter<bool>("use_gpgga", false);
  auto gpgga_data_dir_str = node->declare_parameter<std::string>("gpgga_data_dir", "");
  auto gpgga_stream_name = node->declare_parameter<std::string>("gpgga_stream_name", "");

  fs::path tdcp_dir, gpgga_dir;
  std::shared_ptr<storage::DataStreamReader<TdcpMsg>> tdcp_stream;
  std::shared_ptr<storage::DataStreamReader<GpggaMsg>> gpgga_stream;

  if (use_tdcp) {
    LOG(INFO) << "Using time-differenced carrier phase measurements.";
    tdcp_dir = fs::path{common::utils::expand_user(tdcp_data_dir_str)};
    tdcp_stream = std::make_shared<storage::DataStreamReader<TdcpMsg>>(tdcp_dir.string(), tdcp_stream_name);
  }
  if (use_gpgga) {
    LOG(INFO) << "Logging GNSS position measurements.";
    gpgga_dir = fs::path{common::utils::expand_user(gpgga_data_dir_str)};
    gpgga_stream = std::make_shared<storage::DataStreamReader<GpggaMsg>>(gpgga_dir.string(), gpgga_stream_name);
  }

  ModuleVO vo(node, results_dir);

  vo.setCalibration(std::make_shared<vision::RigCalibration>(rig_calibration));

  std::shared_ptr<storage::VTRMessage> image_msg;
  std::shared_ptr<storage::VTRMessage> gpgga_msg;
  std::shared_ptr<storage::VTRMessage> tdcp_msg;
  int image_idx = 0;
  int tdcp_idx = 0;
  int gpgga_idx = 0;

  // get first image
  bool seek_success = stereo_stream.seekByIndex(static_cast<int32_t>(start_index));
  if (!seek_success) {
    LOG(ERROR) << "Image seek failed!";
    return 0;
  }
  image_msg = stereo_stream.readNextFromSeek();
  if (!image_msg) {
    LOG(ERROR) << "Image msg is nullptr!";
    return 0;
  }
  auto rig_images = image_msg->template get<RigImages>();
  rig_images.vtr_header.sensor_time_stamp.nanoseconds_since_epoch =
      rig_images.channels[0].cameras[0].stamp.nanoseconds_since_epoch;
  auto image_stamp = rig_images.vtr_header.sensor_time_stamp;

  // get first GPS messages
  if (use_tdcp) {
    seek_success = tdcp_stream->seekByTimestamp(image_stamp.nanoseconds_since_epoch);
    if (!seek_success) {
      LOG(ERROR) << "TDCP seek failed!";
      return 0;
    }
    tdcp_msg = tdcp_stream->readNextFromSeek();
  }
  if (use_gpgga) {
    seek_success = gpgga_stream->seekByTimestamp(image_stamp.nanoseconds_since_epoch);
    if (!seek_success) {
      LOG(ERROR) << "TDCP seek failed!";
      return 0;
    }
    gpgga_msg = gpgga_stream->readNextFromSeek();
  }

  // loop through streams, processing whichever of available data is chronologically first
  while (image_idx + start_index < stop_index && rclcpp::ok()) {

    if ((tdcp_msg == nullptr ||
        image_stamp.nanoseconds_since_epoch < tdcp_msg->template get<TdcpMsg>().t_b)
        && (gpgga_msg == nullptr ||
            image_stamp.nanoseconds_since_epoch < gpgga_msg->template get<GpggaMsg>().utc_seconds * 1e9)) {

      // process image
      LOG(INFO) << "Processing image: " << image_idx;
      vo.processImageData(std::make_shared<RigImages>(rig_images), image_stamp);
      image_idx++;

      // get next image
      image_msg = stereo_stream.readNextFromSeek();
      if (!image_msg) {
        LOG(ERROR) << "Image msg is nullptr!";
        break;
      }
      rig_images = image_msg->template get<RigImages>();
      rig_images.vtr_header.sensor_time_stamp.nanoseconds_since_epoch =
          rig_images.channels[0].cameras[0].stamp.nanoseconds_since_epoch;
      image_stamp = rig_images.vtr_header.sensor_time_stamp;
    } else if (tdcp_msg != nullptr && (gpgga_msg == nullptr
        || tdcp_msg->template get<TdcpMsg>().t_b < gpgga_msg->template get<GpggaMsg>().utc_seconds * 1e9)) {

      // process tdcp data
      LOG(INFO) << "Processing TDCP measurement: " << tdcp_idx;
      auto tdcp_meas = tdcp_msg->template get<TdcpMsg>();
      vo.processTdcpData(std::make_shared<TdcpMsg>(tdcp_meas));
      tdcp_idx++;

      // get next tdcp measurement
      tdcp_msg = tdcp_stream->readNextFromSeek();

    } else if (gpgga_msg != nullptr) {

      // process gpgga data
      LOG(INFO) << "Processing GPS position measurement: " << gpgga_idx;
      auto gpgga_meas = gpgga_msg->template get<GpggaMsg>();
      vo.processGpggaData(std::make_shared<GpggaMsg>(gpgga_meas));
      gpgga_idx++;

      // get next gpgga measurement
      gpgga_msg = gpgga_stream->readNextFromSeek();
    }
  }

  LOG(INFO) << "Time to exit!";
}
