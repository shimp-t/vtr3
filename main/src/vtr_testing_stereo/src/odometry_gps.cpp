#include "rclcpp/rclcpp.hpp"

#include <vtr_common/utils/filesystem.hpp>
#include <vtr_logging/logging_init.hpp>
#include <vtr_testing_stereo/odometry.hpp>

using namespace vtr;

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("module_vo_gps");

  /// Log into a subfolder of the data directory (if requested to log)
  auto output_dir = node->declare_parameter<std::string>("output_dir", "/tmp");
  auto to_file = node->declare_parameter<bool>("log_to_file", false);
  std::string log_filename;
  if (to_file) {
    auto log_name = common::timing::toIsoFilename(common::timing::clock::now());
    log_filename = fs::path{common::utils::expand_user(output_dir)} / "logs" /
        (log_name + ".log");
  }
  logging::configureLogging(log_filename, false);
  LOG_IF(to_file, INFO) << "Logging to: " << log_filename;
  LOG_IF(!to_file, WARNING) << "NOT LOGGING TO A FILE.";

  LOG(INFO) << "Starting Odometry with GPS, beep beep beep";
  OdometryNavigator navigator{node, output_dir};


  // gps parameters and setup
  auto use_tdcp = node->declare_parameter<bool>("use_tdcp", false);
#if 0   // GPS code solution not ported from 3.0 yet
  auto use_gpgga = node->declare_parameter<bool>("use_gpgga", false);
  auto gpgga_data_dir_str = node->declare_parameter<std::string>("gpgga_data_dir", "");
  auto gpgga_stream_name = node->declare_parameter<std::string>("gpgga_stream_name", "");
#endif
  std::shared_ptr<storage::DataStreamReader<TdcpMsg>> tdcp_stream;
  if (use_tdcp) {
    LOG(INFO) << "Using time-differenced carrier phase measurements.";
    auto tdcp_data_dir_str =
        common::utils::expand_user(node->declare_parameter<std::string>(
            "tdcp_data_dir",
            ""));
    auto
        tdcp_dataset = node->declare_parameter<std::string>("tdcp_dataset", "");
    tdcp_stream = std::make_shared<storage::DataStreamReader<TdcpMsg>>(
        tdcp_data_dir_str,
        tdcp_dataset);
  }
#if 0
  if (use_gpgga) {
    LOG(INFO) << "Logging GNSS position measurements.";
    gpgga_dir = fs::path{common::utils::expand_user(gpgga_data_dir_str)};
    gpgga_stream = std::make_shared<storage::DataStreamReader<GpggaMsg>>(gpgga_dir.string(), gpgga_stream_name);
  }
#endif

  // image parameters and setup
  auto input_dir = node->declare_parameter<std::string>("input_dir", "");
  storage::DataStreamReader<RigImagesMsg, RigCalibrationMsg> stereo_stream(
      common::utils::expand_user(input_dir), "front_xb3");
  // fetch calibration
  auto calibration_msg =
      stereo_stream.fetchCalibration()->get<RigCalibrationMsg>();
  auto rig_calibration = vtr::messages::copyCalibration(calibration_msg);
  navigator.setCalibration(
      std::make_shared<vision::RigCalibration>(rig_calibration));
  navigator.setCalibration(std::make_shared<vision::RigCalibration>(
      rig_calibration));
  // start playback images
  auto start_index = node->declare_parameter<int>("start_index", 1);
  auto stop_index = node->declare_parameter<int>("stop_index", 99999);

  std::shared_ptr<storage::VTRMessage> image_msg;
#if 0
  std::shared_ptr<storage::VTRMessage> gpgga_msg;
#endif
  std::shared_ptr<storage::VTRMessage> tdcp_msg;
  int image_idx = 0;
  int tdcp_idx = 0;
#if 0
  int gpgga_idx = 0;
#endif

  // get first image
  bool seek_success =
      stereo_stream.seekByIndex(static_cast<int32_t>(start_index));
  if (!seek_success) {
    LOG(ERROR) << "Image seek failed!";
    return 0;
  }
  image_msg = stereo_stream.readNextFromSeek();
  if (!image_msg) {
    LOG(ERROR) << "Image msg is nullptr!";
    return 0;
  }
  auto rig_images = image_msg->template get<RigImagesMsg>();
  rig_images.vtr_header.sensor_time_stamp.nanoseconds_since_epoch =
      rig_images.channels[0].cameras[0].stamp.nanoseconds_since_epoch;
  auto image_stamp = rig_images.vtr_header.sensor_time_stamp;

  // get first GPS messages
  if (use_tdcp) {
    seek_success =
        tdcp_stream->seekByTimestamp(image_stamp.nanoseconds_since_epoch);
    if (!seek_success) {
      LOG(ERROR) << "TDCP seek failed!";
      return 0;
    }
    tdcp_msg = tdcp_stream->readNextFromSeek();
  }
#if 0
  if (use_gpgga) {
    seek_success = gpgga_stream->seekByTimestamp(image_stamp.nanoseconds_since_epoch);
    if (!seek_success) {
      LOG(ERROR) << "TDCP seek failed!";
      return 0;
    }
    gpgga_msg = gpgga_stream->readNextFromSeek();
  }
#endif

  // loop through streams, processing whichever of available data is chronologically first
  while (image_idx + start_index < stop_index && rclcpp::ok()) {

    if ((tdcp_msg == nullptr ||
        image_stamp.nanoseconds_since_epoch
            < tdcp_msg->template get<TdcpMsg>().t_b))
#if 0
      && (gpgga_msg == nullptr ||
          image_stamp.nanoseconds_since_epoch < gpgga_msg->template get<GpggaMsg>().utc_seconds * 1e9)
#endif
    {
      // process image
      LOG(INFO) << "Processing image: " << image_idx;
      navigator.process(std::make_shared<RigImagesMsg>(rig_images));
      image_idx++;

      // get next image
      image_msg = stereo_stream.readNextFromSeek();
      if (!image_msg) {
        LOG(ERROR) << "Image msg is nullptr!";
        break;
      }
      rig_images = image_msg->template get<RigImagesMsg>();
      rig_images.vtr_header.sensor_time_stamp.nanoseconds_since_epoch =
          rig_images.channels[0].cameras[0].stamp.nanoseconds_since_epoch;
      image_stamp = rig_images.vtr_header.sensor_time_stamp;
    } else if (tdcp_msg != nullptr) {
#if 0
      && (gpgga_msg == nullptr
        || tdcp_msg->template get<TdcpMsg>().t_b < gpgga_msg->template get<GpggaMsg>().utc_seconds * 1e9)) {
#endif
      // process tdcp data
      LOG(INFO) << "Processing TDCP measurement: " << tdcp_idx;
      auto tdcp_meas = tdcp_msg->template get<TdcpMsg>();
      navigator.processTdcpData(std::make_shared<TdcpMsg>(tdcp_meas));
      tdcp_idx++;

      // get next tdcp measurement
      tdcp_msg = tdcp_stream->readNextFromSeek();
    }
#if 0
    else if (gpgga_msg != nullptr) {

      // process gpgga data
      LOG(INFO) << "Processing GPS position measurement: " << gpgga_idx;
      auto gpgga_meas = gpgga_msg->template get<GpggaMsg>();
      navigator.processGpggaData(std::make_shared<GpggaMsg>(gpgga_meas));
      gpgga_idx++;

      // get next gpgga measurement
      gpgga_msg = gpgga_stream->readNextFromSeek();
    }
#endif
  }

  LOG(INFO) << "Time to exit!";
}