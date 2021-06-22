#include "rclcpp/rclcpp.hpp"

#include <cpo_interfaces/msg/tdcp.hpp>
#include <vtr_common/utils/filesystem.hpp>
#include <vtr_logging/logging_init.hpp>
#include <vtr_testing_stereo/odometry.hpp>

using namespace vtr;

using RigImagesMsg = vtr_messages::msg::RigImages;
using RigCalibrationMsg = vtr_messages::msg::RigCalibration;
using TdcpMsg = cpo_interfaces::msg::TDCP;

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("navigator");
  auto tdcp_publisher = node->create_publisher<TdcpMsg>("tdcp", 10);

  /// Log into a subfolder of the data directory (if requested to log)
  auto output_dir = node->declare_parameter<std::string>("output_dir", "/tmp");
  auto to_file = node->declare_parameter<bool>("log_to_file", false);
  auto log_debug = node->declare_parameter<bool>("log_debug", false);
  auto clear_output_dir =
      node->declare_parameter<bool>("clear_output_dir", false);
  if (clear_output_dir) {
    LOG(INFO) << "Clearing data directory.";
    fs::remove_all(fs::path{
        common::utils::expand_user(common::utils::expand_env(output_dir))
            + "/graph.index"});
  }
  std::string log_filename;
  if (to_file) {
    auto log_name = common::timing::toIsoFilename(common::timing::clock::now());
    log_filename = fs::path{common::utils::expand_user(
        common::utils::expand_env(output_dir))} /
        "logs" / (log_name + ".log");
  }
  logging::configureLogging(log_filename, log_debug);
  LOG_IF(to_file, INFO) << "Logging to: " << log_filename;
  LOG_IF(!to_file, WARNING) << "NOT LOGGING TO A FILE.";

  LOG(INFO) << "Starting Odometry with GPS, beep beep beep";
  OdometryNavigator navigator{node, output_dir};


  /// gps parameters and setup
  auto use_tdcp = node->declare_parameter<bool>("use_tdcp", false);
#if 0   // GPS code solution not ported from 3.0 yet
  auto use_gpgga = node->declare_parameter<bool>("use_gpgga", false);
  auto gpgga_data_dir_str = node->declare_parameter<std::string>("gpgga_data_dir", "");
  auto gpgga_stream_name = node->declare_parameter<std::string>("gpgga_stream_name", "");
#endif
  std::shared_ptr<storage::DataStreamReader<TdcpMsg>> tdcp_stream;
  if (use_tdcp) {
    LOG(INFO) << "Using time-differenced carrier phase measurements.";
    auto tdcp_data_dir_str = common::utils::expand_user(
        common::utils::expand_env(
            node->declare_parameter<std::string>("tdcp_data_dir", "")));
    auto
        tdcp_dataset = node->declare_parameter<std::string>("tdcp_dataset", "");
    tdcp_stream = std::make_shared<storage::DataStreamReader<TdcpMsg>>(
        tdcp_data_dir_str,
        tdcp_dataset);
  } else {
    LOG(INFO) << "Not using time-differenced carrier phase measurements.";
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
      common::utils::expand_user(common::utils::expand_env(input_dir)),
      "front_xb3");
  // fetch calibration
  auto calibration_msg =
      stereo_stream.fetchCalibration()->get<RigCalibrationMsg>();
  auto rig_calibration = vtr::messages::copyCalibration(calibration_msg);
  navigator.setCalibration(
      std::make_shared<vision::RigCalibration>(rig_calibration));
  // start playback images
  auto start_index = node->declare_parameter<int>("start_index", 1);
  auto stop_index = node->declare_parameter<int>("stop_index", 99999);

  auto blackout_toggles = node->declare_parameter<std::vector<double>>(
      "blackout_toggles",
      std::vector<double>{});
  bool blackout = false;
  std::vector<uint8_t> black_data(589824, 0);     // assuming fixed size image
  LOG(INFO) << "Found " << blackout_toggles.size() << " blackout toggles.";
  for (double toggle : blackout_toggles)
    std::cout << "Toggle point: " << toggle << std::endl;

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
    // todo: this doesn't seek probably because standard rosbag2
    seek_success =
        tdcp_stream->seekByTimestamp(image_stamp.nanoseconds_since_epoch);
    if (!seek_success) {
      LOG(ERROR) << "TDCP seek failed!";
      return 0;
    }
    // loop added because seek above doesn't work properly
    do {
      tdcp_msg = tdcp_stream->readNextFromSeek();
    } while (tdcp_msg != nullptr && tdcp_msg->template get<TdcpMsg>().t_b
        < image_stamp.nanoseconds_since_epoch);
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
      if (!blackout_toggles.empty()
          && image_idx + start_index >= blackout_toggles.front()) {
        // if at toggle point, toggle and remove
        blackout = !blackout;
        blackout_toggles.erase(blackout_toggles.begin());
      }

      // process image
      LOG(INFO) << "Processing image: " << image_idx << "    Blackout? "
                << blackout;
      if (blackout)   // replace left image with zeros
        rig_images.channels[0].cameras[0].data = black_data;

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
      // publish TDCP msg for CPO to process
      LOG(INFO) << "Publishing TDCP measurement: " << tdcp_idx;
      auto tdcp_meas = tdcp_msg->template get<TdcpMsg>();
      tdcp_publisher->publish(tdcp_meas);
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

  LOG(INFO) << "Bye-bye!";
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  LOG(INFO) << "Leaving now!";

  rclcpp::shutdown();
}