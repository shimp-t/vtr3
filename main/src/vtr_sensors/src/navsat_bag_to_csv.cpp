#include <vtr_sensors/navsat_bag_to_csv.hpp>

#include <chrono>
#include <filesystem>
#include <thread>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace fs = std::filesystem;

NavsatBagToCsv::NavsatBagToCsv(const std::string &data_dir,
                               const std::string &stream_name)
    : Node("navsat_bag_to_csv"), reader_(data_dir, stream_name) {

  std::string results_path = "/home/ben/Desktop/gt/" + stream_name + ".csv";
  outstream_.open(results_path);
}

/// @brief Replay XB3 stereo images from a rosbag2
int main(int argc, char *argv[]) {
  // Default path
  fs::path data_dir{"/home/ben/ASRL/data/june16-navsatfix"};
  std::string stream_name = "june16c";
  int start_index = 1;
  int stop_index = 9999999;

  // User specified path
  if (argc >= 3) {
    data_dir = argv[1];
    stream_name = argv[2];
  } else if (argc != 1) {
    throw std::invalid_argument("Wrong number of arguments provided!");
  }

  rclcpp::init(argc, argv);
  auto replay = NavsatBagToCsv(data_dir.string(), stream_name);

  replay.reader_.seekByIndex(start_index);

  int curr_index = start_index;

  replay.outstream_ << std::setprecision(12)
                    << "msg_type,sec,ns,lat,lon,alt,pos_cov_0" << std::endl;

  while (true) {
    if (!rclcpp::ok()) break;

    if (curr_index == stop_index) break;
    curr_index++;

    auto message = replay.reader_.readNextFromSeek();
    if (!message.get()) break;

    auto fix_msg = message->template get<NavSatFix>();

    auto timestamp = fix_msg.header.stamp;
    auto lat = fix_msg.latitude;
    auto lon = fix_msg.longitude;
    auto alt = fix_msg.altitude;
    auto pos_cov_0 = fix_msg.position_covariance.at(0);

    std::cout << std::setprecision(12) << "\ntimestamp.sec     "
              << timestamp.sec << std::endl;
    std::cout << "timestamp.nanosec " << timestamp.nanosec << std::endl;
    std::cout << "lat               " << lat << std::endl;
    std::cout << "lon               " << lon << std::endl;
    std::cout << "alt               " << alt << std::endl;
    std::cout << "pos_cov_0         " << pos_cov_0 << std::endl;

    replay.outstream_ << "$NAVSAT," << timestamp.sec << "," << timestamp.nanosec
                      << "," << lat << "," << lon << "," << alt << ","
                      << pos_cov_0 << std::endl;

  }

  replay.outstream_.close();

  rclcpp::shutdown();
}
