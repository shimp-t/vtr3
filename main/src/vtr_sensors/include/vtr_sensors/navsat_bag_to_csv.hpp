#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>

#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <vtr_storage/data_stream_reader.hpp>
#include <vtr_storage/data_stream_writer.hpp>
#include <fstream>

using NavSatFix = sensor_msgs::msg::NavSatFix;

class NavsatBagToCsv : public rclcpp::Node {
 public:
  NavsatBagToCsv(const std::string &data_dir, const std::string &stream_name);

  vtr::storage::DataStreamReader<NavSatFix> reader_;

  std::ofstream outstream_;
};