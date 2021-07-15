#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vtr_storage/data_stream_writer.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <c++/8/fstream>

using NavSatFix = sensor_msgs::msg::NavSatFix;

/** \brief Subscribes to NavSatFix msgs and saves to rosbag2s */
class NavsatToCsv : public rclcpp::Node {
 public:
  /** \brief Constructor */
  NavsatToCsv();

 private:
  /** \brief Callback for NavSatFix msgs */
  void _navsatCallback(NavSatFix::SharedPtr msg);

  /** \brief Subscriber */
  rclcpp::Subscription<NavSatFix>::SharedPtr data_subscription_;

  /** \brief NavSatFix writer */
  std::ofstream outstream_;
};