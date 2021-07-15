#include <filesystem>
#include <vtr_sensors/navsat_to_csv_2.hpp>

namespace fs = std::filesystem;

NavsatToCsv::NavsatToCsv()
    : Node("xb3_recorder") {
  data_subscription_ = this->create_subscription<NavSatFix>(
      "fix", 10,
      std::bind(&NavsatToCsv::_navsatCallback, this, std::placeholders::_1));

  outstream_.open("/home/ben/Desktop/july5x.csv");   // have to manually change for now
  outstream_ << "msg_type,sec,ns,lat,lon,alt,pos_cov_0\n";
}

void NavsatToCsv::_navsatCallback(const NavSatFix::SharedPtr msg) {
  long sec = msg->header.stamp.sec;
  long ns = msg->header.stamp.nanosec;
  double lat = msg->latitude;
  double lon = msg->longitude;
  double alt = msg->altitude;
  double pos_cov_0 = msg->position_covariance.at(0);

  outstream_ << std::setprecision(12);
  outstream_ << "$NAVSAT" << "," << sec << "," << ns << "," << lat << "," << lon
             << "," << alt << "," << pos_cov_0;
  outstream_ << "\n";
}

/// @brief Record NavSatFix msgs to a rosbag2
int main(int argc, char *argv[]) {

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NavsatToCsv>());
  rclcpp::shutdown();
  return 0;
}