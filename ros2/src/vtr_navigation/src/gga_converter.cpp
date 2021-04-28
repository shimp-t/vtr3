#include <rclcpp/rclcpp.hpp>
#include <nmea_msgs/msg/gpgga.hpp>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <utility>

#define LEAP_SECONDS        18
#define UNIX_GPS_OFFSET     315964800

typedef std::chrono::high_resolution_clock Clock;

struct GgaLine {
  std::string msg_type, time_of_day, lat_deg, lat_dir, lon_deg, lon_dir, fix_type;
  std::string num_sats, hdop, alt, alt_units, undulation, und_units, age, stn_id;
};

class GgaConverter : public rclcpp::Node {
 public:
  GgaConverter(std::string csv_path, uint day_secs)
      : Node("gga_conversion"), csv_path_(std::move(csv_path)), gga_day_seconds_(day_secs) {

    publisher_ = this->create_publisher<nmea_msgs::msg::Gpgga>("gpgga", 10);
  }

  void run() {
    // open csv file
    std::ifstream data(csv_path_);
    std::string line;
    // read in line
    std::getline(data, line);
    GgaLine first_gga = readLine(line);

    nmea_msgs::msg::Gpgga first_msg = toRosMsg(first_gga);
    publisher_->publish(first_msg);

    double first_msg_time = first_msg.utc_seconds;

    // start timer
    auto clock_start = Clock::now();

    while (std::getline(data, line)) {
      GgaLine gga = readLine(line);
      nmea_msgs::msg::Gpgga msg = toRosMsg(gga);
      // wait until the time is right
      while ((double) std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - clock_start).count() / 1000.0
          < (msg.utc_seconds - first_msg_time)) {
        usleep(1000);
      }
      publisher_->publish(msg);
    }

    std::cout << "Finished running :) " << std::endl;
  }

 private:

  static GgaLine readLine(const std::string &line) {
    std::stringstream lineStream(line);
    std::string cell;
    GgaLine gga_line;
    int i = 0;
    // iterate through fields of line
    while (std::getline(lineStream, cell, ',')) {
      switch (i) {
        case 0: gga_line.msg_type = cell;
          break;
        case 1: gga_line.time_of_day = cell;
          break;
        case 2: gga_line.lat_deg = cell;
          break;
        case 3: gga_line.lat_dir = cell;
          break;
        case 4: gga_line.lon_deg = cell;
          break;
        case 5: gga_line.lon_dir = cell;
          break;
        case 6: gga_line.fix_type = cell;
          break;
        case 7: gga_line.num_sats = cell;
          break;
        case 8: gga_line.hdop = cell;
          break;
        case 9: gga_line.alt = cell;
          break;
        case 10: gga_line.alt_units = cell;
          break;
        case 11: gga_line.undulation = cell;
          break;
        case 12: gga_line.und_units = cell;
          break;
        case 13: gga_line.age = cell;
          break;
        case 14: gga_line.stn_id = cell;
          break;
        default:break;
      }
      ++i;
    }
    return gga_line;
  }

  nmea_msgs::msg::Gpgga toRosMsg(const GgaLine &line) const {
    nmea_msgs::msg::Gpgga msg;

    if (line.msg_type != "$GPGGA") {
      std::cout << "Error: line received has msg_type " << line.msg_type << std::endl;
      return msg;
    }

    msg.message_id = line.msg_type;

    // convert from hhmmmss.ss to UTC seconds
    msg.utc_seconds = gga_day_seconds_ + std::stod(line.time_of_day.substr(0, 2)) * 3600.0
        + std::stod(line.time_of_day.substr(2, 2)) * 60.0 + std::stod(line.time_of_day.substr(4));

//    msg.header.stamp.sec = msg.utc_seconds;

    msg.lat_dir = line.lat_dir;
    msg.lon_dir = line.lon_dir;
    // convert to DD.DDDD from DDmm.mm
    msg.lat = (std::stod(line.lat_deg.substr(0, 2)) + std::stod(line.lat_deg.substr(2)) / 60.0);
    if (line.lat_dir == "S") {
      msg.lat *= -1;
    }
    msg.lon = (std::stod(line.lon_deg.substr(0, 3)) + std::stod(line.lon_deg.substr(3)) / 60.0);
    if (line.lon_dir == "W") {
      msg.lon *= -1;
    }

    msg.gps_qual = std::stoi(line.fix_type);
    msg.num_sats = std::stoi(line.num_sats);
    msg.hdop = std::stof(line.hdop);
    msg.alt = std::stof(line.alt);
    msg.altitude_units = line.alt_units;
    msg.undulation = std::stof(line.undulation);
    msg.undulation_units = line.und_units;
    msg.diff_age = std::stoi(line.age);
    msg.station_id = line.stn_id.substr(0, 4);

    return msg;
  }

  std::string csv_path_;

  rclcpp::Publisher<nmea_msgs::msg::Gpgga>::SharedPtr publisher_;

  uint gga_day_seconds_;

};

int main(int argc, char **argv) {

//  std::string csv_path = "/home/ben/CLionProjects/gpso/data/gpgga/feb10a_gga.ASC";
  std::string csv_path = "/home/ben/CLionProjects/gpso/data/gpgga/feb15a_gga.ASC";

//  uint approx_time_in = 1613000000;     // Feb 10
  uint approx_time_in = 1613420000;     // Feb 15

  // GPS time =/= Unix time so convert
  const uint gps_day = floor((double) (approx_time_in - UNIX_GPS_OFFSET) / (60.0 * 60.0 * 24.0));
  const uint gga_day_seconds = UNIX_GPS_OFFSET + LEAP_SECONDS + gps_day * 24 * 3600;

  rclcpp::init(argc, argv);
  auto node = std::make_shared<GgaConverter>(csv_path, gga_day_seconds);
  node->run();
  rclcpp::shutdown();
  return 0;
}


