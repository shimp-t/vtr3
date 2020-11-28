#include <vtr_messages/msg/rig_images.hpp>
#include <vtr_messages/msg/rig_calibration.hpp>
#include <vtr_storage/data_stream_writer.hpp>

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <c++/8/fstream>

namespace fs = std::filesystem;
using RigImages = vtr_messages::msg::RigImages;
using RigCalibration = vtr_messages::msg::RigCalibration;

/// @brief Load .png images and save to rosbag2 format
int main(int argc, char *argv[]) {

  std::string data_dir{"/home/ben/itd_pngs/run_000003"};
  std::string stamps_path{"/home/ben/itd_pngs/run_000003/stamps.csv"};
  std::string bag_dir{"/home/ben/itd_bags/run_000003"};
  std::string stream_name{"front_xb3"};

  // grab timestamps from csv
  std::vector<uint64> stamp_vec;
  std::string stamp_str;
  std::ifstream ss(stamps_path);
  while (std::getline(ss, stamp_str)) {
    uint64 stamp = std::stoull(stamp_str);
    stamp_vec.push_back(stamp);
  }

  vtr::storage::DataStreamWriter<RigImages> writer(bag_dir, stream_name);

  int img_num = 0;
  std::stringstream left_ss;
  std::stringstream right_ss;
  left_ss << data_dir << "/left/" << std::setfill('0') << std::setw(6) << img_num << ".png";
  right_ss << data_dir << "/right/" << std::setfill('0') << std::setw(6) << img_num << ".png";

  // iterate through all images in folder
  for (; fs::exists(fs::path(left_ss.str())) && fs::exists(fs::path(right_ss.str())); ++img_num) {

    vtr_messages::msg::RigImages sensor_message;
    sensor_message.vtr_header.sensor_time_stamp.nanoseconds_since_epoch = stamp_vec[img_num];
    vtr_messages::msg::ChannelImages chan_im;
    for (int i = 0; i < 2; ++i){
      vtr_messages::msg::Image cam_im;

      // get image from png
      cv::Mat im;
      std::string cam_name;
      if (i == 0){
        im = cv::imread(left_ss.str(), cv::IMREAD_COLOR);
        cam_name = "left";
      } else {
        im = cv::imread(right_ss.str(), cv::IMREAD_COLOR);
        cam_name = "right";
      }

//      cam_im.name = cam_name;
      cam_im.height = im.rows;
      cam_im.width = im.cols;
      cam_im.depth = 0; // 3;
      cam_im.encoding = "bgr8";
      cam_im.is_bigendian = true;
      cam_im.stamp.nanoseconds_since_epoch = stamp_vec[img_num];
      cam_im.step = 512; //im.step[0];
      cam_im.data.resize(cam_im.height * cam_im.width * 3);
      cam_im.data.assign(im.data, im.data+im.total()*im.channels());

      chan_im.cameras.push_back(cam_im);
    }
    sensor_message.channels.push_back(chan_im);

    // write to rosbag
    writer.write(sensor_message);

    // update image paths
    left_ss.str(std::string());
    left_ss.clear();
    left_ss << data_dir << "/left/" << std::setfill('0') << std::setw(6) << img_num + 1 << ".png";
    right_ss.str(std::string());
    right_ss.clear();
    right_ss << data_dir << "/right/" << std::setfill('0') << std::setw(6) << img_num + 1 << ".png";
  }

  return 0;
}
