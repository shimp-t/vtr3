#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>

#include "vtr_messages/msg/rig_image_calib.hpp"

using RigImageCalibMsg = vtr_messages::msg::RigImageCalib;

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("xb3_image_viewer");

  auto image_sub = node->create_subscription<RigImageCalibMsg>(
      "/images", rclcpp::SensorDataQoS(),
      [](const RigImageCalibMsg::SharedPtr msg) {
        auto &image = msg->rig_images;

        // Visualization
        auto &left = image.channels[0].cameras[0];
        auto &right = image.channels[0].cameras[1];

        // Get image parameters from left camera and assume right is the same
        int outputmode = -1;
        int datasize = 0;
        if (left.encoding == "bgr8") {
          datasize = left.height * left.width * 3;
          outputmode = CV_8UC3;
        } else if (left.encoding == "mono8") {
          datasize = left.height * left.width;
          outputmode = CV_8UC1;
        }

        // Create OpenCV images to be shown
        // clang-format off
        left.data.resize(datasize);
        const auto& cv_left = cv::Mat(left.height, left.width, outputmode, (void *)left.data.data());
        cv::imshow(image.channels[0].name + "/left", cv_left);
        right.data.resize(datasize);
        const auto& cv_right = cv::Mat(right.height, right.width, outputmode, (void *)right.data.data());
        cv::imshow(image.channels[0].name + "/right", cv_right);
        // clang-format on

        cv::waitKey(1);
      });

  rclcpp::spin(node);
}
