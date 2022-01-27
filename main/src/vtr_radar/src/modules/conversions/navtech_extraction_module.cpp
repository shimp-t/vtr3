// Copyright 2021, Autonomous Space Robotics Lab (ASRL)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * \file navtech_extraction_module.cpp
 * \brief NavtechExtractionModule class methods definition
 *
 * \author Keenan Burnett, Autonomous Space Robotics Lab (ASRL)
 */
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <vtr_radar/modules/conversions/navtech_extraction_module.hpp>

namespace vtr {
namespace radar {

using namespace tactic;
using namespace vtr::lidar;

void NavtechExtractionModule::configFromROS(
    const rclcpp::Node::SharedPtr &node, const std::string param_prefix) {
  config_ = std::make_shared<Config>();
  // clang-format off
  config_->visualize = node->declare_parameter<bool>(param_prefix + ".visualize", config_->visualize);
  config_->detector = node->declare_parameter<std::string>(param_prefix + ".detector", config_->detector);
  config_->kstrong = node->declare_parameter<int>(param_prefix + ".kstrong", config_->kstrong);
  config_->minr = node->declare_parameter<double>(param_prefix + ".minr", config_->minr);
  config_->maxr = node->declare_parameter<double>(param_prefix + ".maxr", config_->maxr);
  config_->zq = node->declare_parameter<double>(param_prefix + ".zq", config_->zq);
  config_->sigma = node->declare_parameter<int>(param_prefix + ".sigma", config_->sigma);
  config_->width = node->declare_parameter<int>(param_prefix + ".width", config_->width);
  config_->guard = node->declare_parameter<int>(param_prefix + ".guard", config_->guard);
  config_->kstat = node->declare_parameter<int>(param_prefix + ".kstat", config_->kstat);
  config_->threshold = node->declare_parameter<double>(param_prefix + ".threshold", config_->threshold);
  config_->radar_resolution = node->declare_parameter<double>(param_prefix + ".radar_resolution", config_->radar_resolution);
  // clang-format on
}

void NavtechExtractionModule::runImpl(QueryCache &qdata0,
                                       const Graph::ConstPtr &) {
  auto &qdata = dynamic_cast<RadarQueryCache &>(qdata0);

  /// Input
  const auto &msg = qdata.scan_msg.ptr();
  cv::Mat raw_data = cv_bridge::toCvShare(msg, "mono8")->image;
  raw_data.convertTo(raw_data, CV_32F);

  /// Output
  auto &raw_pointcloud_time = *qdata.raw_pointcloud_time.fallback();
  auto &raw_pointcloud_cart = *qdata.raw_pointcloud_cart.fallback();
  auto &raw_pointcloud_pol = *qdata.raw_pointcloud_pol.fallback();
  auto &raw_scan = *qdata.raw_scan.fallback();
  auto &azimuth_times = *qdata.azimuth_times.fallback();
  auto &azimuth_angles = *qdata.azimuth_angles.fallback();

  // Load scan, times, azimuths from raw_data
  qdata.radar_resolution = config_->radar_resolution;
  load_radar(raw_data, azimuth_times, azimuth_angles, raw_scan);

  // Extract keypoints and times
  Detector detector;
  switch(config_->detector) {
    case "kstrongest":
      detector = KStrongest(config_->kstrong, config_->threshold, config_->minr,
        config_->maxr);
    case "cen2018":
      detector = Cen2018(config_->zq, config_->sigma, config_->minr, config_->maxr);
    case "cacfar":
      detector = CACFAR(config_->width, config_->guard, config_->threshold,
        config_->minr, config_->maxr);
    case "oscfar":
      detector = OSCFAR(config_->width, config_->guard, config_->kstat,
        config_->threshold, config_->minr, config_->maxr);
  }
  detector.run(*qdata0);

  // Convert to cartesian format
  pol2Cart(raw_pointcloud_pol, raw_pointcloud_cart);
}

}  // namespace radar
}  // namespace vtr