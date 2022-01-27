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

namespace {

/**
 * \brief Converts cartesian to polar and force polar coordinates to be
 * continuous.
 */
void cart2Pol(const std::vector<PointXYZ> &cart, std::vector<PointXYZ> &polar) {
  polar.clear();
  polar.reserve(cart.size());

  for (size_t i = 0; i < cart.size(); i++) {
    const auto &p = cart[i];

    const float rho = sqrt(p.sq_norm());
    const float theta = atan2(sqrt(p.x * p.x + p.y * p.y), p.z);
    float phi = atan2(p.y, p.x) + M_PI / 2;

    if (i > 0 && (phi - polar[i - 1].z) > 1.5 * M_PI)
      phi -= 2 * M_PI;
    else if (i > 0 && (phi - polar[i - 1].z) < -1.5 * M_PI)
      phi += 2 * M_PI;

    polar.emplace_back(rho, theta, phi);
  }
}

}  // namespace

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
  // const auto &msg = qdata.scan_msg.ptr();
  qdata.raw_scan = cv_bridge::toCvShare(msg, "mono8")->image;
  qdata.raw_scan.convertTo(qdata.raw_scan, CV_32F);

  /// Output
  auto &raw_pointcloud_time = *qdata.raw_pointcloud_time.fallback();
  auto &raw_pointcloud_cart = *qdata.raw_pointcloud_cart.fallback();
  auto &raw_pointcloud_pol = *qdata.raw_pointcloud_pol.fallback();

  // Extract keypoints and times

  // Copy over points and time
  const auto N = (size_t)(msg->width * msg->height);
  raw_pointcloud_time.reserve(N);
  raw_pointcloud_cart.reserve(N);
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x"),
      iter_y(*msg, "y"), iter_z(*msg, "z");
  sensor_msgs::PointCloud2ConstIterator<double> iter_time(*msg, "t");
  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_time) {
    raw_pointcloud_cart.push_back(PointXYZ(*iter_x, *iter_y, *iter_z));
    raw_pointcloud_time.push_back(*iter_time);
  }

  // Velodyne has no polar coordinates, so compute them manually.
  cart2Pol(raw_pointcloud_cart, raw_pointcloud_pol);
}

}  // namespace lidar
}  // namespace vtr