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
 * \file navtech_extraction_module.hpp
 * \brief NavtechExtractionModule class definition
 *
 * \author Keenan Burnett, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <pcl_conversions/pcl_conversions.h>

#include <vtr_radar/cache.hpp>
#include <vtr_radar/utils/utils.hpp>
#include <vtr_tactic/modules/base_module.hpp>

namespace vtr {
namespace radar {

/**
 * \brief Extracts keypoints from Navtech radar scans.
 */
class NavtechExtractionModule : public tactic::BaseModule {
 public:
  /** \brief Static module identifier. */
  static constexpr auto static_name = "radar.navtech_extractor";

  /** \brief Config parameters. */
  struct Config {
    bool visualize = false;
    std::string detector = "kstrongest";
    int kstrong = 10;
    double minr = 2;
    double maxr = 100;
    double zq = 3;
    int sigma = 17;
    int width = 40;
    int guard = 2;
    int kstat = 20;
    double threshold = 1.0;
    double radar_resolution = 0.0438;
  };

  NavtechExtractionModule(const std::string &name = static_name)
      : tactic::BaseModule{name}, config_(std::make_shared<Config>()) {}

  void configFromROS(const rclcpp::Node::SharedPtr &node,
                     const std::string param_prefix) override;

 private:
  void runImpl(tactic::QueryCache &qdata,
               const tactic::Graph::ConstPtr &graph) override;

  std::shared_ptr<Config> config_;
};

}  // namespace lidar
}  // namespace vtr