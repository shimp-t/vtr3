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
 * \file utils.hpp
 * \brief Utility functions for working with radar data
 *
 * \author Keenan Burnett, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <>

namespace vtr{
namespace radar{

void load_radar(const std::string &path, std::vector<double> &timestamps,
    std::vector<double> &azimuths, cv::Mat &fft_data);

void load_radar(const cv::Mat &raw_data, std::vector<double> &timestamps,
    std::vector<double> &azimuths, cv::Mat &fft_data);

void pol2Cart(const std::vector<PointXYZ> &polar, std::vector<PointXYZ> &cart);

}  // namespace vtr
}  // namespace radar