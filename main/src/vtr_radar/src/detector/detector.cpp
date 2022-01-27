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
 * \file detector.cpp
 * \brief Keypoint extraction methods for Navtech radar
 *
 * \author Keenan Burnett, Autonomous Space Robotics Lab (ASRL)
 */
#include <vtr_radar/detector/detector.hpp>


namespace vtr {
namespace radar {

KStrongest::run(RadarQueryCache qdata) {
    for (uint i = 0; i < qdata->raw_scan.rows; ++i) {
        double mean = 0;
        for (uint j = 0; j < qdata->raw_scan.cols; ++j) {

        }
    }
    


}

}  // namespace radar
}  // namespace vtr