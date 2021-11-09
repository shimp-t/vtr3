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
 * \file conversions.cpp
 * \brief lgmath class to corresponding ROS message conversion
 *
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */

#include "vtr_common/lgmath/conversions.hpp"

namespace vtr {

namespace common {

void toROSMsg(const lgmath::se3::TransformationWithCovariance& T,
              vtr_msgs::msg::LieGroupTransform& T_msg) {
  // transform
  T_msg.xi.clear();
  T_msg.xi.reserve(6);
  auto vec = T.vec();
  for (int row = 0; row < 6; ++row) T_msg.xi.push_back(vec(row));

  // covariance
  T_msg.cov.clear();
  T_msg.cov.reserve(36);
  if (!T.covarianceSet()) {
    T_msg.cov_set = false;
  } else {
    auto cov = T.cov();
    for (int row = 0; row < 6; row++)
      for (int col = 0; col < 6; col++) T_msg.cov.push_back(cov(row, col));
    T_msg.cov_set = true;
  }
}

void fromROSMsg(const vtr_msgs::msg::LieGroupTransform& T_msg,
                lgmath::se3::TransformationWithCovariance& T) {
  using TransformT = lgmath::se3::TransformationWithCovariance;
  using TransformVecT = Eigen::Matrix<double, 6, 1>;

  if (!T_msg.cov_set)
    T = TransformT(TransformVecT(T_msg.xi.data()));
  else {
    Eigen::Matrix<double, 6, 6> cov;
    for (int row = 0; row < 6; ++row)
      for (int col = 0; col < 6; ++col)
        cov(row, col) = T_msg.cov[row * 6 + col];
    T = TransformT(TransformVecT(T_msg.xi.data()), cov);
  }
}

}  // namespace common
}  // namespace vtr