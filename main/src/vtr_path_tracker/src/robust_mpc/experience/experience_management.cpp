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
 * \file experience_management.cpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#include <vtr_path_tracker/robust_mpc/experience/experience_management.hpp>

namespace vtr {
namespace path_tracker {

void RCExperienceManagement::computeVelocitiesForExperienceKm1() {
  // Transform the robot poses
  tf2::Transform T_km1_k =
      experience_km1_.T_0_v.inverse() * experience_k_.T_0_v;
  tf2::Vector3 p_km1_k_km1 = T_km1_k.getOrigin();
  tf2::Transform C_km1_k(T_km1_k.getRotation());
  tf2::Vector3 xhat(1, 0, 0);
  tf2::Vector3 th_vec = C_km1_k * xhat;
  float th_k = atan2(th_vec.getY(), th_vec.getX());

  // Arrange the change in pose
  Eigen::VectorXf x_km1;
  x_km1 = Eigen::VectorXf::Zero(STATE_SIZE);

  Eigen::VectorXf x_k(STATE_SIZE);
  x_k << p_km1_k_km1.getX(), p_km1_k_km1.getY(), th_k;

  // Compute the change in time
  rclcpp::Duration dt_ros =
      experience_k_.transform_time - experience_km1_.transform_time;
  auto d_t = (float)dt_ros.seconds();

  // Compute velocities
  if (d_t > 0.01) {
    computeVelocitiesFromState(experience_km1_.velocity_k, x_km1, x_k, d_t);
    experience_k_.x_k.velocity_km1 = experience_km1_.velocity_k;
    experience_km1_.velocity_is_valid = true;
  } else {
    // Pose estimate is not new, copy v and w from previous time
    experience_km1_.velocity_k = experience_km2_.velocity_k;
    experience_km1_.velocity_is_valid = false;
  }
}

void RCExperienceManagement::computeVelocitiesFromState(
    Eigen::VectorXf &velocity, const Eigen::VectorXf &state_km1,
    const Eigen::VectorXf &state_k, const float d_t) {
  Eigen::VectorXf d_x = state_k - state_km1;

  velocity = Eigen::VectorXf::Zero(VELOCITY_SIZE);

  float v_pos = d_x.head<2>().norm() / d_t;

  velocity(0) = d_x(0) > 0 ? v_pos : -v_pos;
  velocity(1) = d_x(2) / d_t;
}

bool RCExperienceManagement::computeDisturbancesForExperienceKm2() {
  return nominal_model_.computeDisturbancesForExperienceKm2(experience_km2_,
                                                            experience_km1_);
}

}  // namespace path_tracker
}  // namespace vtr
