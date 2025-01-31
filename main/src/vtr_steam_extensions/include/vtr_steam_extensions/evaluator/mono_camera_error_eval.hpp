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
 * \file mono_camera_error_eval.hpp
 * \brief
 * \details
 *
 * \author Michael Warren, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <steam.hpp>
#include <steam/problem/NoiseModel.hpp>

namespace vtr {
namespace steam_extensions {
namespace mono {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Simple structure to hold the stereo camera intrinsics
//////////////////////////////////////////////////////////////////////////////////////////////
struct CameraIntrinsics {
  /// Convenience typedefs
  typedef std::shared_ptr<CameraIntrinsics> Ptr;
  typedef std::shared_ptr<const CameraIntrinsics> ConstPtr;

  /// \brief Focal length in the u-coordinate (horizontal)
  double fu;

  /// \brief Focal length in the v-coordinate (vertical)
  double fv;

  /// \brief Focal center offset in the u-coordinate (horizontal)
  double cu;

  /// \brief Focal center offset in the v-coordinate (vertical)
  double cv;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Calculates the mono Camera model Jacobian
/// \param intrinsics The mono camera intrinsic properties.
/// \param point The homogeneous point the jacobian is being evaluated at.
/// \return the jacobian of the camera model, evaluated at the given point.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 2, 4> cameraModelJacobian(
    const CameraIntrinsics::ConstPtr& intrinsics, const Eigen::Vector4d& point);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluates the noise of an uncertain map landmark, which has been
/// reprojected into the
///        a query coordinate frame using a steam transform evaluator.
//////////////////////////////////////////////////////////////////////////////////////////////
class LandmarkNoiseEvaluator : public steam::NoiseEvaluator<Eigen::Dynamic> {
 public:
  /// \brief Constructor
  /// \param landmark_mean The landmark mean, in the query frame.
  /// \param landmark_cov The landmark covariance, in the query frame.
  /// \param meas_noise The noise on the landmark measurement.
  /// \param intrinsics The mono camera intrinsics.
  /// \param T_query_map The steam transform evaluator that takes points from
  /// the landmark frame into the query frame.
  LandmarkNoiseEvaluator(
      const Eigen::Vector4d& landmark_mean, const Eigen::Matrix3d& landmark_cov,
      const Eigen::Matrix2d& meas_noise,
      const CameraIntrinsics::ConstPtr& intrinsics,
      const steam::se3::TransformEvaluator::ConstPtr& T_query_map);

  /// \brief Default destructor
  ~LandmarkNoiseEvaluator() = default;

  /// \brief Evaluates the reprojection covariance
  /// \return the 4x4 covariance of the landmark reprojected into the query mono
  ///         camera frame.
  virtual Eigen::MatrixXd evaluate();

 private:
  /// \brief The stereo camera intrinsics.
  CameraIntrinsics::ConstPtr intrinsics_;

  /// \brief The landmark covariance.
  Eigen::Matrix2d meas_noise_;

  /// \brief the landmark mean.
  Eigen::Vector4d mean_;

  /// \brief The steam transform evaluator that takes points from the landmark
  /// frame
  ///        into the query frame.
  steam::se3::TransformEvaluator::ConstPtr T_query_map_;

  /// \brief The 3x3 landmark covariance (phi) dialated into a 3x3 matrix.
  /// @details dialated_phi_ = D*phi*D^T, where D is a 4x3 dialation matrix.
  Eigen::Matrix4d dialated_phi_;

  /// \brief The mono camara jacobian, evaluated at landmark mean, j.
  Eigen::Matrix<double, 2, 4> camera_jacobian_j_;

  /// \brief the last computed covariance
  Eigen::Matrix2d last_computed_cov_;
};

}  // end namespace mono

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Stereo camera error function evaluator
///
/// *Note that we fix MAX_STATE_DIM to 6. Typically the performance benefits of
/// fixed size
///  matrices begin to die if larger than 6x6. Size 6 allows for transformation
///  matrices and 6D velocities. If you have a state-type larger than this,
///  consider writing an error evaluator that extends from the dynamically sized
///  ErrorEvaluatorX.
//////////////////////////////////////////////////////////////////////////////////////////////
class MonoCameraErrorEval : public steam::ErrorEvaluatorX {
 public:
  /// Convenience typedefs
  typedef std::shared_ptr<MonoCameraErrorEval> Ptr;
  typedef std::shared_ptr<const MonoCameraErrorEval> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  MonoCameraErrorEval(
      const Eigen::Vector2d& meas,
      const mono::CameraIntrinsics::ConstPtr& intrinsics,
      const steam::se3::TransformEvaluator::ConstPtr& T_cam_landmark,
      const steam::se3::LandmarkStateVar::Ptr& landmark);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state
  /// variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 2-d measurement error (ul vl ur vr)
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::VectorXd evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 2-d measurement error (u v) and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::VectorXd evaluate(const Eigen::MatrixXd& lhs,
                                   std::vector<steam::Jacobian<> >* jacs) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Camera model Jacobian
  //////////////////////////////////////////////////////////////////////////////////////////////
  //  Eigen::Matrix<double, 2, 4> cameraModelJacobian() const {
  //    return mono::cameraModelJacobian(intrinsics_, meas_);
  //  };

 private:
  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Camera model
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Vector2d cameraModel(const Eigen::Vector4d& point) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Measurement coordinates extracted from images (u v)
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Vector2d meas_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Camera instrinsics
  //////////////////////////////////////////////////////////////////////////////////////////////
  mono::CameraIntrinsics::ConstPtr intrinsics_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Point evaluator (evaluates the point transformed into the camera
  /// frame)
  //////////////////////////////////////////////////////////////////////////////////////////////
  steam::se3::ComposeLandmarkEvaluator::ConstPtr eval_;
};

}  // namespace steam_extensions
}  // namespace vtr