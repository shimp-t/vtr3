/**
 * \file scale_error_eval.hpp
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

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Scale error Jacobian
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 1, 6> scaleJacobian(const Eigen::Matrix<double, 6, 1> d);

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Scale error function evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
class ScaleErrorEval : public steam::ErrorEvaluatorX {
 public:
  /// Convenience typedefs
  typedef boost::shared_ptr<ScaleErrorEval> Ptr;
  typedef boost::shared_ptr<const ScaleErrorEval> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor - error is difference between 'T' and identity (in Lie
  /// algebra space)
  //////////////////////////////////////////////////////////////////////////////////////////////
  ScaleErrorEval(double inmeas,
                 const steam::se3::TransformEvaluator::ConstPtr& T);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state
  /// variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 1-d measurement error
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::VectorXd evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 6-d measurement error and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::VectorXd evaluate(const Eigen::MatrixXd& lhs,
                                   std::vector<steam::Jacobian<> >* jacs) const;

 private:
  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Measurement coordinates extracted from images (u v)
  //////////////////////////////////////////////////////////////////////////////////////////////
  double meas_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Error evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  steam::se3::LogMapEvaluator::ConstPtr errorEvaluator_;
};

}  // namespace steam_extensions
}  // namespace vtr
