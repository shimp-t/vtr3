#include <vtr_steam_extensions/evaluator/unicycle_error_eval.hpp>

namespace vtr {
namespace steam_extensions {

UnicycleErrorEval::UnicycleErrorEval(const steam::VectorSpaceStateVar::Ptr &state_vec)
    : state_vec_(state_vec) {}

bool UnicycleErrorEval::isActive() const {
  return !state_vec_->isLocked();
}

Eigen::Matrix<double, 4, 1> UnicycleErrorEval::evaluate() const {
  Eigen::Matrix<double, 4, 6> P;
  P << Eigen::Matrix<double, 4, 1>::Zero(),
      Eigen::Matrix4d::Identity(), Eigen::Matrix<double, 4, 1>::Zero();

  return P * state_vec_->getValue();
}

Eigen::Matrix<double, 4, 1> UnicycleErrorEval::evaluate(
    const Eigen::Matrix<double, 4, 4> &lhs,
    std::vector<steam::Jacobian<4, 6> > *jacs) const {
  // Check and initialize Jacobian array
  if (jacs == nullptr) {
    throw std::invalid_argument(
        "Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  Eigen::Matrix<double, 4, 6> P;
  P << Eigen::Matrix<double, 4, 1>::Zero(),
      Eigen::Matrix4d::Identity(), Eigen::Matrix<double, 4, 1>::Zero();

  if (!state_vec_->isLocked()) {
    jacs->push_back(steam::Jacobian<4, 6>());
    steam::Jacobian<4, 6> &jacref = jacs->back();
    jacref.key = state_vec_->getKey();
    jacref.jac = lhs * P;
  }
  return P * state_vec_->getValue();
}

}  // namespace steam_extensions
}  // namespace vtr
