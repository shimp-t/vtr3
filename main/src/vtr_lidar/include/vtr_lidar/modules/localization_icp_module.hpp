/**
 * \file localization_icp_module.hpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <random>

#include <steam.hpp>

#include <vtr_common/timing/stopwatch.hpp>
#include <vtr_lidar/cache.hpp>
#include <vtr_tactic/modules/base_module.hpp>

namespace vtr {

namespace lidar {

/** \brief ICP algorithm for localization. */
class LocalizationICPModule : public tactic::BaseModule {
 public:
  /** \brief Static module identifier. */
  static constexpr auto static_name = "lidar.localization_icp";

  /** \brief Config parameters. */
  struct Config : public steam::VanillaGaussNewtonSolver::Params {
    /// Success criteria
    float min_matched_ratio = 0.4;

    /// Prior terms
    bool use_pose_prior = false;

    /// ICP parameters
    // number of threads for nearest neighbor search
    int num_threads = 8;
    // initial alignment config
    size_t first_num_steps = 3;
    size_t initial_max_iter = 100;
    size_t initial_num_samples = 1000;
    float initial_max_pairing_dist = 2.0;
    float initial_max_planar_dist = 0.3;
    // refined stage
    size_t refined_max_iter = 10;  // we use a fixed number of iters for now
    size_t refined_num_samples = 5000;
    float refined_max_pairing_dist = 2.0;
    float refined_max_planar_dist = 0.1;
    // error calculation
    float averaging_num_steps = 5;
    float trans_diff_thresh = 0.01;              // threshold on variation of T
    float rot_diff_thresh = 0.1 * M_PI / 180.0;  // threshold on variation of R
  };

  LocalizationICPModule(const std::string &name = static_name)
      : tactic::BaseModule{name}, config_(std::make_shared<Config>()){};

  void configFromROS(const rclcpp::Node::SharedPtr &node,
                     const std::string param_prefix) override;

 private:
  void runImpl(tactic::QueryCache &qdata,
               const tactic::Graph::ConstPtr &graph) override;

  void addPosePrior(
      const tactic::EdgeTransform &T_r_m,
      const steam::se3::TransformEvaluator::Ptr &T_r_m_eval,
      const steam::ParallelizedCostTermCollection::Ptr &prior_cost_terms);

  std::shared_ptr<Config> config_;
};

}  // namespace lidar
}  // namespace vtr