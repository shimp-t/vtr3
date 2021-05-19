#pragma once

#include <lgmath.hpp>
#include <steam.hpp>

#include <vtr_navigation/modules/base_module.hpp>
#include <vtr_navigation/modules/optimization/steam_module.hpp>

#if false
#include <asrl/messages/TrajectoryStatus.pb.h>
#endif

namespace vtr {
namespace navigation {

/** 
 * \brief Reject outliers and estimate a preliminary transform 
 * \details
 * requires: 
 *   qdata.[rig_calibrations, rig_features, T_sensor_vehicle, steam_mutex, *T_q_m_prior]
 *   mdata.[map_landmarks, T_sensor_vehicle_map, ransac_matches]
 * outputs: 
 *   mdata.[steam_failure, success, T_q_m]
 */
class KeyframeOptimizationModule : public SteamModule {
 public:
  /**
   * \brief Static module identifier.
   * \todo change this to static_name
   */
  static constexpr auto type_str_ = "keyframe_optimization";

  /** \brief Collection of config parameters */
  struct Config : SteamModule::Config {
    bool depth_prior_enable;
    double depth_prior_weight;
    bool pose_prior_enable;
    bool use_migrated_points;
  };

  KeyframeOptimizationModule(std::string name = type_str_) : SteamModule(name) {
    LOG(INFO) << "Constructing keyframe-frame steam module";
  }

  void setConfig(std::shared_ptr<Config> &config);

 protected:
  /** \brief Saves the trajectory. */
  void updateGraph(QueryCache &qdata, MapCache &mdata,
                   const std::shared_ptr<Graph> &graph, VertexId id) override;

  /** \brief Given two frames, builds a sensor specific optimization problem. */
  std::shared_ptr<steam::OptimizationProblem> generateOptimizationProblem(
      QueryCache &qdata, MapCache &mdata,
      const std::shared_ptr<const Graph> &graph);

  virtual void updateCaches(QueryCache &qdata, MapCache &mdata);

 private:
  /**
   * \brief Verifies the input data being used in the optimization problem,
   * namely, the inlier matches and initial estimate.
   * \param mdata The map data.
   */
  virtual bool verifyInputData(QueryCache &, MapCache &mdata);

  /**
   * \brief Verifies the output data generated by the optimization problem
   * \param qdata The query data.
   * \param mdata The map data.
   */
  virtual bool verifyOutputData(QueryCache &qdata, MapCache &mdata);
#if false
  /** \brief samples and saves the trajectory results to disk. */
  void saveTrajectory(QueryCache &qdata, MapCache &mdata,
                      const std::shared_ptr<Graph> &graph, VertexId id);
#endif
  /**
   * \brief performs sanity checks on the landmark
   * \param point The landmark.
   * \return true if the landmark meets all checks, false otherwise.
   */
  bool isLandmarkValid(const Eigen::Vector3d &point);

  /**
   * \brief Initializes the problem based on an initial condition.
   * \param T_q_m The initial guess at the transformation between the query
   * frame and the map frame.
   */
  void resetProblem(EdgeTransform &T_q_m);

  /**
   * \brief Adds a depth cost associated with this landmark to the depth cost
   * terms.
   * \param landmark The landmark in question.
   */
  void addDepthCost(steam::se3::LandmarkStateVar::Ptr landmark);

  /**
   * \brief Adds a steam trajectory for the state variables in the problem.
   * \param qdata The query data
   * \param mdata The map data
   * \param graph The pose graph.
   */
  void computeTrajectory(QueryCache &qdata, MapCache &mdata,
                         const std::shared_ptr<const Graph> &graph);

  void addPosePrior(MapCache &mdata);

  /** \brief the cost terms associated with landmark observations. */
  steam::ParallelizedCostTermCollection::Ptr cost_terms_;

  /** \brief The cost terms associated with landmark depth. */
  steam::ParallelizedCostTermCollection::Ptr depth_cost_terms_;

  /** \brief The loss function used for the depth cost. */
  steam::LossFunctionBase::Ptr sharedDepthLossFunc_;

  /** \brief the loss function assicated with observation cost. */
  steam::LossFunctionBase::Ptr sharedLossFunc_;

  /** \brief the locked map pose. */
  steam::se3::TransformStateVar::Ptr map_pose_;

  /** \brief Transform evaluator for the map pose. */
  steam::se3::TransformEvaluator::Ptr tf_map_;

  /** \brief the unlocked query pose. */
  steam::se3::TransformStateVar::Ptr query_pose_;

  /** \brief Transform evaluate for the query pose. */
  steam::se3::TransformEvaluator::Ptr tf_query_;

  /** \brief Algorithm Configuration */
  std::shared_ptr<Config> keyframe_config_;

  /** \brief The steam problem. */
  std::shared_ptr<steam::OptimizationProblem> problem_;

  /**
   * \brief Maps velocity variable pointers to their respective vertices
   * \note a value of -1 is used for the live frame.
   */
  std::map<VertexId, steam::VectorSpaceStateVar::Ptr> velocity_map_;
#if false
  /**
   * \brief Status message containing the pre- and post-optimized trajectory
   * information
   */
  asrl::status_msgs::TrajectoryStatus trajectory_status_;
#endif
};

}  // namespace navigation
}  // namespace vtr