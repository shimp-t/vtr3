#pragma once

#include <asrl/messages/Matches.pb.h>
#include <asrl/navigation/modules/base_module.h>

//#include <asrl/vision/matching/SensorModel/SensorModelBase.h>
//#include <asrl/vision/matching/Sampler/BasicSampler.h>

namespace asrl {
namespace navigation {

/** \brief Reject outliers and estimate a preliminary transform
 */
class VertexCreationModule : public BaseModule {
 public:
  static constexpr auto type_str_ = "vertex_creation";
  struct Config {
    double distance_threshold_min;
    double distance_threshold_max;
    double rotation_threshold_min;
    double rotation_threshold_max;
    int match_threshold_min_count;
    int match_threshold_fail_count;
  };
  /** \brief TODO Construct with settings...
   */
  VertexCreationModule() {}

  /** \brief Given two frames and matches detects the inliers that fit
  ///        the given model, and provides an initial guess at transform T_q_m.
   */
  virtual void run(QueryCache &qdata, MapCache &mdata,
                   const std::shared_ptr<const Graph> &graph) = 0;

  /** \brief Update the graph with the frame data for the live vertex
   */
  virtual void updateGraph(QueryCache &, MapCache &,
                           const std::shared_ptr<Graph> &, VertexId){};

  void setConfig(std::shared_ptr<Config> &config) { config_ = config; }

 protected:
 private:
  /** \brief Algorithm Configuration
   */
  std::shared_ptr<Config> config_;
};

}  // namespace navigation
}  // namespace asrl
