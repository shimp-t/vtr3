#pragma once

#include <vtr_tactic/modules/base_module.hpp>
#include <vtr_vision/messages/bridge.hpp>

namespace vtr {
namespace tactic {

/**
 * \brief A module that retrieves landmarks from multiple graph vertices and
 * store them into map cache.
 * \details
 */
class StereoWindowedRecallModule : public BaseModule {
 public:
  /** \brief Static module identifier. */
  static constexpr auto static_name = "stereo_windowed_recall";

  /** \brief Collection of config parameters */
  struct Config {
    int window_size;
    bool tdcp_enable;
    Eigen::Matrix<double, 6, 6> default_T_0g_cov;
  };

  StereoWindowedRecallModule(const std::string &name = static_name)
      : BaseModule{name}, config_(std::make_shared<Config>()){};
  ~StereoWindowedRecallModule() = default;

  /** \brief Sets the module's configuration. */
  void setConfig(std::shared_ptr<Config> &config) { config_ = config; }

 private:
  /**
   * \brief Given a window size, and a start vertex, recall all of the
   * landmarks and observations within the window, and set up a chain of poses
   * in a single coordinate frame.
   */
  void runImpl(QueryCache &qdata, MapCache &mdata,
               const Graph::ConstPtr &graph) override;

  /** \brief Does nothing? */
  void updateGraphImpl(QueryCache &, MapCache &, const Graph::Ptr &, VertexId);

  /**
   * \brief Loads a specific vertex's landmarks and observations into the
   * landmark and pose map.
   * \param[in,out] lm_map A map containing all currently observed landmarks
   with observations.
   * \param[in,out] poses A map containing poses associated with each vertex.
   * \param transforms TODO
   * \param current_vertex The current vertex
   * \param rig_name TODO
   * \param graph The pose graph.
   */
  void loadVertexData(LandmarkMap &lm_map, SteamPoseMap &poses,
                      SensorVehicleTransformMap &transforms,
                      const pose_graph::RCVertex::Ptr &current_vertex,
                      const std::string &rig_name,
                      const std::shared_ptr<const Graph> &graph);

  /**
   * \brief Loads a all of the landmarks and observations for a specific
   * vertex's channel.
   * \param[in,out] lm_map A map containing all currently observed landmarks
   with observations.
   * \param[in,out] poses A map containing poses associated with each vertex.
   * \param transforms TODO
   * \param current_vertex The current vertex
   * \param channel_obs TODO
   * \param rig_name TODO
   * \param graph The pose graph.
   */
  void loadLandmarksAndObs(
      LandmarkMap &lm_map, SteamPoseMap &poses,
      SensorVehicleTransformMap &transforms,
      const pose_graph::RCVertex::Ptr &current_vertex,
      const vtr_messages::msg::ChannelObservations &channel_obs,
      const std::string &rig_name, const std::shared_ptr<const Graph> &graph);

  /**
   * \brief Given a set of vertices, computes poses for each vertex in a single
   * global coordinate frame.
   * \param[in,out] poses A map containing poses associated with each vertex.
   * \param graph The pose graph.
   */
  void computePoses(SteamPoseMap &poses,
                    const std::shared_ptr<const Graph> &graph);

  void getTimesandVelocities(SteamPoseMap &poses,
                             const std::shared_ptr<const Graph> &graph);

  /**
   * \brief Loads the sensor transform from robochunk via a vertex ID
   * \param vid The Vertex ID of the vertex we need to load the transform from.
   * \param transforms The map of vertex ID to T_s_v's
   * \param rig_name the name of the current rig
   * \param graph A pointer to the pose graph.
   */
  void loadSensorTransform(const VertexId &vid,
                           SensorVehicleTransformMap &transforms,
                           const std::string &rig_name,
                           const Graph::ConstPtr &graph);

  /**
   * \brief Retrieve GPS-related measurements & prior on global orientation from
   * pose graph and return to put in cache for use in optimization
   * \param msgs The pseudo-measurements
   * \param T_0g_prior The prior (we only care about the orientation part)
   * \param graph The pose graph
   * \param vertex_0 The vertex the prior is with respect to
 */
  void getTdcpMeas(std::vector<cpo_interfaces::msg::TDCP::SharedPtr> &msgs,
                   std::pair<pose_graph::VertexId,
                             lgmath::se3::TransformationWithCovariance> &T_0g_prior,
                   const std::shared_ptr<const Graph> &graph,
                   const pose_graph::RCVertex::Ptr &vertex_0);

  /**
   * \brief a map that keeps track of the pointers into the vertex landmark
   * messages.
   */
  std::map<VertexId, std::shared_ptr<vtr_messages::msg::RigLandmarks>>
      vertex_landmarks_;

  /** \brief Module configuration. */
  std::shared_ptr<Config> config_;
};

}  // namespace tactic
}  // namespace vtr