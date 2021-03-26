#pragma once

#include <memory>
#include <atomic>
#include <future>

#include <geometry_msgs/TwistStamped.h>

#include <asrl/pose_graph/id/VertexId.hpp>
#include <asrl/common/timing/TimeUtils.hpp>
#include <asrl/pose_graph/path/LocalizationChain.hpp>

#include <lgmath.hpp>
#include <steam/trajectory/SteamTrajInterface.hpp>

namespace asrl::pose_graph {
class RCGraph;
}

namespace vtr {

namespace path_tracker {

// common type shortcuts
typedef ::asrl::pose_graph::LocalizationChain Chain;
typedef ::asrl::pose_graph::RCGraph Graph;
typedef lgmath::se3::Transformation Tf;
typedef lgmath::se3::TransformationWithCovariance TfCov;
typedef ::asrl::pose_graph::VertexId Vid;
typedef ::asrl::pose_graph::RCVertex Vertex;
typedef ::asrl::pose_graph::RCRun::IdType RunId;
typedef ::asrl::common::timing::time_point Stamp;
typedef ::asrl::common::timing::clock Clock;

/** \brief Path tracking state for external control of the path tracker
 */
enum class State {
  STOP,   ///< thread exits, path aborted
  PAUSE,  ///< skip control loop, path kept
  RUN,    ///< control output, follow path
};

// Output types
class SafetyStatus;
typedef geometry_msgs::TwistStamped Command;

/** \brief Base path tracker class that implements a generic path tracker interface
 */
class Base {
 public:

  /** \brief Construct the base path tracker
   *
   * \param graph The pose graph
   * \param control_period_ms The period for the control loop
 */
  Base(const std::shared_ptr<Graph> & graph,
       double control_period_ms);

  /** \brief Destruct the path tracker, stopping any active paths
 */
  virtual ~Base();

  /** \brief Start a new thread that will follow the path
 *
 * \param state Initial following state
 * \param chain The path to follow
*/
  void followPathAsync(const State & state,
                       Chain& chain);

  /** \brief
 */
  virtual void finishControlLoop();

  /** \brief Load configuration parameters and do any pre-processing
 */
  virtual void loadConfigs() = 0;

  /** \brief Notify the path tracker about a new leaf (vision-based path localization)
   *
   * \param chain Ref to the loc. chain with the most recent pose estimate
   * \param leaf_stamp The timestamp for the pose
   * \param live_vid Vertex ID of the current vertex in the live run
 */
  ///
  virtual void notifyNewLeaf(const Chain & chain, ///<
                             Stamp leaf_stamp,///<
                             Vid live_vid ///< Vid of the current vertex in the live run
  ) = 0;

  /** \brief Notify the path tracker about a new leaf including a STEAM trajectory.
 */
  virtual void notifyNewLeaf(const Chain & chain,
                             const steam::se3::SteamTrajInterface & trajectory,
                             const Vid live_vid,
                             const uint64_t image_stamp) = 0;

#if 0
  /** \brief Pass a new IMU state to the path tracker
   *
   * \note Never implemented
 */
  virtual void updateImuState(const Eigen::Quaterniond & global_orientation,
                              const Eigen::Vector3d & global_accel,
                              const Stamp & stamp) {}
#endif

  /** \brief Determine if we are currently following a path
   *
   * \return True if following path
 */
  bool isRunning() {
    return control_loop_.valid() &&
        control_loop_.wait_for(std::chrono::milliseconds(0))
            != std::future_status::ready;
  }

/** \brief Set the path-following state for the active control loop
*
* \note Not thread-safe for multiple writers,
* only control the path tracker from a single thread.
 */
  void setState(const State & state) {
    // don't change the state if we've already commanded a stop
    if (state_ != State::STOP) {
      std::lock_guard<std::mutex> lock(state_mtx_);
      state_ = state;
    }
  }

/** \brief Get the path-following state
*/
  State getState() {
    return state_;
  }

  /** \brief Stop the path tracker and wait for the thread to finish
   */
  void stopAndJoin() {
    LOG(INFO) << "Path tracker stopping and joining";
    if (control_loop_.valid()) {
      setState(State::STOP);
      control_loop_.wait();
    }
  }

  /** \brief Pause the path tracker but keep the current path
   */
  void pause() {
    if (control_loop_.valid()) {
      LOG(INFO) << "Pausing the path tracker thread";
      setState(State::PAUSE);
      LOG(INFO) << "Path tracker paused";
    }
  }

  /** \brief Resume the goal after pause() called
  */
  void resume() {
    if (control_loop_.valid()) {
      LOG(INFO) << "Resuming the path tracker thread";
      setState(State::RUN);
      LOG(INFO) << "Path tracker thread resumed";
    }
  }

  /** \brief Constructor method for the factory
  */
  static std::shared_ptr<Base> Create();

  /** \brief The future for the async control loop task
  */
  std::future<void> control_loop_;

 protected:

  /** \brief Follow the path specified by the chain
   *
   * This is the synchronous implementation of the control loop
 */
  void controlLoop();

  /** \brief Sleep the remaining time in the control loop (currently fixed sleep)
   */
  virtual void controlLoopSleep();

  /** \brief This is the main control step that is left to derived classes
   */
  virtual Command controlStep() = 0;

  /** \brief Publish a drive command to the vehicle
   */
  virtual void publishCommand(Command &command);

  /** \brief Reset before a new run
   */
  virtual void reset() = 0;

  /** \brief The path tracker state (RUN/PAUSE/STOP) that guides the control loop
   */
  std::atomic<State> state_;

  /** \brief
   */
  std::mutex state_mtx_;

  /** \brief The control loop period in ms
   */
  double control_period_ms_;

  /** \brief
   */
  ::asrl::common::timing::SimpleTimer step_timer_;

  /** \brief The latest command produces by the controlStep
   */
  Command latest_command_;

  /** \brief The last time an update was received from the safety monitor
   */
  Stamp t_last_safety_monitor_update_;

  /** \brief The publisher for control messages
   *
   * \todo Not implemented
   */
// std::shared_ptr<asrl__common__babelfish::Publisher<geometry_msgs::Twist>> publisher_;

  /** \brief The localization chain
   */
  std::shared_ptr<Chain> chain_;

  /** \brief The pose graph
   */
  std::shared_ptr<const Graph> graph_;

};

}
}