#pragma once

#include <stdint.h>
#include <ostream>

#include <vtr_pose_graph/id/graph_id.hpp>

namespace vtr {
namespace mission_planning {
namespace state {

/**
 * \brief Enumerates direct commands to the StateMachine.  These represent very
 * low level commands that don't require any complex logic or safety checking.
 */
enum class Action : int8_t {
  Abort = -2,    // Kill it, state abort and drop into idle
  Reset = -1,    // Kill it and drop into idle
  Continue = 0,  // Keep going with this state (default action)
  NewGoal,       // Set the goal state, replacing the current goal if necessary
  SwapGoal,      // Replace the goal at the top of the stack.  Like NewGoal, but
                 // only replaces
  AppendGoal,    // Append a new temporary goal state (eg. taking a brief detour
                 // to reteach a bad looking area)
  EndGoal  // User indication that a goal is complete (eg. mapping/repair is
           // done now)
};

std::ostream& operator<<(std::ostream& os, const Action& action);

/**
 * \brief Enumerates descriptive signals, or world events that might require a
 * response.  These represent higher level objectives or conditions that a state
 * may need to check and respond to appropriately.
 */
enum class Signal : int8_t {
  Continue = 0,  // Keep going with this state (default action)
#if 0
  // [teach::topo_loc?, repeat::topo_loc?]
  CantLocalize,  // Hard failure on localization after trying many different
                 // things
  // [teach::topo_loc?, repeat::topo_loc?]
  VertexFound,   // Topological localization was successful
  // [repeat::plan?]
  RobotLost,     // We are really lost and need to redo topological localization
  CantPlan,  // There is no route to the goal, and we cannot do repair right now
  PlanSuccess,     // Path planning to goal succeeded
  DoRepair,        // User is indicating a need to repair
  // [repeat::metric_loc?]
  LocalizeObs,     // A localization type obstruction has occurred
  Localized,       // We have successfully localized (either metrically or
                   // topologically)
  // [repeat::follow?]
  LocalizeFail,    // Localization has failed
  Obstructed,      // Path is obstructed and we need to replan
#endif
  // [repeat::follow]
  GoalReached,  // The robot has reached the desired goal in following
  // [teach::branch?, teach::merge]
  AttemptClosure,  // Attempt to link back to the existing map
  // [teach::merge]
  ContinueTeach  // After linking to the map, continue teaching
};

std::ostream& operator<<(std::ostream& os, const Signal& signal);

// Forward declaration
class BaseState;

struct Event {
  using StatePtr = std::shared_ptr<BaseState>;
  using VertexId = vtr::pose_graph::VertexId;

  // Constructors for high level command events
  static Event StartIdle();
  static Event StartTeach();
  static Event StartRepeat(const std::list<VertexId>& waypoints);
  static Event StartMerge(const std::vector<VertexId>& matchWindow,
                          const VertexId& targetVertex);
  static Event StartMerge(const std::list<VertexId>& matchWindow,
                          const VertexId& targetVertex);
  static Event StartLocalize(const std::list<VertexId>& matchWindow,
                             const VertexId& targetVertex);
  static Event StartLocalize(const std::vector<VertexId>& matchWindow,
                             const VertexId& targetVertex);
#if 0
  static Event Pause();
#endif
  static Event Reset();

  Event(const Signal& signal);
  Event(const Action& type = Action::Continue, const StatePtr& goal = nullptr);

  std::string signalName() const;
  std::string actionName() const;

  friend std::ostream& operator<<(std::ostream& os, const Event& e);

  Action type_;
  StatePtr goal_;
  Signal signal_;
};

}  // namespace state
}  // namespace mission_planning
}  // namespace vtr