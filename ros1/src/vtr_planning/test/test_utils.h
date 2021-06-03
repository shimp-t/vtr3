#include <vtr/planning/state_machine.h>

using namespace vtr::planning;
using namespace vtr::path_planning;
using state::Action;
using state::BaseState;
using state::Event;
using state::Signal;
using state::StateMachine;

/** Test tactic to ensure that the state machine makes the correct calls to the
 * tactic.
 */
struct TestTactic : public StateMachineInterface {
 public:
  PTR_TYPEDEFS(TestTactic)

  TestTactic()
      : pipeline_(PipelineType::Idle),
        closest_(VertexId::Invalid()),
        current_(VertexId::Invalid()) {
    status_.localization_ = LocalizationStatus::DeadReckoning;
    status_.safety_ = SafetyStatus::NotThatSafe;
  }

  void setPipeline(const PipelineType& pipeline) { pipeline_ = pipeline; }
  LockType lockPipeline() { return LockType(); }
  void setPath(const PathType&, bool) {}
  const Localization& persistentLoc() const { return loc_; }
  const Localization& targetLoc() const { return loc_; }
#if 0
  bool startHover(const PathType&) { return true; }
  bool startFollow(const PathType&) { return true; }
#endif
  void setTrunk(const VertexId&) {}  // not important for state machine testing
  double distanceToSeqId(const uint64_t&) { return 9001; }
  TacticStatus status() const { return status_; }
  const VertexId& closestVertexID() const { return closest_; }  // not important
  const VertexId& currentVertexID() const { return current_; }  // not important
  const VertexId& connectToTrunk(bool) { return closest_; }
  void addRun(bool, bool, bool) {}
#if 0
  void removeEphemeralRuns() {}
#endif
  void relaxGraph() {}
  void saveGraph() {}

  PipelineType pipeline_;
  VertexId closest_;  // not important for state machine testing
  VertexId current_;  // not important for state machine testing
  TacticStatus status_;
  Localization loc_;
};

/** Test path planner to ensure that the state machine makes the correct
 * callbacks to the path planner.
 */
class TestPathPlanner : public vtr::path_planning::PlanningInterface {
 public:
  PTR_TYPEDEFS(TestPathPlanner)
  PathType path(const VertexId& from, const VertexId& to) { return PathType{}; }
  PathType path(const VertexId&, const VertexId::List& to,
                std::list<uint64_t>* idx) {
    return PathType();
  }
  void updatePrivileged() {}
  double cost(const VertexId&) { return 0.0; }
  double cost(const EdgeId&) { return 0.0; }
  EvalPtr cost() {
    return asrl::pose_graph::Eval::Weight::Const::MakeShared(0, 0);
  }
};