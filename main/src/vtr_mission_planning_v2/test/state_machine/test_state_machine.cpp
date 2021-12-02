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
 * \file test_state_machine.cpp
 * \brief
 *
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#include <gtest/gtest.h>

#include "vtr_logging/logging_init.hpp"
#include "vtr_mission_planning_v2/state_machine/base_state.hpp"
#include "vtr_mission_planning_v2/state_machine/state_machine.hpp"
#include "vtr_mission_planning_v2/test_utils.hpp"

using namespace ::testing;
using namespace vtr::mission_planning;

struct StateGenerator {
  // clang-format off
  static StateInterface::Ptr idle() { return std::make_shared<Idle>(); }
  // teach states
  static StateInterface::Ptr teach_branch() { return std::make_shared<teach::Branch>(); }
  static StateInterface::Ptr teach_merge() { return std::make_shared<teach::Merge>(); }
  static StateInterface::Ptr teach_topoloc() { return std::make_shared<teach::TopologicalLocalize>(); }
  // repeat states
  static StateInterface::Ptr repeat_follow() { return std::make_shared<repeat::Follow>(); }
  static StateInterface::Ptr repeat_metrloc() { return std::make_shared<repeat::MetricLocalize>(); }
  static StateInterface::Ptr repeat_topoloc() { return std::make_shared<repeat::TopologicalLocalize>(); }
  static StateInterface::Ptr repeat_plan() { return std::make_shared<repeat::Plan>(); }
  // clang-format on
};

class StateMachineTest : public Test {
 protected:
  // clang-format off
  StateMachine::Tactic::Ptr tactic = std::make_shared<TestTactic>();
  StateMachine::RoutePlanner::Ptr planner = std::make_shared<TestRoutePlanner>();
  StateMachineCallback::Ptr callback = std::make_shared<TestCallback>();
  // clang-format on
};

TEST_F(StateMachineTest, constructor_destructor) {
  StateMachine sm(tactic, planner, callback);
}

#if false
/** Ensure the state machine can handle all events properly. */
TEST(EventHandling, eventHandling) {
  StateMachine::Ptr state_machine = StateMachine::InitialState();

  TestCallbacks::Ptr callbacks(new TestCallbacks());
  state_machine->setCallbacks(callbacks.get());
  TestTactic::Ptr tactic(new TestTactic());
  state_machine->setTactic(tactic.get());
  state_machine->setPlanner(TestPathPlanner::Ptr(new TestPathPlanner()));

  // Start in idle
  EXPECT_EQ(state_machine->name(), "::Idle");
  EXPECT_EQ(state_machine->goals().size(), (unsigned)1);

  // Handle idle -> idle: nothing should have changed
  state_machine->handleEvents(Event::StartIdle());
  EXPECT_EQ(state_machine->name(), "::Idle");
  EXPECT_EQ(state_machine->goals().size(), (unsigned)1);
#if 0
  // Handle pause from idle:
  //   Goal size is increased with another idle in goal stack.
  //     \todo Confirm that this is the intended result.
  state_machine->handleEvents(Event::Pause());
  EXPECT_EQ(state_machine->name(), "::Idle");
  EXPECT_EQ(state_machine->goals().size(), (unsigned)2);
#endif
  // Handle idle -> teach::branch:
  //   Goes into topological localization state first (entry state of teach)
  //   Trigger stateChanged callback saying it's in topological localization
  //   Call tactic to LockPipeline
  //   Perform idle onExit, topological localization setPipeline and onEntry
  //     Call tactic to addRun \todo there is a ephermeral flag seems never used
  //   Trigger stateChanged callback saying it's in branch
  //   Perform topological localization onExit, teach setPipeline and onEntry
  //   Pipeline unlocked (out scope)
  state_machine->handleEvents(Event::StartTeach());
  EXPECT_EQ(state_machine->name(), "::Teach::Branch");
  EXPECT_EQ(state_machine->goals().size(), (unsigned)1);

  // Handle teach::branch -> teach::merge:
  //   Trigger stateChanged callback saying it's in merge (change directly)
  //   Call tactic to LockPipeline
  //   Perform branch onExit, merge setPipeline and onEntry
  //      Call tactic to setPath, setting merge target
  //      Reset cancelled_ to false (cancelled_ says merge is cancelled/failed)
  //   Pipeline unlocked (out scope)
  // \todo the second argument is necessary?
  state_machine->handleEvents(
      Event::StartMerge(std::vector<VertexId>{{1, 50}, {1, 300}}, {1, 50}));
  EXPECT_EQ(state_machine->name(), "::Teach::Merge");
  EXPECT_EQ(state_machine->goals().size(), (unsigned)1);

  // Handle signal AttemptClosure in merge without successful localization:
  //   AttemptClosure failed so fall back to ContinueTeach via swap goal
  //   Trigger stateChanged callback saying it's in branch (change directly)
  //   Perform merge onExit, branch setPipeline and onEntry
  //   Pipeline unlocked (out scope)
  state_machine->handleEvents(Event(Signal::AttemptClosure));
  EXPECT_EQ(state_machine->name(), "::Teach::Branch");
  EXPECT_EQ(state_machine->goals().size(), (unsigned)1);

  // \todo Need tests for AttemptClusure in merge with successful localization

  // Handle end goal event in teach:
  //   triggerSuccess
  //   Trigger stateChanged callback saying it's in idle
  //   Call tactic to LockPipeline
  //   Perform branch onExit, idle setPipeline and onEntry
  //     call tactic to lockPipeline, relaxGraph and saveGraph
  //     call path planner to updatePrivileged
  //     call tactic setPath to clear the path when entering Idle
  state_machine->handleEvents(Event(Action::EndGoal));
  EXPECT_EQ(state_machine->name(), "::Idle");
  EXPECT_EQ(state_machine->goals().size(), (unsigned)1);

  // Handle idle -> repeat (without persistent_loc):
  //   Goes into topological localization state first (entry state of repeat)
  //   Trigger stateChanged callback saying it's in topological localization
  //   Call tactic to LockPipeline
  //   Perform idle onExit, topological localization setPipeline and onEntry
  //     Call tactic to addRun \todo there is a ephermeral flag seems never used
  //   Check tactic->persistentLoc, found vertex not set and call Action::Abort
  //   Trigger stateChanged callback saying it's in Idle
  //   Pipeline unlocked (out scope)
  state_machine->handleEvents(Event::StartRepeat({{1, 50}, {1, 300}}));
  EXPECT_EQ(state_machine->name(), "::Idle");
  EXPECT_EQ(state_machine->goals().size(), (unsigned)1);

  // \todo Need tests to handle idle -> repeat with persistent_loc
}
#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}