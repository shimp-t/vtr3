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
 * \file follow.cpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#include <vtr_mission_planning/states/repeat/follow.hpp>

namespace vtr {
namespace mission_planning {
namespace state {

namespace repeat {

auto Follow::nextStep(const Base *newState) const -> BasePtr {
  // If where we are going is not a child, delegate to the parent
  if (!InChain(newState)) return Parent::nextStep(newState);

  // If we aren't changing to a different chain, there is no intermediate step
  return nullptr;
}

void Follow::processGoals(Tactic *tactic, UpgradableLockGuard &goal_lock,
                          const Event &event) {
  switch (event.signal_) {
    case Signal::Continue:
      break;
    case Signal::LocalizeFail: {
      // We do not know where we are, so detour back to MetricLocalize
      Event tmp(Action::AppendGoal, BasePtr(new repeat::MetricLocalize()),
                event.signal_);
      return Parent::processGoals(tactic, goal_lock, tmp);
    }
#if 0
    case Signal::Obstructed: {
      // We cannot continue along this path, so detour back to planning
      Event tmp(Action::AppendGoal, BasePtr(new repeat::Plan()));
      tmp.signal_ = event.signal_;
      return Parent::processGoals(tactic, goal_lock, tmp);
    }
#endif
    case Signal::GoalReached: {
      // The path is finished, and we can safely exit
      Event tmp(Action::EndGoal);
      tmp.signal_ = event.signal_;
      return Parent::processGoals(tactic, goal_lock, tmp);
    }
    default:
      // Any unhandled signals percolate upwards
      return Parent::processGoals(tactic, goal_lock, event);
  }

  switch (event.type_) {
    case Action::Continue: {
      /// \todo yuchen add this check back
#if false
      auto loc_status = tactic->status().localization_;
      if (loc_status != LocalizationStatus::Confident &&
          loc_status != LocalizationStatus::DeadReckoning) {
        auto T = tactic->persistentLoc().T;
        LOG(WARNING) << "Lost localization!! Has cov? " << T.covarianceSet();
        LOG_IF(T.covarianceSet(), WARNING) << T.cov();
        Event tmp(Action::AppendGoal, BasePtr(new repeat::MetricLocalize()));
        tmp.signal_ = event.signal_;
        return Parent::processGoals(tactic, goal_lock, tmp);
      }
#endif
      if (!waypoints_.empty()) {
        LOG(DEBUG) << "Front waypoint is: " << waypoints_.front()
                   << ", id: " << waypointSeq_.front() << ", distance:"
                   << tactic->distanceToSeqId(waypointSeq_.front());
      }

      // TODO: Right now the path tracker needs to say when its done....
      // If we have passed a waypoint, remove it from the list
      while (!waypointSeq_.empty() &&
             tactic->distanceToSeqId(waypointSeq_.front()) <= 0) {
        LOG(INFO) << "Popping waypoint " << waypoints_.front()
                  << " with distance "
                  << tactic->distanceToSeqId(waypointSeq_.front());
        waypoints_.pop_front();
        waypointSeq_.pop_front();
      }

      // We are done when there are no waypoints left
      if (waypoints_.empty()) {
        if (tactic->pathFollowingDone()) {
          CLOG(INFO, "state_machine")
              << "Path following completed; ending the current goal.";
          return Parent::processGoals(tactic, goal_lock,
                                      Event(Action::EndGoal));
        } else {
          CLOG_EVERY_N(16, INFO, "state_machine")
              << "All waypoints complete; waiting on path tracker to finish";
        }
      } else {
        const double travelled = -1 * tactic->distanceToSeqId(0);
        const double remained = tactic->distanceToSeqId(waypointSeq_.back());
        const double percent = travelled / (travelled + remained);
        container_->callbacks()->stateUpdate(percent * 100);
        LOG_EVERY_N(10, INFO) << "Percent complete is: " << percent;
      }
    }
      // NOTE: the lack of a break statement here is intentional, to allow
      // unhandled cases to percolate up the chain
      [[fallthrough]];
    default:
      // Delegate all goal swapping/error handling to the base class
      return Parent::processGoals(tactic, goal_lock, event);
  }
}

void Follow::onExit(Tactic *tactic, Base *newState) {
  // If the new target is a derived class, we are not exiting
  if (InChain(newState)) return;

  // Note: This is called *before* we call up the tree, as we destruct from
  // leaves to root
  // TODO: Exit from repeating metastate

  // Recursively call up the inheritance chain until we get to the least common
  // ancestor
  Parent::onExit(tactic, newState);
}

void Follow::onEntry(Tactic *tactic, Base *oldState) {
  // If the previous state was a derived class, we did not leave
  if (InChain(oldState)) return;

  // Recursively call up the inheritance chain until we get to the least common
  // ancestor
  Parent::onEntry(tactic, oldState);

  // Note: This is called after we call up the tree, as we construct from root
  // to leaves
  // TODO: The stuff
}

}  // namespace repeat
}  // namespace state
}  // namespace mission_planning
}  // namespace vtr
