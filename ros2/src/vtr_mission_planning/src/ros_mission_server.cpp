#include <vtr_mission_planning/ros_mission_server.hpp>

#if 0
#include <actionlib/client/simple_goal_state.h>
#include <asrl/messages/Command.pb.h>
#include <asrl/messages/Goal.pb.h>
#include <vtr_planning/MissionGoal.h>
#endif

std::ostream &operator<<(std::ostream &os, const rclcpp_action::GoalUUID &id) {
  for (auto i : id) os << std::to_string(i);
  return os;
}

namespace vtr {
namespace mission_planning {

RosMissionServer::RosMissionServer(const std::shared_ptr<rclcpp::Node> node,
                                   const typename StateMachine::Ptr &state)
    : BaseMissionServer(state), node_(node) {
  action_server_ = rclcpp_action::create_server<Mission>(
      node_, "manager",
      std::bind(&RosMissionServer::_handleGoal, this, std::placeholders::_1,
                std::placeholders::_2),
      std::bind(&RosMissionServer::_handleCancel, this, std::placeholders::_1),
      std::bind(&RosMissionServer::_handleAccepted, this,
                std::placeholders::_1));
#if 0
  /// reorderService_ = nh_.advertiseService(
  ///     "reorder", &RosMissionServer::_reorderCallback, this);
  reorder_service_ = node->create_service<GoalReorder>(
      "reorder", std::bind(&RosMissionServer::_reorderCallback, this,
                           std::placeholders::_1, std::placeholders::_2));
#endif
  /// pauseService_ =
  ///     nh_.advertiseService("pause", &RosMissionServer::_pauseCallback,
  ///     this);
  pause_service_ = node->create_service<MissionPause>(
      "pause", std::bind(&RosMissionServer::_pauseCallback, this,
                         std::placeholders::_1, std::placeholders::_2));
#if 0
  /// cmdService_ =
  ///     nh_.advertiseService("cmd", &RosMissionServer::_cmdCallback, this);
  cmd_service_ = node->create_service<MissionCmd>(
      "cmd", std::bind(&RosMissionServer::_cmdCallback, this,
                       std::placeholders::_1, std::placeholders::_2));
#endif
  /// statusPublisher_ =
  ///     nh_.advertise<vtr_planning::MissionStatus>("status", 1, true);
  status_publisher_ = node_->create_publisher<MissionStatus>("status", 1);
  /// statusTimer_ = nh.createTimer(ros::Duration(0.25),
  ///                               &RosMissionServer::_publishStatus, this);
  status_timer_ = node_->create_wall_timer(
      1s, std::bind(&RosMissionServer::_publishStatus, this));
#if 0
  /// uiPublisher_ = nh_.advertise<asrl__messages::UILog>("out/ui_log", 10,
  /// true);
  ui_publisher_ = node_->create_publisher<UILog>("out/ui_log", 10);
#endif
}

void RosMissionServer::stateUpdate(double percent_complete) {
  _setFeedback(Iface::id(top()), false, percent_complete);
  _publishFeedback(Iface::id(top()));
}

void RosMissionServer::abortGoal(GoalHandle gh, const std::string &msg) {
  LockGuard lck(lock_);
  // Notify the client of the cancellation
  auto result = std::make_shared<Mission::Result>();
  result->return_code = Mission::Result::EXCEPTION;
  gh->abort(result);

  _publishFeedback(Iface::id(gh));
  feedback_.erase(Iface::id(gh));

  Parent::abortGoal(gh, msg);
  _publishStatus();
}

void RosMissionServer::cancelGoal(GoalHandle gh) {
  LockGuard lck(lock_);
  while (!gh->is_canceling())
    ;  // wait until the ros server says the goal is canceling.
  _publishFeedback(Iface::id(gh));
  feedback_.erase(Iface::id(top()));

  Parent::cancelGoal(gh);

  // Notify the client of the cancellation
  auto result = std::make_shared<Mission::Result>();
  result->return_code = Mission::Result::USER_INTERRUPT;
  gh->canceled(result);

  _publishStatus();
}

void RosMissionServer::executeGoal(GoalHandle gh) {
  LockGuard lck(lock_);

  // Notify the client that the goal is in progress (potentially waiting)
  gh->execute();
  _publishStatus();

  // Accept the goal internally and deal with pauses.  NOTE: async = false
  Parent::executeGoal(gh);
}

void RosMissionServer::finishGoal(GoalHandle gh) {
  // Notify the client of the success
  auto result = std::make_shared<Mission::Result>();
  result->return_code = Mission::Result::SUCCESS;
  gh->succeed(result);

  // Publish updated goal queue
  _publishStatus();
}

void RosMissionServer::transitionToNextGoal(GoalHandle gh) {
  LockGuard lck(lock_);

  // Publish a feedback message at 100%
  _setFeedback(Iface::id(gh), 100.0);
  _publishFeedback(Iface::id(gh));
  feedback_.erase(Iface::id(gh));

  // Remove the goal from the queue and do any pauses
  Parent::transitionToNextGoal(gh);

  // Publish updated goal queue
  _publishStatus();
}

void RosMissionServer::setGoalWaiting(GoalHandle gh, bool waiting) {
  LockGuard lck(lock_);
  _setFeedback(Iface::id(gh), waiting);
  _publishFeedback(Iface::id(gh));
}

rclcpp_action::GoalResponse RosMissionServer::_handleGoal(
    const typename Iface::Id &uuid, std::shared_ptr<const Mission::Goal>) {
  LOG(INFO) << "Found new goal: " << uuid;
  if (isTracking(uuid)) return rclcpp_action::GoalResponse::REJECT;
  return rclcpp_action::GoalResponse::ACCEPT_AND_DEFER;
}

rclcpp_action::CancelResponse RosMissionServer::_handleCancel(GoalHandle gh) {
  if (!isTracking(Iface::id(gh))) return rclcpp_action::CancelResponse::REJECT;

  // Launch a separate thread to cancel the goal after ros sets it to canceling.
  // Check if we have a goal to cancel, and block if we do.
  if (cancel_goal_future_.valid()) cancel_goal_future_.get();
  cancel_goal_future_ =
      std::async(std::launch::async, [this, gh] { cancelGoal(gh); });
  return rclcpp_action::CancelResponse::ACCEPT;
}

void RosMissionServer::_handleAccepted(GoalHandle gh) {
  if (Iface::target(gh) == Target::Unknown) {
    // Check to make sure the goal definition is in range
    auto result = std::make_shared<Mission::Result>();
    result->return_code = Mission::Result::UNKNOWN_GOAL;
    LOG(ERROR) << "Goal target not in {IDLE, TEACH, REPEAT}";
    gh->execute();
    gh->abort(result);
  } else if (Iface::target(gh) == Target::Repeat &&
             Iface::path(gh).size() == 0) {
    // Check to make sure that we got a path to repeat
    auto result = std::make_shared<Mission::Result>();
    result->return_code = Mission::Result::PATH_INVALID;
    LOG(ERROR) << "Issued a REPEAT Target without specifying a path";
    gh->execute();
    gh->abort(result);
  } else if (Iface::target(gh) == Target::Merge &&
             Iface::path(gh).size() == 0 &&
             Iface::vertex(gh) == VertexId::Invalid()) {
    auto result = std::make_shared<Mission::Result>();
    result->return_code = Mission::Result::PATH_INVALID;
    LOG(ERROR) << "Cannot merge without a target vertex and/or target path";
    gh->execute();
    gh->abort(result);
  } else if (Iface::target(gh) == Target::Localize &&
             Iface::path(gh).size() == 0 &&
             Iface::vertex(gh) == VertexId::Invalid()) {
    auto result = std::make_shared<Mission::Result>();
    result->return_code = Mission::Result::PATH_INVALID;
    LOG(ERROR) << "Cannot localize without a target vertex and/or target path";
    gh->execute();
    gh->abort(result);
  } else {
    // Otherwise we can accept this goal
    LOG(INFO) << "Adding goal: " << Iface::id(gh);
    addGoal(gh);
    _setFeedback(Iface::id(gh), false, 0);
    _publishFeedback(Iface::id(gh));
  }
}

#if 0
bool RosMissionServer::_reorderCallback(GoalReorder::Request &request,
                                        GoalReorder::Response &response) {
  // Republish service request for UI logging
  asrl::ui_msgs::ReorderGoal msg;
  msg.set_goal_id(request.goalId);
  msg.set_before_id(request.beforeId);
  msg.set_to_idx(request.toIdx);

  for (auto &&it : request.totalOrder) {
    *msg.mutable_total_order()->Add() = it;
  }

  _publishUI(msg);

  LockGuard lck(this->lock_);
  if (request.goalId != "") {
    if (this->goal(request.goalId).getGoalStatus().status !=
        actionlib_msgs::GoalStatus::PENDING) {
      response.returnCode = GoalReorder::Response::GOAL_IN_PORGRESS;
      return true;
    } else if (request.beforeId != "") {
      // Relative case: move the goal to just before another goal
      this->moveGoal(request.goalId, request.beforeId);
    } else {
      // Index case: move the goal to a position, or the end if toIdx is not in
      // [0, goal_queue_.size()]
      this->moveGoal(request.goalId, request.toIdx);
    }

    //    this->_publishFeedback();
    this->_publishStatus();
    return true;
  } else if (request.totalOrder.size() > 0) {
    this->reorderGoals(std::list<std::string>(request.totalOrder.begin(),
                                              request.totalOrder.end()));
    //    this->_publishFeedback();
    this->_publishStatus();
    return true;
  } else {
    response.returnCode = 255;
    return true;
  }

  // If something goes really wrong, don't send a response
  return false;
}
#endif

void RosMissionServer::_pauseCallback(
    std::shared_ptr<MissionPause::Request> request,
    std::shared_ptr<MissionPause::Response> response) {
#if 0
  // Republish service request for UI logging
  asrl::ui_msgs::MissionPause msg;
  msg.set_pause(request.paused);
  _publishUI(msg);
#endif
  /// ROS_INFO_NAMED("_pauseCallback", "In callback...");
  setPause(request->pause);

  /// ROS_INFO_NAMED("_pauseCallback", "Parent pause finished");

  if (request->pause) {
    /// ROS_INFO_NAMED("_pauseCallback", "Requested a pause");
    switch (status()) {
      case ServerState::Processing:
      case ServerState::Empty:
        response->response_code = MissionPause::Response::FAILURE;
        return;
      case ServerState::Paused:
        response->response_code = MissionPause::Response::SUCCESS;
        _publishStatus();
        return;
      case ServerState::PendingPause:
        response->response_code = MissionPause::Response::PENDING;
        _publishStatus();
        return;
    }
  } else {
    /// ROS_INFO_NAMED("_pauseCallback", "Requested a continue");
    switch (status()) {
      case ServerState::Processing:
      case ServerState::Empty:
        response->response_code = MissionPause::Response::SUCCESS;
        _publishStatus();
        return;
      case ServerState::Paused:
      case ServerState::PendingPause:
        response->response_code = MissionPause::Response::FAILURE;
        return;
    }
  }
}

#if 0
bool RosMissionServer::_cmdCallback(MissionCmd::Request &request,
                                    MissionCmd::Response &response) {
  // Republish service request for UI logging
  asrl::ui_msgs::MissionCmd msg;
  msg.set_vertex(request.vertex);

  for (auto &&it : request.path) {
    msg.mutable_path()->Add(it);
  }

  LockGuard lck(this->lock_);
  std::string name = stateMachine()->name();

  switch (request.action) {
    case MissionCmd::Request::ADD_RUN: {
      msg.set_action(asrl::ui_msgs::MissionCmd::ADD_RUN);
      _publishUI(msg);

      if (name == "::Idle") {
        this->addRun();
        response.success = true;
      } else {
        response.success = false;
        response.message = "Cannot add a run while not in ::Idle";
      }
      return true;
    }
    case MissionCmd::Request::LOCALIZE: {
      msg.set_action(asrl::ui_msgs::MissionCmd::LOCALIZE);
      _publishUI(msg);

      if (name == "::Idle") {
        LOG(INFO) << "Persistent vertex being set to: "
                  << VertexId(request.vertex);
        this->stateMachine()->tactic()->setTrunk(request.vertex);
        response.success = true;
      } else {
        response.success = false;
        response.message = "Cannot set the localization while not in ::Idle";
      }
      return true;
    }
    case MissionCmd::Request::START_MERGE: {
      msg.set_action(asrl::ui_msgs::MissionCmd::START_MERGE);
      _publishUI(msg);

      if (name == "::Teach::Branch") {
        VertexId::List tmp(request.path.begin(), request.path.end());
        this->stateMachine()->handleEvents(
            Event::StartMerge(tmp, request.vertex));
        response.success = true;
      } else {
        response.success = false;
        response.message =
            "Must be in ::Teach::Branch to move to ::Teach::Merge";
      }
      return true;
    }
    case MissionCmd::Request::CONFIRM_MERGE: {
      msg.set_action(asrl::ui_msgs::MissionCmd::CONFIRM_MERGE);
      _publishUI(msg);

      if (name == "::Teach::Merge") {
        this->stateMachine()->handleEvents(
            Event(state::Signal::AttemptClosure));
        response.success = true;
      } else {
        response.success = false;
        response.message =
            "Must be in ::Teach::Merge to confirm a loop closure";
      }
      return true;
    }
    case MissionCmd::Request::LOC_SEARCH: {
      msg.set_action(asrl::ui_msgs::MissionCmd::LOC_SEARCH);
      _publishUI(msg);

      if (name == "::Repeat::Follow") {
        this->stateMachine()->handleEvents(Event(state::Signal::LocalizeFail));
        response.success = true;
      } else {
        response.success = false;
        response.message =
            "Must be in ::Repeat::Follow to force a localization search";
      }
      return true;
    }
  }

  LOG(ERROR) << "[RosMissionServer] Unhandled action received: "
             << request.action;
  return false;
}
#endif

void RosMissionServer::_publishFeedback(const Iface::Id &id) {
  LockGuard lck(lock_);
  try {
    if (feedback_[id] == nullptr) return;
    (*goal_map_.at(id))->publish_feedback(feedback_[id]);
  } catch (const std::out_of_range &e) {
    LOG(ERROR) << "Couldn't find goal in map: " << e.what();
  }
}

void RosMissionServer::_setFeedback(const Iface::Id &id, bool waiting,
                                    double percent_complete) {
  LockGuard lck(lock_);
  if (feedback_[id] == nullptr)
    feedback_[id] = std::make_shared<Mission::Feedback>();
  feedback_[id]->waiting = waiting;
  feedback_[id]->percent_complete = percent_complete;
}

void RosMissionServer::_publishStatus() {
  LockGuard lck(lock_);
  auto msg = MissionStatus{};
  switch (status()) {
    case ServerState::Empty:
      msg.status = MissionStatus::EMPTY;
      break;
    case ServerState::Paused:
      msg.status = MissionStatus::PAUSED;
      break;
    case ServerState::PendingPause:
      msg.status = MissionStatus::PENDING_PAUSE;
      break;
    case ServerState::Processing:
      msg.status = MissionStatus::PROCESSING;
      break;
  }

  msg.mission_queue.clear();
#if 0  /// \todo (yuchen) figure out mission_queue type
  for (auto &&it : goal_queue_) msg.mission_queue.push_back(Iface::id(it));
#endif
  status_publisher_->publish(msg);
}

}  // namespace mission_planning
}  // namespace vtr