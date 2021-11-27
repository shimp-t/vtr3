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
 * \file template_module.hpp
 * \brief TemplateModule class definition
 *
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include "vtr_tactic/modules/base_module.hpp"
#include "vtr_tactic/modules/module_factory.hpp"  /// include this header if this module launches other modules
#include "vtr_tactic/task_queue.hpp"  /// include this header if using the task queue

namespace vtr {
namespace tactic {

/** \brief A tactic module template */
class TemplateModule : public BaseModule {
 public:
  /** \brief Static module identifier. */
  static constexpr auto static_name = "template";

  /** \brief Collection of config parameters */
  struct Config {
    std::string parameter = "default value";
  };

  TemplateModule(const std::string &name = static_name)
      : BaseModule{name}, config_(std::make_shared<Config>()) {}

  void configFromROS(const rclcpp::Node::SharedPtr &node,
                     const std::string param_prefix) override {
    /// Configure your module from ROS
    config_ = std::make_shared<Config>();
    // clang-format off
    config_->parameter = node->declare_parameter<std::string>(param_prefix + ".parameter", config_->parameter);
    // clang-format on
    CLOG(INFO, "tactic.module")
        << "Template module parameter set to: " << config_->parameter;
  }

 private:
  void runImpl(QueryCache &qdata, const Graph::Ptr &,
               const TaskExecutor::Ptr &executor) override {
    /// Pure virtual method that must be overriden.
    /// Do the actual work of your module. Load data from and store data to
    /// QueryCache.
    CLOG(INFO, "tactic.module") << "Running the template module...";
    /// You can use the executor to run some task later in a non-blocking way.
    /// Task constructor accepts a priority, a dependency ID of this task (uuid)
    /// and a set of dependencies (uuid) that this task depends on.
    /// The dependency ID is auto-generated by default, keep track of it in case
    /// your subsequent task depends on it.
    executor->dispatch(
        std::make_shared<Task>(shared_from_this(), qdata.shared_from_this()));
  }

  void runAsyncImpl(QueryCache &, const Graph::Ptr &, const TaskExecutor::Ptr &,
                    const Task::Priority &priority,
                    const Task::DepId &dep_id) override {
    /// Task that runs asynchronously.
    CLOG(INFO, "tactic.module")
        << "Running the async part of template module with priority: "
        << priority << " and dependency id: " << dep_id;
  }

  /** \brief Module configuration. */
  std::shared_ptr<Config> config_;  /// \todo no need to be a shared pointer.
};

}  // namespace tactic
}  // namespace vtr