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
 * \file types.hpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <cstdint>

namespace vtr {
namespace pose_graph {

/**
 * \brief Window type used for picking start/stop time when reindexing streams
 */
enum class WindowType : uint8_t { Before, Center, After };

/** \brief Mode for registering a vertex stream */
enum class RegisterMode : uint8_t {
  Create,    // No data exists; create a read/write stream
  Existing,  // Data exists but no stream; create a read-only stream
};

}  // namespace pose_graph
}  // namespace vtr