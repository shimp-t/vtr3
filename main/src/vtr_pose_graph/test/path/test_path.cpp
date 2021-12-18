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
 * \file path_tests.hpp
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#include <gtest/gtest.h>

#include <random>

#include "vtr_logging/logging_init.hpp"
#include "vtr_pose_graph/path/path.hpp"

using namespace ::testing;  // NOLINT
using namespace vtr::logging;
using namespace vtr::pose_graph;

using IntRandType =
    decltype(std::bind(std::uniform_int_distribution<int64_t>{0, 1000},
                       std::mt19937(std::random_device{}())));
using DoubleRandType =
    decltype(std::bind(std::uniform_real_distribution<double>{0.f, 100.f},
                       std::mt19937(std::random_device{}())));

class PathTest : public Test {
 public:
  PathTest()
      : graph_(new BasicGraph()),
        irnd_(std::bind(std::uniform_int_distribution<int64_t>{0, 1000},
                        std::mt19937(std::random_device{}()))),
        drnd_(std::bind(std::uniform_real_distribution<double>{0.f, 100.f},
                        std::mt19937(std::random_device{}()))) {}
  ~PathTest() override {}

  void SetUp() override {
    /* Create the following graph
     * R0: 0 --- 1 --- 2 --- ...
     *             \
     *              \
     *               \
     * R1: 0 --- 1 --- 2 --- ...
     */

    // Add a graph with 2 runs and 5 vertices per run.
    for (int major_idx = 0; major_idx < 2; ++major_idx) {
      // Create the robochunk directories
      graph_->addRun();
      graph_->addVertex();
      for (int minor_idx = 0; minor_idx < 5 - 1; ++minor_idx) {
        graph_->addVertex();
        graph_->addEdge(VertexId(major_idx, minor_idx),
                        VertexId(major_idx, minor_idx + 1), Temporal);
      }
    }
    // Add spatial edge across runs.
    graph_->addEdge(VertexId(1, 2), VertexId(0, 1), Spatial);

    // set the edge's transform to something special;
    auto edge_map = graph_->edges()->unlocked().get();
    for (auto itr = edge_map.begin(); itr != edge_map.end(); ++itr) {
      Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
      transform(0, 3) = itr->second->id().majorId2();
      transform(1, 3) = itr->second->id().minorId2();
      transform(2, 3) = itr->second->id().type();
      itr->second->setTransform(lgmath::se3::Transformation(transform));
    }
  }

  void TearDown() override {}

 protected:
  BasicGraph::Ptr graph_;
  IntRandType irnd_;
  DoubleRandType drnd_;
};

TEST_F(PathTest, PathTest) {
  using BasicPath = Path<BasicGraph>;
  using Sequence = BasicPath::Sequence;
  auto path = std::make_shared<BasicPath>(graph_);

  // SECTION("A valid path")
  {
    // run 0 ids
    Sequence seq = {0ul, 1ul, 2ul, 3ul};
    // copy the sequence into the path
    path->setSequence(seq);
    LOG(INFO) << "About to verify the sequence: " << path->sequence();
    EXPECT_TRUE(path->verifySequence());
  }

  // SECTION("A valid multi-run path")
  {
    // run 0 ids
    Sequence seq = {VertexId(0, 0), VertexId(0, 1), VertexId(1, 2),
                    VertexId(1, 3)};
    // copy the sequence into the path
    path->setSequence(seq);
    LOG(INFO) << "About to verify the sequence: " << path->sequence();
    EXPECT_TRUE(path->verifySequence());
  }

  // SECTION("A valid cyclic path")
  {
    // run 0 ids
    Sequence seq = {0ul, 1ul, 2ul, 1ul, 0ul};
    // copy the sequence into the path
    path->setSequence(seq);
    LOG(INFO) << "About to verify the sequence: " << path->sequence();
    EXPECT_TRUE(path->verifySequence());
  }

  // SECTION("An invalid path with valid vertices")
  {
    // run 0 ids
    Sequence seq = {0ul, 1ul, 4ul, 1ul};
    // copy the sequence into the path
    path->setSequence(seq);
    LOG(INFO) << "About to verify the sequence: " << path->sequence();
    EXPECT_FALSE(path->verifySequence());
  }

  // SECTION("An invalid path with invalid vertices")
  {
    // run 0 ids
    Sequence seq = {0ul, 1ul, 10ul, 1ul};
    // copy the sequence into the path
    path->setSequence(seq);
    LOG(INFO) << "About to verify the sequence: " << path->sequence();
    EXPECT_FALSE(path->verifySequence());
  }

  // SECTION("Expand path to get poses")
  {
    // run 0 ids
    Sequence seq = {0ul, 1ul, 2ul, 1ul, 0ul};
    // copy the sequence into the path
    path->setSequence(seq);
    EXPECT_TRUE(path->verifySequence());

    // The pose at the far end of the path
    Eigen::Matrix<double, 6, 1> end_vec;
    end_vec.setZero();
    end_vec(1) = -3.;

    EXPECT_TRUE(path->pose(0).vec() == path->pose(4).vec());
    EXPECT_TRUE(path->pose(1).vec() == path->pose(3).vec());
    EXPECT_TRUE(path->pose(2).vec() == end_vec);
  }
}

int main(int argc, char** argv) {
  configureLogging("", true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
