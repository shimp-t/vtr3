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
 * \file evaluator_common_tests.cpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#include <gtest/gtest.h>

#include <random>

#include <vtr_logging/logging_init.hpp>
#include <vtr_pose_graph/evaluator/evaluators.hpp>

using namespace ::testing;  // NOLINT
using namespace vtr::logging;
using namespace vtr::pose_graph;

using IntRandType =
    decltype(std::bind(std::uniform_int_distribution<int64_t>{0, 1000},
                       std::mt19937(std::random_device{}())));
using DoubleRandType =
    decltype(std::bind(std::uniform_real_distribution<double>{0.f, 100.f},
                       std::mt19937(std::random_device{}())));

class EvaluatorTestFixture : public Test {
 public:
  EvaluatorTestFixture()
      : graph_(new BasicGraph()),
        irnd_(std::bind(std::uniform_int_distribution<int64_t>{0, 1000},
                        std::mt19937(std::random_device{}()))),
        drnd_(std::bind(std::uniform_real_distribution<double>{0.f, 100.f},
                        std::mt19937(std::random_device{}()))) {}

  ~EvaluatorTestFixture() override {}

  void SetUp() override {
    /* Create the following graph
     * R0: 0 --- 1 --- 2
     *       \
     *        \
     *         \
     * R1: 0 --- 1 --- 2
     *                 |
     * R2: 0 --- 1 --- 2
     *           |
     * R3: 0 --- 1 --- 2
     *                 |
     * R4: 0 --- 1 --- 2
     */

    // Add a graph with 5 runs and 3 vertices per run.
    for (int idx = 0; idx < 5; ++idx) {
      graph_->addRun();
      graph_->addVertex();
      graph_->addVertex();
      graph_->addVertex();
      graph_->addEdge(VertexId(idx, 0), VertexId(idx, 1), Temporal);
      graph_->addEdge(VertexId(idx, 1), VertexId(idx, 2), Temporal);
    }
    // Add spatial edges across runs.
    graph_->addEdge(VertexId(1, 1), VertexId(0, 0), Spatial);
    graph_->addEdge(VertexId(2, 2), VertexId(1, 2), Spatial);
    graph_->addEdge(VertexId(3, 1), VertexId(2, 1), Spatial);
    graph_->addEdge(VertexId(4, 2), VertexId(3, 2), Spatial);

    // set the edge's transform to something special;
    const auto& edge_map = graph_->edges()->unlocked().get();
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
  BasicGraph::UniquePtr graph_;
  IntRandType irnd_;
  DoubleRandType drnd_;
};
// clang-format off
TEST_F(EvaluatorTestFixture, DistanceEvaluator) {
  using namespace eval::Weight;

  const auto& edges = graph_->edges()->unlocked().get();
  const auto& vertices = graph_->vertices()->unlocked().get();

  // SECTION("Direct", "[direct]")
  {
    auto eval = Distance<BasicGraph>::Direct::MakeShared();

    eval->setGraph(graph_.get());

    for (auto it = vertices.begin(); it != vertices.end(); ++it) {
      EXPECT_EQ(eval->at(it->first), 0);
      EXPECT_EQ(eval->at(it->second->id()), 0);
      EXPECT_EQ((*eval)[it->first], 0);
      EXPECT_EQ((*eval)[it->second->id()], 0);
    }
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      auto tmpId = it->second->id();
      double norm = std::sqrt(std::pow(tmpId.majorId2(), 2) + std::pow(tmpId.minorId2(), 2) + std::pow(tmpId.idx(), 2));

      EXPECT_EQ(eval->at(it->first), norm);
      EXPECT_EQ(eval->at(it->second->id()), norm);

      EXPECT_EQ((*eval)[it->first], norm);
      EXPECT_EQ((*eval)[it->second->id()], norm);
    }
  }

  // SECTION("Caching", "[caching]")
  {
    auto eval = Distance<BasicGraph>::Caching::MakeShared();

    eval->setGraph(graph_.get());

    for (auto it = vertices.begin(); it != vertices.end(); ++it) {
      // Const access before computation should throw
      EXPECT_THROW(eval->at(it->first), std::out_of_range);
      EXPECT_THROW(eval->at(it->second->id()), std::out_of_range);

      // Non-const access before computation should compute and cache
      EXPECT_EQ((*eval)[it->first], 0);
      EXPECT_EQ((*eval)[it->second->id()], 0);

      // Const access after computation should succeed
      EXPECT_EQ(eval->at(it->first), 0);
      EXPECT_EQ(eval->at(it->second->id()), 0);
    }
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      auto tmpId = it->second->id();
      double norm = std::sqrt(std::pow(tmpId.majorId2(), 2) + std::pow(tmpId.minorId2(), 2) + std::pow(tmpId.idx(), 2));

      // Const access before computation should throw
      EXPECT_THROW(eval->at(it->first), std::out_of_range);
      EXPECT_THROW(eval->at(it->second->id()), std::out_of_range);

      // Non-const access before computation should compute and cache
      EXPECT_EQ((*eval)[it->first], norm);
      EXPECT_EQ((*eval)[it->second->id()], norm);

      // Const access after computation should succeed
      EXPECT_EQ(eval->at(it->first), norm);
      EXPECT_EQ(eval->at(it->second->id()), norm);
    }
  }

  // SECTION("Windowed", "[windowed]")
  {
    auto eval = Distance<BasicGraph>::Windowed::MakeShared(5);

    eval->setGraph(graph_.get());

    for (auto it = vertices.begin(); it != vertices.end(); ++it) {
      // Const access before computation should throw
      EXPECT_THROW(eval->at(it->first), std::out_of_range);
      EXPECT_THROW(eval->at(it->second->id()), std::out_of_range);

      // Non-const access before computation should compute and cache
      EXPECT_EQ((*eval)[it->first], 0);
      EXPECT_EQ((*eval)[it->second->id()], 0);

      // Const access after computation should succeed
      EXPECT_EQ(eval->at(it->first), 0);
      EXPECT_EQ(eval->at(it->second->id()), 0);
    }
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      auto tmpId = it->second->id();
      double norm = std::sqrt(std::pow(tmpId.majorId2(), 2) + std::pow(tmpId.minorId2(), 2) + std::pow(tmpId.idx(), 2));

      // Const access before computation should throw
      EXPECT_THROW(eval->at(it->first), std::out_of_range);
      EXPECT_THROW(eval->at(it->second->id()), std::out_of_range);

      // Non-const access before computation should compute and cache
      EXPECT_EQ((*eval)[it->first], norm);
      EXPECT_EQ((*eval)[it->second->id()], norm);

      // Const access after computation should succeed
      EXPECT_EQ(eval->at(it->first), norm);
      EXPECT_EQ(eval->at(it->second->id()), norm);
    }

    for (auto it = vertices.begin(); it != vertices.end(); ++it) {
      // Const access should fail as the vertex will be out of the cache by now
      EXPECT_THROW(eval->at(it->first), std::out_of_range);
      EXPECT_THROW(eval->at(it->second->id()), std::out_of_range);

      // Move the cache forward
      EXPECT_EQ((*eval)[it->first], 0);
    }
    for (auto it = edges.begin(); it != edges.end();
         ++it) {
      EdgeId tmpId = it->second->id();
      double norm = std::sqrt(std::pow(tmpId.majorId2(), 2) + std::pow(tmpId.minorId2(), 2) + std::pow(tmpId.idx(), 2));

      // Const access should fail as the vertex will be out of the cache by now
      EXPECT_THROW(eval->at(it->first), std::out_of_range);
      EXPECT_THROW(eval->at(it->second->id()), std::out_of_range);

      // Move the cache forward
      EXPECT_EQ((*eval)[it->first], norm);
    }
  }
}
// clang-format on
int main(int argc, char** argv) {
  configureLogging("", true);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}