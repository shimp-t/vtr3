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
 * \file kruskal_mst_functions.hpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <list>
#include <map>
#include <unordered_map>
#include <vector>

namespace vtr {
namespace pose_graph {
namespace simple {
namespace kruskal {

using SimpleVertex = uint64_t;

/** \brief A structure to represent a weighted edge in graph */
struct WeightedEdge {
  using SimpleEdge = std::pair<SimpleVertex, SimpleVertex>;

  WeightedEdge() {}
  WeightedEdge(const SimpleEdge &edge, double weight)
      : edge(edge), weight(weight) {}

  SimpleEdge edge;
  double weight;

  bool operator<(const WeightedEdge &other) const {
    return (weight < other.weight);
  }
};

/** \brief A structure to represent a subset for union-find */
struct UnionFindSubset {
  SimpleVertex parent;
  SimpleVertex rank;
};

/**
 * \brief A utility function to find set of an element i (uses path compression
 * technique)
 */
SimpleVertex find(std::unordered_map<SimpleVertex, UnionFindSubset> *subsetMap,
                  SimpleVertex i) {
  // Find root and make root as parent of i (path compression)
  if (subsetMap->at(i).parent != i) {
    subsetMap->at(i).parent = find(subsetMap, subsetMap->at(i).parent);
  }
  return subsetMap->at(i).parent;
}

/**
 * \brief A function that does union of two sets of x and y (uses union by
 * rank)
 */
void unionSubsets(std::unordered_map<SimpleVertex, UnionFindSubset> *subsetMap,
                  SimpleVertex x, SimpleVertex y) {
  SimpleVertex xroot = find(subsetMap, x);
  SimpleVertex yroot = find(subsetMap, y);

  // Attach smaller rank tree under root of high rank tree (Union by Rank)
  if (subsetMap->at(xroot).rank < subsetMap->at(yroot).rank) {
    subsetMap->at(xroot).parent = yroot;
  } else if (subsetMap->at(xroot).rank > subsetMap->at(yroot).rank) {
    subsetMap->at(yroot).parent = xroot;
  } else {
    // Ranks are the same, make one root and increment its rank by one
    subsetMap->at(yroot).parent = xroot;
    subsetMap->at(xroot).rank++;
  }
}

}  // namespace kruskal
}  // namespace simple
}  // namespace pose_graph
}  // namespace vtr
