#include <filesystem>
// #include <c++/8/fstream>
#include <fstream>
#include <iostream>
#include <random>

#include <vtr_logging/logging_init.hpp>
#include <vtr_messages/msg/rig_landmarks.hpp>
#include <vtr_messages/msg/localization_status.hpp>
#include <vtr_messages/msg/exp_recog_status.hpp>
#include <vtr_pose_graph/index/rc_graph/rc_graph.hpp>
#include <vtr_pose_graph/path/path.hpp>
#include <vtr_tactic/types.hpp> // TemporalEvaluator

namespace fs = std::filesystem;
using namespace vtr::pose_graph;

void GetVOPosesTeach(std::string graph_dir, std::string results_dir) {

  // Load the graph
  std::shared_ptr<RCGraph> graph;
  graph = vtr::pose_graph::RCGraph::LoadOrCreate(graph_dir, 0);
  LOG(INFO) << "Loaded graph has " << graph->vertices()->size() << " vertices";

  

  // Set up CSV files for wititing the data.
  std::ofstream vo_file;
  fs::path results_path{results_dir};
  fs::create_directory(results_path);
  vo_file.open(fs::path{results_path / "vo_poses_teach.csv"});
  
  vo_file << "priv_id,x, y\n";

  // Create a temporal mask
  vtr::tactic::TemporalEvaluator::Ptr tempeval(new vtr::tactic::TemporalEvaluator());
  tempeval->setGraph((void *)graph.get());  

  // get the transformation of the current vertex and its parent
    // this gives you parent to current transformation

  auto iter = graph->runs().begin();
  auto graph_run = graph->getSubgraph(vtr::tactic::VertexId(iter->first, 0), tempeval);

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  auto T_next_world = lgmath::se3::Transformation(transform);
    
  int ind = 0;
  for (auto it = graph_run->begin(vtr::tactic::VertexId(iter->first, 0)); it != graph_run->end(); ++it) {
    if (ind == 0) it++;
    ind = 1;

    const auto &T_next_previous = it->e()->T();

    T_next_world = T_next_previous * T_next_world;

    LOG(ERROR) << "vo: " << T_next_previous;
    LOG(ERROR) << "integrated: " << T_next_world;
    LOG(ERROR) << "integrated: " << T_next_world.r_ab_inb();

    vo_file << it->to().minorId() << ","
            << T_next_world.r_ba_ina()(0) << ","
            << T_next_world.r_ba_ina()(1) << "\n";
  }

  vo_file.close();
}

// Run this twice. Second time tests retrieval from disk.
int main(int argc, char** argv) {

  vtr::logging::configureLogging();

  std::string path_name = argv[argc-1];

  LOG(INFO) << "Path name: " << path_name;

  std::stringstream graph_dir;
  std::stringstream results_dir;
  graph_dir << path_name << "/graph.index/";
  results_dir << path_name << "/graph.index/repeats/vo_teach";

  fs::path graph_dir_path{graph_dir.str()};
  if(fs::exists(graph_dir_path)) {

    GetVOPosesTeach(graph_dir.str(), results_dir.str());
  } else {
    LOG(ERROR) << "Path to graph does not exist: " << graph_dir.str();
  } 
}