#include <filesystem>
#include <vtr_common/utils/filesystem.hpp>
#include <vtr_logging/logging_init.hpp>
#include <vtr_tactic/types.hpp>  // TemporalEvaluator
#include "rclcpp/rclcpp.hpp"
#include <vtr_messages/msg/localization_status.hpp>
#include <vtr_pose_graph/index/rc_graph/rc_graph.hpp>
#include <vtr_pose_graph/path/path.hpp>

namespace fs = std::filesystem;

using namespace vtr;
using namespace vtr::pose_graph;

void GetDistOnVO(std::string graph_dir, 
                 std::string results_dir, 
                 int* total_load_fail) { 

  // Set up CSV files for wititing the data.
  std::ofstream dist_file;
  fs::path results_path{results_dir};
  fs::create_directory(results_path);
  dist_file.open(fs::path{results_path / "dist.csv"});
  dist_file << "timestamp,live_id,priv_id,success,distance\n";

  auto graph = pose_graph::RCGraph::LoadOrCreate(graph_dir, 0);

  LOG(INFO) << "Loaded pose graph has " << graph->numberOfRuns() << " runs and "
            << graph->numberOfVertices() << " vertices in total.";
  
  if (!graph->numberOfVertices()) return;

  // Register the stream so we can read messages from it
  std::string stream_name_loc = "results_localization";

  int r_ind = 0;
  for (const auto& run : graph->runs()) {
    if (r_ind > 0) {  
      run.second->registerVertexStream<vtr_messages::msg::LocalizationStatus>(
        stream_name_loc, true, RegisterMode::Existing);
    }
    r_ind++;
  }

  /// Create a temporal evaluator
  tactic::TemporalEvaluator::Ptr evaluator(new tactic::TemporalEvaluator());
  evaluator->setGraph(graph.get());

  /// Iterate over all runs
  double total_length = 0;

  r_ind = 0;
  // Iterate over all the runs in the graph (teach and repeat)
  for (auto iter = graph->runs().begin(); iter != graph->runs().end(); iter++) {
    
    // Only interested in the repeat, not the teach.
    if (r_ind > 0) {
      
      if (iter->second->vertices().empty()) continue;
      
      // Get the repeat run
      auto graph_run =
          graph->getSubgraph(tactic::VertexId(iter->first, 0), evaluator);
      
      // Iterate overt the vertices on the repeat
      tactic::LocalizationChain chain(graph);
      tactic::VertexId::Vector sequence;
      uint seq_id = 0;
      
      for (auto it = graph_run->begin(tactic::VertexId(iter->first, 0));
                it != graph_run->end(); ++it) {

        // Get the distance along the path so far
        sequence.push_back(it->v()->id());
        chain.setSequence(sequence);
        chain.expand();
        auto distance = chain.dist(seq_id);
        seq_id++;

        // Get localization info for the vertex
        try {
          it->v()->load(stream_name_loc);
          auto msg = it->v()->retrieveKeyframeData<vtr_messages::msg::LocalizationStatus>(
                       stream_name_loc);

          dist_file << msg->keyframe_time << "," 
                    << msg->query_id << "," 
                    << msg->map_id << "," 
                    << msg->success << ","
                    << msg->inlier_channel_matches[0] << "," 
                    << msg->inlier_channel_matches[1] << "," 
                    << msg->inlier_channel_matches[2] << "," 
                    << distance << "\n";
        } catch (const std::exception& e){
            LOG(ERROR) << "Could not load loc results, run: " 
                       << r_ind << ", vertex: " << seq_id - 1;
            *total_load_fail += 1;   
            continue;
        }
      }
    }

    r_ind++;
  }

  dist_file.close();
}

// Run this twice. Second time tests retrieval from disk.
int main(int argc, char** argv) {

  vtr::logging::configureLogging();

  std::string path_name = argv[argc-3];
  int start = atoi(argv[argc-2]);
  int end = atoi(argv[argc-1]);

  LOG(INFO) << "Path name: " << path_name;
  LOG(INFO) << "Start: " << start;
  LOG(INFO) << "End: " << end;

  int total_load_fail = 0;
  
  for(int i = start; i <= end; i++) {

    std::stringstream graph_dir;
    std::stringstream results_dir;
    graph_dir << path_name << "/graph.index/repeats/" << i << "/graph.index";
    results_dir << path_name << "/graph.index/repeats/" << i << "/results";

    fs::path graph_dir_path{graph_dir.str()};
    if(fs::exists(graph_dir_path)) {

      GetDistOnVO(graph_dir.str(), results_dir.str(), &total_load_fail);
    } else {
      LOG(ERROR) << "Path to graph does not exist: " << graph_dir.str();
    } 
  }

  LOG(INFO) << "Loc results load fail: " << total_load_fail;
}