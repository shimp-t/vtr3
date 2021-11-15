#include <filesystem>
// #include <c++/8/fstream>
#include <fstream>
#include <iostream>
#include <random>
#include <typeinfo>

#include <vtr_logging/logging_init.hpp>
#include <vtr_messages/msg/rig_landmarks.hpp>
#include <vtr_messages/msg/localization_status.hpp>
#include <vtr_messages/msg/matches.hpp>
#include <vtr_pose_graph/index/rc_graph/rc_graph.hpp>
#include <vtr_pose_graph/path/path.hpp>
#include "rclcpp/rclcpp.hpp"

#include <vtr_common/timing/time_utils.hpp>
#include <vtr_common/utils/filesystem.hpp>

namespace fs = std::filesystem;
using namespace vtr::pose_graph;

float distance(const float *d1, const float *d2, unsigned size) {

  // This code is run so often, that we need to optimize it:
  float score = 0.f;
  for (unsigned i = 0; i < size; ++i)
    score += d1[i]*d2[i];

  // Practically, it does the same as this more readable version:
  // typedef Eigen::Matrix<float,Eigen::Dynamic,1> MatrixType;
  // Eigen::Map<const MatrixType> m1(d1,(int)size);
  // Eigen::Map<const MatrixType> m2(d2,(int)size);
  // // size is the number of bytes used for the descriptor
  // float score =  m1.transpose()*m2;

  // LOG(INFO) << score;
  score = score / size;
  // LOG(INFO) << score;
  //return the value
  return 1.f-score;
}

void ReadFeatureMatches(std::string graph_dir, 
                        std::string results_dir,
                        int* num_fail_read_all) {

  // Load the graph
  std::shared_ptr<RCGraph> graph;
  graph = vtr::pose_graph::RCGraph::LoadOrCreate(graph_dir, 0);
  LOG(INFO) << "Loaded graph has " << graph->vertices()->size() << " vertices";

  // Register the streams so we can read messages from them.
  std::string stream_name_loc = "results_localization";
  std::string stream_name_landmark = "front_xb3_landmarks";
  std::string stream_name_match = "front_xb3_landmarks_matches";

  for (const auto& run : graph->runs()) {
    // We are iterating over the teach and one repeat, run_000000 and run_000001
    run.second->registerVertexStream<vtr_messages::msg::RigLandmarks>(
        stream_name_landmark, true, RegisterMode::Existing);
 
    if (! run.second->isManual()) { 
      run.second->registerVertexStream<vtr_messages::msg::LocalizationStatus>(
        stream_name_loc, true, RegisterMode::Existing);
      run.second->registerVertexStream<vtr_messages::msg::Matches>(
        stream_name_match, true, RegisterMode::Existing);
    }
  }

  // Set up CSV files for wititing the data.
  std::ofstream fm_file;
  fs::path results_path{results_dir};
  fs::create_directory(results_path);
  fm_file.open(fs::path{results_path / "feature_matches.csv"});
  fm_file << "timestamp,live_id,map_id,from_id,to_id,16,32,64,128,256,total\n";

  int r_ind = 0;
  for (const auto& run : graph->runs()) {

    if (run.second->isManual()) {
      r_ind++;
      continue;
    }

    int num_vertices = run.second->vertices().size();

    for (int v_ind = 0; v_ind < num_vertices; v_ind++) {

      auto v = graph->at(VertexId(r_ind, v_ind));
               
      try {
        // Load localization status message to get query and map id.
        v->load(stream_name_loc);
        auto msg = v->retrieveKeyframeData<vtr_messages::msg::LocalizationStatus>(
          stream_name_loc);
        uint64_t query_id =  msg->query_id;
        uint64_t map_id =  msg->map_id; 
        auto timestamp = msg->keyframe_time;

        LOG(INFO) << "Query id: " << query_id;
        LOG(INFO) << "Map id: " << map_id;

        // Load landmarks messages for the repeat
        v->load(stream_name_landmark);
        auto landmark_msg = v->retrieveKeyframeData<vtr_messages::msg::RigLandmarks>(
          stream_name_landmark);
        auto rgb_channel_landmarks = landmark_msg->channels[0];
        auto &descriptors_string_repeat = rgb_channel_landmarks.descriptors;
        const auto &step_size = rgb_channel_landmarks.desc_type.bytes_per_desc;
        
        // Load inlier matches message
        v->load(stream_name_match);
        auto matches_msg = v->retrieveKeyframeData<vtr_messages::msg::Matches>(
          stream_name_match);
        auto inlier_matches = matches_msg->matches; 

        int num_landmarks_repeat = rgb_channel_landmarks.valid.size();
        int num_inlier_matches = inlier_matches.size();
        LOG(INFO) << "Num landmarks teach: " << num_landmarks_repeat;
        LOG(INFO) << "Num landmark inliers: " << num_inlier_matches;

        // Find the corresponding teach vertex that we have localized to and 
        // load the land marks from the teach so we can acces the descriptors.
        auto v_teach = graph->at(VertexId(map_id));

        // Load landmarks messages for the repeat
        v_teach->load(stream_name_landmark);
        auto landmark_msg_teach = 
            v_teach->retrieveKeyframeData<vtr_messages::msg::RigLandmarks>(
            stream_name_landmark);
        auto rgb_channel_landmarks_teach = landmark_msg_teach->channels[0];
        auto &descriptors_string_teach = rgb_channel_landmarks_teach.descriptors;
        
        int num_landmarks_teach = rgb_channel_landmarks_teach.valid.size();
        LOG(INFO) << "Num landmarks teach: " << num_landmarks_teach;

        // Store the landmark matches in a csv file.                               
        for (int m_ind = 0; m_ind < num_inlier_matches; m_ind++) {
            
          auto from_id = inlier_matches[m_ind].from_id;
          auto to_ids = inlier_matches[m_ind].to_id;
          int num_inlier_matches = to_ids.size();

          if (num_inlier_matches > 0) {

            // We should only match to the teach.
            if (num_inlier_matches > 1) {
              LOG(ERROR) << "Matched to " << num_inlier_matches << " landmarks";
            }                

            // Check that landmark ids are in range.
            if (from_id.idx >= num_landmarks_repeat) {
              LOG(ERROR) << "from id: " << from_id.idx;
              LOG(ERROR) << "num_landmarks_repeat: " << num_landmarks_repeat;
            }

            if (to_ids[0].idx >= num_landmarks_teach) {
              LOG(ERROR) << "to id: " << to_ids[0].idx;
              LOG(ERROR) << "num_landmarks_teach: " << num_landmarks_teach;
            }

            // Write to csv file.
            fm_file << timestamp << ","
                    << query_id << "," 
                    << map_id << "," 
                    << from_id.idx << ","
                    << to_ids[0].idx << ",";

            std::vector<int> layer_start_inds = {0, 16, 32, 64, 128, 0};
            std::vector<int> layer_size = {16, 32, 64, 128, 256, 496};

            for (int l_ind=0; l_ind < layer_start_inds.size(); l_ind++) {
              
              auto pointer_index_repeat = (step_size * from_id.idx) + 
                                          (step_size * layer_start_inds[l_ind]);

              auto pointer_index_teach = (step_size * to_ids[0].idx) + 
                                         (step_size * layer_start_inds[l_ind]);

              auto *descriptor_repeat = 
               (float *)&descriptors_string_repeat.data()[pointer_index_repeat];
        
              auto *descriptor_teach = 
               (float *)&descriptors_string_teach.data()[pointer_index_teach];

              double score = distance(descriptor_repeat, 
                                      descriptor_teach, 
                                      layer_size[l_ind]);

              fm_file << score << ",";            
            }

            fm_file << "\n";        
          }
        }

      } catch (const std::exception& e){
        LOG(ERROR) << "COULD NOT MESSAGES: " << v_ind << "/" << num_vertices;
        *num_fail_read_all += 1;
      }
    }

    fm_file.close();    
    r_ind++;
  }
}

// Run this twice. Second time tests retrieval from disk.
int main(int argc, char** argv) {

  LOG(INFO) << "here";

  std::string path_name = argv[argc-3];
  int start = atoi(argv[argc-2]);
  int end = atoi(argv[argc-1]);

  LOG(INFO) << "Path name: " << path_name;
  LOG(INFO) << "Start: " << start;
  LOG(INFO) << "End: " << end;

  int num_fail_read_all = 0;
 
  for(int i = start; i <= end; i++) {

    vtr::logging::configureLogging();

    std::stringstream graph_dir;
    std::stringstream results_dir;
    
    graph_dir << path_name << "/graph.index/repeats/" << i << "/graph.index";
    results_dir << path_name << "/graph.index/repeats/" << i << "/results";
    
    fs::path graph_dir_path{graph_dir.str()};
    LOG(INFO) << graph_dir_path;
    if(fs::exists(graph_dir_path)) {

      LOG(INFO) << "RUN: " << i;

      ReadFeatureMatches(graph_dir.str(), 
                         results_dir.str(), 
                         &num_fail_read_all); 
    }
  }

  LOG(INFO) << "Total failed read: " << num_fail_read_all; 
}