#include <filesystem>
// #include <c++/8/fstream>
#include <fstream>
#include <iostream>
#include <random>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vtr_logging/logging_init.hpp>
#include <vtr_messages/msg/image.hpp>
#include <vtr_messages/msg/localization_status.hpp>
#include <vtr_pose_graph/index/rc_graph/rc_graph.hpp>
#include <vtr_pose_graph/path/path.hpp>
#include "rclcpp/rclcpp.hpp"

#include <vtr_common/timing/time_utils.hpp>
#include <vtr_common/utils/filesystem.hpp>

namespace fs = std::filesystem;
using namespace vtr::pose_graph;

cv::Mat wrapImage(const vtr_messages::msg::Image &asrl_image) {
  const auto & data = asrl_image.data;

  //assert(data != nullptr);

  // Convert to opencv
  uint32_t width = asrl_image.width;
  uint32_t height =  asrl_image.height;
  std::string encoding = asrl_image.encoding;

  if(encoding == "mono8") {
    return cv::Mat(cv::Size(width,height),CV_8UC1,(void*)data.data());
  } else if (encoding == "bgr8") {
    return  cv::Mat(cv::Size(width,height),CV_8UC3,(void*)data.data());
  } else {
    return cv::Mat();
  }
}

cv::Mat setupDisplayImage(cv::Mat input_image) {
  // create a visualization image to draw on.
  cv::Mat display_image;
  if (input_image.type() == CV_8UC1) {
    cv::cvtColor(input_image, display_image, cv::COLOR_GRAY2RGB);
  } else if (input_image.type() == CV_16S) {
    input_image.convertTo(display_image, CV_8U, 255/(48*16.));
  } else {
    display_image = input_image.clone();
  }
  return display_image;
}

void ReadVisualizationImages(std::string graph_dir, std::string results_dir) {

  // Load the graph
  std::shared_ptr<RCGraph> graph;
  graph = vtr::pose_graph::RCGraph::LoadOrCreate(graph_dir, 0);
  LOG(INFO) << "Loaded graph has " << graph->vertices()->size() << " vertices";

  // Register the stream so we can read messages from it
  std::string stream_name_loc = "results_localization";
  std::string stream_name = "front_xb3_visualization_images";

  int r_ind = 0;
  for (const auto& run : graph->runs()) {
    // We are iterating over the teach and one repeat, run_000000 and run_000001
     
    run.second->registerVertexStream<vtr_messages::msg::Image>(stream_name, 
                                                  true, RegisterMode::Existing);
    if (r_ind > 0) {  
      run.second->registerVertexStream<vtr_messages::msg::LocalizationStatus>(
        stream_name_loc, true, RegisterMode::Existing);
    }
    r_ind++;
  }

  // Set up CSV files for wititing the data.
  fs::path results_path{results_dir};
  fs::create_directory(results_path);
  fs::path results_img_path{fs::path{results_path / "images"}};
  fs::create_directory(results_img_path);

  r_ind = 0;
  for (const auto& run : graph->runs()) {

    // if (r_ind > 0) {
      int num_vertices = run.second->vertices().size();
      
      for (int v_ind = 0; v_ind < num_vertices; v_ind++) {

        auto vertex_id = VertexId(r_ind, v_ind);
        // auto single_uint64_id = vertex_id->uint64_t();
        auto v = graph->at(vertex_id);
      
        try {

          uint64_t q_id = 0;
          uint64_t m_id = 0; 

          if (r_ind > 0) {
            v->load(stream_name_loc);
            auto loc_msg = v->retrieveKeyframeData<vtr_messages::msg::LocalizationStatus>(
               stream_name_loc);

            q_id = loc_msg->query_id;
            m_id = loc_msg->map_id;
          }

          v->load(stream_name);
          auto ros_image = 
                  v->retrieveKeyframeData<vtr_messages::msg::Image>(stream_name);

          cv::Mat input_image = wrapImage(*ros_image);
          cv::Mat display_image = setupDisplayImage(input_image);
                  
          std::stringstream img_file;
          img_file << results_img_path.u8string() << "/" 
                                                  << q_id << "_"
                                                  << r_ind << "_" 
                                                  << v_ind << "_"
                                                  << m_id << ".png";

          cv::imwrite(img_file.str(), display_image);

        } catch (const std::exception& e){
            LOG(ERROR) << "Could not load visualization image, run: " 
                       << r_ind << ", vertex: " << v_ind;
            continue;
        }
      }
    // }
    
    r_ind++;
  }
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
  
  for(int i = start; i <= end; i++) {
    
    std::stringstream graph_dir;
    std::stringstream results_dir;
    graph_dir << path_name << "/graph.index/repeats/" << i << "/graph.index";
    results_dir << path_name << "/graph.index/repeats/" << i << "/results";

    fs::path graph_dir_path{graph_dir.str()};
    LOG(INFO) << graph_dir_path;
    if(fs::exists(graph_dir_path)) {
    
      LOG(INFO) << "graph_dir: " << graph_dir.str();
      LOG(INFO) << "results_dir: " << results_dir.str();

      ReadVisualizationImages(graph_dir.str(), results_dir.str()); 
    } else {
      LOG(ERROR) << "Path does not exist: " << graph_dir_path;
    }
  }
}