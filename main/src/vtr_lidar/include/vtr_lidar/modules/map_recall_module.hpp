/**
 * \file map_recall_module.hpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <pcl_conversions/pcl_conversions.h>

#include <vtr_lidar/cache.hpp>
#include <vtr_tactic/modules/base_module.hpp>

// temp
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vtr_messages/msg/movability.hpp>
#include <vtr_messages/msg/point_map.hpp>

namespace vtr {
namespace lidar {

using PointCloudMsg = sensor_msgs::msg::PointCloud2;
using PointXYZMsg = vtr_messages::msg::PointXYZ;
using MovabilityMsg = vtr_messages::msg::Movability;
using PointMapMsg = vtr_messages::msg::PointMap;

/** \brief */
class MapRecallModule : public tactic::BaseModule {
 public:
  /** \brief Static module identifier. */
  static constexpr auto static_name = "lidar.map_recall";

  /** \brief Config parameters. */
  struct Config {
    float map_voxel_size = 0.2;
    bool visualize = false;
  };

  MapRecallModule(const std::string &name = static_name)
      : tactic::BaseModule{name}, config_(std::make_shared<Config>()) {}

  void configFromROS(const rclcpp::Node::SharedPtr &node,
                     const std::string param_prefix) override;

 private:
  void runImpl(tactic::QueryCache &qdata,
               const tactic::Graph::ConstPtr &graph) override;

  void visualizeImpl(tactic::QueryCache &,
                     const tactic::Graph::ConstPtr &) override;

  /** \brief Module configuration. */
  std::shared_ptr<Config> config_;

  /** \brief for visualization only */
  rclcpp::Publisher<PointCloudMsg>::SharedPtr map_pub_;
};

}  // namespace lidar
}  // namespace vtr