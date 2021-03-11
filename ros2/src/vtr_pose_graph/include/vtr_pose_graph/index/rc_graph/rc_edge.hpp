#pragma once

#include <vtr_common/utils/container_tools.hpp>
#include <vtr_messages/msg/graph_edge.hpp>
#include <vtr_messages/msg/graph_edge_header.hpp>
#include <vtr_pose_graph/index/edge_base.hpp>
#include <vtr_pose_graph/interface/rc_point_interface.hpp>

namespace vtr {
namespace pose_graph {

class RCEdge : public EdgeBase, public RCPointInterface {
 public:
  // Helper typedef to find the base class corresponding to edge data
  using Base = EdgeBase;

  // Message typedefs, used for retreiving a message type for an arbitrary graph
  // object
  using Msg = vtr_messages::msg::GraphEdge;
  using HeaderMsg = vtr_messages::msg::GraphEdgeHeader;

  // When loading
  using RunFilter = std::unordered_set<BaseIdType>;

  /** \brief Typedefs for shared pointers to edges */
  PTR_TYPEDEFS(RCEdge)

  /**
   * \brief Interface to downcast base class pointers
   * \details This allows us to do DerivedPtrType = Type::Cast(BasePtrType)
   */
  PTR_DOWNCAST_OPS(RCEdge, EdgeBase)

  /** \brief Typedefs for containers of edges */
  CONTAINER_TYPEDEFS(RCEdge)

  /** \brief Pseudo-constructors for making shared pointers to edges */
  static Ptr MakeShared() {
    return Ptr(new RCEdge());
  }
  static Ptr MakeShared(const IdType& id) {
    return Ptr(new RCEdge(id));
  }
  static Ptr MakeShared(const IdType& id, const VertexId& fromId,
                        const VertexId& toId, bool manual = false) {
    return Ptr(new RCEdge(id, fromId, toId, manual));
  }
  static Ptr MakeShared(const IdType& id, const VertexId& fromId,
                        const VertexId& toId, const TransformType& T_to_from,
                        bool manual = false) {
    return Ptr(new RCEdge(id, fromId, toId, T_to_from, manual));
  }
  static Ptr MakeShared(
      const vtr_messages::msg::GraphEdge& msg, BaseIdType runId,
      const LockableFieldMapPtr& streamNames,
      const RCPointInterface::LockableDataStreamMapPtr& streamMap) {
    return Ptr(new RCEdge(msg, runId, streamNames, streamMap));
  }
  /** \brief Default constructor */
  RCEdge() = default;
  explicit RCEdge(const IdType& id) : EdgeBase(id), RCPointInterface(){};
  RCEdge(const IdType id, const VertexId& fromId, const VertexId& toId,
         bool manual = false)
      : EdgeBase(id, fromId, toId, manual), RCPointInterface(){};

  RCEdge(const IdType& id, const VertexId& fromId, const VertexId& toId,
         const TransformType& T_to_from, bool manual = false)
      : EdgeBase(id, fromId, toId, T_to_from, manual), RCPointInterface(){};

  RCEdge(const vtr_messages::msg::GraphEdge& msg, BaseIdType runId,
         const LockableFieldMapPtr& streamNames,
         const RCPointInterface::LockableDataStreamMapPtr& streamMap);

  /** \brief Default constructor */
  virtual ~RCEdge() = default;

  /** \brief Serialize to a ros message, as a temporal edge */
  Msg toRosMsg();

  /** \brief Helper for run filtering while loading */
  static inline bool MeetsFilter(const Msg& m, const RunFilter& r) {
    return (m.to_run_id == -1) || common::utils::contains(r, m.to_run_id);
  }

  /** \brief String name for file saving */
  const std::string name() const;
};
}  // namespace pose_graph
}  // namespace vtr
