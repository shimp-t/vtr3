#pragma once

#include <memory>
#include <mutex>

/// #include <robochunk/base/ChunkSerializer.hpp>
/// #include <robochunk/base/DataBubble.hpp>
#include <vtr_pose_graph/robochunk/base/chunk_serializer.hpp>
#include <vtr_pose_graph/robochunk/base/data_stream.hpp>
#include <vtr_storage/DataStreamReader.hpp>
#include <vtr_storage/DataStreamWriter.hpp>

namespace vtr {
namespace pose_graph {

#if 1
struct RobochunkIO {
  using StreamPtr = std::shared_ptr<robochunk::base::ChunkStream>;
  using SerializerPtr = std::shared_ptr<robochunk::base::ChunkSerializer>;

  StreamPtr first;
  SerializerPtr second;
  std::recursive_mutex read_mtx;
  std::recursive_mutex write_mtx;

  using Guard = std::unique_lock<std::recursive_mutex>;
  struct RWGuard {
    Guard read, write;
  };
  RWGuard lock(bool read = true, bool write = true) {
    RWGuard rwg{{read_mtx, std::defer_lock}, {write_mtx, std::defer_lock}};
    if (read and write) {
      std::lock(rwg.read, rwg.write);
    } else if (read) {
      rwg.read.lock();
    } else if (write) {
      rwg.write.lock();
    }
    return rwg;
  }
};
#endif

struct RosBagIO {
  using DataStreamReader = std::shared_ptr<storage::DataStreamReader>;
  using DataStreamWriter = std::shared_ptr<storage::DataStreamWriter>;

  DataStreamReader first;
  DataStreamWriter second;
  std::recursive_mutex read_mtx;
  std::recursive_mutex write_mtx;

  using Guard = std::unique_lock<std::recursive_mutex>;
  struct RWGuard {
    Guard read, write;
  };
  RWGuard lock(bool read = true, bool write = true) {
    RWGuard rwg{{read_mtx, std::defer_lock}, {write_mtx, std::defer_lock}};
    if (read and write) {
      std::lock(rwg.read, rwg.write);
    } else if (read) {
      rwg.read.lock();
    } else if (write) {
      rwg.write.lock();
    }
    return rwg;
  }
};
}  // namespace pose_graph
}  // namespace vtr