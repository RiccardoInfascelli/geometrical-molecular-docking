#ifndef GPU_LIGAND_BATCH_HPP
#define GPU_LIGAND_BATCH_HPP

#include "gpu_pose_search.hpp"

#include <cstddef>
#include <string>
#include <vector>

// Explicit planning-limit failure, distinct from CUDA allocation/device errors.
// The caller may retry a smaller complete batch after the failed call unwinds.
class GpuBatchMemoryError : public GpuLigandSearchError {
public:
    GpuBatchMemoryError(std::size_t required, std::size_t limit, const std::string& phase)
        : GpuLigandSearchError("CUDA batch explicit memory budget exceeded at " + phase +
              " (required=" + std::to_string(required) + ", limit=" + std::to_string(limit) + ")", 0, 0),
          required_bytes(required), limit_bytes(limit), stage(phase) {}
    std::size_t required_bytes, limit_bytes;
    std::string stage;
};

struct GpuLigandBatchSearchTimings {
    double search_seconds = 0.0;
    // Conservative allocation estimate, including an all-survivors population;
    // this is not a measured physical GPU memory high-water mark.
    std::size_t explicit_device_bytes = 0;
};

// Aggregate prepared ligands into the ordinary lookup/alignment pipeline, then
// return every Fast-scored survivor in each ligand's original alignment order.
// Packet order is result order; local alignment IDs restart at zero per ligand.
// With a device owner, retain the aggregate pool and return only counters on the
// host. Only alignment IDs are downloaded to validate order and ligand identity;
// pipeline.poses stays empty. Resident alignment IDs remain global.
// The target and packets are borrowed until this blocking call returns. One
// consumer may use the target/workspace at a time. Quaternion is required.
// The budget covers live explicit batch buffers and retained compaction scratch,
// excluding shared protein resources, context and wrapper-local query/CUB scratch.
// No candidate is truncated or processed through a hidden serial fallback.
std::vector<GpuLigandSearchResult> run_gpu_ligand_batch_search(
    const std::vector<GpuPreparedLigand*>& ligands,
    const PreparedGpuTarget& target,
    const RunConfig& config,
    std::size_t max_device_bytes,
    GpuLigandBatchSearchTimings* timings = nullptr,
    GpuPosePipelineDeviceResult* device_result = nullptr);

#endif
