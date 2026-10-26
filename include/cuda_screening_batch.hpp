#ifndef CUDA_SCREENING_BATCH_HPP
#define CUDA_SCREENING_BATCH_HPP

#include "cuda_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class GpuScreeningBatchStrategy
{
    WarpPerLigand,
    BlockPerLigand,
    MultiBlockPerLigand
};

// All arrays retain the order used by the ordinary CUDA screening pipeline.
// Matching-point indices in orders address matching_points, not atom indices.
struct GpuScreeningBatchLigand
{
    std::vector<GpuTripletKey> queries;
    std::vector<GpuLigandTripletOrders> orders;
    std::vector<GpuPoint3> matching_points;
    std::vector<GpuPoint3> atoms;
    std::vector<GpuFastMatchingArea> fast_areas;
};

struct GpuScreeningBatchOptions
{
    GpuScreeningBatchStrategy strategy = GpuScreeningBatchStrategy::BlockPerLigand;
    std::uint32_t block_size = 128;
    double clash_threshold = 0.05;
    // Exact budget for this call's explicit device buffers, excluding borrowed
    // protein resources, the CUDA context, and compiler-managed thread storage.
    std::size_t max_device_bytes = std::size_t{256} * 1024 * 1024;
    // Only MultiBlockPerLigand uses this value. Records from each query are
    // striped over these blocks, without limiting the number of candidates.
    std::uint32_t blocks_per_ligand = 8;
};

struct GpuScreeningBatchResult
{
    std::uint64_t candidate_matches{};
    std::uint64_t transforms_generated{};
    std::uint64_t valid_transforms{};
    std::uint64_t survivors{};
    bool has_top_pose = false;
    GpuRigidTransform top_transform{};
    GpuClashScore top_clash{};
    GpuFastScore top_fast_score{};
    // Lexicographic generation identity: query, neighbor slot, absolute shared
    // index record, equivalent ligand order. Equal Fast scores keep the first.
    std::uint32_t top_query_index{};
    std::uint32_t top_neighbor_slot{};
    std::uint32_t top_record_index{};
    std::uint32_t top_order_index{};
};

struct GpuScreeningBatchTimings
{
    double pack_seconds{};
    double allocation_seconds{};
    double h2d_seconds{};
    // Includes the device summary reduction in multi-block mode.
    double kernel_seconds{};
    double d2h_seconds{};
    // Final host merge only; distinct from the device reduction above.
    double reduction_seconds{};
    double cleanup_seconds{};
    std::size_t batch_device_bytes{};
    std::size_t h2d_bytes{};
    std::size_t d2h_bytes{};
    // Device-wide free-memory samples, not a precise physical high-water mark.
    std::size_t min_free_device_bytes{};
    std::size_t total_device_bytes{};
};

// Warp/block modes use one block per ligand and strided queries. Multi-block
// mode assigns several blocks to each ligand: warps share queries and blocks
// stripe the matching records. Every record/order is evaluated, without
// candidate truncation or per-ligand index rebuilding. Only counts and top-1
// are returned; multi-block partial results are reduced on device and host.
// This experiment always uses quaternion fitting; it does not read the Kabsch
// environment selector. The caller supplies a sorted, previously uploaded
// protein index and consistent grid/target buffers, as for the baseline API.
// This blocking call borrows the persistent protein resources. Invalid input,
// CUDA errors, budget excess, or uint64 count overflow return false, clear the
// results, and describe the failure in error. Kernel time uses completed CUDA
// events; copy times are completed host wall intervals, including staging costs.
bool cuda_screen_ligand_batch(
    const GpuFlatTripletIndexDevice& protein_index,
    const GpuPointArrayDevice& protein_targets,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuScreeningBatchLigand>& ligands,
    const GpuScreeningBatchOptions& options,
    std::vector<GpuScreeningBatchResult>& results,
    GpuScreeningBatchTimings& timings,
    std::string& error
);

#endif
