#ifndef GPU_POSE_SEARCH_HPP
#define GPU_POSE_SEARCH_HPP

#include "cuda_pose_pipeline.hpp"
#include "ligand_preparation.hpp"
#include "protein_preparation.hpp"
#include "pose_generation.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

// One owner for the shared host target, fixed-cutoff index and uploaded buffers.
// A missing clearance disables pruning; zero is an active clearance threshold.
// The device handles are borrowed by searches and must not be freed by callers.
struct PreparedGpuTarget {
    PreparedProtein protein;
    GpuFlatTripletIndexDevice device_index{};
    GpuPointArrayDevice device_target_points{};
    GpuProteinGridDevice device_grid{};
    std::size_t index_records = 0;
    std::size_t index_keys = 0;
    double target_prep_seconds = 0.0;
    double index_seconds = 0.0;
    double gpu_upload_seconds = 0.0;

    PreparedGpuTarget(const RunConfig& config, double cutoff,
                      std::optional<double> minimum_target_clearance = std::nullopt,
                      bool timings = true);
    PreparedGpuTarget(const PreparedGpuTarget&) = delete;
    PreparedGpuTarget& operator=(const PreparedGpuTarget&) = delete;
    ~PreparedGpuTarget();

private:
    void release() noexcept;
};

struct GpuLigandSearchResult {
    GpuPosePipelineResult pipeline;
    std::size_t candidate_matches = 0;
    std::size_t alignments = 0;
    double query_preparation_seconds = 0.0;
    int query_threads_requested = 1;
    int query_threads_used = 1;
};

// Adapt all downloaded survivors to host rescoring without changing order,
// transforms, clash information or the precomputed Fast score.
std::vector<RankedPoseDiagnostic> make_rescoring_pose_pool(
    const std::vector<GpuPipelinePose>& poses);

// Preserve counters from completed device stages when a later stage fails.
// This lets the diagnostic screening driver retain its partial-error CSV data.
class GpuLigandSearchError : public std::runtime_error {
public:
    GpuLigandSearchError(const std::string& message, std::size_t matches,
                        std::size_t alignment_count, std::size_t valid_count = 0,
                        std::size_t survivor_count = 0)
        : std::runtime_error(message), candidate_matches(matches), alignments(alignment_count),
          valid_transforms(valid_count), survivors(survivor_count) {}
    std::size_t candidate_matches;
    std::size_t alignments;
    std::size_t valid_transforms;
    std::size_t survivors;
};

// Search a sanitized molecule prepared by prepare_ligand(). No preparation,
// ranking, truncation, rescoring, RMSD or export occurs here. Return every clash
// survivor in original alignment order, including its CUDA Fast score. By default
// the pool is downloaded; device_result instead retains it with no bulk download
// and leaves pipeline.poses empty. Its owner may outlive this search call.
// Empty triplets or an empty index return an empty result without device lookup.
// The blocking call completes device work and releases search inputs/intermediates.
// Non-null stages enables the same diagnostic stage synchronizations as the
// screening driver. Download intervals are collected only when requested;
// pipeline stage timings are appended to stages, preserving its existing items.
// collect_query_timings measures host packing without adding device synchronization.
GpuLigandSearchResult run_gpu_ligand_search(
    const RDKit::ROMol& molecule,
    const PreparedLigand& ligand,
    const PreparedGpuTarget& target,
    const RunConfig& config,
    StageProfile* stages = nullptr,
    GpuPosePipelineTimings* download_timings = nullptr,
    GpuPosePipelineDeviceResult* device_result = nullptr,
    bool collect_query_timings = false
);

#endif
