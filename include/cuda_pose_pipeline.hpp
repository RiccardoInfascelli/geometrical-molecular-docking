#ifndef CUDA_POSE_PIPELINE_HPP
#define CUDA_POSE_PIPELINE_HPP

#include "cuda_backend.hpp"
#include "stage_profile.hpp"

#include <cstdint>
#include <vector>


struct GpuPipelinePose
{
    std::uint32_t alignment_index{};

    GpuRigidTransform transform{};

    GpuClashScore clash{};

    GpuFastScore fast_score{};
};


struct GpuPosePipelineResult
{
    std::uint32_t input_alignment_count{};
    std::uint32_t valid_transform_count{};
    std::uint32_t clash_survivor_count{};

    std::vector<GpuPipelinePose> poses;
};

// Optional owner for the compacted pool and Fast scores. The CUDA rescoring path
// keeps these allocations until ranking and downloads only its selected top-k.
// Existing callers continue to receive the complete host pool.
struct GpuPosePipelineDeviceResult
{
    GpuClashSurvivorArrayDevice survivors{};
    GpuFastScoreArrayDevice fast_scores{};
    GpuPosePipelineDeviceResult() = default;
    GpuPosePipelineDeviceResult(const GpuPosePipelineDeviceResult&) = delete;
    GpuPosePipelineDeviceResult& operator=(const GpuPosePipelineDeviceResult&) = delete;
    ~GpuPosePipelineDeviceResult();
    void clear() noexcept;
};

// Optional host wall times around the existing blocking download wrappers.
// Include host allocation and any device wait inside those calls, not just copy
// bandwidth. Those two fields add no synchronization; skipped downloads stay zero.
// collect_stages additionally synchronizes the existing device stage boundaries
// and records completed host-wall stages. It is off by default and is preserved
// when the output metrics are reset.
struct GpuPosePipelineTimings
{
    double survivor_download_seconds{};
    double fast_score_download_seconds{};
    bool collect_stages = false;
    StageProfile stages;
};

// Borrow uploaded inputs, fit ordered triples, filter clash and score survivors
// on the device. Return surviving poses and Fast scores in alignment order.
// With device_result, transfer their ownership there instead of downloading;
// result still contains counts, but its host pose vector remains empty.
// This function performs no ranking or RMSD.
// When supplied, timing outputs are reset even if validation fails or no poses
// survive; the collect_stages request is retained.
// False reports invalid arguments or CUDA failure; partial counters are not a
// completed search result and must not be used as scientific output.
bool cuda_run_pose_pipeline(
    const GpuPointArrayDevice& ligand_matching_points,
    const GpuPointArrayDevice& protein_target_points,
    const GpuPointArrayDevice& ligand_atoms,
    const GpuFastMatchingAreaArrayDevice& ligand_fast_areas,
    const GpuProteinGridDevice& protein_grid,
    const GpuAlignmentCandidateArrayDevice& alignments,
    double clash_threshold,
    GpuPosePipelineResult& result,
    GpuPosePipelineTimings* timings = nullptr,
    GpuPosePipelineDeviceResult* device_result = nullptr
);


// Convenience overload: additionally upload the host alignment vector.
bool cuda_run_pose_pipeline(
    const GpuPointArrayDevice& ligand_matching_points,
    const GpuPointArrayDevice& protein_target_points,
    const GpuPointArrayDevice& ligand_atoms,
    const GpuFastMatchingAreaArrayDevice& ligand_fast_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuAlignmentCandidate>& alignments,
    double clash_threshold,
    GpuPosePipelineResult& result,
    GpuPosePipelineTimings* timings = nullptr
);


#endif
