#ifndef CUDA_POSE_RANKING_HPP
#define CUDA_POSE_RANKING_HPP

#include "cuda_pose_pipeline.hpp"
#include "cuda_weighted_interaction_scoring.hpp"
#include "cuda_vinardo_scoring.hpp"

// Owns device IDs into the compacted survivor pool, not original alignment IDs.
// Release with cuda_free_rescoring_selection(); copied handles alias that storage.
struct GpuRescoringSelection {
    std::uint32_t* pose_ids{};
    std::uint32_t count{};
};

struct GpuRankedRescoredPose {
    std::uint32_t pose_id{};
    GpuPipelinePose pose{};
    GpuWeightedInteractionScore weighted_interaction{};
    GpuVinardoScore vinardo{};
    double adjusted{};
    double hybrid{};
};

struct GpuRankingTimings {
    double score_preparation_seconds{};
    double hybrid_seconds{};
    double ranking_seconds{};
    double download_seconds{};
};

// Validate every Fast score, then optionally select top-N on device. Zero means
// no prefilter. Selected IDs retain original survivor order before rescoring.
bool cuda_select_rescoring_pool(const GpuFastScoreArrayDevice& fast,
    std::uint32_t top_n, GpuRescoringSelection& selection);
void cuda_free_rescoring_selection(GpuRescoringSelection& selection);

// Methods match PoseScoreMethod: 1 WeightedInteraction, 2 Vinardo, 3 Hybrid. Only top-k poses
// and scores are downloaded. Hybrid normalizes the complete selected pool with
// population variance, no epsilon, and zero z-scores for constant components.
// CUDA uses parallel double reductions; the CPU reference uses long double.
// Ties use original survivor ID. At lambda 0/1, sort by raw Vinardo/WeightedInteraction.
// Invalid/nonfinite input fails the entire call, including rows outside top-k.
// Scorer arrays must follow selection order. The call blocks until output is
// downloaded; borrowed input buffers must remain valid throughout the call.
bool cuda_rank_rescored_poses(const GpuPosePipelineDeviceResult& pool,
    const GpuRescoringSelection& selection, const GpuWeightedInteractionScoresDevice& weighted_interaction,
    const GpuVinardoScoresDevice& vinardo, int method, double alpha, double lambda,
    std::uint32_t top_k, std::vector<GpuRankedRescoredPose>& output,
    GpuRankingTimings* timings = nullptr);

#endif
