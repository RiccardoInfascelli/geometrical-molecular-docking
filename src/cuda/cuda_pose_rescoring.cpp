#include "cuda_pose_rescoring.hpp"

#include <chrono>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
Clock::time_point start_timer(bool enabled) { return enabled ? Clock::now() : Clock::time_point{}; }
double seconds(Clock::time_point start, bool enabled) {
    return enabled ? std::chrono::duration<double>(Clock::now() - start).count() : 0.0;
}
void require(bool ok, const char* operation) {
    if (!ok) throw std::runtime_error(std::string("CUDA rescoring failed: ") + operation);
}
struct LigandScoringBuffers {
    GpuWeightedInteractionAreasDevice interaction_areas{};
    GpuVinardoAtomsDevice atoms{};
    GpuRescoringSelection selection{};
    GpuWeightedInteractionScoresDevice weighted_interaction{};
    GpuVinardoScoresDevice vinardo{};
    ~LigandScoringBuffers() {
        cuda_free_weighted_interaction_areas(interaction_areas);
        cuda_free_vinardo_atoms(atoms);
        cuda_free_rescoring_selection(selection);
        cuda_free_weighted_interaction_scores(weighted_interaction);
        cuda_free_vinardo_scores(vinardo);
    }
};
}

CudaRescoringTarget::CudaRescoringTarget(const RescoringTarget& target, PoseRescoringMethods methods) {
    try {
        if (methods.weighted_interaction)
            require(cuda_upload_weighted_interaction_areas(make_gpu_weighted_interaction_areas(target.interaction_areas), weighted_interaction), "receptor WeightedInteraction upload");
        if (methods.vinardo)
            require(cuda_upload_vinardo_atoms(make_gpu_vinardo_receptor_atoms(target.vinardo), vinardo), "receptor Vinardo upload");
    } catch (...) {
        cuda_free_weighted_interaction_areas(weighted_interaction);
        cuda_free_vinardo_atoms(vinardo);
        throw;
    }
}
CudaRescoringTarget::~CudaRescoringTarget() {
    cuda_free_weighted_interaction_areas(weighted_interaction);
    cuda_free_vinardo_atoms(vinardo);
}

GpuDockingResult rescore_pose_pool_cuda(const GpuPosePipelineDeviceResult& pool,
    const RDKit::ROMol& molecule, const PreparedLigand& ligand,
    const CudaRescoringTarget& receptor, const GpuDockingSettings& settings, std::size_t top_k) {
    if (!top_k) throw std::invalid_argument("top-k must be positive");
    if (settings.fast_prefilter && *settings.fast_prefilter > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Fast prefilter exceeds CUDA index range");
    GpuDockingResult result;
    const auto method = settings.ranking.method;
    const bool weighted_interaction = method == PoseScoreMethod::WeightedInteraction || method == PoseScoreMethod::Hybrid;
    const bool vinardo = method == PoseScoreMethod::Vinardo || method == PoseScoreMethod::Hybrid;
    const bool timed = settings.collect_timings;
    LigandScoringBuffers buffers;
    auto start = start_timer(timed);
    require(cuda_select_rescoring_pool(pool.fast_scores,
        settings.fast_prefilter ? static_cast<std::uint32_t>(*settings.fast_prefilter) : 0,
        buffers.selection), "Fast pool selection or nonfinite Fast score");
    result.timings.prefilter_seconds = seconds(start, timed);
    result.candidate_pool = buffers.selection.count;
    if (!buffers.selection.count) return result;

    start = start_timer(timed);
    const auto interaction_areas = weighted_interaction ? make_gpu_weighted_interaction_areas(ligand.areas) : std::vector<GpuWeightedInteractionArea>{};
    const auto atoms = vinardo ? make_gpu_vinardo_ligand_atoms(molecule) : std::vector<GpuVinardoAtom>{};
    result.timings.chemical_preparation_seconds = seconds(start, timed);
    start = start_timer(timed);
    if (weighted_interaction) require(cuda_upload_weighted_interaction_areas(interaction_areas, buffers.interaction_areas), "ligand WeightedInteraction upload");
    if (vinardo) require(cuda_upload_vinardo_atoms(atoms, buffers.atoms), "ligand Vinardo upload");
    result.timings.rescoring_upload_seconds = seconds(start, timed);

    const GpuRigidTransformArrayDevice transforms{pool.survivors.transforms, pool.survivors.count};
    if (weighted_interaction) {
        start = start_timer(timed);
        require(cuda_score_weighted_interaction_device(receptor.weighted_interaction, buffers.interaction_areas, transforms,
            buffers.selection.pose_ids, buffers.selection.count, make_gpu_weighted_interaction_weights(WeightedInteractionWeights{}),
            buffers.weighted_interaction), "WeightedInteraction scoring");
        result.timings.weighted_interaction_seconds = seconds(start, timed);
    }
    if (vinardo) {
        start = start_timer(timed);
        require(cuda_score_vinardo_device(receptor.vinardo, buffers.atoms, transforms,
            buffers.selection.pose_ids, buffers.selection.count, buffers.vinardo), "Vinardo scoring");
        result.timings.vinardo_seconds = seconds(start, timed);
    }
    std::vector<GpuRankedRescoredPose> ranked;
    GpuRankingTimings ranking_times;
    const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(top_k, buffers.selection.count));
    require(cuda_rank_rescored_poses(pool, buffers.selection, buffers.weighted_interaction, buffers.vinardo,
        static_cast<int>(method), settings.ranking.vinardo_alpha, settings.ranking.hybrid_lambda,
        count, ranked, timed ? &ranking_times : nullptr), "ranking, normalization or nonfinite score");
    if (timed) {
        result.timings.hybrid_seconds = ranking_times.hybrid_seconds;
        result.timings.score_preparation_seconds = ranking_times.score_preparation_seconds;
        result.timings.selection_seconds = ranking_times.ranking_seconds;
        result.timings.rescoring_download_seconds = ranking_times.download_seconds;
    }
    for (const auto& row : ranked) {
        result.poses.push_back(row.pose);
        GpuDockingPoseScores scores;
        scores.pose_id = row.pose_id;
        if (weighted_interaction) scores.weighted_interaction = weighted_interaction_score_from_gpu(row.weighted_interaction);
        if (vinardo) {
            scores.vinardo = vinardo_score_from_gpu(row.vinardo);
            scores.vinardo_adjusted = row.adjusted;
        }
        if (method == PoseScoreMethod::Hybrid) scores.hybrid = row.hybrid;
        result.scores.push_back(scores);
    }
    return result;
}
