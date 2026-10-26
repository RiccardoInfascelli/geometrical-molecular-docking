#include "gpu_docking.hpp"

#include "gpu_pose_search.hpp"
#include "ligand_preparation.hpp"
#include "pose_rescoring.hpp"
#include "cuda_pose_rescoring.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {
using Clock = std::chrono::steady_clock;
Clock::time_point start_timer(bool enabled) {
    return enabled ? Clock::now() : Clock::time_point{};
}
double elapsed(Clock::time_point start, bool enabled) {
    return enabled ? std::chrono::duration<double>(Clock::now() - start).count() : 0.0;
}
GpuDockingSettings legacy_settings(double cutoff, bool timed) {
    GpuDockingSettings result;
    result.cutoff = cutoff;
    result.collect_timings = timed;
    return result;
}
void validate(const RunConfig& config, const GpuDockingSettings& settings) {
    if (settings.rescoring_backend != GpuRescoringBackend::Cpu &&
        settings.rescoring_backend != GpuRescoringBackend::Cuda)
        throw std::invalid_argument("Rescoring backend must be cpu or cuda");
    if (!std::isfinite(settings.cutoff) || settings.cutoff <= 0.0 ||
        !std::isfinite(config.triplet_bin_size) || config.triplet_bin_size <= 0.0 ||
        !std::isfinite(config.grid_voxel_size) || config.grid_voxel_size <= 0.0)
        throw std::invalid_argument("cutoff, triplet bin and grid voxel size must be finite and positive");
    if (!std::isfinite(config.grid_margin) || config.grid_margin < 0.0 ||
        !std::isfinite(config.pocket_centroid_distance) || config.pocket_centroid_distance < 0.0)
        throw std::invalid_argument("grid margin and pocket distance must be finite and non-negative");
    if (!std::isfinite(config.operational_clash_threshold) ||
        config.operational_clash_threshold < 0.0 || config.operational_clash_threshold > 1.0)
        throw std::invalid_argument("clash threshold must be in [0, 1]");
    if (settings.min_target_clearance && (!std::isfinite(*settings.min_target_clearance) ||
                                        *settings.min_target_clearance < 0.0))
        throw std::invalid_argument("minimum target clearance must be finite and non-negative");
    if (settings.fast_prefilter && *settings.fast_prefilter == 0)
        throw std::invalid_argument("Fast prefilter must be positive or disabled");
    if (settings.ranking.method == PoseScoreMethod::Fast && settings.fast_prefilter)
        throw std::invalid_argument("Fast prefilter is only supported before rescoring");
    if (settings.interaction_full_protein && settings.ranking.method != PoseScoreMethod::WeightedInteraction &&
        settings.ranking.method != PoseScoreMethod::Hybrid)
        throw std::invalid_argument("WeightedInteraction region is only applicable to WeightedInteraction or Hybrid");
    if (settings.ranking.method != PoseScoreMethod::Vinardo &&
        settings.ranking.method != PoseScoreMethod::Hybrid && settings.ranking.vinardo_alpha != 1.0)
        throw std::invalid_argument("Vinardo alpha is only applicable to Vinardo or Hybrid");
    if (settings.ranking.method != PoseScoreMethod::Hybrid && settings.ranking.hybrid_lambda != 0.3)
        throw std::invalid_argument("Hybrid lambda is only applicable to Hybrid");
    PoseRescoringResult empty;
    empty.weighted_interaction_calculated = empty.vinardo_calculated = true;
    (void)rank_rescored_poses(empty, settings.ranking, 1);
}
}

struct GpuDockingScreen::Impl {
    RunConfig config;
    GpuDockingSettings settings;
    std::unique_ptr<PreparedGpuTarget> target;
    std::optional<RescoringTarget> rescoring_target;
    std::unique_ptr<CudaRescoringTarget> device_rescoring_target;
    double startup = 0.0;

    Impl(const RunConfig& input, const GpuDockingSettings& requested)
        : config(input), settings(requested) {
        validate(config, settings);
        const auto start = start_timer(settings.collect_timings);
        target = std::make_unique<PreparedGpuTarget>(config, settings.cutoff,
            settings.min_target_clearance, settings.collect_timings);
        if (settings.ranking.method != PoseScoreMethod::Fast)
            rescoring_target.emplace(config, settings.interaction_full_protein);
        if (rescoring_target && settings.rescoring_backend == GpuRescoringBackend::Cuda) {
            const auto method = settings.ranking.method;
            device_rescoring_target = std::make_unique<CudaRescoringTarget>(*rescoring_target,
                PoseRescoringMethods{method != PoseScoreMethod::Vinardo, method != PoseScoreMethod::WeightedInteraction});
        }
        startup = elapsed(start, settings.collect_timings);
    }
};

GpuDockingScreen::GpuDockingScreen(const RunConfig& config, double cutoff, bool timed)
    : GpuDockingScreen(config, legacy_settings(cutoff, timed)) {}
GpuDockingScreen::GpuDockingScreen(const RunConfig& config, const GpuDockingSettings& settings)
    : impl_(std::make_unique<Impl>(config, settings)) {}
GpuDockingScreen::~GpuDockingScreen() = default;
std::size_t GpuDockingScreen::index_records() const { return impl_->target->index_records; }
double GpuDockingScreen::startup_seconds() const { return impl_->startup; }
std::size_t GpuDockingScreen::targets_before_pruning() const {
    return impl_->target->protein.targets_before_pruning;
}
std::size_t GpuDockingScreen::targets_after_pruning() const {
    return impl_->target->protein.targets.size();
}
std::size_t GpuDockingScreen::vinardo_skipped_receptor_atoms() const {
    return impl_->rescoring_target ? impl_->rescoring_target->vinardo.receptor_skipped_unsupported() : 0;
}

GpuDockingResult GpuDockingScreen::dock(RDKit::RWMol& molecule, std::size_t k) {
    if (k == 0) throw std::invalid_argument("top-k must be positive");
    GpuDockingResult output;
    const auto& settings = impl_->settings;
    const bool timed = settings.collect_timings;
    auto start = start_timer(timed);
    PreparedLigand ligand;
    prepare_ligand(molecule, ligand);
    output.diameter = ligand.diameter;
    output.ligand_triplets = ligand.triplets.size();
    output.timings.preparation_seconds = elapsed(start, timed);

    start = start_timer(timed);
    GpuPosePipelineDeviceResult device_pool;
    const bool device_rescoring = settings.ranking.method != PoseScoreMethod::Fast &&
                                 settings.rescoring_backend == GpuRescoringBackend::Cuda;
    auto search = run_gpu_ligand_search(molecule, ligand, *impl_->target, impl_->config,
        settings.diagnostic_stages ? &output.stages : nullptr, nullptr,
        device_rescoring ? &device_pool : nullptr, timed);
    output.timings.search_seconds = elapsed(start, timed);
    output.timings.query_preparation_seconds = timed ? search.query_preparation_seconds : 0.0;
    output.timings.gpu_search_seconds = timed
        ? output.timings.search_seconds - search.query_preparation_seconds : 0.0;
    output.timings.query_threads_requested = search.query_threads_requested;
    output.timings.query_threads_used = search.query_threads_used;
    output.candidate_matches = search.candidate_matches;
    output.alignments = search.alignments;
    output.valid_transforms = search.pipeline.valid_transform_count;
    output.survivors = search.pipeline.clash_survivor_count;

    if (settings.ranking.method == PoseScoreMethod::Fast) {
        start = start_timer(timed);
        output.poses = select_top_gpu_poses(search.pipeline.poses, k);
        output.candidate_pool = output.survivors;
        output.timings.selection_seconds = elapsed(start, timed);
        return output;
    }

    if (device_rescoring) {
        auto rescored = rescore_pose_pool_cuda(device_pool, molecule, ligand,
            *impl_->device_rescoring_target, settings, k);
        output.poses = std::move(rescored.poses);
        output.scores = std::move(rescored.scores);
        output.candidate_pool = rescored.candidate_pool;
        const auto prep_s = output.timings.preparation_seconds;
        const auto search_s = output.timings.search_seconds;
        const auto query_s = output.timings.query_preparation_seconds;
        const auto gpu_s = output.timings.gpu_search_seconds;
        const auto requested_threads = output.timings.query_threads_requested;
        const auto used_threads = output.timings.query_threads_used;
        output.timings = rescored.timings;
        output.timings.preparation_seconds = prep_s;
        output.timings.search_seconds = search_s;
        output.timings.query_preparation_seconds = query_s;
        output.timings.gpu_search_seconds = gpu_s;
        output.timings.query_threads_requested = requested_threads;
        output.timings.query_threads_used = used_threads;
        return output;
    }

    // The joint API receives every survivor before its optional Fast prefilter.
    const auto candidates = make_rescoring_pose_pool(search.pipeline.poses);
    const auto method = settings.ranking.method;
    const PoseRescoringMethods methods{
        method == PoseScoreMethod::WeightedInteraction || method == PoseScoreMethod::Hybrid,
        method == PoseScoreMethod::Vinardo || method == PoseScoreMethod::Hybrid};
    const auto& receptor = *impl_->rescoring_target;
    const auto rescored = rescore_poses(candidates, molecule, ligand,
        impl_->target->protein.grid, receptor.interaction_areas, receptor.vinardo,
        settings.fast_prefilter, true, methods);
    output.candidate_pool = rescored.poses.size();
    if (timed) {
        output.timings.prefilter_seconds = rescored.selection_seconds;
        output.timings.weighted_interaction_seconds = rescored.weighted_interaction_scoring_seconds;
        output.timings.vinardo_seconds = rescored.vinardo_scoring_seconds;
    }
    start = start_timer(timed);
    const auto ranked = rank_rescored_poses(rescored, settings.ranking, k,
        timed ? &output.timings.hybrid_seconds : nullptr);
    for (const auto& row : ranked) {
        const auto& scored = row.scored;
        output.poses.push_back(search.pipeline.poses.at(scored.pose_id));
        GpuDockingPoseScores scores;
        scores.pose_id = scored.pose_id;
        if (methods.weighted_interaction) scores.weighted_interaction = scored.pose.weighted_interaction_score;
        if (methods.vinardo) scores.vinardo = scored.vinardo;
        scores.vinardo_adjusted = row.vinardo_adjusted;
        scores.hybrid = row.hybrid;
        output.scores.push_back(scores);
    }
    output.timings.selection_seconds = elapsed(start, timed);
    return output;
}
