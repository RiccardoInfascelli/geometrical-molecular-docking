#include "gpu_docking.hpp"

#include "gpu_pose_search.hpp"
#include "gpu_ligand_batch.hpp"
#include "ligand_preparation.hpp"
#include "pose_rescoring.hpp"
#include "cuda_pose_rescoring.hpp"
#include "cuda_ligand_batch_rescoring.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
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
    return dock_impl(molecule, k, nullptr);
}

GpuDockingResult GpuDockingScreen::dock_prepared(GpuPreparedLigand& prepared, std::size_t k) {
    if (!prepared.molecule) throw GpuQueryPreparationError("Prepared GPU packet has no molecule");
    return dock_impl(*prepared.molecule, k, &prepared);
}

void GpuDockingScreen::release_batch_scratch() noexcept {
    impl_->target->compaction_workspace.clear();
}

std::vector<GpuDockingResult> GpuDockingScreen::dock_batch_prepared(
    const std::vector<GpuPreparedLigand*>& ligands, std::size_t k,
    std::size_t max_device_bytes, GpuDockingBatchTimings* timings,
    std::vector<std::exception_ptr>* record_errors) {
    if (timings) *timings = {};
    if (record_errors) record_errors->assign(ligands.size(), nullptr);
    if (k == 0) throw std::invalid_argument("top-k must be positive");
    const bool rescoring = impl_->settings.ranking.method != PoseScoreMethod::Fast;
    if (rescoring && impl_->settings.rescoring_backend != GpuRescoringBackend::Cuda)
        throw std::invalid_argument("GPU batch rescoring requires the CUDA backend");
    const char* method = std::getenv("DOCKING_KABSCH_METHOD");
    if (!method || std::strcmp(method, "quaternion") != 0)
        throw std::invalid_argument("GPU batch requires quaternion Kabsch");
    if (impl_->settings.diagnostic_stages)
        throw std::invalid_argument("GPU batch does not support per-record stage diagnostics");

    GpuLigandBatchSearchTimings batch;
    GpuPosePipelineDeviceResult device_pool;
    auto searches = run_gpu_ligand_batch_search(ligands, *impl_->target, impl_->config,
        max_device_bytes, impl_->settings.collect_timings || timings ? &batch : nullptr,
        rescoring ? &device_pool : nullptr);
    if (timings) {
        timings->search_seconds = impl_->settings.collect_timings ? batch.search_seconds : 0.0;
        timings->explicit_device_bytes = batch.explicit_device_bytes;
    }
    std::vector<GpuDockingResult> results;
    if (rescoring) {
        const auto& scratch = impl_->target->compaction_workspace;
        const auto scratch_bytes = scratch.capacity * 2 * sizeof(std::uint32_t) +
            (scratch.valid_count ? sizeof(std::uint32_t) : 0) + scratch.scan_capacity_bytes;
        const auto rescoring_start = start_timer(impl_->settings.collect_timings);
        results = rescore_ligand_batch_cuda(device_pool, ligands, searches,
            *impl_->device_rescoring_target, impl_->settings, k, max_device_bytes,
            scratch_bytes, timings, record_errors);
        if (timings) timings->rescoring_seconds = elapsed(rescoring_start, impl_->settings.collect_timings);
    } else results.resize(ligands.size());
    for (std::size_t i = 0; i < ligands.size(); ++i) {
        auto& result = results[i];
        const auto& prepared = *ligands[i];
        const auto& search = searches[i];
        result.diameter = prepared.ligand.diameter;
        result.ligand_triplets = prepared.ligand.triplets.size();
        result.candidate_matches = search.candidate_matches;
        result.alignments = search.alignments;
        result.valid_transforms = search.pipeline.valid_transform_count;
        result.survivors = search.pipeline.clash_survivor_count;
        if (!rescoring) result.candidate_pool = result.survivors;
        const bool timed = impl_->settings.collect_timings;
        result.timings.preparation_seconds = timed ? prepared.preparation_seconds : 0.0;
        result.timings.query_preparation_seconds = timed ? prepared.query_preparation_seconds : 0.0;
        result.timings.query_threads_requested = search.query_threads_requested;
        result.timings.query_threads_used = search.query_threads_used;
        if (rescoring) continue;
        const auto start = start_timer(timed);
        try {
            result.poses = select_top_gpu_poses(search.pipeline.poses, k);
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception&) {
            if (!record_errors) throw;
            (*record_errors)[i] = std::current_exception();
        }
        result.timings.selection_seconds = elapsed(start, timed);
    }
    return results;
}

GpuDockingResult GpuDockingScreen::dock_impl(
    RDKit::RWMol& molecule, std::size_t k, const GpuPreparedLigand* prepared) {
    if (k == 0) throw std::invalid_argument("top-k must be positive");
    GpuDockingResult output;
    const auto& settings = impl_->settings;
    const bool timed = settings.collect_timings;
    auto start = start_timer(timed);
    PreparedLigand local_ligand;
    if (!prepared) prepare_ligand(molecule, local_ligand);
    const auto& ligand = prepared ? prepared->ligand : local_ligand;
    output.diameter = ligand.diameter;
    output.ligand_triplets = ligand.triplets.size();
    output.timings.preparation_seconds = prepared && timed
        ? prepared->preparation_seconds : elapsed(start, timed);

    start = start_timer(timed);
    GpuPosePipelineDeviceResult device_pool;
    const bool device_rescoring = settings.ranking.method != PoseScoreMethod::Fast &&
                                 settings.rescoring_backend == GpuRescoringBackend::Cuda;
    const bool record_completed_stages = prepared && timed && device_rescoring;
    std::optional<GpuOverlapTraceRange> search_trace;
    if (record_completed_stages && prepared->trace_enabled)
        search_trace.emplace(true, "gpu_geometry_search");
    auto search = run_gpu_ligand_search(molecule, ligand, *impl_->target, impl_->config,
        settings.diagnostic_stages ? &output.stages : nullptr, nullptr,
        device_rescoring ? &device_pool : nullptr, timed, prepared);
    search_trace.reset();
    const auto geometry_end = record_completed_stages ? Clock::now() : Clock::time_point{};
    const auto geometry_start = record_completed_stages ? start : Clock::time_point{};
    const double completed_search_s = elapsed(start, timed);
    // Prepared queries run on a different host thread. Preserve search_seconds
    // as total search work (query + completed wrapper), not elapsed pipeline wall.
    output.timings.search_seconds = timed && prepared
        ? completed_search_s + search.query_preparation_seconds : completed_search_s;
    output.timings.query_preparation_seconds = timed ? search.query_preparation_seconds : 0.0;
    output.timings.gpu_search_seconds = timed
        ? (prepared ? completed_search_s : output.timings.search_seconds - search.query_preparation_seconds) : 0.0;
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
        const auto rescoring_start = record_completed_stages ? Clock::now() : Clock::time_point{};
        std::optional<GpuOverlapTraceRange> rescoring_trace;
        if (record_completed_stages && prepared->trace_enabled)
            rescoring_trace.emplace(true, "gpu_rescoring_and_ranking");
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
        if (record_completed_stages) {
            output.timings.geometry_start = geometry_start;
            output.timings.geometry_end = geometry_end;
            output.timings.rescoring_start = rescoring_start;
            output.timings.rescoring_end = Clock::now();
        }
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
