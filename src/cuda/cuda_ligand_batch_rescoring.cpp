#include "cuda_ligand_batch_rescoring.hpp"
#include "cuda_scoring_range.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start, bool timed) {
    return timed ? std::chrono::duration<double>(Clock::now() - start).count() : 0.0;
}
void require(bool success, const char* operation) {
    if (!success) throw GpuLigandSearchError(std::string("CUDA batch rescoring failed: ") + operation, 0, 0);
}
std::size_t size_bytes(std::size_t count, std::size_t width) {
    if (!count || !width) return 0;
    if (count > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("CUDA batch rescoring byte count overflow");
    return count * width;
}
std::size_t sum(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("CUDA batch rescoring byte count overflow");
    return a + b;
}
std::uint32_t index_count(std::size_t count) {
    if (count > static_cast<std::size_t>(INT_MAX))
        throw std::overflow_error("CUDA batch rescoring exceeds the index limit");
    return static_cast<std::uint32_t>(count);
}
struct Budget {
    std::size_t limit, live, peak;
    void check(std::size_t extra, const char* stage) {
        const auto needed = sum(live, extra);
        if (needed > limit)
            throw GpuBatchMemoryError(needed, limit, stage);
        peak = std::max(peak, needed);
    }
    void allocated(std::size_t bytes) { live = sum(live, bytes); }
};
struct Selection {
    GpuRescoringSelection value{};
    ~Selection() { if (value.pose_ids) cuda_free_rescoring_selection(value); }
};
struct ScoringBuffers {
    GpuWeightedInteractionAreasDevice areas{};
    GpuVinardoAtomsDevice atoms{};
    GpuWeightedInteractionScoresDevice interaction{};
    GpuVinardoScoresDevice vinardo{};
    std::uint32_t* ids{};
    GpuLigandScoringRange* interaction_ranges{};
    GpuLigandScoringRange* vinardo_ranges{};
    ~ScoringBuffers() {
        if (interaction.scores) cuda_free_weighted_interaction_scores(interaction);
        if (vinardo.scores) cuda_free_vinardo_scores(vinardo);
        if (areas.areas) cuda_free_weighted_interaction_areas(areas);
        if (atoms.atoms) cuda_free_vinardo_atoms(atoms);
        if (ids) cudaFree(ids);
        if (interaction_ranges) cudaFree(interaction_ranges);
        if (vinardo_ranges) cudaFree(vinardo_ranges);
    }
};

// The production ranking API accepts an owner. Reset aliases before its
// destructor runs; neither success nor an exception may free interior pointers.
struct BorrowedPool {
    GpuPosePipelineDeviceResult value;
    BorrowedPool(const GpuPosePipelineDeviceResult& source, std::uint32_t begin, std::uint32_t count) {
        if (!count) return;
        value.survivors = {source.survivors.transforms + begin, source.survivors.clash_scores + begin,
                           source.survivors.alignment_indices + begin, count};
        value.fast_scores = {source.fast_scores.scores + begin, count};
    }
    ~BorrowedPool() { value.survivors = {}; value.fast_scores = {}; }
    BorrowedPool(const BorrowedPool&) = delete;
    BorrowedPool& operator=(const BorrowedPool&) = delete;
};
struct LigandSlice {
    std::uint32_t survivor_begin{}, selected_begin{};
    std::size_t alignment_begin{};
    GpuLigandScoringRange interaction{}, vinardo{};
};

template<class T>
void upload(const std::vector<T>& host, T*& device, Budget& budget) {
    const auto bytes = size_bytes(host.size(), sizeof(T));
    if (!bytes) return;
    budget.check(bytes, "selection/range upload");
    require(cudaMalloc(reinterpret_cast<void**>(&device), bytes) == cudaSuccess, "selection/range allocation");
    budget.allocated(bytes);
    require(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess, "range upload");
}

template<class T>
std::vector<T> download(const T* device, std::size_t count, const char* operation) {
    std::vector<T> host(count);
    if (count)
        require(cudaMemcpy(host.data(), device, size_bytes(count, sizeof(T)), cudaMemcpyDeviceToHost)
                    == cudaSuccess, operation);
    return host;
}

std::vector<GpuDockingResult> score_batch(
    const GpuPosePipelineDeviceResult& pool,
    const std::vector<GpuPreparedLigand*>& ligands,
    const std::vector<GpuLigandSearchResult>& searches,
    const CudaRescoringTarget& receptor, const GpuDockingSettings& settings,
    std::size_t top_k, std::size_t max_device_bytes,
    std::size_t retained_scratch_bytes, GpuDockingBatchTimings* timings,
    std::vector<std::exception_ptr>* record_errors,
    std::vector<GpuLigandSearchResult>* downloaded_searches,
    std::vector<PoseRescoringResult>* raw_results) {
    const auto method = settings.ranking.method;
    if (!top_k || !max_device_bytes || method == PoseScoreMethod::Fast || settings.rescoring_backend != GpuRescoringBackend::Cuda)
        throw std::invalid_argument("Batch rescoring requires a CUDA rescoring method and positive top-k");
    if (ligands.size() != searches.size() ||
        (record_errors && record_errors->size() != ligands.size()) ||
        pool.survivors.count != pool.fast_scores.count ||
        (pool.survivors.count && (!pool.survivors.transforms || !pool.survivors.clash_scores ||
            !pool.survivors.alignment_indices || !pool.fast_scores.scores)))
        throw std::invalid_argument("Inconsistent CUDA batch rescoring inputs");
    if (settings.fast_prefilter && *settings.fast_prefilter > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Fast prefilter exceeds CUDA index range");
    index_count(pool.survivors.count);
    const bool interaction = method != PoseScoreMethod::Vinardo;
    const bool vinardo = method != PoseScoreMethod::WeightedInteraction;
    const bool timed = settings.collect_timings;
    const auto started = timed ? Clock::now() : Clock::time_point{};
    GpuDockingTimings aggregate;
    const auto resident = sum(retained_scratch_bytes, size_bytes(pool.survivors.count,
        sizeof(GpuRigidTransform) + sizeof(GpuClashScore) + sizeof(std::uint32_t) + sizeof(GpuFastScore)));
    Budget budget{max_device_bytes, resident, resident};
    budget.check(0, "resident rescoring buffers");
    std::vector<GpuDockingResult> results(ligands.size());
    std::vector<std::unique_ptr<Selection>> selections;
    std::vector<LigandSlice> slices(ligands.size());
    std::vector<GpuWeightedInteractionArea> areas;
    std::vector<GpuVinardoAtom> atoms;
    std::size_t survivor_count = 0, selected_count = 0, alignment_count = 0;

    for (std::size_t i = 0; i < ligands.size(); ++i) {
        if (!ligands[i] || !ligands[i]->molecule)
            throw std::invalid_argument("Missing prepared ligand in CUDA rescoring batch");
        auto& slice = slices[i];
        slice.survivor_begin = index_count(survivor_count);
        slice.selected_begin = index_count(selected_count);
        slice.alignment_begin = alignment_count;
        const auto count = index_count(searches[i].pipeline.clash_survivor_count);
        survivor_count = sum(survivor_count, count);
        alignment_count = sum(alignment_count, searches[i].alignments);
        index_count(alignment_count);
        if (survivor_count > pool.survivors.count)
            throw std::invalid_argument("CUDA batch survivor segments exceed the resident pool");
        selections.push_back(std::make_unique<Selection>());
        auto& selection = selections.back()->value;
        const auto selection_bytes = size_bytes(count, sizeof(std::uint32_t));
        budget.check(selection_bytes, "per-ligand Fast selection");
        const GpuFastScoreArrayDevice fast{count ? pool.fast_scores.scores + slice.survivor_begin : nullptr, count};
        auto start = timed ? Clock::now() : Clock::time_point{};
        require(cuda_select_rescoring_pool(fast, settings.fast_prefilter
            ? static_cast<std::uint32_t>(*settings.fast_prefilter) : 0, selection), "per-ligand Fast prefilter");
        budget.allocated(selection_bytes);
        results[i].timings.prefilter_seconds = elapsed(start, timed);
        aggregate.prefilter_seconds += results[i].timings.prefilter_seconds;
        results[i].candidate_pool = selection.count;
        if (!selection.count) continue;
        start = timed ? Clock::now() : Clock::time_point{};
        try {
            const auto local_areas = interaction ? make_gpu_weighted_interaction_areas(ligands[i]->ligand.areas)
                                                : std::vector<GpuWeightedInteractionArea>{};
            const auto local_atoms = vinardo ? make_gpu_vinardo_ligand_atoms(*ligands[i]->molecule)
                                            : std::vector<GpuVinardoAtom>{};
            slice.interaction = {index_count(areas.size()), index_count(local_areas.size()), slice.survivor_begin};
            slice.vinardo = {index_count(atoms.size()), index_count(local_atoms.size()), slice.survivor_begin};
            index_count(sum(areas.size(), local_areas.size()));
            index_count(sum(atoms.size(), local_atoms.size()));
            index_count(sum(selected_count, selection.count));
            areas.insert(areas.end(), local_areas.begin(), local_areas.end());
            atoms.insert(atoms.end(), local_atoms.begin(), local_atoms.end());
            selected_count += selection.count;
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::overflow_error&) {
            throw;
        } catch (const std::exception&) {
            if (!record_errors) throw;
            (*record_errors)[i] = std::current_exception();
            cuda_free_rescoring_selection(selection);
            budget.live -= selection_bytes;
        }
        results[i].timings.chemical_preparation_seconds = elapsed(start, timed);
        aggregate.chemical_preparation_seconds += results[i].timings.chemical_preparation_seconds;
    }
    if (survivor_count != pool.survivors.count)
        throw std::invalid_argument("CUDA batch survivor segments do not cover the resident pool");

    ScoringBuffers buffers;
    if (selected_count) {
        auto start = timed ? Clock::now() : Clock::time_point{};
        const auto id_bytes = size_bytes(selected_count, sizeof(std::uint32_t));
        budget.check(id_bytes, "packed selected IDs");
        require(cudaMalloc(reinterpret_cast<void**>(&buffers.ids), id_bytes) == cudaSuccess, "selected IDs allocation");
        budget.allocated(id_bytes);
        std::vector<GpuLigandScoringRange> interaction_ranges, vinardo_ranges;
        if (interaction) interaction_ranges.reserve(selected_count);
        if (vinardo) vinardo_ranges.reserve(selected_count);
        for (std::size_t i = 0; i < slices.size(); ++i) {
            const auto& selection = selections[i]->value;
            if (!selection.count) continue;
            require(cudaMemcpy(buffers.ids + slices[i].selected_begin, selection.pose_ids,
                size_bytes(selection.count, sizeof(std::uint32_t)), cudaMemcpyDeviceToDevice) == cudaSuccess,
                "selected IDs copy");
            if (interaction) interaction_ranges.insert(interaction_ranges.end(), selection.count, slices[i].interaction);
            if (vinardo) vinardo_ranges.insert(vinardo_ranges.end(), selection.count, slices[i].vinardo);
        }
        if (interaction) {
            const auto bytes = size_bytes(areas.size(), sizeof(GpuWeightedInteractionArea));
            budget.check(bytes, "ligand interaction-area upload");
            require(cuda_upload_weighted_interaction_areas(areas, buffers.areas), "packed ligand area upload");
            budget.allocated(bytes);
            upload(interaction_ranges, buffers.interaction_ranges, budget);
        }
        if (vinardo) {
            const auto bytes = size_bytes(atoms.size(), sizeof(GpuVinardoAtom));
            budget.check(bytes, "ligand Vinardo-atom upload");
            require(cuda_upload_vinardo_atoms(atoms, buffers.atoms), "packed ligand atom upload");
            budget.allocated(bytes);
            upload(vinardo_ranges, buffers.vinardo_ranges, budget);
        }
        aggregate.rescoring_upload_seconds = elapsed(start, timed);
        const GpuRigidTransformArrayDevice transforms{pool.survivors.transforms, pool.survivors.count};
        if (interaction) {
            const auto score_bytes = size_bytes(selected_count, sizeof(GpuWeightedInteractionScore));
            budget.check(sum(score_bytes, size_bytes(selected_count, receptor.weighted_interaction.count)),
                         "WeightedInteraction scoring and coverage");
            start = timed ? Clock::now() : Clock::time_point{};
            require(cuda_score_weighted_interaction_batch_device(receptor.weighted_interaction, buffers.areas,
                transforms, buffers.ids, index_count(selected_count), buffers.interaction_ranges,
                make_gpu_weighted_interaction_weights(WeightedInteractionWeights{}), buffers.interaction), "aggregated WeightedInteraction");
            budget.allocated(score_bytes);
            aggregate.weighted_interaction_seconds = elapsed(start, timed);
        }
        if (vinardo) {
            const auto score_bytes = size_bytes(selected_count, sizeof(GpuVinardoScore));
            budget.check(score_bytes, "Vinardo scoring");
            start = timed ? Clock::now() : Clock::time_point{};
            require(cuda_score_vinardo_batch_device(receptor.vinardo, buffers.atoms, transforms,
                buffers.ids, index_count(selected_count), buffers.vinardo_ranges, buffers.vinardo), "aggregated Vinardo");
            budget.allocated(score_bytes);
            aggregate.vinardo_seconds = elapsed(start, timed);
        }
    }

    const auto finish_timings = [&] {
        if (timings) {
            timings->rescoring_seconds = elapsed(started, timed);
            timings->rescoring = aggregate;
            timings->explicit_device_bytes = std::max(timings->explicit_device_bytes, budget.peak);
        }
    };
    if (raw_results) {
        const auto download_start = timed ? Clock::now() : Clock::time_point{};
        std::vector<GpuRigidTransform> transforms;
        std::vector<GpuClashScore> clashes;
        std::vector<std::uint32_t> alignment_ids;
        std::vector<GpuFastScore> fast;
        require(cuda_download_clash_survivors(pool.survivors, transforms, clashes, alignment_ids),
                "complete raw survivor download");
        require(cuda_download_fast_scores(pool.fast_scores, fast), "complete raw Fast download");
        if (transforms.size() != pool.survivors.count || clashes.size() != transforms.size() ||
            alignment_ids.size() != transforms.size() || fast.size() != transforms.size())
            throw GpuLigandSearchError("CUDA raw batch survivor/Fast array sizes differ", 0, 0);
        for (std::size_t i = 1; i < alignment_ids.size(); ++i)
            if (alignment_ids[i] <= alignment_ids[i - 1])
                throw GpuLigandSearchError("CUDA raw batch compaction changed alignment order", 0, 0);
        const auto ids = download(buffers.ids, selected_count, "raw selected IDs download");
        const auto interaction_scores = interaction
            ? download(buffers.interaction.scores, selected_count, "raw WeightedInteraction download")
            : std::vector<GpuWeightedInteractionScore>{};
        const auto vinardo_scores = vinardo
            ? download(buffers.vinardo.scores, selected_count, "raw Vinardo download")
            : std::vector<GpuVinardoScore>{};
        raw_results->resize(ligands.size());
        for (std::size_t i = 0; i < ligands.size(); ++i) {
            const auto& slice = slices[i];
            const auto count = searches[i].pipeline.clash_survivor_count;
            auto& poses = (*downloaded_searches)[i].pipeline.poses;
            poses.clear();
            poses.reserve(count);
            for (std::size_t local = 0; local < count; ++local) {
                const auto offset = slice.survivor_begin + local;
                const auto alignment = alignment_ids[offset];
                if (alignment < slice.alignment_begin || alignment - slice.alignment_begin >= searches[i].alignments)
                    throw GpuLigandSearchError("CUDA raw batch survivor has an invalid ligand identity", 0, 0);
                poses.push_back({static_cast<std::uint32_t>(alignment - slice.alignment_begin),
                                 transforms[offset], clashes[offset], fast[offset]});
            }
            const auto candidates = make_rescoring_pose_pool(poses);
            auto& result = (*raw_results)[i];
            result.input_pose_count = count;
            result.weighted_interaction_calculated = interaction;
            result.vinardo_calculated = vinardo;
            result.selection_seconds = results[i].timings.prefilter_seconds;
            const auto& selection = selections[i]->value;
            result.poses.reserve(selection.count);
            for (std::size_t selected = 0; selected < selection.count; ++selected) {
                const auto offset = slice.selected_begin + selected;
                const auto pose_id = ids[offset];
                if (pose_id >= count || (selected && pose_id <= ids[offset - 1]))
                    throw GpuLigandSearchError("CUDA raw batch selection changed local survivor order", 0, 0);
                RescoredPose row{pose_id, candidates[pose_id], {}};
                if (interaction) {
                    if (!std::isfinite(interaction_scores[offset].weighted_score))
                        throw GpuLigandSearchError("Nonfinite CUDA raw WeightedInteraction score", 0, 0);
                    row.pose.weighted_interaction_score = weighted_interaction_score_from_gpu(interaction_scores[offset]);
                }
                if (vinardo) row.vinardo = vinardo_score_from_gpu(vinardo_scores[offset]);
                result.poses.push_back(row);
            }
        }
        aggregate.rescoring_download_seconds = elapsed(download_start, timed);
        finish_timings();
        return results;
    }

    for (std::size_t i = 0; i < ligands.size(); ++i) {
        const auto& selection = selections[i]->value;
        if (!selection.count) continue;
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(top_k, selection.count));
        // Known vectors in the existing ranker coexist. Algorithm-internal
        // Thrust sort/reduction storage remains outside the explicit budget.
        budget.check(sum(size_bytes(selection.count, (method == PoseScoreMethod::Hybrid ? 3 : 2) * sizeof(double) +
            sizeof(std::uint32_t)), size_bytes(count, sizeof(GpuRankedRescoredPose))), "per-ligand ranking workspace");
        BorrowedPool local(pool, slices[i].survivor_begin,
            index_count(searches[i].pipeline.clash_survivor_count));
        const GpuWeightedInteractionScoresDevice interaction_scores{
            interaction ? buffers.interaction.scores + slices[i].selected_begin : nullptr,
            interaction ? selection.count : 0};
        const GpuVinardoScoresDevice vinardo_scores{
            vinardo ? buffers.vinardo.scores + slices[i].selected_begin : nullptr,
            vinardo ? selection.count : 0};
        std::vector<GpuRankedRescoredPose> ranked;
        GpuRankingTimings ranking_times;
        // The bool API also reports CUDA/Thrust failures. Do not classify an
        // undifferentiated ranking failure as a recoverable molecular error.
        require(cuda_rank_rescored_poses(local.value, selection, interaction_scores, vinardo_scores,
            static_cast<int>(method), settings.ranking.vinardo_alpha, settings.ranking.hybrid_lambda,
            count, ranked, timed ? &ranking_times : nullptr), "per-ligand ranking or normalization");
        auto& result = results[i];
        result.timings.score_preparation_seconds = ranking_times.score_preparation_seconds;
        result.timings.hybrid_seconds = ranking_times.hybrid_seconds;
        result.timings.selection_seconds = ranking_times.ranking_seconds;
        result.timings.rescoring_download_seconds = ranking_times.download_seconds;
        aggregate.score_preparation_seconds += ranking_times.score_preparation_seconds;
        aggregate.hybrid_seconds += ranking_times.hybrid_seconds;
        aggregate.selection_seconds += ranking_times.ranking_seconds;
        aggregate.rescoring_download_seconds += ranking_times.download_seconds;
        for (auto& row : ranked) {
            if (row.pose.alignment_index < slices[i].alignment_begin ||
                row.pose.alignment_index - slices[i].alignment_begin >= searches[i].alignments ||
                row.pose_id >= searches[i].pipeline.clash_survivor_count)
                throw GpuLigandSearchError("CUDA batch ranked pose has an invalid ligand identity", 0, 0);
            row.pose.alignment_index -= static_cast<std::uint32_t>(slices[i].alignment_begin);
            result.poses.push_back(row.pose);
            GpuDockingPoseScores scores;
            scores.pose_id = row.pose_id;
            if (interaction) scores.weighted_interaction = weighted_interaction_score_from_gpu(row.weighted_interaction);
            if (vinardo) {
                scores.vinardo = vinardo_score_from_gpu(row.vinardo);
                scores.vinardo_adjusted = row.adjusted;
            }
            if (method == PoseScoreMethod::Hybrid) scores.hybrid = row.hybrid;
            result.scores.push_back(scores);
        }
    }
    finish_timings();
    return results;
}
} // namespace

std::vector<GpuDockingResult> rescore_ligand_batch_cuda(
    const GpuPosePipelineDeviceResult& pool,
    const std::vector<GpuPreparedLigand*>& ligands,
    const std::vector<GpuLigandSearchResult>& searches,
    const CudaRescoringTarget& receptor, const GpuDockingSettings& settings,
    std::size_t top_k, std::size_t max_device_bytes,
    std::size_t retained_scratch_bytes, GpuDockingBatchTimings* timings,
    std::vector<std::exception_ptr>* record_errors) {
    return score_batch(pool, ligands, searches, receptor, settings, top_k,
        max_device_bytes, retained_scratch_bytes, timings, record_errors, nullptr, nullptr);
}

std::vector<PoseRescoringResult> score_ligand_batch_cuda(
    const GpuPosePipelineDeviceResult& pool,
    const std::vector<GpuPreparedLigand*>& ligands,
    std::vector<GpuLigandSearchResult>& searches,
    const CudaRescoringTarget& receptor, const GpuDockingSettings& settings,
    std::size_t max_device_bytes, std::size_t retained_scratch_bytes,
    GpuDockingBatchTimings* timings,
    std::vector<std::exception_ptr>* record_errors) {
    std::vector<PoseRescoringResult> raw;
    score_batch(pool, ligands, searches, receptor, settings, 1, max_device_bytes,
        retained_scratch_bytes, timings, record_errors, &searches, &raw);
    return raw;
}
