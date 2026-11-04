#include "conformer_docking.hpp"
#include "gpu_docking_output.hpp"
#include "gpu_ligand_prefetch.hpp"
#include "gpu_batch_retry.hpp"
#include "cuda_ligand_batch_rescoring.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
}

ConformerDockingResult dock_conformer(
    std::size_t conformer_id, RDKit::RWMol& ligand, const RDKit::ROMol& reference,
    const PreparedGpuTarget& target, const RescoringTarget& scoring,
    const RunConfig& config, std::optional<std::size_t> fast_prefilter) {
    validate_gpu_docking_reference(ligand, reference);
    ConformerDockingResult result;
    auto start = Clock::now();
    PreparedLigand prepared;
    prepare_ligand(ligand, prepared);
    result.preparation_s = seconds(start);
    result.diameter = prepared.diameter;
    result.triplets = prepared.triplets.size();

    start = Clock::now();
    const auto search = run_gpu_ligand_search(ligand, prepared, target, config,
                                            nullptr, nullptr, nullptr, true);
    result.search_s = seconds(start); // Completed wrapper, including queries and copies.
    result.query_s = search.query_preparation_seconds;
    result.matches = search.candidate_matches;
    result.alignments = search.alignments;
    result.valid_transforms = search.pipeline.valid_transform_count;
    result.survivors = search.pipeline.clash_survivor_count;
    if (result.survivors != search.pipeline.poses.size())
        throw std::runtime_error("CUDA survivor count differs from the downloaded pool");

    const auto candidates = make_rescoring_pose_pool(search.pipeline.poses);
    start = Clock::now();
    const auto rescored = rescore_poses(candidates, ligand, prepared, target.protein.grid,
        scoring.interaction_areas, scoring.vinardo, fast_prefilter, true);
    result.rescoring_s = seconds(start);
    result.prefilter_s = rescored.selection_seconds;
    result.weighted_interaction_s = rescored.weighted_interaction_scoring_seconds;
    result.vinardo_s = rescored.vinardo_scoring_seconds;
    result.poses.reserve(candidates.size());
    start = Clock::now();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double rmsd = gpu_pose_heavy_atom_rmsd(ligand, search.pipeline.poses[i].transform, reference);
        result.poses.push_back({conformer_id, i, {i, candidates[i], {}}, rmsd,
                               search.pipeline.poses[i].alignment_index});
        if (!result.oracle_all_survivors || rmsd < *result.oracle_all_survivors)
            result.oracle_all_survivors = rmsd;
    }
    for (const auto& row : rescored.poses) {
        result.poses.at(row.pose_id).scored = row;
        result.rescored_ids.push_back(row.pose_id);
        const double rmsd = result.poses.at(row.pose_id).rmsd;
        if (!result.oracle_after_prefilter || rmsd < *result.oracle_after_prefilter)
            result.oracle_after_prefilter = rmsd;
    }
    result.rmsd_s = seconds(start);
    return result;
}

std::vector<ConformerDockingResult> dock_conformers(
    const std::vector<ConformerDockingInput>& inputs, const RDKit::ROMol& reference,
    const PreparedGpuTarget& target, const RescoringTarget& scoring,
    const RunConfig& config, std::optional<std::size_t> fast_prefilter,
    const ConformerExecutionOptions& options, ConformerExecutionTimings* timings,
    const CudaRescoringTarget* device_scoring) {
    if (timings) *timings = {};
    const auto started = Clock::now();
    if (options.batch_size == 0 || options.batch_size > 64 || options.batch_memory_bytes == 0)
        throw std::invalid_argument("Invalid conformer batch size or memory budget");
    if (options.gpu_batch && options.rescoring_backend != GpuRescoringBackend::Cuda)
        throw std::invalid_argument("Conformer batching requires CUDA raw rescoring");
    const auto* method = std::getenv("DOCKING_KABSCH_METHOD");
    const auto* query = std::getenv("DOCKING_QUERY_METHOD");
    if (!method || std::strcmp(method, "quaternion") || !query || options.query_method != query)
        throw std::invalid_argument("Freeze quaternion/query selectors before conformer preparation");
    std::set<std::size_t> identities;
    for (const auto& input : inputs) {
        if (!input.molecule || !identities.insert(input.conformer_id).second)
            throw std::invalid_argument("Conformer inputs require unique IDs and owned source molecules");
        validate_gpu_docking_reference(*input.molecule, reference);
    }
    std::vector<ConformerDockingResult> results(inputs.size());
    if (inputs.empty()) return results;
    std::unique_ptr<CudaRescoringTarget> local_device_scoring;
    if (options.rescoring_backend == GpuRescoringBackend::Cuda && !device_scoring) {
        local_device_scoring = std::make_unique<CudaRescoringTarget>(scoring, PoseRescoringMethods{});
        device_scoring = local_device_scoring.get();
    }
    std::size_t next = 0;
    GpuLigandSource source = [&]() -> std::unique_ptr<GpuLigandInput> {
        if (next == inputs.size()) return nullptr;
        auto input = std::make_unique<GpuLigandInput>();
        input->index = next + 1;
        input->name = "conformer-" + std::to_string(inputs[next].conformer_id);
        input->molecule = std::make_unique<RDKit::RWMol>(*inputs[next].molecule);
        ++next;
        return input;
    };
    const auto group_size = options.gpu_batch ? options.batch_size : std::size_t{1};
    GpuLigandPrefetch prefetch(std::move(source), config, inputs.size(), target.index_records != 0,
        options.cpu_gpu_overlap.value_or(false), true, false,
        options.gpu_batch ? 2 * group_size : 1,
        options.gpu_batch ? std::size_t{256} * 1024 * 1024 : 0);
    std::size_t consumed = 0;
    double consumer_wait = 0.0;
    while (consumed < inputs.size()) {
        std::vector<std::unique_ptr<GpuPreparedRecord>> packets;
        while (packets.size() < group_size && consumed + packets.size() < inputs.size()) {
            double wait = 0.0;
            auto packet = prefetch.next(&wait);
            consumer_wait += wait;
            if (!packet) throw std::runtime_error("Conformer producer ended before all inputs were prepared");
            if (packet->index != consumed + packets.size() + 1)
                throw std::runtime_error("Conformer producer changed source order");
            if (packet->error) {
                try { std::rethrow_exception(packet->error); }
                catch (const std::exception& error) {
                    throw std::runtime_error("Conformer " +
                        std::to_string(inputs[packet->index - 1].conformer_id) + ": " + error.what());
                }
            }
            packets.push_back(std::move(packet));
        }
        const auto work = [&](std::size_t first, std::size_t last, bool single_retry) {
            std::vector<GpuPreparedLigand*> prepared;
            for (auto i = first; i < last; ++i) prepared.push_back(&packets[i]->prepared);
            GpuPosePipelineDeviceResult device_pool;
            std::vector<GpuLigandSearchResult> searches;
            GpuLigandBatchSearchTimings search_times;
            auto start = Clock::now();
            bool scoring_started = false;
            try {
                if (options.gpu_batch) {
                    searches = run_gpu_ligand_batch_search(prepared, target, config,
                        options.batch_memory_bytes, &search_times, &device_pool);
                } else {
                    auto& input = *prepared.front();
                    searches.push_back(run_gpu_ligand_search(*input.molecule, input.ligand, target,
                        config, nullptr, nullptr,
                        options.rescoring_backend == GpuRescoringBackend::Cuda ? &device_pool : nullptr,
                        true, &input));
                }
                const double search_s = seconds(start);
                if (timings) timings->search_s += search_s;
                start = Clock::now();
                scoring_started = true;
                std::vector<PoseRescoringResult> scores;
                GpuDockingBatchTimings score_times;
                if (options.rescoring_backend == GpuRescoringBackend::Cuda) {
                    GpuDockingSettings settings;
                    settings.collect_timings = true;
                    settings.rescoring_backend = GpuRescoringBackend::Cuda;
                    settings.ranking.method = PoseScoreMethod::Hybrid; // Request both raw terms, never rank here.
                    settings.fast_prefilter = fast_prefilter;
                    const auto& scratch = target.compaction_workspace;
                    const auto scratch_bytes = scratch.capacity * 2 * sizeof(std::uint32_t) +
                        (scratch.valid_count ? sizeof(std::uint32_t) : 0) + scratch.scan_capacity_bytes;
                    scores = score_ligand_batch_cuda(device_pool, prepared, searches, *device_scoring,
                        settings, options.gpu_batch ? options.batch_memory_bytes :
                            std::numeric_limits<std::size_t>::max(), scratch_bytes, &score_times);
                } else {
                    const auto& input = *prepared.front();
                    scores.push_back(rescore_poses(make_rescoring_pose_pool(searches.front().pipeline.poses),
                        *input.molecule, input.ligand, target.protein.grid, scoring.interaction_areas,
                        scoring.vinardo, fast_prefilter, true));
                }
                const double rescoring_s = seconds(start);
                if (timings) timings->rescoring_s += rescoring_s;
                if (scores.size() != prepared.size() || searches.size() != prepared.size())
                    throw std::runtime_error("Conformer batch result count differs from its input");
                for (std::size_t i = 0; i < prepared.size(); ++i) {
                    auto& result = results[consumed + first + i];
                    const auto& search = searches[i];
                    const auto& input = *prepared[i];
                    result.diameter = input.ligand.diameter;
                    result.triplets = input.ligand.triplets.size();
                    result.matches = search.candidate_matches;
                    result.alignments = search.alignments;
                    result.valid_transforms = search.pipeline.valid_transform_count;
                    result.survivors = search.pipeline.clash_survivor_count;
                    result.preparation_s = input.preparation_seconds;
                    result.query_s = input.query_preparation_seconds;
                    result.shared_search = prepared.size() > 1;
                    result.shared_rescoring = prepared.size() > 1;
                    if (!result.shared_search) result.search_s = search_s + result.query_s;
                    if (!result.shared_rescoring) {
                        result.rescoring_s = rescoring_s;
                        result.weighted_interaction_s = options.rescoring_backend == GpuRescoringBackend::Cuda
                            ? score_times.rescoring.weighted_interaction_seconds : scores[i].weighted_interaction_scoring_seconds;
                        result.vinardo_s = options.rescoring_backend == GpuRescoringBackend::Cuda
                            ? score_times.rescoring.vinardo_seconds : scores[i].vinardo_scoring_seconds;
                    }
                    result.prefilter_s = scores[i].selection_seconds;
                    if (search.pipeline.poses.size() != result.survivors)
                        throw std::runtime_error("Conformer execution omitted clash survivors");
                    const auto candidates = make_rescoring_pose_pool(search.pipeline.poses);
                    const auto rmsd_start = Clock::now();
                    for (std::size_t pose = 0; pose < candidates.size(); ++pose) {
                        const auto rmsd = gpu_pose_heavy_atom_rmsd(*input.molecule,
                            search.pipeline.poses[pose].transform, reference);
                        result.poses.push_back({inputs[consumed + first + i].conformer_id, pose,
                            {pose, candidates[pose], {}}, rmsd, search.pipeline.poses[pose].alignment_index});
                        if (!result.oracle_all_survivors || rmsd < *result.oracle_all_survivors)
                            result.oracle_all_survivors = rmsd;
                    }
                    for (const auto& row : scores[i].poses) {
                        result.poses.at(row.pose_id).scored = row;
                        result.rescored_ids.push_back(row.pose_id);
                        const auto rmsd = result.poses.at(row.pose_id).rmsd;
                        if (!result.oracle_after_prefilter || rmsd < *result.oracle_after_prefilter)
                            result.oracle_after_prefilter = rmsd;
                    }
                    result.rmsd_s = seconds(rmsd_start);
                    if (timings) {
                        timings->preparation_s += result.preparation_s;
                        timings->query_s += result.query_s;
                    }
                }
            } catch (const GpuBatchMemoryError& error) {
                if (timings) {
                    if (scoring_started) timings->rescoring_s += seconds(start);
                    else timings->search_s += seconds(start);
                    ++timings->memory_retries;
                    if (last - first > 1) ++timings->adaptive_splits;
                }
                if (single_retry)
                    throw std::runtime_error("Conformer " + std::to_string(inputs[consumed + first].conformer_id) +
                        " exceeds the explicit memory budget: " + error.what());
                throw;
            }
        };
        if (options.gpu_batch)
            run_gpu_batch_adaptive(0, packets.size(), work, [&] { target.compaction_workspace.clear(); });
        else work(0, packets.size(), false);
        consumed += packets.size();
    }
    prefetch.stop();
    if (timings) {
        const auto stats = prefetch.statistics();
        timings->consumer_wait_s = consumer_wait;
        timings->producer_wait_s = prefetch.producer_queue_wait_seconds();
        timings->queue_capacity = stats.capacity;
        timings->queue_high_water_records = stats.high_water_records;
        timings->queue_high_water_bytes = stats.high_water_bytes;
        timings->wall_s = seconds(started);
    }
    return results;
}
