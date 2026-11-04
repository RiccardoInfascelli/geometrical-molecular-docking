#include "gpu_ligand_batch.hpp"

#include "cuda_geometry.hpp"
#include "cuda_ligand_batch.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;

std::size_t bytes(std::size_t count, std::size_t width) {
    if (count > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("Batch byte count overflow");
    return count * width;
}

std::size_t checked_sum(std::initializer_list<std::size_t> values) {
    std::size_t result = 0;
    for (const auto value : values) {
        if (value > std::numeric_limits<std::size_t>::max() - result)
            throw std::overflow_error("Batch byte count overflow");
        result += value;
    }
    return result;
}

std::uint32_t index_count(std::size_t count, const char* description) {
    if (count > static_cast<std::size_t>(INT_MAX))
        throw GpuQueryPreparationError(std::string(description) + " exceeds the CUDA batch index limit");
    return static_cast<std::uint32_t>(count);
}

struct BatchBuffers {
    GpuTripletNeighborCountArrayDevice neighbor_counts{};
    GpuTripletNeighborOffsetArrayDevice neighbor_offsets{};
    GpuTripletCandidateArrayDevice candidates{};
    GpuAlignmentCountArrayDevice alignment_counts{};
    GpuAlignmentOffsetArrayDevice alignment_offsets{};
    GpuAlignmentCandidateArrayDevice alignments{};
    GpuPointArrayDevice matching_points{}, atoms{};
    GpuFastMatchingAreaArrayDevice areas{};
    GpuLigandBatchSegmentsDevice segments{};
    GpuRigidTransformArrayDevice transforms{};
    GpuClashScoreArrayDevice clashes{};
    GpuClashSurvivorArrayDevice survivors{};
    GpuFastScoreArrayDevice fast_scores{};

    ~BatchBuffers() { clear(); }
    void clear() noexcept {
        if (fast_scores.scores) cuda_free_fast_scores(fast_scores);
        if (survivors.transforms || survivors.clash_scores || survivors.alignment_indices)
            cuda_free_clash_survivors(survivors);
        if (clashes.scores) cuda_free_clash_scores(clashes);
        if (transforms.transforms) cuda_free_rigid_transforms(transforms);
        if (segments.segments) cuda_free_ligand_batch_segments(segments);
        if (areas.areas) cuda_free_fast_matching_areas(areas);
        if (atoms.points) cuda_free_points(atoms);
        if (matching_points.points) cuda_free_points(matching_points);
        if (alignments.alignments) cuda_free_alignment_candidates(alignments);
        if (alignment_offsets.offsets) cuda_free_alignment_offsets(alignment_offsets);
        if (alignment_counts.counts) cuda_free_alignment_counts(alignment_counts);
        if (candidates.candidates) cuda_free_triplet_candidates(candidates);
        if (neighbor_offsets.offsets) cuda_free_triplet_neighbor_offsets(neighbor_offsets);
        if (neighbor_counts.counts) cuda_free_triplet_neighbor_counts(neighbor_counts);
    }

    std::size_t live_bytes() const {
        return checked_sum({
            bytes(neighbor_counts.count, sizeof(std::uint32_t)),
            bytes(neighbor_offsets.count, sizeof(std::uint32_t)),
            bytes(candidates.count, sizeof(GpuTripletCandidate)),
            bytes(alignment_counts.count, sizeof(std::uint32_t)),
            bytes(alignment_offsets.count, sizeof(std::uint32_t)),
            bytes(alignments.count, sizeof(GpuAlignmentCandidate)),
            bytes(matching_points.count, sizeof(GpuPoint3)),
            bytes(atoms.count, sizeof(GpuPoint3)),
            bytes(areas.count, sizeof(GpuFastMatchingArea)),
            bytes(segments.count, sizeof(GpuLigandBatchSegment)),
            bytes(transforms.count, sizeof(GpuRigidTransform)),
            bytes(clashes.count, sizeof(GpuClashScore)),
            bytes(survivors.count, sizeof(GpuRigidTransform) + sizeof(GpuClashScore) + sizeof(std::uint32_t)),
            bytes(fast_scores.count, sizeof(GpuFastScore))});
    }
};
}

std::vector<GpuLigandSearchResult> run_gpu_ligand_batch_search(
    const std::vector<GpuPreparedLigand*>& ligands,
    const PreparedGpuTarget& target, const RunConfig& config,
    std::size_t max_device_bytes, GpuLigandBatchSearchTimings* timings,
    GpuPosePipelineDeviceResult* device_result) {
    if (timings) *timings = {};
    if (device_result) device_result->clear();
    const auto start = timings ? Clock::now() : Clock::time_point{};
    if (max_device_bytes == 0)
        throw std::invalid_argument("CUDA batch memory budget must be positive");
    const auto* method = std::getenv("DOCKING_KABSCH_METHOD");
    if (!method || std::strcmp(method, "quaternion") != 0)
        throw GpuQueryPreparationError("CUDA ligand batching requires quaternion Kabsch");
    if (!std::isfinite(config.operational_clash_threshold) ||
        config.operational_clash_threshold < 0.0 || config.operational_clash_threshold > 1.0)
        throw std::invalid_argument("CUDA batch clash threshold must be in [0, 1]");
    index_count(ligands.size(), "Ligand count");
    const int requested_threads = gpu_query_requested_threads();
    std::vector<GpuLigandSearchResult> results(ligands.size());
    for (std::size_t i = 0; i < ligands.size(); ++i) {
        const auto* prepared = ligands[i];
        if (!prepared || !prepared->molecule || prepared->bin_size != config.triplet_bin_size ||
            prepared->query_info.requested_threads != requested_threads)
            throw GpuQueryPreparationError("CUDA batch contains an incompatible prepared ligand");
        results[i].query_threads_requested = requested_threads;
        results[i].query_threads_used = prepared->query_info.used_threads;
        results[i].query_preparation_seconds = prepared->query_preparation_seconds;
        if (prepared->orders.size() != prepared->queries.size() ||
            (target.index_records &&
             (prepared->queries.size() != prepared->ligand.triplets.size() ||
              (!prepared->ligand.triplets.empty() && !prepared->queries_enabled))))
            throw GpuQueryPreparationError("CUDA batch query/order count does not match the prepared ligand");
    }

    BatchBuffers device;
    const auto scratch_bytes = [&] {
        const auto& scratch = target.compaction_workspace;
        return checked_sum({bytes(scratch.capacity, 2 * sizeof(std::uint32_t)),
                          scratch.valid_count ? sizeof(std::uint32_t) : 0,
                          scratch.scan_capacity_bytes});
    };
    std::size_t peak = 0;
    const auto budget = [&](std::size_t extra, const char* stage) {
        const auto total = checked_sum({device.live_bytes(), scratch_bytes(), extra});
        if (total > max_device_bytes)
            throw GpuBatchMemoryError(total, max_device_bytes, stage);
        peak = std::max(peak, total);
    };
    const auto require = [](bool success, const char* stage) {
        if (!success) throw GpuLigandSearchError(std::string("CUDA batch failed: ") + stage, 0, 0);
    };
    const auto finish = [&] {
        if (device_result) {
            device_result->survivors = device.survivors;
            device_result->fast_scores = device.fast_scores;
            device.survivors = {};
            device.fast_scores = {};
        }
        device.clear();
        if (timings) {
            timings->search_seconds = std::chrono::duration<double>(Clock::now() - start).count();
            timings->explicit_device_bytes = peak;
        }
    };
    budget(0, "retained compaction workspace");
    if (ligands.empty() || target.index_records == 0) {
        finish();
        return results;
    }

    std::vector<GpuTripletKey> queries;
    std::vector<GpuLigandTripletOrders> orders;
    std::vector<GpuPoint3> points, atoms;
    std::vector<GpuFastMatchingArea> areas;
    std::vector<std::uint32_t> query_prefix{0};
    std::vector<GpuLigandBatchSegment> segments(ligands.size());
    for (std::size_t i = 0; i < ligands.size(); ++i) {
        const auto& prepared = *ligands[i];
        const auto local_points = make_gpu_ligand_matching_points(prepared.ligand.matching_areas);
        const auto point_begin = index_count(points.size(), "Matching point offset");
        index_count(checked_sum({points.size(), local_points.size()}), "Matching point count");
        index_count(checked_sum({queries.size(), prepared.queries.size()}), "Query count");
        queries.insert(queries.end(), prepared.queries.begin(), prepared.queries.end());
        for (auto order : prepared.orders) {
            if (order.equivalent_order_count > 6)
                throw GpuQueryPreparationError("CUDA batch equivalent-order count exceeds six");
            for (std::uint32_t equivalent = 0; equivalent < order.equivalent_order_count; ++equivalent)
                for (int vertex = 0; vertex < 3; ++vertex) {
                    auto& index = order.equivalent_orders[equivalent][vertex];
                    if (index >= local_points.size())
                        throw GpuQueryPreparationError("CUDA batch matching-point index is out of range");
                    index += point_begin;
                }
            orders.push_back(order);
        }
        points.insert(points.end(), local_points.begin(), local_points.end());
        query_prefix.push_back(index_count(queries.size(), "Query count"));
        auto& segment = segments[i];
        segment.atom_begin = index_count(atoms.size(), "Atom offset");
        segment.atom_count = index_count(prepared.molecule->getNumAtoms(), "Ligand atom count");
        index_count(checked_sum({atoms.size(), segment.atom_count}), "Atom count");
        const auto local_atoms = make_gpu_ligand_atom_points(*prepared.molecule);
        atoms.insert(atoms.end(), local_atoms.begin(), local_atoms.end());
        segment.area_begin = index_count(areas.size(), "Fast area offset");
        segment.area_count = index_count(prepared.ligand.matching_areas.size(), "Ligand Fast area count");
        index_count(checked_sum({areas.size(), segment.area_count}), "Fast area count");
        const auto local_areas = make_gpu_fast_matching_areas(prepared.ligand.matching_areas, FastScoreWeights{});
        areas.insert(areas.end(), local_areas.begin(), local_areas.end());
    }
    if (queries.empty()) {
        finish();
        return results;
    }

    budget(bytes(queries.size(), sizeof(std::uint32_t)), "neighbor counts");
    require(cuda_count_triplet_neighbor_records_device(target.device_index, queries, device.neighbor_counts), "neighbor counts");
    budget(bytes(queries.size() + 1, sizeof(std::uint32_t)), "neighbor offsets");
    require(cuda_exclusive_triplet_neighbor_prefix_sum_device(device.neighbor_counts, device.neighbor_offsets), "neighbor scan");
    std::vector<std::uint32_t> candidate_prefix(query_prefix.size());
    for (std::size_t i = 0; i < query_prefix.size(); ++i)
        require(cudaMemcpy(&candidate_prefix[i], device.neighbor_offsets.offsets + query_prefix[i],
                           sizeof(std::uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess, "candidate boundary download");
    const auto candidate_count = index_count(candidate_prefix.back(), "Candidate count");
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (candidate_prefix[i + 1] < candidate_prefix[i])
            throw GpuLigandSearchError("CUDA batch candidate boundaries are not monotonic", 0, 0);
        results[i].candidate_matches = candidate_prefix[i + 1] - candidate_prefix[i];
    }
    if (candidate_count == 0) {
        finish();
        return results;
    }
    budget(bytes(candidate_count, sizeof(GpuTripletCandidate)), "candidate expansion");
    require(cuda_expand_triplet_neighbor_records_device(target.device_index, queries,
        device.neighbor_counts, device.neighbor_offsets, device.candidates), "candidate expansion");
    cuda_free_triplet_neighbor_offsets(device.neighbor_offsets);
    cuda_free_triplet_neighbor_counts(device.neighbor_counts);
    budget(bytes(candidate_count, sizeof(std::uint32_t)), "alignment counts");
    require(cuda_count_alignment_candidates_device(device.candidates, orders, device.alignment_counts), "alignment counts");
    budget(bytes(static_cast<std::size_t>(candidate_count) + 1, sizeof(std::uint32_t)), "alignment offsets");
    require(cuda_exclusive_prefix_sum_device(device.alignment_counts, device.alignment_offsets), "alignment scan");
    for (std::size_t i = 0; i < results.size(); ++i) {
        auto& segment = segments[i];
        require(cudaMemcpy(&segment.alignment_begin, device.alignment_offsets.offsets + candidate_prefix[i],
                           sizeof(std::uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess, "alignment boundary download");
        require(cudaMemcpy(&segment.alignment_end, device.alignment_offsets.offsets + candidate_prefix[i + 1],
                           sizeof(std::uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess, "alignment boundary download");
        if (segment.alignment_end < segment.alignment_begin)
            throw GpuLigandSearchError("CUDA batch alignment boundaries are not monotonic", 0, 0);
        results[i].alignments = segment.alignment_end - segment.alignment_begin;
        results[i].pipeline.input_alignment_count = static_cast<std::uint32_t>(results[i].alignments);
    }
    const auto alignment_count = index_count(segments.back().alignment_end, "Alignment count");
    if (alignment_count == 0) {
        finish();
        return results;
    }
    budget(bytes(alignment_count, sizeof(GpuAlignmentCandidate)), "alignment expansion");
    require(cuda_expand_alignment_candidates_device(device.candidates, orders, device.alignment_offsets,
                                                    device.alignments), "alignment expansion");
    cuda_free_alignment_offsets(device.alignment_offsets);
    cuda_free_alignment_counts(device.alignment_counts);
    cuda_free_triplet_candidates(device.candidates);
    budget(bytes(points.size(), sizeof(GpuPoint3)), "matching-point upload");
    require(cuda_upload_points(points, device.matching_points), "matching-point upload");
    budget(bytes(alignment_count, sizeof(GpuRigidTransform)), "quaternion alignment");
    require(cuda_compute_kabsch_transforms_device(device.matching_points, target.device_target_points,
                                                device.alignments, device.transforms), "quaternion alignment");
    cuda_free_alignment_candidates(device.alignments);
    cuda_free_points(device.matching_points);
    budget(bytes(atoms.size(), sizeof(GpuPoint3)), "atom upload");
    require(cuda_upload_points(atoms, device.atoms), "atom upload");
    budget(bytes(segments.size(), sizeof(GpuLigandBatchSegment)), "segment upload");
    require(cuda_upload_ligand_batch_segments(segments, device.segments), "segment upload");
    budget(checked_sum({bytes(alignment_count, sizeof(GpuClashScore)),
                      bytes(segments.size(), sizeof(std::uint32_t))}), "segmented clash");
    std::vector<std::uint32_t> valid_counts;
    require(cuda_score_ligand_batch_clash_device(device.atoms, target.device_grid, device.segments,
        device.transforms, device.clashes, valid_counts), "segmented clash");
    if (valid_counts.size() != results.size())
        throw GpuLigandSearchError("CUDA batch valid-transform count size differs", 0, 0);
    std::uint64_t expected_valid = 0;
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (valid_counts[i] > results[i].alignments)
            throw GpuLigandSearchError("CUDA batch valid count exceeds its alignment segment", 0, 0);
        results[i].pipeline.valid_transform_count = valid_counts[i];
        expected_valid += valid_counts[i];
    }
    const auto& scratch = target.compaction_workspace;
    const auto growth_bytes = alignment_count > scratch.capacity
        ? bytes(alignment_count, 2 * sizeof(std::uint32_t)) : 0;
    std::size_t scan_bytes = 0;
    require(cuda_ligand_batch_scan_storage_bytes(alignment_count, scan_bytes), "compaction storage query");
    const auto scan_growth_bytes = scan_bytes > scratch.scan_capacity_bytes ? scan_bytes : 0;
    // Growth temporarily owns old and replacement scratch. Conservatively allow
    // that peak together with an all-survivors output before allocating either.
    budget(checked_sum({bytes(alignment_count,
        sizeof(GpuRigidTransform) + sizeof(GpuClashScore) + sizeof(std::uint32_t)),
        growth_bytes, scan_growth_bytes,
        scratch.valid_count ? 0 : sizeof(std::uint32_t)}), "stable compaction worst case");
    std::uint32_t aggregate_valid = 0;
    require(cuda_compact_clash_survivors(device.transforms, device.clashes,
        config.operational_clash_threshold, device.survivors, aggregate_valid,
        &target.compaction_workspace), "stable compaction");
    budget(0, "retained compaction scratch");
    if (aggregate_valid != expected_valid)
        throw GpuLigandSearchError("CUDA batch valid-transform counts differ after compaction", 0, 0);
    cuda_free_clash_scores(device.clashes);
    cuda_free_rigid_transforms(device.transforms);
    cuda_free_points(device.atoms);
    if (device.survivors.count == 0) {
        finish();
        return results;
    }
    budget(bytes(areas.size(), sizeof(GpuFastMatchingArea)), "Fast-area upload");
    require(cuda_upload_fast_matching_areas(areas, device.areas), "Fast-area upload");
    budget(bytes(device.survivors.count, sizeof(GpuFastScore)), "segmented Fast");
    require(cuda_score_ligand_batch_fast_device(device.areas, target.device_grid, device.segments,
        device.survivors, device.fast_scores), "segmented Fast");
    budget(0, "completed Fast buffers");
    std::vector<GpuRigidTransform> transforms;
    std::vector<GpuClashScore> clashes;
    std::vector<std::uint32_t> alignment_ids;
    std::vector<GpuFastScore> fast;
    if (device_result) {
        alignment_ids.resize(device.survivors.count);
        require(cudaMemcpy(alignment_ids.data(), device.survivors.alignment_indices,
            bytes(alignment_ids.size(), sizeof(std::uint32_t)), cudaMemcpyDeviceToHost) == cudaSuccess,
            "survivor identity download");
    } else {
        require(cuda_download_clash_survivors(device.survivors, transforms, clashes, alignment_ids), "survivor download");
        require(cuda_download_fast_scores(device.fast_scores, fast), "Fast download");
        if (transforms.size() != alignment_ids.size() || clashes.size() != alignment_ids.size() ||
            fast.size() != alignment_ids.size())
            throw GpuLigandSearchError("CUDA batch survivor/Fast array sizes differ", 0, 0);
    }
    std::size_t ligand_index = 0;
    for (std::size_t i = 0; i < alignment_ids.size(); ++i) {
        const auto alignment = alignment_ids[i];
        if (i && alignment <= alignment_ids[i - 1])
            throw GpuLigandSearchError("CUDA batch compaction changed alignment order", 0, 0);
        while (ligand_index < segments.size() && alignment >= segments[ligand_index].alignment_end)
            ++ligand_index;
        if (ligand_index == segments.size() || alignment < segments[ligand_index].alignment_begin)
            throw GpuLigandSearchError("CUDA batch survivor has no ligand segment", 0, 0);
        auto& pipeline = results[ligand_index].pipeline;
        if (!device_result)
            pipeline.poses.push_back({alignment - segments[ligand_index].alignment_begin,
                                      transforms[i], clashes[i], fast[i]});
        ++pipeline.clash_survivor_count;
    }
    require(cudaDeviceSynchronize() == cudaSuccess, "batch completion");
    finish();
    return results;
}
