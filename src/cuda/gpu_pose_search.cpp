#include "gpu_pose_search.hpp"

#include "cuda_geometry.hpp"
#include "cuda_triplet_index.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void require_cuda(bool ok, const char* operation) {
    if (!ok) throw std::runtime_error(std::string("CUDA failed: ") + operation);
}

// The target owns the persistent protein data. Every allocation here belongs to
// this ligand and is released on success and on every exception path.
struct LigandDeviceBuffers {
    GpuPointArrayDevice matching_points{};
    GpuPointArrayDevice atoms{};
    GpuFastMatchingAreaArrayDevice fast_areas{};
    GpuTripletNeighborCountArrayDevice neighbor_counts{};
    GpuTripletNeighborOffsetArrayDevice neighbor_offsets{};
    GpuTripletCandidateArrayDevice candidates{};
    GpuAlignmentCountArrayDevice alignment_counts{};
    GpuAlignmentOffsetArrayDevice alignment_offsets{};
    GpuAlignmentCandidateArrayDevice alignments{};

    LigandDeviceBuffers() = default;
    LigandDeviceBuffers(const LigandDeviceBuffers&) = delete;
    LigandDeviceBuffers& operator=(const LigandDeviceBuffers&) = delete;
    ~LigandDeviceBuffers() {
        cuda_free_alignment_candidates(alignments);
        cuda_free_alignment_offsets(alignment_offsets);
        cuda_free_alignment_counts(alignment_counts);
        cuda_free_triplet_candidates(candidates);
        cuda_free_triplet_neighbor_offsets(neighbor_offsets);
        cuda_free_triplet_neighbor_counts(neighbor_counts);
        cuda_free_fast_matching_areas(fast_areas);
        cuda_free_points(atoms);
        cuda_free_points(matching_points);
    }
};
} // namespace

PreparedGpuTarget::PreparedGpuTarget(const RunConfig& config, double cutoff,
                                   std::optional<double> minimum_target_clearance,
                                   bool timings) {
    if (!std::isfinite(cutoff) || cutoff <= 0.0)
        throw std::invalid_argument("cutoff must be finite and positive");
    if (minimum_target_clearance &&
        (!std::isfinite(*minimum_target_clearance) || *minimum_target_clearance < 0.0))
        throw std::invalid_argument("Minimum target clearance must be finite and non-negative");
    try {
        const auto target_start = timings ? Clock::now() : Clock::time_point{};
        protein = prepare_protein(config, minimum_target_clearance);
        if (timings) target_prep_seconds = seconds_since(target_start);

        const auto index_start = timings ? Clock::now() : Clock::time_point{};
        const auto shared_hash = build_protein_triplet_hash_table(
            protein.targets, cutoff, config.triplet_bin_size);
        const auto flat = flatten_protein_triplet_index(shared_hash);
        index_records = flat.records.size();
        index_keys = flat.keys.size();
        const auto gpu_index = make_gpu_flat_triplet_index(flat);
        if (timings) index_seconds = seconds_since(index_start);

        const auto upload_start = timings ? Clock::now() : Clock::time_point{};
        require_cuda(cuda_upload_triplet_index(gpu_index, device_index),
                     "shared triplet index upload");
        require_cuda(cuda_upload_points(make_gpu_pose_target_points(protein.targets),
                                        device_target_points),
                     "protein target point upload");
        require_cuda(cuda_upload_protein_grid(make_gpu_protein_grid(protein.grid), device_grid),
                     "protein grid upload");
        require_cuda(cudaDeviceSynchronize() == cudaSuccess, "shared target synchronization");
        if (timings) gpu_upload_seconds = seconds_since(upload_start);
    } catch (...) {
        release();
        throw;
    }
}

PreparedGpuTarget::~PreparedGpuTarget() { release(); }

std::vector<RankedPoseDiagnostic> make_rescoring_pose_pool(
    const std::vector<GpuPipelinePose>& poses) {
    std::vector<RankedPoseDiagnostic> result;
    result.reserve(poses.size());
    for (const auto& gpu : poses) {
        if (!gpu.transform.valid || !gpu.clash.valid || !gpu.fast_score.valid)
            throw std::runtime_error("Invalid GPU survivor supplied to rescoring");
        RankedPoseDiagnostic pose;
        for (std::size_t i = 0; i < 9; ++i) pose.transform.rotation[i] = gpu.transform.rotation[i];
        for (std::size_t i = 0; i < 3; ++i) pose.transform.translation[i] = gpu.transform.translation[i];
        pose.occupied_atoms = gpu.clash.occupied_atoms;
        pose.clash_fraction = gpu.clash.fraction;
        pose.fast_weighted_score = gpu.fast_score.weighted_score;
        result.push_back(pose);
    }
    return result;
}

void PreparedGpuTarget::release() noexcept {
    cuda_free_protein_grid(device_grid);
    cuda_free_points(device_target_points);
    cuda_free_triplet_index(device_index);
}

GpuLigandSearchResult run_gpu_ligand_search(
    const RDKit::ROMol& molecule, const PreparedLigand& ligand,
    const PreparedGpuTarget& target, const RunConfig& config,
    StageProfile* stages, GpuPosePipelineTimings* download_timings,
    GpuPosePipelineDeviceResult* device_result) {
    GpuLigandSearchResult result;
    if (device_result) device_result->clear();
    std::size_t completed_valid_transforms = 0;
    std::size_t completed_survivors = 0;
    GpuPosePipelineTimings local_timings;
    GpuPosePipelineTimings* pipeline_timings = download_timings;
    if (stages && !pipeline_timings) pipeline_timings = &local_timings;
    if (pipeline_timings) {
        *pipeline_timings = GpuPosePipelineTimings{};
        pipeline_timings->collect_stages = stages != nullptr;
    }
    if (ligand.triplets.empty() || target.index_records == 0) return result;
    const auto gpu_stage = [&](const char* phase, std::size_t count, const char* scope,
                               auto&& operation) {
        if (!stages) return operation();
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        const auto start = Clock::now();
        const bool success = operation();
        const bool completed = cudaDeviceSynchronize() == cudaSuccess;
        if (success && completed)
            stages->push_back({phase, seconds_since(start) * 1000.0, count, scope});
        return success && completed;
    };
    try {
        LigandDeviceBuffers device;
        std::vector<GpuTripletKey> queries;
        std::vector<GpuLigandTripletOrders> orders;
        const auto query_start = stages ? Clock::now() : Clock::time_point{};
        make_gpu_ligand_triplet_queries(ligand.triplets, config.triplet_bin_size, queries, orders);
        if (stages)
            stages->push_back({"query_preparation", seconds_since(query_start) * 1000.0,
                               queries.size(), "observed_host_queries"});
        if (queries.size() != orders.size())
            throw std::runtime_error("GPU query/order mismatch");

        // Count/scan/expand reserves disjoint ranges, preserving query, neighbor,
        // bucket and equivalent-order traversal despite parallel execution.
        require_cuda(gpu_stage("lookup_count", queries.size(), "mixed_wrapper_internal_copies",
            [&] { return cuda_count_triplet_neighbor_records_device(
                target.device_index, queries, device.neighbor_counts); }), "neighbor count");
        require_cuda(gpu_stage("lookup_scan", queries.size(), "mixed_wrapper_internal_copies",
            [&] { return cuda_exclusive_triplet_neighbor_prefix_sum_device(
                device.neighbor_counts, device.neighbor_offsets); }), "neighbor scan");
        require_cuda(gpu_stage("matching_expansion", queries.size(), "mixed_wrapper_internal_copies",
            [&] { return cuda_expand_triplet_neighbor_records_device(
                target.device_index, queries, device.neighbor_counts, device.neighbor_offsets,
                device.candidates); }), "candidate expansion");
        result.candidate_matches = device.candidates.count;

        require_cuda(gpu_stage("alignment_count", result.candidate_matches, "mixed_wrapper_internal_copies",
            [&] { return cuda_count_alignment_candidates_device(
                device.candidates, orders, device.alignment_counts); }), "alignment count");
        require_cuda(gpu_stage("alignment_scan", result.candidate_matches, "mixed_wrapper_internal_copies",
            [&] { return cuda_exclusive_prefix_sum_device(
                device.alignment_counts, device.alignment_offsets); }), "alignment scan");
        require_cuda(gpu_stage("alignment_expansion", result.candidate_matches, "mixed_wrapper_internal_copies",
            [&] { return cuda_expand_alignment_candidates_device(
                device.candidates, orders, device.alignment_offsets, device.alignments); }),
            "alignment expansion");
        result.alignments = device.alignments.count;

        require_cuda(gpu_stage("matching_points_h2d", ligand.matching_areas.size(), "host_pack_alloc_h2d",
            [&] { return cuda_upload_points(make_gpu_ligand_matching_points(ligand.matching_areas),
                                            device.matching_points); }), "ligand matching point upload");
        require_cuda(gpu_stage("ligand_atoms_h2d", molecule.getNumAtoms(), "host_pack_alloc_h2d",
            [&] { return cuda_upload_points(make_gpu_ligand_atom_points(molecule), device.atoms); }),
            "ligand atom point upload");
        const FastScoreWeights weights{};
        require_cuda(gpu_stage("fast_areas_h2d", ligand.matching_areas.size(), "host_pack_alloc_h2d",
            [&] { return cuda_upload_fast_matching_areas(
                make_gpu_fast_matching_areas(ligand.matching_areas, weights), device.fast_areas); }),
            "ligand fast-area upload");

        require_cuda(cuda_run_pose_pipeline(device.matching_points, target.device_target_points,
                     device.atoms, device.fast_areas, target.device_grid, device.alignments,
                     config.operational_clash_threshold, result.pipeline, pipeline_timings, device_result),
                     "device-resident pose pipeline");
        if (stages)
            stages->insert(stages->end(), pipeline_timings->stages.begin(), pipeline_timings->stages.end());
        completed_valid_transforms = result.pipeline.valid_transform_count;
        completed_survivors = device_result ? device_result->survivors.count : result.pipeline.poses.size();
        if (completed_survivors != result.pipeline.clash_survivor_count ||
            result.alignments != result.pipeline.input_alignment_count)
            throw std::runtime_error("GPU output count mismatch");
        require_cuda(cudaDeviceSynchronize() == cudaSuccess, "end of ligand synchronization");
    } catch (const std::exception& exception) {
        throw GpuLigandSearchError(exception.what(), result.candidate_matches, result.alignments,
                                  completed_valid_transforms, completed_survivors);
    }
    return result;
}
